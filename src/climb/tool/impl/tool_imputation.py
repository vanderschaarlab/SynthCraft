import os
from typing import Any, Dict, List, Optional

import pandas as pd
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.utils.serialization import save

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


def update_dataframe(df_orig: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    # Get the list of columns in df_new that exist in df_orig.
    common_columns = [col for col in df_new.columns if col in df_orig.columns]

    # Update df_orig with values from df_new for the common columns.
    df_out = df_orig.copy()
    df_out.update(df_new[common_columns])

    return df_out


class BasicHook:
    def __init__(self, iters: int) -> None:
        self.iters = iters
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.count += 1
        print(f"HyperImpute: Iterations completed {self.count}/{self.iters}")


def hyperimpute_impute(
    tc: ToolCommunicator,
    data_file_path: str,
    imputed_file_path: str,
    workspace: str,  # pylint: disable=unused-argument
    subset: Optional[List[str]] = None,
) -> None:
    df = pd.read_csv(data_file_path)
    df_use = df[subset] if subset else df
    if subset:
        tc.print("Imputing subset of columns:")
        for col in subset:
            tc.print(f"* {col}")
    else:
        tc.print("Imputing all columns")

    tc.print("Setting up HyperImpute Imputer...")
    n_inner_iter = 3
    imputer = Imputers().get(
        "hyperimpute",
        n_inner_iter=n_inner_iter,
        classifier_seed=[
            "random_forest",
            "logistic_regression",
            "catboost",
            # "xgboost",
            # NOTE: xgboost fails for e.g. CF dataset - fails with error related to lack of values of certain classes.
            # TODO: requires proper investigation and fixing.
        ],
        inner_loop_hook=BasicHook(n_inner_iter),
    )

    tc.print("Imputing data...")
    tc.print("Missing values before imputation:")
    tc.print(df_use.isnull().sum())
    n_missing_before = df_use.isnull().sum().sum()

    out = imputer.fit_transform(df_use.copy())
    if subset:
        out = update_dataframe(df, out)

    # Save the imputation model:
    imputed_file_basename = os.path.splitext(os.path.basename(imputed_file_path))[0]
    model_file_path = os.path.join(workspace, f"hyperimpute__{imputed_file_basename}.pkl")
    tc.print("Saving imputation model to:", model_file_path)
    bytes = save(imputer)
    with open(model_file_path, "wb") as f:
        f.write(bytes)

    tc.print("Missing values after imputation:")
    tc.print(out.isnull().sum())
    n_missing_after = out.isnull().sum().sum()

    tc.print("Saving imputed data...")
    out.to_csv(imputed_file_path, index=False)

    tc.set_returns(
        tool_return=(
            f"{n_missing_before - n_missing_after} missing values were imputed. "
            f"The imputed data has been saved to {imputed_file_path}"
        ),
        files_in=[os.path.basename(data_file_path)],
        files_out=[os.path.basename(imputed_file_path)],
    )


class HyperImputeImputation(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        real_imputed_path = os.path.join(self.working_directory, kwargs["imputed_file_path"])
        subset = kwargs.get("subset", None)
        thrd, out_stream = execute_tool(
            hyperimpute_impute,
            wd=self.working_directory,
            data_file_path=real_path,
            imputed_file_path=real_imputed_path,
            workspace=self.working_directory,
            subset=subset,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "hyperimpute_imputation"

    @property
    def description(self) -> str:
        return """
        Uses the **HyperImpute** library to automatically impute missing values in your data.
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "imputed_file_path": {
                            "type": "string",
                            "description": "Path to the imputed data file.",
                        },
                        "subset": {
                            "type": "array",
                            "description": "Optional subset of columns to impute. If not provided, all columns will be imputed.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["data_file_path", "imputed_file_path"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "uses the **HyperImpute** library to automatically impute missing values in your data."


# === TRAIN-TEST SPLIT VERSION (TODO: Consolidate) ===


def hyperimpute_impute_train_test(
    tc: ToolCommunicator,
    training_data_path: str,
    workspace: str,  # pylint: disable=unused-argument
    test_data_path: Optional[str],
    subset: Optional[List[str]] = None,
) -> None:
    def modify_basename_only(file_path: str, pattern: str) -> str:
        filename_no_extension = os.path.splitext(os.path.basename(file_path))[0]
        new_name = pattern.replace("<filename_no_extension>", filename_no_extension)
        return os.path.join(workspace, new_name)

    training_data_path = os.path.join(workspace, training_data_path)
    training_imputed_file_path = modify_basename_only(training_data_path, "<filename_no_extension>_imputed.csv")
    tc.print("Training data path:", training_data_path)
    tc.print("Imputed training data path to be used:", training_imputed_file_path)
    if test_data_path:
        test_data_path = os.path.join(workspace, test_data_path)
        test_imputed_file_path = modify_basename_only(test_data_path, "<filename_no_extension>_imputed.csv")
        tc.print("\nTest data path:", test_data_path)
        tc.print("Imputed test data path to be used:", test_imputed_file_path)

    df_train = pd.read_csv(training_data_path)
    df_train_use = df_train[subset] if subset else df_train
    if test_data_path:
        df_test = pd.read_csv(test_data_path)
        df_test_use = df_test[subset] if subset else df_test

    model_file_path = modify_basename_only(training_data_path, "hyperimpute__<filename_no_extension>.pkl")
    tc.print("\nHyperImpute model file path:", model_file_path)

    if subset:
        tc.print("\nImputing subset of columns:")
        for col in subset:
            tc.print(f"* {col}")
    else:
        tc.print("\nImputing all columns")

    tc.print("\nSetting up HyperImpute Imputer...")
    n_inner_iter = 3
    imputer = Imputers().get(
        "hyperimpute",
        n_inner_iter=n_inner_iter,
        classifier_seed=[
            "random_forest",
            "logistic_regression",
            "catboost",
            # "xgboost",
            # NOTE: xgboost fails for e.g. CF dataset - fails with error related to lack of values of certain classes.
            # TODO: requires proper investigation and fixing.
        ],
        inner_loop_hook=BasicHook(n_inner_iter),
    )

    tc.print("\nMissing values before imputation (training set):")
    tc.print(df_train_use.isnull().sum())
    n_missing_before = df_train_use.isnull().sum().sum()

    tc.print("\nTraining HyperImpute model...")

    out = imputer.fit_transform(df_train_use.copy())
    if subset:
        out = update_dataframe(df_train, out)

    # Save the imputation model:
    tc.print("Saving imputation model to:", model_file_path)
    bytes = save(imputer)
    with open(model_file_path, "wb") as f:
        f.write(bytes)

    tc.print("\nMissing values after imputation (training set):")
    tc.print(out.isnull().sum())
    n_missing_after = out.isnull().sum().sum()

    tc.print(f"\nSaving imputed data (training dataset)...\nSaving imputed data to: {training_imputed_file_path}")
    out.to_csv(training_imputed_file_path, index=False)

    if test_data_path:
        tc.print("\nImputing test data...")

        tc.print("\nMissing values before imputation (test set):")
        tc.print(df_test_use.isnull().sum())
        n_missing_before_test = df_test_use.isnull().sum().sum()

        out_test = imputer.transform(df_test_use.copy())
        if subset:
            out_test = update_dataframe(df_test, out_test)

        tc.print("\nMissing values after imputation (test set):")
        tc.print(out_test.isnull().sum())
        n_missing_after_test = out_test.isnull().sum().sum()

        tc.print(f"\nSaving imputed data (test dataset)...\nSaving imputed data to: {test_imputed_file_path}")
        out_test.to_csv(test_imputed_file_path, index=False)

    tc.set_returns(
        tool_return=(
            f"Training dataset:"
            f"{n_missing_before - n_missing_after} missing values were imputed. "
            f"The imputed data has been saved to {training_imputed_file_path}"
            + (
                f"\n\nTest dataset:"
                f"{n_missing_before_test - n_missing_after_test} missing values were imputed. "
                f"The imputed data has been saved to {test_imputed_file_path}"
                if test_data_path
                else ""
            )
        ),
        files_in=[os.path.basename(training_data_path)]
        + ([os.path.basename(test_data_path)] if test_data_path else []),
        files_out=[os.path.basename(training_imputed_file_path)]
        + ([os.path.basename(test_imputed_file_path)] if test_data_path else []),
    )


class HyperImputeImputationTrainTest(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        thrd, out_stream = execute_tool(
            hyperimpute_impute_train_test,
            wd=self.working_directory,
            training_data_path=kwargs["training_data_path"],
            test_data_path=kwargs.get("test_data_path", None),
            workspace=self.working_directory,
            subset=kwargs.get("subset", None),
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "hyperimpute_imputation_train_test"

    @property
    def description(self) -> str:
        return """
        Uses the **HyperImpute** library to automatically impute missing values in your data (fit on and transform the 
        training data, transform the test data). The imputed data is saved with `_imputed` appended to the filename(s).
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "training_data_path": {"type": "string", "description": "Path to the training data file."},
                        "test_data_path": {"type": "string", "description": "Optional path to the test data file."},
                        "subset": {
                            "type": "array",
                            "description": "Optional subset of columns to impute. If not provided, all columns will be imputed.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["training_data_path"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "uses the **HyperImpute** library to automatically impute missing values in your data."
