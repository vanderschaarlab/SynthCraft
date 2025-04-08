import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from autoprognosis.utils.serialization import load_model_from_file

# Import MAPIE classes
from mapie.classification import MapieClassifier
from mapie.regression import MapieRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


# Convert prediction sets to a user-friendly string.
def to_hashable(s):
    # Convert each element to a tuple if needed, then sort and return as tuple.
    return tuple(sorted(tuple(x) if isinstance(x, list) else x for x in s))


# ----- ModelWrapper Definition -----
class ModelWrapper(BaseEstimator):
    """
    A lightweight wrapper that delegates fit, predict, and predict_proba
    to the underlying model. It stores the classifier's classes_ attribute explicitly,
    so that cloning in prefit mode preserves it.

    For binary classification, if the underlying model returns a one-column probability array,
    it converts it into a two-column array.
    """

    def __init__(self, model, fitted: bool = True, classes_: Optional[np.ndarray] = None):
        self.model = model
        self.fitted_ = fitted
        if fitted and hasattr(model, "classes_"):
            self.classes_ = model.classes_
        else:
            self.classes_ = classes_

    def get_params(self, deep: bool = True):
        return {"model": self.model, "fitted": self.fitted_, "classes_": self.classes_}

    def set_params(self, **params):
        if "model" in params:
            self.model = params["model"]
        if "fitted" in params:
            self.fitted_ = params["fitted"]
        if "classes_" in params:
            self.classes_ = params["classes_"]
        return self

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted_ = True
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        else:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        if hasattr(proba, "values"):
            proba = proba.values
        if proba.ndim == 2 and proba.shape[1] == 1:
            proba = np.hstack([1 - proba, proba])
        return proba


# ----- SurvivalToClassificationWrapper Definition -----
class SurvivalToClassificationWrapper(ModelWrapper):
    """
    A wrapper that converts a survival model (e.g. a RiskEnsemble) into a binary classifier
    at a chosen time horizon T0. It inherits from ModelWrapper so that prefit functionality and
    parameter handling are preserved.

    It implements:
      - predict_survival_function: If the underlying model has that method, it uses it.
        Otherwise, it calls the risk ensemble's predict with eval_time_horizons=[T0] to obtain a risk score,
        then approximates the survival probability as S(T0) = exp(-risk_score).
      - predict_proba: Evaluates survival probability at T0 and returns a two-column array:
           Column 0: S(T0) (i.e. probability that the event has NOT occurred by T0)
           Column 1: 1 - S(T0) (i.e. probability that the event has occurred by T0)
      - predict: Returns argmax of predict_proba.
      - classes_: Always returns np.array([0, 1]).
    """

    def __init__(self, survival_model, T0, fitted: bool = True):
        super().__init__(survival_model, fitted=fitted, classes_=np.array([0, 1]))
        self.T0 = T0

    def predict_survival_function(self, X):
        if hasattr(self.model, "predict_survival_function"):
            return self.model.predict_survival_function(X)
        else:
            # Use the risk ensemble's predict method with eval_time_horizons = [T0]
            risk_pred_df = self.model.predict(X, eval_time_horizons=[self.T0])
            risk_scores = risk_pred_df.iloc[:, 0].values
            S_T0 = np.exp(-risk_scores)
            return [lambda T, s=s: s for s in S_T0]

    def predict_proba(self, X):
        surv_funcs = self.predict_survival_function(X)
        S_T0 = np.array([fn(self.T0) for fn in surv_funcs])
        proba = np.column_stack((S_T0, 1 - S_T0))
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def set_to_string(s):
    """
    Given a prediction set s (which may be nested), flatten it,
    remove duplicates while preserving order, and return a string
    of values separated by commas.
    """
    flat = np.array(s).flatten().tolist()
    unique = []
    for x in flat:
        if x not in unique:
            unique.append(x)
    return ", ".join(str(x) for x in unique)


# ----- Conformal Prediction Function -----
def conformal_prediction_function(
    tc: ToolCommunicator,
    model_file_path: str,
    train_data_file_path: str,
    test_data_file_path: Optional[str],
    task_type: str,
    target_column: str,
    workspace: str,
    alpha: float = 0.1,
    time_to_event_column: Optional[str] = None,
) -> None:
    """
    Apply conformal prediction on a pre-trained model.

    Parameters:
        model_file_path (str): Path to the pre-trained model file.
        train_data_file_path (str): Path to the CSV file with training data.
        test_data_file_path (str, optional): Path to the CSV file with test data. If not provided,
            the function creates a test split from the training data.
        task_type (str): One of 'classification', 'regression', or 'survival'.
        target_column (str): Name of the target variable column.
        workspace (str): Path to the workspace directory.
        alpha (float): Miscoverage level (e.g., 0.1 for 90% coverage).
        time_to_event_column (str, optional): Name of the time-to-event column (required for survival).

    Returns:
        For classification and survival: DataFrame with a "Predictions_in_conf_interval" column listing the classes included.
        For regression: DataFrame with "lower_bound" and "upper_bound" columns.

    Raises:
        ValueError: if the train and test data do not share the same feature columns.
    """
    workspace = Path(workspace)  # pyright: ignore
    df_train = pd.read_csv(train_data_file_path)

    # If no test file is provided, split the training data into train/test (80/20 split)
    if not test_data_file_path or test_data_file_path.strip() == "":
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)
    else:
        df_test = pd.read_csv(test_data_file_path)

    # Determine expected feature columns based on task type.
    if task_type.lower() in ["classification", "regression"]:
        expected_features = set(df_train.columns) - {target_column}
    elif task_type.lower() == "survival":
        if time_to_event_column is None:
            raise ValueError("For survival tasks, time_to_event_column must be provided.")
        expected_features = set(df_train.columns) - {target_column, time_to_event_column}
    else:
        raise ValueError("Task type not recognized. Choose 'classification', 'regression', or 'survival'.")

    # Check that test data contains the same features.
    test_features = set(df_test.columns)
    # Remove extra columns (e.g., if a binary target was added later) by intersecting with expected_features.
    test_features = test_features & expected_features
    if expected_features != test_features:
        raise ValueError("Training and testing data do not share the same feature columns.")

    tc.print("Training the conformal prediction model...")
    if task_type.lower() == "classification":
        model = load_model_from_file(Path(model_file_path))
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column])
        classes = np.unique(y_train)
        wrapped_model = ModelWrapper(model, fitted=True, classes_=classes)

        clf = MapieClassifier(estimator=wrapped_model, cv="prefit", method="score")  # pyright: ignore
        clf.fit(X_train, y_train)
        _, prediction_sets = clf.predict(X_test, alpha=alpha)
        output = pd.DataFrame(
            {"Predictions_in_conf_interval": [set_to_string(ps.tolist()) for ps in prediction_sets]},
            index=df_test.index,
        )

        # Concatenate X_test with the prediction sets.
        result_df = pd.concat([X_test.reset_index(drop=True), output.reset_index(drop=True)], axis=1)
        # Convert the result DataFrame to a formatted string.
        result_str = result_df.to_csv(sep="\t", index=True)
        tc.print("Conformal prediction completed.")
        tc.print(output)
        tc.set_returns(
            tool_return=(
                "The conformal prediction model has been trained and predictions have been made on the test set."
            ),
            user_report=[
                "**Conformal Prediction:**\n",
                result_str,
            ],
        )

    elif task_type.lower() == "regression":
        model = load_model_from_file(Path(model_file_path))
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column])
        wrapped_model = ModelWrapper(model, fitted=True)

        reg = MapieRegressor(estimator=wrapped_model, cv="prefit", method="plus")  # pyright: ignore
        reg.fit(X_train, y_train)
        _, prediction_intervals = reg.predict(X_test, alpha=alpha)
        prediction_intervals = np.squeeze(prediction_intervals, axis=-1)
        output = pd.DataFrame(prediction_intervals, columns=["lower_bound", "upper_bound"], index=df_test.index)
        # Concatenate X_test with the prediction sets.
        result_df = pd.concat([X_test.reset_index(drop=True), output.reset_index(drop=True)], axis=1)
        # Convert the result DataFrame to a formatted string.
        result_str = result_df.to_string(index=False)
        tc.print("Conformal prediction completed.")
        tc.print(output)
        tc.set_returns(
            tool_return=(
                "The conformal prediction model has been trained and predictions have been made on the test set."
            ),
            user_report=[
                "**Conformal Prediction:**\n",
                result_str,
            ],
        )

    elif task_type.lower() == "survival":
        model = load_model_from_file(Path(model_file_path))
        # Determine features: all columns except target and time_to_event.
        feature_cols = [col for col in df_train.columns if col not in {target_column, time_to_event_column}]
        X_train = df_train[feature_cols]
        df_train["binary_event"] = (
            df_train[time_to_event_column] <= df_train[df_train[target_column] == 1][time_to_event_column].median()
        ).astype(int)
        y_train_binary = df_train["binary_event"]
        X_test = df_test[feature_cols]

        # Choose T0 as the median time among subjects with an event.
        T0 = df_train[df_train[target_column] == 1][time_to_event_column].median()
        wrapped_model = SurvivalToClassificationWrapper(model, T0, fitted=True)

        clf = MapieClassifier(estimator=wrapped_model, cv="prefit", method="score")  # pyright: ignore
        clf.fit(X_train, y_train_binary)
        _, prediction_sets = clf.predict(X_test, alpha=alpha)
        output = pd.DataFrame(
            {"Predictions_in_conf_interval": [set_to_string(ps.tolist()) for ps in prediction_sets]},
            index=df_test.index,
        )
        # Concatenate X_test with the prediction sets.
        result_df = pd.concat([X_test.reset_index(drop=True), output.reset_index(drop=True)], axis=1)
        # Convert the result DataFrame to a formatted string.
        result_str = result_df.to_csv(sep="\t", index=True)
        tc.print("Conformal prediction completed.")
        tc.print(output)
        tc.set_returns(
            tool_return=(
                "The conformal prediction model has been trained and predictions have been made on the test set."
            ),
            user_report=[
                "**Conformal Prediction:**\n",
                result_str,
            ],
        )

    else:
        raise ValueError("Task type not recognized. Choose 'classification', 'regression', or 'survival'.")


# ----- Tool Class Definition -----
class ConformalPrediction(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        model_file_path = os.path.join(self.working_directory, kwargs["model_file_path"])
        train_data_file_path = os.path.join(self.working_directory, kwargs["train_data_file_path"])
        test_data_file_path = os.path.join(self.working_directory, kwargs["test_data_file_path"])
        task_type = kwargs["task_type"]
        target_column = kwargs["target_column"]
        workspace = self.working_directory  # using working_directory as workspace
        time_to_event_column = kwargs.get("time_to_event_column")
        alpha = kwargs.get("alpha", 0.1)

        thrd, out_stream = execute_tool(
            conformal_prediction_function,
            model_file_path=model_file_path,
            train_data_file_path=train_data_file_path,
            test_data_file_path=test_data_file_path,
            task_type=task_type,
            target_column=target_column,
            workspace=workspace,
            alpha=alpha,
            time_to_event_column=time_to_event_column,
            # ---
            wd=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "conformal_prediction"

    @property
    def description(self) -> str:
        return """
        Applies conformal prediction on a pre-trained model (provided via model_file_path) using 
        training and test CSV files. For classification and survival tasks, returns a DataFrame 
        with a 'prediction_set' column that lists the classes included in the confidence interval.
        For regression, returns a DataFrame with 'lower_bound' and 'upper_bound' columns.
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
                        "model_file_path": {"type": "string", "description": "Path to the pre-trained model file."},
                        "train_data_file_path": {
                            "type": "string",
                            "description": "Path to the training data CSV file.",
                        },
                        "test_data_file_path": {"type": "string", "description": "Path to the test data CSV file."},
                        "task_type": {
                            "type": "string",
                            "description": "The type of task: 'classification', 'regression', or 'survival'.",
                        },
                        "target_column": {"type": "string", "description": "Name of the target variable column."},
                        "time_to_event_column": {
                            "type": "string",
                            "description": "Name of the time-to-event column (required for survival tasks).",
                        },
                        "alpha": {"type": "number", "description": "Miscoverage level (e.g., 0.1 for 90% coverage)."},
                    },
                    "required": [
                        "model_file_path",
                        "train_data_file_path",
                        "test_data_file_path",
                        "task_type",
                        "target_column",
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Applies conformal prediction on a pre-trained model and returns prediction intervals or sets."
