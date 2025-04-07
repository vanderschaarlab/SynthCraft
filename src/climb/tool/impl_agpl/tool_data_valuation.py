import os
import pickle
from typing import Any, Dict

import pandas as pd
from pydvl.utils.dataset import Dataset  # noqa: F401  # type: ignore
from pydvl.utils.utility import Utility  # noqa: F401  # type: ignore
from pydvl.value import compute_shapley_values  # noqa: F401  # type: ignore
from pydvl.value.shapley import ShapleyMode  # noqa: F401  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import Bunch

from climb.common.utils import raise_if_extra_not_available

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

raise_if_extra_not_available()


# TODO: Consider improving this function by running multiple times and taking mean shapely values as
# The number of records under the threshold is quite volatile.
def knn_shapley_valuation(
    tc: ToolCommunicator,
    data_file_path: str,
    target_variable: str,
    workspace: str,
) -> None:
    # General pre-processing
    df = pd.read_csv(data_file_path)
    df = clean_dataframe(df)

    # pydvl pre-processing
    data, X, y = preprocess_dataframe(df, target_variable)
    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Create the Utility object
    utility = Utility(knn, data)  # pyright: ignore

    tc.print("Running knn...")
    shapley_values = compute_shapley_values(utility, mode=ShapleyMode.KNN, progress=True)

    # NOTE: The thresholds here are heuristics.
    threshold = 0

    df_shapley = pd.DataFrame(
        {
            "value": shapley_values.values,
            "index": shapley_values.indices,
        }
    )
    # Filter Shapley values greater than 0
    positive_df = df_shapley[df_shapley["value"] >= threshold]

    # Filter Shapley values less than 0
    negative_df = df_shapley[df_shapley["value"] < threshold]

    # Save the results.
    tc.print("Saving the results...")
    results = {
        "shapley_values": shapley_values.values,
        "good_samples": positive_df["index"].values,
        "bad_samples": negative_df["index"].values,
        "df": df,
    }
    results_path = os.path.join(workspace, "knn_shapley_valuation_results.p")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # Load in all the data.
    tc.print("Preparing the plot...")

    tc.set_returns(
        tool_return=f"""
Results saved to: `{results_path}`.
This is a pickle file containing a dictionary with keys:
{{
"shapley_values": the `ValuationResult` object containing all the Shapley values,
"good_samples": numpy array with indices of good samples,
"bad_samples": numpy array with indices of bad samples,
"df": the DataFrame used for the analysis
}}
""",
    )


class KNNShapleyValuation(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_data_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        target_variable = kwargs["target_variable"]
        thrd, out_stream = execute_tool(
            knn_shapley_valuation,
            wd=self.working_directory,
            data_file_path=real_data_path,
            target_variable=target_variable,
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "knn_shapley_data_valuation"

    @property
    def description(self) -> str:
        return (
            "This is a data valuation tool. It tool uses the KNN algorithm to compute Shapley values for each feature in the dataset. "
            "The tool returns a list of features with positive Shapley values, which are considered good predictors, "
            "and a list of features with negative Shapley values, which are considered bad predictors."
            "The user may want to exclude the bad predictors from their model to improve its performance."
        )

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
                        "target_variable": {"type": "string", "description": "Name of the target variable."},
                    },
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "This is a data valuation tool. It uses the KNN algorithm to compute Shapley values for each feature in the dataset. "
            "The Shapley value of a feature is a measure of its importance in predicting the target variable. "
            "The tool returns a list of features with positive Shapley values, which are considered good predictors, "
            "and a list of features with negative Shapley values, which are considered bad predictors."
            "You may want to exclude the bad predictors from your model to improve its performance."
        )


def clean_dataframe(df, unique_threshold=15):
    # Identify column data types
    inferred_categorical_columns = []
    inferred_numerical_columns = []
    inferred_boolean_columns = []

    for col in df.columns:
        unique_values = df[col].dropna().unique()  # Drop NA to get unique values
        num_unique_values = len(unique_values)

        if df[col].dtype == "bool":
            inferred_boolean_columns.append(col)
        elif num_unique_values < unique_threshold or df[col].dtype == "object":
            inferred_categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            inferred_numerical_columns.append(col)
        else:
            # Handle mixed or unexpected data types
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                inferred_numerical_columns.append(col)
            except ValueError:
                inferred_categorical_columns.append(col)

    numerical_columns = [
        col
        for col in inferred_numerical_columns
        if col not in inferred_categorical_columns and col not in inferred_boolean_columns
    ]
    categorical_columns = inferred_categorical_columns
    boolean_columns = inferred_boolean_columns

    # Convert categorical columns to category indices
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes

    # Clean numerical columns
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Handle missing values - example: fill with the median
        df[col] = df[col].fillna(df[col].median())

    # Convert boolean columns to integers
    for col in boolean_columns:
        df[col] = df[col].astype(int)

    return df


def preprocess_dataframe(df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
    # Split the DataFrame into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Optionally, you can split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert the training data into a scikit-learn-like Bunch object
    sklearn_dataset = Bunch(data=X_train.values, target=y_train.values, feature_names=X_train.columns.tolist())

    # Create a PyDVL Dataset from the scikit-learn dataset
    pydvl_dataset = Dataset.from_sklearn(sklearn_dataset)

    return pydvl_dataset, X_test.values, y_test.values
