import os
from collections import Counter
from typing import Any, Dict

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


def compute_sampling_strategy(y, method, desired_ratio=1.0):
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]

    if method == "undersample":
        sampling_strategy = minority_count / (majority_count * desired_ratio)
    elif method == "oversample":
        sampling_strategy = {minority_class: int(majority_count * desired_ratio)}
    elif method == "combine":
        undersample_ratio = 0.5 * desired_ratio
        sampling_strategy = {"undersample": minority_count / (majority_count * undersample_ratio), "smote": "auto"}
    else:
        raise ValueError("Unsupported method. Choose from 'undersample', 'oversample', or 'combine'.")

    return sampling_strategy


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


def balance_data(
    tc: ToolCommunicator,
    data_file_path: str,
    balanced_data_file_path: str,
    target_column: str,
    method: str,
    sampling_strategy: str,
    desired_ratio: float,
    workspace: str,  # pylint: disable=unused-argument
) -> None:
    """balance_data

    Args:
        tc (ToolCommunicator): The tool communicator object.
        data_file_path (str): The path to the input CSV file.
        balanced_data_file_path (str): The path to the output CSV file with balanced data.
        method (str): The balancing method to use. Options are 'over' for oversampling, 'under' for \
            undersampling, and 'smote' for SMOTE.
        sampling_strategy (str): The sampling strategy to use. Options are:
            - 'minority' to balance the minority class,
            - 'not minority' to balance all classes except the minority class,
            - 'not majority' to balance all classes except the majority class,
            - 'all' to balance all classes, 
            - a float to specify the desired ratio of minority to majority samples.
            - a dict where the keys correspond to the targeted classes and the values correspond to \
                the desired number of samples for each targeted class.
        workspace (str): The workspace directory path.
    """
    # Load the data
    df = pd.read_csv(data_file_path)
    df = clean_dataframe(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if sampling_strategy is None:
        sampling_strategy = compute_sampling_strategy(y, method=method, desired_ratio=desired_ratio)

    if method == "smote":
        sampler = SMOTE(sampling_strategy=sampling_strategy)
    elif method == "oversample":
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy)
    elif method == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
    elif method == "combine":
        # First, apply under-sampling, then apply SMOTE
        X, y = RandomUnderSampler(sampling_strategy=0.5).fit_resample(X, y)
        sampler = SMOTE(sampling_strategy=sampling_strategy)
    else:
        raise ValueError("Invalid method. Choose from 'smote', 'oversample', 'undersample', or 'combine'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    tc.print(f"Balanced class distribution: {Counter(y_resampled)}")

    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled_df = pd.DataFrame(y_resampled, columns=[target_column])
    # Combine features and target into a single DataFrame
    balanced_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)

    balanced_df.to_csv(balanced_data_file_path, index=False)

    tc.set_returns(
        tool_return=(
            f"Dataset has been balanced." f"The new balanced dataset has been saved to {balanced_data_file_path}"
        ),
        files_in=[os.path.basename(data_file_path)],
        files_out=[os.path.basename(balanced_data_file_path)],
    )


class BalanceData(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        out_path = os.path.join(self.working_directory, kwargs["balanced_data_file_path"])
        thrd, out_stream = execute_tool(
            balance_data,
            wd=self.working_directory,
            data_file_path=real_path,
            balanced_data_file_path=out_path,
            target_column=kwargs["target_column"],
            method=kwargs["method"],
            sampling_strategy=kwargs["sampling_strategy"],
            desired_ratio=kwargs["desired_ratio"],
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "balance_data"

    @property
    def description(self) -> str:
        return """
        Uses the `balance_data` tool to rebalance target distribution of the data.
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
                        "balanced_data_file_path": {
                            "type": "string",
                            "description": "Path to the data file with extracted features, which this function creates.",
                        },
                        "method": {
                            "type": "string",
                            "description": " The balancing method to use. Options are 'over' for oversampling, 'under' for undersampling, and 'smote' for SMOTE.",
                        },
                        "target_column": {
                            "type": "string",
                            "description": "The target column to predict in the research task. For survival analysis this should be the event column.",
                        },
                        "sampling_strategy": {
                            "type": "string",
                            "description": """
The sampling strategy to use. Options are:
- 'minority' to balance the minority class,
- 'not minority' to balance all classes except the minority class,
- 'not majority' to balance all classes except the majority class,
- 'all' to balance all classes, 
- a float to specify the desired ratio of minority to majority samples.
- a dict where the keys correspond to the targeted classes and the values correspond to the desired number of samples for each targeted class.
""",
                        },
                        "desired_ratio": {
                            "type": "number",
                            "description": """
The desired ratio of minority to majority samples. Here is a brief description of the options:
1. Perfect Balance (desired_ratio = 1.0)
This should be used as the default option unless there is a specific reason to deviate. 
Disadvantages:
- May lead to overfitting in small datasets if oversampling is used excessively.
- Can reduce majority class information with aggressive undersampling.
2. Imbalanced Classes ( 1.5 < desired_ratio < 2.0)
When to Use:
- Preserving majority information: When the majority class has important patterns that might be lost with perfect balance.
- Preventing overfitting: When oversampling the minority class might lead to duplicates or overfitting.
- Natural imbalance: If the problem inherently has an imbalanced distribution (e.g., rare event prediction).
3. Severe Imbalance (desired_ratio > 2.0)
The majority class remains significantly larger than the minority class.
When to Use:
- Large datasets: When the dataset is large, the minority class can still have sufficient samples despite remaining imbalanced.
- Majority-dominated problems: When the majority class contains critical information that must be preserved.
- Extreme imbalance: In cases like fraud detection or medical diagnosis, where the minority class is inherently rare.
""",
                        },
                    },
                    "required": [
                        "data_file_path",
                        "balanced_data_file_path",
                        "method",
                        "sampling_strategy",
                        "desired_ratio",
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Uses the `balance_data` tool to rebalance target distribution of the data."
