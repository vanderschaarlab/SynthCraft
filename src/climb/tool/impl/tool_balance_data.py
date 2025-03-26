import os
from collections import Counter
from typing import Any, Dict, Optional, Union

import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

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


def clean_dataframe(df: pd.DataFrame, unique_threshold: int = 15):
    """
    Cleans the dataframe by encoding categorical variables, handling missing values, and converting data types.

    Parameters:
    - df (pd.DataFrame): The input dataframe to clean.
    - unique_threshold (int): Threshold to decide if a numerical column should be treated as categorical.

    Returns:
    - df_cleaned (pd.DataFrame): The cleaned dataframe.
    - encoders (dict): Dictionary of LabelEncoders for categorical columns.
    """
    # Initialize encoders
    encoders = {}

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
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().all():
                inferred_categorical_columns.append(col)
            else:
                inferred_numerical_columns.append(col)

    numerical_columns = [
        col
        for col in inferred_numerical_columns
        if col not in inferred_categorical_columns and col not in inferred_boolean_columns
    ]
    categorical_columns = inferred_categorical_columns
    boolean_columns = inferred_boolean_columns

    # Convert categorical columns using LabelEncoder
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("Missing"))
        encoders[col] = le

    # Clean numerical columns
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Handle missing values - fill with the median
        df[col] = df[col].fillna(df[col].median())

    # Convert boolean columns to integers
    for col in boolean_columns:
        df[col] = df[col].astype(int)

    return df, encoders


def balance_data(
    tc: ToolCommunicator,
    data_file_path: str,
    balanced_data_file_path: str,
    target_column: str,
    method: str,
    sampling_strategy: Optional[Union[str, float, Dict]],
    desired_ratio: float,
    workspace: str,  # pylint: disable=unused-argument
) -> None:
    """balance_data

    Args:
        tc (ToolCommunicator): The tool communicator object.
        data_file_path (str): The path to the input CSV file.
        balanced_data_file_path (str): The path to the output CSV file with balanced data.
        method (str): The balancing method to use. Options are 'over' for oversampling, 'under' for \
            undersampling, 'smote' for SMOTE, and 'combine' for combining under-sampling and SMOTE.
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

    df_original = df.copy()

    # Clean a separate copy of the data for re-balancing
    df_cleaned, encoders = clean_dataframe(df.copy())

    X_cleaned = df_cleaned.drop(columns=[target_column])
    y_cleaned = df_cleaned[target_column]

    if sampling_strategy is None:
        sampling_strategy = compute_sampling_strategy(y_cleaned, method=method, desired_ratio=desired_ratio)
        tc.print(f"Computed sampling strategy: {sampling_strategy}")
    else:
        tc.print(f"Using provided sampling strategy: {sampling_strategy}")

    # Identify categorical feature indices for SMOTENC
    categorical_features = [
        i for i, col in enumerate(X_cleaned.columns) if col in encoders
    ]  # Identify categorical feature indices for SMOTENC

    if method == "smote":
        sampler = SMOTENC(
            sampling_strategy=sampling_strategy,
            categorical_features=categorical_features,
            random_state=42,
        )
    elif method == "oversample":
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy)
    elif method == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
    elif method == "combine":
        # First, apply under-sampling, then apply SMOTE
        undersampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy.get("undersample", 0.5),
            random_state=42,
        )
        sampler = SMOTENC(
            sampling_strategy=sampling_strategy.get("smote", "auto"),
            categorical_features=categorical_features,
            random_state=42,
        )
    else:
        raise ValueError("Invalid method. Choose from 'smote', 'oversample', 'undersample', or 'combine'.")

    # Apply the sampler on the cleaned data
    tc.print("Applying the re-balancing algorithm...")
    if method == "combine":
        # First apply undersampling, then SMOTENC
        X_under, y_under = undersampler.fit_resample(X_cleaned, y_cleaned)
        X_resampled, y_resampled = sampler.fit_resample(X_under, y_under)
    else:
        X_resampled, y_resampled = sampler.fit_resample(X_cleaned, y_cleaned)

    tc.print(f"Balanced class distribution: {Counter(y_resampled)}")

    # Initialize DataFrame for balanced data
    df_balanced = pd.DataFrame(X_resampled, columns=X_cleaned.columns)
    df_balanced[target_column] = y_resampled

    # Inverse transform categorical columns
    if method in ["smote", "combine"]:
        # Calculate number of synthetic samples
        num_original = len(X_cleaned)
        num_resampled = len(X_resampled)
        num_synthetic = num_resampled - num_original

        if num_synthetic > 0:
            # Extract synthetic samples
            synthetic_X = X_resampled[-num_synthetic:]
            synthetic_y = y_resampled[-num_synthetic:]

            # Create DataFrame for synthetic samples
            synthetic_df = pd.DataFrame(synthetic_X, columns=X_cleaned.columns)
            synthetic_df[target_column] = synthetic_y

            # Inverse transform categorical columns in synthetic samples
            for col, le in encoders.items():
                # Ensure synthetic samples have integer values for categorical columns
                synthetic_df[col] = synthetic_df[col].round().astype(int)
                # Handle potential out-of-range values by clipping
                synthetic_df[col] = synthetic_df[col].clip(0, len(le.classes_) - 1)
                synthetic_df[col] = le.inverse_transform(synthetic_df[col])

            # Append synthetic samples to the original dataset
            df_balanced_final = pd.concat(
                [df_original, synthetic_df], ignore_index=True
            )  # Concatenate synthetic samples with original data
        else:
            # No synthetic samples were generated
            tc.print("No synthetic samples were generated.")
            df_balanced_final = df_original.copy()
    else:
        # For 'oversample' and 'undersample', inverse transform categorical columns
        for col, le in encoders.items():  # Inverse transform categorical features
            df_balanced[col] = le.inverse_transform(df_balanced[col].astype(int))

        # Set balanced data as resampled data
        df_balanced_final = df_balanced

    tc.print("Saving the balanced data...")
    df_balanced_final.to_csv(balanced_data_file_path, index=False)

    # Log the results
    tc.set_returns(
        tool_return=(
            f"Dataset has been balanced using '{method}' method. "
            f"The balanced dataset has been saved to {balanced_data_file_path}."
        ),
        user_report=[
            "📊 **Data Balancing**",
            f"Resampled class distribution: {Counter(y_resampled)}",
            f"Balanced data saved to: {balanced_data_file_path}",
        ],
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
                        "target_column",
                        "sampling_strategy",
                        "desired_ratio",
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Uses the `balance_data` tool to rebalance target distribution of the data."
