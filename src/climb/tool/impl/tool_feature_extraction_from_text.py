import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm  # Optional: For progress visualization

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


def feature_extraction_from_text(
    tc: ToolCommunicator,
    data_file_path: str,
    extracted_data_file_path: str,
    topics_dict: str,
    workspace: str,  # pylint: disable=unused-argument
) -> None:
    """
    Extract specified categorical topics from free-text fields in a pandas DataFrame.

    Parameters:
    - data_file_path (str): Path to the input CSV file.
    - extracted_data_file_path (str): Path to the output CSV file with extracted features.
    - topics_dict (str): A nested dictionary where keys are free-text column names,
                          and values are dictionaries mapping topics to their synonyms.
                          e.g.
                          topics_dict = {
                            "column1": {
                              "topic1": ["synonym1", "synonym2"],
                              "topic2": ["synonym3", "synonym4"]
                            },
                            "column2": {
                              "topic1": ["synonym1", "synonym2"],
                              "topic3": ["synonym5", "synonym6"]
                            },
                          }
    - workspace (str): The path to the workspace directory.
    """
    # Load spaCy model with disabled components for speed
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        # If the model is not found, download it
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Load the data
    workspace = Path(workspace)
    data_file_path = workspace / data_file_path
    extracted_data_file_path = workspace / extracted_data_file_path
    df = pd.read_csv(data_file_path)
    df = clean_dataframe(df)

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Initialize the Matcher
    matcher = Matcher(nlp.vocab)

    # Dictionary to map matcher IDs to topic names
    matcher_id_to_topic = {}

    # Convert topics_dict from JSON string to Python dictionary
    topics_dict = json.loads(topics_dict)
    tc.print(f"Extracting topics from free text fields in the DataFrame using these concepts: \n{topics_dict}")

    # Initialize a dictionary to count the number of matches per field
    field_match_count = {field: 0 for field in topics_dict.keys()}

    # Iterate through each specified field in topics_dict
    for field, topics in topics_dict.items():
        tc.print(f"\nProcessing field: '{field}'")

        # Check if the field exists in the DataFrame
        if field not in df.columns:
            tc.print(f"Warning: Field '{field}' not found in DataFrame.")
            continue

        # Initialize new columns for each topic
        for topic, synonyms in topics.items():
            sanitized_topic = topic.replace(" ", "_")
            column_name = f"{field}_{sanitized_topic}"
            df[column_name] = 0  # Binary indicator

            # Create patterns based on lemmas and lowercase synonyms
            for synonym in synonyms:
                doc = nlp(synonym.lower())
                pattern = []
                for token in doc:
                    pattern.append({"LEMMA": token.lemma_})
                matcher.add(sanitized_topic, [pattern])
                matcher_id_to_topic[matcher.vocab.strings[sanitized_topic]] = sanitized_topic

        # Process texts with tqdm for progress visualization
        tc.print(f"\nExtracting topics from {field}...")
        for idx, text in tqdm(df[field].items(), desc="Processing texts"):
            # Ensure the text is a string
            if not isinstance(text, str):
                text = str(text)

            # Process the text with spaCy
            doc = nlp(text)

            # Find matches in the text
            matches = matcher(doc)

            # Set the corresponding topic columns to 1 if any synonym is found
            for match_id, start, end in matches:
                string_id = nlp.vocab.strings[match_id]  # Get string representation
                column_name = f"{field}_{string_id}"
                df.at[idx, column_name] = 1  # Mark presence

            # Increment the match count for the field
            field_match_count[field] += len(matches)

    # Drop the original text fields
    df.drop(columns=[field for field in topics_dict.keys()], inplace=True)

    df.to_csv(extracted_data_file_path, index=False)

    match_count_string = "\n".join(
        [f"Number of matches found in '{field}': {count}" for field, count in field_match_count.items()]
    )

    tc.set_returns(
        tool_return=(
            f"Features extracted from free text."
            f"\n\n{match_count_string}\n\n"
            f"The new dataset with extracted features has been saved to {extracted_data_file_path}"
        ),
        files_in=[os.path.basename(data_file_path)],
        files_out=[os.path.basename(extracted_data_file_path)],
    )


class FeatureExtractionFromText(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        out_path = os.path.join(self.working_directory, kwargs["extracted_data_file_path"])
        thrd, out_stream = execute_tool(
            feature_extraction_from_text,
            wd=self.working_directory,
            data_file_path=real_path,
            extracted_data_file_path=out_path,
            topics_dict=kwargs["topics_dict"],
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "feature_extraction_from_text"

    @property
    def description(self) -> str:
        return """
        Uses the `feature_extraction_from_text` tool to extract the features from free text fields.
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
                        "extracted_data_file_path": {
                            "type": "string",
                            "description": "Path to the data file with extracted features, which this function creates.",
                        },
                        "topics_dict": {
                            "type": "string",
                            "description": """
A json formatted string structured as a nested dictionary where keys are free-text column names, and values are dictionaries mapping topics to their synonyms. The \
synonyms should be a list of the top ten words associated with the topic. The dictionary should be structured as follows,

```json
{
    "column1": {
        "topic1": ["synonym1", "synonym2", ...],
        "topic2": ["synonym3", "synonym4", ...]
    },
    "column2": {
        "topic1": ["synonym1", "synonym2", ...],
        "topic3": ["synonym5", "synonym6", ...]
    },
}
```
""",
                        },
                    },
                    "required": ["data_file_path", "extracted_data_file_path", "topics_dict"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Uses an LLM to extract the features from free text fields."


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
