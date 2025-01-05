import os
from typing import Any, Dict, List, cast

from climb.common.data_structures import UploadedFileAbstraction

from ..tool_comms import ToolOutput, ToolReturnIter
from ..tools import ToolBase, UserInputRequest


class UploadDataFile(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        if self.user_input is None:
            raise ValueError("No user input obtained")
        if self.working_directory == "TO_BE_SET_BY_ENGINE":
            raise ValueError("Working directory not set")
        # for user_input in self.user_input:
        if self.user_input.kind == "file":
            uploaded_file = cast(UploadedFileAbstraction, self.user_input.received_input)
            with open(os.path.join(self.working_directory, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.content)
        else:
            raise NotImplementedError("Expected user input of kind 'file' but got something else.")
        tool_output = ToolOutput()
        tool_output.tool_return = f"File uploaded successfully: `{uploaded_file.name}`"
        return [tool_output]

    @property
    def name(self) -> str:
        return "upload_data_file"

    @property
    def description(self) -> str:
        return """
        The user will be prompted to upload a file via the UI. The uploaded file will then be placed in the \
        working directory. The name of the file will be returned as a string. Make use of this file name in further \
        steps of the workflow!
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
                    "properties": {},
                },
            },
        }

    @property
    def user_input_requested(self) -> List[UserInputRequest]:
        return [
            UserInputRequest(
                key="file",
                kind="file",
                description="Please upload your file",
                extra={"file_types": ["csv"], "multiple": False},
            ),
        ]

    @property
    def description_for_user(self) -> str:
        return "allows you to upload your data file in CSV format."


class UploadDataMultipleFiles(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        if self.user_input is None:
            raise ValueError("No user input obtained")
        if self.working_directory == "TO_BE_SET_BY_ENGINE":
            raise ValueError("Working directory not set")
        if self.user_input.kind == "multiple_files":
            uploaded_files = cast(List[UploadedFileAbstraction], self.user_input.received_input)
            for uploaded_file in uploaded_files:
                with open(os.path.join(self.working_directory, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.content)
        else:
            raise NotImplementedError("Expected user input of kind 'multiple_files' but got something else.")
        tool_output = ToolOutput()
        tool_output.tool_return = ""
        for uploaded_file in uploaded_files:
            tool_output.tool_return += f"File uploaded successfully: `{uploaded_file.name}`\n"
        return [tool_output]

    @property
    def name(self) -> str:
        return "upload_data_multiple_files"

    @property
    def description(self) -> str:
        return """
        The user will be prompted to upload file(s) (training and optionally test set files) via the UI. \
        The uploaded file will then be placed in the working directory. The name(s) of the file(s) will be returned as \
        a string. Make use of these file names in further steps of the workflow!
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
                    "properties": {},
                },
            },
        }

    @property
    def user_input_requested(self) -> List[UserInputRequest]:
        return [
            UserInputRequest(
                key="multiple_files",
                kind="multiple_files",
                description="Please upload your file(s)",
                extra={"file_types": ["csv"], "multiple": True},
            ),
        ]

    @property
    def description_for_user(self) -> str:
        return (
            "allows you to upload your data files (training dataset, and if you have it, a "
            "test/evaluation dataset) in CSV format."
        )
