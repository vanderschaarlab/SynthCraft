from typing import Any, Dict

# synthcity
from synthcity.plugins import Plugins

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

def list_available_generators(
    tc: ToolCommunicator,
    use_pravacy_plugins: bool = False,
) -> None:
    """list_available_generators

    Args:
        pravacy_plugins (bool, optional): Boolean value to indicate if the user is interested in privacy preserving synthetic data. Defaults to False.
    """
    plugins = Plugins(categories=["generic"]).list()

    if not use_pravacy_plugins:
        tc.set_returns(
            tool_return=(
                f"The list of available plugins is: {plugins}."
            ),
            user_report=[
                "**Avalable plugins**",
                f"The list of available plugins is: {plugins}. ",
            ],
        )
    else:
        privacy_plugins = Plugins(categories=["privacy"]).list()
    
        # Log the results
        tc.set_returns(
            tool_return=(
                f"The list of available general purpose plugins is: {plugins}. "
                f"The list of available privacy plugins is: {privacy_plugins}."
            ),
            user_report=[
                "**Avalable privacy plugins**",
                f"The list of available general purpose plugins is: {plugins}. ",
                f"The list of available privacy plugins is: {privacy_plugins}. ",
            ],
        )



class ListGenerators(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        thrd, out_stream = execute_tool(
            list_available_generators,
            wd=self.working_directory,
            # ---
            use_pravacy_plugins=kwargs.get("use_pravacy_plugins", False),
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "list_available_generators"

    @property
    def description(self) -> str:
        return """
        Uses the `list_available_generators` tool to list the available generators.
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
                    "properties": {"use_pravacy_plugins": {"type": "boolean", "description": "Boolean value to indicate if the user is interested in privacy preserving synthetic data. Defaults to False."},},
                    "required": ["use_pravacy_plugins"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Uses the `list_available_generators` tool to list the available generators."
