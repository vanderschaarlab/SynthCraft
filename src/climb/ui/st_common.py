import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from dotenv import dotenv_values
from streamlit.components.v1 import html
from streamlit_js import st_js_blocking

from climb.common.utils import replace_str_from_dict, ui_log
from climb.db.tinydb_db import DB
from climb.engine import ENGINE_MAP

PAGE_TITLES = {
    "main_emoji": "⚕️ SynthCraft",
    "main_plain": "SynthCraft",
    "research_management_emoji": "🗨️ Research Management",
    "research_management_plain": "Research Management",
    "settings_emoji": "⚙️ Settings",
    "settings_plain": "Settings",
}

TITLE = PAGE_TITLES["main_plain"]
VERSION = "alpha"

SHOW_ROLES = ["user", "assistant", "tool", "code_execution"]  # "system"
SHOW_VISIBILITIES = ["all", "ui_only"]

COMMON_SETTINGS = {
    "display": {
        "show_tool_call_logs": {
            "name": "Show **tool** logs",
            "help": (
                "Whether to show the execution logs (e.g. progress percentage) generated as the "
                f"**tools** available to *{TITLE}* are being executed. "
                "If this is unchecked, you can still expand the tool call logs section to view the logs."
            ),
        },
        "show_tool_call_return": {
            "name": "Show **tool** output",
            "help": (
                "Whether to show the final output (e.g. a dictionary of final evaluation metrics) generated as the "
                f"**tools** available to *{TITLE}* are being executed."
                "If this is unchecked, you can still expand the tool output section to view the output."
            ),
        },
        "show_code": {
            "name": "Show **generated code**",
            "help": (
                f"Whether to show the **code** dynamically generated by *{TITLE}*. "
                "If this is unchecked, you can still expand the generated code section to view the code."
            ),
        },
        "show_code_out": {
            "name": "Show **generated code** logs",
            "help": (
                "Whether to show the output (including any errors) produced as the **code** dynamically generated by "
                f"*{TITLE}* is being executed."
                "If this is unchecked, you can still expand the generated code logs section to view the logs."
            ),
        },
        "show_planning_details": {
            "name": "Show **planning details**",
            "help": (
                "Whether to show the details of the planning process (e.g. the steps taken to generate a plan) "
                f"as *{TITLE}* is planning."
            ),
        },
        "show_full_message_history": {
            "name": "Show full message history",
            "help": ("Whether to show the full message history in the chat section."),
        },
        "show_message_history_length": {
            "name": "Show history length",
            "help": (
                "The number of messages to show in the chat section. If 'Show full message history' is checked, "
                "will show all messages regardless."
            ),
            "min_value": 1,
            "max_value": 9999,
            "step": 1,
        },
    }
}


def kill_executing_tool():
    # Kill an active tool thread if it exists, in order to free up resources and prevent problems with stdout/stderr,
    # exceptions, etc. (which can be caused by the tool thread being active).
    if "engine" in st.session_state:
        st.session_state.engine.stop_tool_execution()


def run_global_markdown_css_hack() -> None:
    SIDEBAR_WIDTH_PX = 224

    REPLACE_IN_STYLE_STR = {
        "{SIDEBAR_WIDTH_PX}": SIDEBAR_WIDTH_PX,
    }

    style_str = """
    <style>
    section[data-testid="stSidebar"] {
        width: {SIDEBAR_WIDTH_PX}px !important;
    }
    </style>
    """

    style_str = replace_str_from_dict(style_str, REPLACE_IN_STYLE_STR)
    st.markdown(style_str, unsafe_allow_html=True)


def menu() -> None:
    # --- Execute any clean-up actions when setting up a new page:
    kill_executing_tool()
    # ---

    run_global_markdown_css_hack()  # NOTE: We fix the sidebar width (can't be adjusted).

    st.sidebar.page_link("pages/main.py", label=PAGE_TITLES["main_emoji"])
    st.sidebar.page_link("pages/research_management.py", label=PAGE_TITLES["research_management_emoji"])
    st.sidebar.page_link("pages/settings.py", label=PAGE_TITLES["settings_emoji"])

    with st.sidebar:
        st.markdown("---")
        st.image("./entry/st/SynthCraft.png")
        st.markdown(
            f"""
            **Synth**Craft is a part of the **CliMB** project. It is a tool for generating synthetic data
            Version: `{VERSION}`
            """
        )
        st.markdown("by [van der Schaar Lab](https://www.vanderschaar-lab.com/)")
        st.image("./entry/st/vds.png")
        st.markdown(
            "and [Cambridge Centre for AI in Medicine](https://www.damtp.cam.ac.uk/new-cambridge-centre-ai-medicine-ccaim-0)"
        )
        st.image("./entry/st/ccaim.png", width=150)


def initialize_common_st_state(db: DB) -> None:
    if "session_reload" not in st.session_state:
        st.session_state.session_reload = False
    if "active_session_key" not in st.session_state:
        st.session_state.active_session_key = db.get_user_settings().active_session
    if "new_session_settings" not in st.session_state:
        st.session_state.new_session_settings = {
            "session_name": None,
            "engine_name": list(ENGINE_MAP.keys())[0],
            "engine_params": dict(),  # All defaults.
        }


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()  # type: ignore
    return encoded


def img_to_html(img_path):
    # Check if img_path exists
    if not Path(img_path).exists():
        return "<img src='#' class='img-fluid' width='100%' alt='[Image not found]'>"
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='100%'>".format(img_to_bytes(img_path))
    return img_html


TRY_DOTENV_PATHS = [
    ".env",
    "keys.env",
]


def get_dotenv_config(try_dotenv_files: List[str] = TRY_DOTENV_PATHS) -> Dict[str, Union[str, None]]:
    dotenv_found = False
    dotenv_found_error_msg = ""

    for path in try_dotenv_files:
        if not Path(path).exists():
            dotenv_found_error_msg += f"`.env` file not found: {Path(path).absolute()}.\n"
        else:
            # print(f"Using `.env` file: {Path(path).absolute()}.")
            dotenv_config = dotenv_values(path)
            dotenv_found = True
            break
    if not dotenv_found:
        raise FileNotFoundError(dotenv_found_error_msg + "No more acceptable .env paths to try.")

    return dotenv_config


class JSExecutor:
    def __init__(
        self, js_code: str = "", container: Any = None, use_st_js: bool = False, log_text: Optional[str] = None
    ) -> None:
        self.use_st_js = use_st_js
        self._js_code = js_code
        self._container = container
        self._log_text = log_text

    def execute_js(self, replacements: Optional[Dict[str, str]] = None) -> Any:
        # Ensure no double quotes in the replacement values.
        # This is because we assume that double quotes are used to wrap the replacement values in the JS code.
        if replacements is not None:
            for replace_with in replacements.values():
                if '"' in replace_with:
                    raise ValueError(f"Replacement value contains double quotes: {replace_with}")

        actual_js_code = replace_str_from_dict(self._js_code, replacements or dict())

        def _exec(js_code: str) -> Any:
            if not self.use_st_js:
                return html(f"<script>{js_code}</script>", height=0)
            else:
                return st_js_blocking(js_code)

        if self._log_text is not None:
            ui_log(self._log_text)

        if self._container is not None:
            with self._container:
                return _exec(actual_js_code)
        else:
            return _exec(actual_js_code)
