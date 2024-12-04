from typing import Any, Dict, List, Optional, Tuple

import pydantic

from climb.common import (
    Agent,
    EngineState,
    Message,
    Session,
    ToolSpecs,
)
from climb.common.utils import d2m, m2d
from climb.db import DB

from ._azure_config import AzureOpenAIConfig
from ._engine import EngineAgent
from ._engine_openai import AzureOpenAIEngineMixin
from .engine_openai_nextgen import MESSAGE_OPTIONS, OpenAINextGenEngine, filter_messages_by_agent

# region: === Prompt templates ===

# SIMULATED_USER_SYSTEM_MESSAGE = f"""
# You are acting as a HUMAN USER of a system where a user interacts with an AI assistant.

# The COORDINATOR has a plan for the overall project, and you are responsible for working with the user on part of this \
# (your TASK).

# ### Record of the conversations
# You will receive the full record of the conversations between other agents and the user so far, so you can pick up \
# where they left off. You should use that conversation record to inform your work on the TASK.

# The record of conversations will look similar to this EXAMPLE:
# system: {LOG_PREAMBLE}
# assistant: What is your favorite cuisine?
# user: Italian.
# assistant: What is your favorite Italian dish?
# user: Spaghetti Carbonara.
# assistant: {TASK_COMPLETED_INDICATOR}
# system: {CONTINUE_INDICATOR}.

# NOTE: If you are the first agent to work on the project, you will NOT receive the conversation record.


# ### Your CAPABILITIES
# {WORKER_CAPABILITIES}


# ### Your RULES: You must follow these EXACTLY and NEVER violate them.
# {WORKER_RULES}

# ### EXAMPLES
# * These show how you should go about your work.
# * These are EXAMPLES ONLY. NOT the actual problem!
# * The actual TASK is given in the "Your TASK" section later.

# === EXAMPLES ===
# {WORKER_EXAMPLES}
# === END of EXAMPLES ===


# ### CURRENT WORKING DIRECTORY CONTENTS (for your information, do not send this to the user):
# ```text
# {WD_CONTENTS_REPLACE_MARKER}
# ```


# ================================================================================
# ### Your TASK
# Your TASK, given by the COORDINATOR is:

# {WORKER_ACTUAL_TASK_REPLACE_MARKER}

# ================================================================================


# ===============
# VERY IMPORTANT:
# - Remember, the previous agents have ALREADY COMPLETED their work steps! Do NOT REDO their work. Start from where \
# they left off!
# - DO NOT work on anything beyond what is asked in the TASK. You must however, make sure that the specific TASK is \
# completed fully and correctly, and issue the {TASK_COMPLETED_INDICATOR} message.
# - Always refer to CURRENT WORKING DIRECTORY CONTENTS to see which files are available to you!
# ===============
# """

# WORKER_STARTING_MESSAGE = f"""
# Now please start with your task.

# Unless you are the first agent, you will receive the conversation record to help you pick up where they left off.

# Remember:
# 1. Do NOT restart with asking for the user's data UNLESS you're the first agent whose task is to do so.
# 2. The previous agents have already completed their work steps. Do NOT REDO their work.
# 3. Your first message to the user MUST briefly explain what you plan to do and seek confirmation from the user. \
# Do NOT jump straight into using a tool or generating code! If the COORDINATOR has instructed you to minimize user \
# interaction, SKIP this.
# 4. If the work looks like it is moving to a task mentioned in "{TASKS_PLANNED_AFTER}", STOP and issue \
# {TASK_COMPLETED_INDICATOR}. The coordinator and the next agent will pick up from there. Generally avoid going into \
# the tasks listed under "{TASKS_PLANNED_AFTER}".
# 5. Check "CURRENT WORKING DIRECTORY CONTENTS" carefully and keep track of the files that have been created so far.
# """

# WORKER_REMINDER = f"""
# **Reminder!**
# 1. Where is my task description?
#     - Your TASK is given in the first system message, but here is a reminder:
#     ================================================================================
#     ### Your TASK
#     {WORKER_ACTUAL_TASK_REPLACE_MARKER}
#     ================================================================================
# 2. When to mark my task as completed?
#     - Check that you are not proceeding beyond the subtasks you have been given!
#     - Check the list of SUBTASKS given to you in the system message.
#     - Check the list of "{TASKS_PLANNED_AFTER}" in the system message.
#     - Are you in danger of getting into the "{TASKS_PLANNED_AFTER}" tasks?
#     - If so, just issue the {TASK_COMPLETED_INDICATOR}.
#     **But - IF THERE ARE ERRORS!**
#     - If some subtask you are executing is raising an error, you SHOULD try to make it work.
#     - If there is an error that you can fix, DO NOT issue the {TASK_COMPLETED_INDICATOR}.
#     - Attempt to fix the error, which often involves generating some code.
#     - Investigate the possible reasons for failure step by step. Generate code that could help you debug the issue.
#     - Then generate code that could help you fix the issue.
#     - If after several attempts you CANNOT make it work, you can issue the {TASK_COMPLETED_INDICATOR}.
# 3. Code generation:
#     - Do not write any text after the code section. The code section must be the last part of your message.
#     - NEVER issue more than one code snippet in a single message.
#     - Do not overwrite any existing files in the working directory, always create new files with unique names.
#         * Even if you are doing a slight modification to a file, save the modified file with a new name!
#         * Adding a suffix like v2, v3 can work.
#         * E.g. if you are slightly modifying "data_processed.csv", save the modified file as "data_processed_v2.csv".
# """

MESSAGE_OPTIONS["simulated_user"] = {
    "system_message_template": "You are a clinician with a dataset test.csv. You want to predict the column 'label'.",
    "first_message_content": "",
}

# endregion


# region: === Engine helper functions (may be considered for moving to a separate module) ===

# endregion


class AgentStore(pydantic.BaseModel):
    coordinator: EngineAgent
    worker: EngineAgent
    simulated_user: EngineAgent

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


class SimulatedUserState(pydantic.BaseModel):
    something: str


class OpenAINextGenEngineSim(OpenAINextGenEngine):
    def __init__(
        self,
        db: DB,
        session: Session,
        conda_path: Optional[str] = None,
        *,
        api_key: str,
        # ---
        **kwargs: Any,
    ):
        super().__init__(
            db=db,
            session=session,
            conda_path=conda_path,
            api_key=api_key,
        )

        self.simulated_user = True
        if self._new_session:
            self.session.engine_state.agent_state["simulated_user"] = m2d(SimulatedUserState(something=""))
            print(self.session.engine_state)
            self.update_state()

    def _dispatch_worker(self, agent: EngineAgent) -> EngineState:
        engine_state = super()._dispatch_worker(agent)

        if self.session.engine_state.user_message_requested is True:
            engine_state.agent_switched = True
            engine_state.agent = "simulated_user"

            # Update simulated_user state:
            simulated_user_state = d2m(self.session.engine_state.agent_state["simulated_user"], SimulatedUserState)
            # TODO: ...
            self.session.engine_state.agent_state["simulated_user"] = m2d(simulated_user_state)
            # print("~" * 80)
            self._set_initial_messages(agent=self.agents_.simulated_user)
            # print(self.get_message_history())

            self.update_state()

        return self.session.engine_state

    def _gather_messages_simulated_user(self, agent: EngineAgent) -> Tuple[List[Message], ToolSpecs]:
        simulated_user_messages = filter_messages_by_agent(self.get_message_history(), agent.agent_type)
        print("=" * 80)
        print(simulated_user_messages)
        print("=" * 80)
        return simulated_user_messages, []

    def _dispatch_simulated_user(self, agent: EngineAgent) -> EngineState:
        self.session.engine_state.user_message_requested = False
        return self.session.engine_state

    def define_agents(self) -> Dict[Agent, EngineAgent]:
        # AgentStore just to act as a dotdict for convenient access.
        self.agents_ = AgentStore(
            coordinator=EngineAgent(
                "coordinator",
                first_message_content=MESSAGE_OPTIONS["coordinator"]["first_message_content"],
                system_message_template=MESSAGE_OPTIONS["coordinator"]["system_message_template"],
                first_message_role="assistant",
                set_initial_messages=OpenAINextGenEngineSim._set_initial_messages,  # type: ignore
                gather_messages=OpenAINextGenEngineSim._gather_messages_coordinator,  # type: ignore
                dispatch=OpenAINextGenEngineSim._dispatch_coordinator,  # type: ignore
            ),
            worker=EngineAgent(
                "worker",
                system_message_template=MESSAGE_OPTIONS["worker"]["system_message_template"],
                first_message_content=MESSAGE_OPTIONS["worker"]["first_message_content"],
                first_message_role="assistant",
                set_initial_messages=OpenAINextGenEngineSim._set_initial_messages,  # type: ignore
                gather_messages=OpenAINextGenEngineSim._gather_messages_worker,  # type: ignore
                dispatch=OpenAINextGenEngineSim._dispatch_worker,  # type: ignore
            ),
            simulated_user=EngineAgent(
                "simulated_user",
                system_message_template=MESSAGE_OPTIONS["simulated_user"]["system_message_template"],
                first_message_content=MESSAGE_OPTIONS["simulated_user"]["first_message_content"],
                first_message_role="user",
                set_initial_messages=OpenAINextGenEngineSim._set_initial_messages,  # type: ignore
                gather_messages=OpenAINextGenEngineSim._gather_messages_simulated_user,  # type: ignore
                dispatch=OpenAINextGenEngineSim._dispatch_simulated_user,  # type: ignore
            ),
        )
        as_dict = self.agents_.model_dump()  # {"coordinator": coordinator EngineAgent, ...}
        return as_dict  # type: ignore

    def _reset_engine_state_after_api_call(self):
        simulated_user_state = d2m(self.session.engine_state.agent_state["simulated_user"], SimulatedUserState)
        super()._reset_engine_state_after_api_call()
        self.session.engine_state.agent_state["simulated_user"] = m2d(simulated_user_state)

    def define_initial_agent(self) -> Agent:
        return "coordinator"

    @staticmethod
    def get_engine_name() -> str:
        return "openai_nextgen_sim"


class AzureOpenAINextGenEngineSim(
    AzureOpenAIEngineMixin,  # Mixing needs to come first to override the methods correctly.
    OpenAINextGenEngineSim,
):
    def __init__(
        self,
        db: DB,
        session: Session,
        conda_path: Optional[str] = None,
        *,
        api_key: str,
        azure_openai_config: AzureOpenAIConfig,
        # ---
        **kwargs: Any,
    ):
        # Initialize the Mixin.
        AzureOpenAIEngineMixin.__init__(
            self,
            azure_openai_config=azure_openai_config,
        )
        # Initialize the Base class.
        OpenAINextGenEngineSim.__init__(
            self,
            db=db,
            session=session,
            conda_path=conda_path,
            api_key=api_key,
            # ---
            **kwargs,
        )

    @staticmethod
    def get_engine_name() -> str:
        return "azure_openai_nextgen_sim"
