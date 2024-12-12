from typing import Any, Callable, Dict, List, Optional, Tuple

import pydantic

from climb.common import (
    Agent,
    EngineState,
    KeyGeneration,
    Message,
    ResponseKind,
    Session,
    ToolSpecs,
)
from climb.common.utils import d2m, m2d, update_templates
from climb.db import DB

from ._azure_config import AzureOpenAIConfig
from ._engine import EngineAgent
from ._engine_openai import AzureOpenAIEngineMixin
from .engine_openai_nextgen import (
    MESSAGE_OPTIONS,
    OpenAINextGenEngine,
)

# region: === Prompt templates ===

WD_CONTENTS_REPLACE_MARKER = "{WD_CONTENTS}"
SIMULATED_USER_ACTUAL_TASK_REPLACE_MARKER = "{SIMULATED_USER_ACTUAL_TASK}"

SIMULATED_USER_LOG_PREAMBLE = "HERE IS THE LOG OF THE CONVERSATION SO FAR"
SIMULATED_USER_CONTINUE_INDICATOR = "CONTINUE YOUR CONVERSATION FROM HERE"

SIMULATED_USER_RULES = """
1. Remember: you are not an AI assistant, but a (simulated) human user!
2. You must respond as if you are a human user, with capabilities as described.
3. Do not do anything that exceeds the knowledge, capabilities, or role of a human user. The more you are like a human \
user, with the described capabilities, the better.
4. Review the conversation so far, and continue it as if you were the human user.
5. Only the AI assistant(s) you interact with are able to generate code or use tools. You cannot do this!
6. Check the last few messages and ask yourself: "Am I repeating myself?", "Is the conversation looping unnaturally?". \
Humans don't do that. Hence, you MUST break this loop. Ask a question, provide a new piece of information, rephrase, etc.
"""

# TODO: This needs to be made customizable.
SIMULATED_USER_CAPABILITIES = """
You are a human, with the following background:
- You are a clinician with a dataset of patient records.
- The dataset is in CSV format, in a file named "heart.csv".
- Each row in the dataset represents a patient.
- Each column in the dataset represents a feature of the patient.
- The dataset contains information about heart disease.
- The target column you want to predict is "HeartDisease" and contains binary values (0 or 1).
"""

SIMULATED_USER_EXAMPLES = f"""
#### Example:

- Given a conversation log like this:

system: {SIMULATED_USER_LOG_PREAMBLE}
assistant: I would like to find out your culinary preferences.
user: Okay.
assistant: What is your favorite Italian dish?
system: {SIMULATED_USER_CONTINUE_INDICATOR}.

- You would continue the conversation like this:

Spaghetti Carbonara.

"""

SIMULATED_USER_SYSTEM_MESSAGE = f"""
You are acting as a HUMAN USER of a system where a user interacts with AI assistant(s).

You have a specific task to complete, which is given below. Work with the AI assistant(s) to complete this task.


### Record of the conversations
You will receive the full record of the conversations between you and AI assistant(s).

The record of conversations will look similar to this EXAMPLE:
system: {SIMULATED_USER_LOG_PREAMBLE}
assistant: What is your favorite cuisine?
user: Italian.
assistant: What is your favorite Italian dish?
user: Spaghetti Carbonara.
assistant: What is your favorite dessert?
system: {SIMULATED_USER_CONTINUE_INDICATOR}.

NOTE: Your past responses are shown as "user" messages. You will need to continue the conversation.


### Your CAPABILITIES
{SIMULATED_USER_CAPABILITIES}


### Your RULES: You must follow these EXACTLY and NEVER violate them.
{SIMULATED_USER_RULES}

### EXAMPLES
* These show how you should go about your work.
* These are EXAMPLES ONLY. NOT the actual problem!
* The actual TASK is given in the "Your TASK" section later.

=== EXAMPLES ===
{SIMULATED_USER_EXAMPLES}
=== END of EXAMPLES ===


### CURRENT WORKING DIRECTORY CONTENTS:
```text
{WD_CONTENTS_REPLACE_MARKER}
```


================================================================================
### Your TASK
Your specific task is:

{SIMULATED_USER_ACTUAL_TASK_REPLACE_MARKER}

================================================================================
"""

SIMULATED_USER_REMINDER = f"""
**Reminder!**
1. Where is my task description?
    - Your TASK is given in the first system message, but here is a reminder:
    ================================================================================
    ### Your TASK
    {SIMULATED_USER_ACTUAL_TASK_REPLACE_MARKER}
    ================================================================================
2. Make sure to follow the following rules:
    ================================================================================
    ### Rules
    {SIMULATED_USER_RULES}
    ================================================================================
"""

MESSAGE_OPTIONS["simulated_user"] = {
    "system_message_template": SIMULATED_USER_SYSTEM_MESSAGE,
    "record_preamble": SIMULATED_USER_LOG_PREAMBLE,
    "record_continuation": SIMULATED_USER_CONTINUE_INDICATOR,
    "reminder": SIMULATED_USER_REMINDER,
}

# TODO: Needs to be "settable".
SIMULATED_USER_ACTUAL_TASK = """
Your task is to work with the AI assistant to create and investigate a predictive model \
for heart disease using the dataset in the file 'heart.csv'.
"""

# endregion


# region: === Engine helper functions (may be considered for moving to a separate module) ===


def get_messages_like(messages: List[Message], like: Callable[[Message], bool]) -> List[Message]:
    found_messages = []
    for message in messages:
        if like(message):
            found_messages.append(message)
    return found_messages


# User-visible worker messages, tool and code execution messages,
# and simulated user messages that are not system messages.
def _simulated_user_message_history(m: Message) -> bool:
    return (
        # User-visible worker messages:
        (
            m.agent == "worker"
            and (
                m.visibility in ("all", "ui_only")
                or
                # Needed to include the tool request messages - otherwise OpenAI API will complain.
                m.incoming_tool_calls is not None
            )
            # Including tool and code execution messages:
            and m.role in ("assistant", "tool", "code_execution")
        )
        or
        # Simulated user messages:
        (
            m.agent == "simulated_user"
            and
            # Not system messages:
            m.role != "system"
        )
    )


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

        if (
            self.session.engine_state.user_message_requested is True
            and self.session.engine_state.response_kind == ResponseKind.TEXT_MESSAGE
        ):
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
        messages_to_process = []

        historic_messages = get_messages_like(self.get_message_history(), like=_simulated_user_message_history)

        # Just re-generate the initial system message for the simulated user.
        system_message_text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                SIMULATED_USER_ACTUAL_TASK_REPLACE_MARKER: SIMULATED_USER_ACTUAL_TASK,
            },
        )
        system_message = Message(
            key=KeyGeneration.generate_message_key(),
            role="system",
            visibility="llm_only",
            new_reasoning_cycle=True,  # NOTE: Not relevant for simulated user.
            text=system_message_text,
            agent=agent.agent_type,
        )

        tools = []  # No tools for simulated user.

        # Structure the messages to process.

        # [system]
        messages_to_process.append(system_message)

        if not historic_messages:
            raise ValueError("No historic messages found for simulated user, but are required.")

        # <separator>
        messages_to_process.append(
            # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                text=MESSAGE_OPTIONS["simulated_user"]["record_preamble"],
                agent=agent.agent_type,
            )
        )
        # [historic record]
        messages_to_process.extend(historic_messages)
        # <separator>
        messages_to_process.append(
            # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                text=MESSAGE_OPTIONS["simulated_user"]["record_continuation"],
                agent=agent.agent_type,
            )
        )

        # Reminder message.
        reminder_message_text = update_templates(
            body_text=MESSAGE_OPTIONS["simulated_user"]["reminder"],
            templates={
                SIMULATED_USER_ACTUAL_TASK_REPLACE_MARKER: SIMULATED_USER_ACTUAL_TASK,
            },
        )
        reminder_message = Message(
            key=KeyGeneration.generate_message_key(),
            role="system",
            text=reminder_message_text,
            agent=agent.agent_type,
            visibility="llm_only_ephemeral",
            engine_state=self.session.engine_state,
        )
        # TODO: Investigate the below line.
        self._append_message(reminder_message)  # Add to history.
        messages_to_process.append(reminder_message)  # And add to list of messages to send to LLM straight away too.
        # Reminder message [END].

        print("=" * 80)
        import rich.pretty

        rich.pretty.pprint(messages_to_process)
        print("=" * 80)
        # raise ValueError("STOP HERE")

        return messages_to_process, tools

    def _dispatch_simulated_user(self, agent: EngineAgent) -> EngineState:
        self.session.engine_state.user_message_requested = False

        self.session.engine_state.agent_switched = True
        self.session.engine_state.agent = "worker"

        # There shouldn't be a need to update worker state here, as it is unchanged.

        self.update_state()

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
                first_message_content=None,
                first_message_role=None,
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
