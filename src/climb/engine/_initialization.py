from typing import Any, Dict, Type

from climb.common.data_structures import Session
from climb.common.utils import fix_windows_path_backslashes
from climb.db import DB

from ._azure_config import (
    AZURE_OPENAI_CONFIG_PATH,
    get_api_key_for_azure_openai,
    load_azure_openai_config_item,
)
from ._config import get_dotenv_config
from ._engine import EngineBase
from ._engine_openai import AzureOpenAIEngineMixin
from .engine_openai_dc import AzureOpenAIDCEngine, OpenAIDCEngine
from .engine_openai_dc_v2 import AzureOpenAIDCv2Engine, OpenAIDCv2Engine
from .engine_openai_min_baseline import AzureOpenAIMinBaselineEngine, OpenAIMinBaselineEngine
from .engine_openai_nextgen import (
    AzureOpenAINextGenEngine,
    AzureOpenAINextGenReasoningAblationEngine,
    OpenAINextGenEngine,
    OpenAINextGenReasoningAblationEngine,
)
from .engine_openai_sim import AzureOpenAINextGenEngineSim, OpenAINextGenEngineSim
from .engine_openai_tool_baseline import AzureOpenAIToolBaselineEngine, OpenAIToolBaselineEngine
from .engine_openai_synthetic import (
    AzureOpenAIDeepSeekSyntheticEngine,
    AzureOpenAIGPT5SyntheticEngine,
    AzureOpenAIGPT5SyntheticReasoningAblationEngine,
    AzureOpenAISyntheticEngine,
    AzureOpenAISyntheticReasoningAblationEngine,
    OpenAIGPT5SyntheticEngine,
    OpenAISyntheticEngine,
    OpenAISyntheticReasoningAblationEngine,
)

dotenv_config = get_dotenv_config()


ENGINE_MAP = {
    # # Current engines:
    # OpenAINextGenEngine.get_engine_name(): OpenAINextGenEngine,
    # AzureOpenAINextGenEngine.get_engine_name(): AzureOpenAINextGenEngine,
    # OpenAINextGenReasoningAblationEngine.get_engine_name(): OpenAINextGenReasoningAblationEngine,
    # AzureOpenAINextGenReasoningAblationEngine.get_engine_name(): AzureOpenAINextGenReasoningAblationEngine,
    # # CoT engines:
    # OpenAIDCEngine.get_engine_name(): OpenAIDCEngine,
    # AzureOpenAIDCEngine.get_engine_name(): AzureOpenAIDCEngine,
    # # CoT v2 engines:
    # OpenAIDCv2Engine.get_engine_name(): OpenAIDCv2Engine,
    # AzureOpenAIDCv2Engine.get_engine_name(): AzureOpenAIDCv2Engine,
    # synthetic data engines:
    OpenAISyntheticEngine.get_engine_name(): OpenAISyntheticEngine,
    AzureOpenAISyntheticEngine.get_engine_name(): AzureOpenAISyntheticEngine,
    OpenAIGPT5SyntheticEngine.get_engine_name(): OpenAIGPT5SyntheticEngine,
    AzureOpenAIGPT5SyntheticEngine.get_engine_name(): AzureOpenAIGPT5SyntheticEngine,
    # AzureOpenAIDeepSeekSyntheticEngine.get_engine_name(): AzureOpenAIDeepSeekSyntheticEngine,
    OpenAISyntheticReasoningAblationEngine.get_engine_name(): OpenAISyntheticReasoningAblationEngine,
    AzureOpenAISyntheticReasoningAblationEngine.get_engine_name(): AzureOpenAISyntheticReasoningAblationEngine,
    AzureOpenAIGPT5SyntheticReasoningAblationEngine.get_engine_name(): AzureOpenAIGPT5SyntheticReasoningAblationEngine,

}

if dotenv_config.get("BASELINE_METHODS", "False") == "True":
    ENGINE_MAP.update(
        {
            # Minimal baseline versions:
            OpenAIMinBaselineEngine.get_engine_name(): OpenAIMinBaselineEngine,
            AzureOpenAIMinBaselineEngine.get_engine_name(): AzureOpenAIMinBaselineEngine,
            # Tool baseline versions:
            OpenAIToolBaselineEngine.get_engine_name(): OpenAIToolBaselineEngine,
            AzureOpenAIToolBaselineEngine.get_engine_name(): AzureOpenAIToolBaselineEngine,
        }
    )

if dotenv_config.get("SIMULATED_USER_METHODS", "False") == "True":
    ENGINE_MAP.update(
        {
            # Minimal baseline versions:
            OpenAINextGenEngineSim.get_engine_name(): OpenAINextGenEngineSim,
            AzureOpenAINextGenEngineSim.get_engine_name(): AzureOpenAINextGenEngineSim,
        }
    )


def _is_azure_engine(engine_class: Type[EngineBase]) -> bool:
    """Centralized check to determine whether an engine uses Azure-specific plumbing."""
    return issubclass(engine_class, AzureOpenAIEngineMixin)


azure_engines = [engine_name for engine_name, EngineClass in ENGINE_MAP.items() if _is_azure_engine(EngineClass)]
non_azure_engines = [engine_name for engine_name in ENGINE_MAP.keys() if engine_name not in azure_engines]


def create_engine(db: DB, session: Session, config: Dict[str, Any]) -> EngineBase:
    EngineClass = ENGINE_MAP[session.engine_name]
    conda_path = config.get("CONDA_PATH", None)
    if conda_path is not None:
        conda_path = fix_windows_path_backslashes(conda_path)
    extra_kwargs: Dict[str, Any] = {
        "conda_path": conda_path,
    }
    if not _is_azure_engine(EngineClass):
        extra_kwargs["api_key"] = config["OPENAI_API_KEY"]
    else:
        extra_kwargs["azure_openai_config"] = load_azure_openai_config_item(
            AZURE_OPENAI_CONFIG_PATH,
            session.engine_params["config_item_name"],  # pyright: ignore
        )
        extra_kwargs["api_key"] = get_api_key_for_azure_openai(extra_kwargs["azure_openai_config"], config)
    return EngineClass(
        db=db,
        session=session,
        **extra_kwargs,
    )
