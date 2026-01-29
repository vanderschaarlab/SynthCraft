from typing import Any, Dict

from openai import AzureOpenAI, OpenAI

from climb.common import Session
from climb.engine.const import MODEL_MAX_MESSAGE_TOKENS


def create_llm_client(
    session: Session,
    additional_kwargs_required: Dict[str, Any],
) -> Any:
    if session.engine_name in (
        "openai",
        "openai_nextgen",
        "openai_min_baseline",
        "openai_nextgen_sim",
        "openai_cot",
        "openai_dc",
    ):
        client = OpenAI(api_key=additional_kwargs_required["api_key"])
    elif session.engine_name in (
        "azure_openai",
        "azure_openai_nextgen",
        "azure_openai_min_baseline",
        "azure_openai_nextgen_sim",
        "azure_openai_cot",
        "azure_openai_dc",
    ):
        client = AzureOpenAI(
            azure_endpoint=additional_kwargs_required["azure_endpoint"],
            api_version=additional_kwargs_required["api_version"],
            api_key=additional_kwargs_required["api_key"],
        )
    else:
        raise ValueError(f"Unknown engine name: {session.engine_name}")
    return client


def get_llm_chat(
    client: Any,
    session: Session,
    additional_kwargs_required: Dict[str, Any],
    chat_kwargs: Dict,
) -> str:
    if session.engine_name in (
        "openai",
        "openai_nextgen",
        "openai_min_baseline",
        "openai_nextgen_sim",
        "openai_cot",
        "openai_dc",
    ):
        model_type = additional_kwargs_required["engine_params"]["model_id"]
        lookup_model = str(model_type).lower()
        max_tokens = MODEL_MAX_MESSAGE_TOKENS.get(model_type, MODEL_MAX_MESSAGE_TOKENS.get(lookup_model, 4_096))
        out = client.chat.completions.create(
            model=model_type,
            max_tokens=max_tokens,
            temperature=additional_kwargs_required["engine_params"]["temperature"],
            # ---
            messages=chat_kwargs["messages"],
            stream=chat_kwargs["stream"],
        )
    elif session.engine_name in (
        "azure_openai",
        "azure_openai_nextgen",
        "azure_openai_min_baseline",
        "azure_openai_nextgen_sim",
        "azure_openai_cot",
        "azure_openai_dc",
    ):
        model_type = additional_kwargs_required["azure_openai_config"].model
        lookup_model = str(model_type).lower()
        max_tokens = MODEL_MAX_MESSAGE_TOKENS.get(model_type, MODEL_MAX_MESSAGE_TOKENS.get(lookup_model, 4_096))
        out = client.chat.completions.create(
            model=additional_kwargs_required["azure_openai_config"].deployment_name,
            max_tokens=max_tokens,
            # ---
            messages=chat_kwargs["messages"],
            stream=chat_kwargs["stream"],
        )
    else:
        raise ValueError(f"Unknown engine name: {session.engine_name}")
    out_text = out.choices[0].message.content  # type: ignore
    return out_text
