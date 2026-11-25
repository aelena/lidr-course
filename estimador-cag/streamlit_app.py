"""Chat interface for the estimator.

Paste a meeting transcription and watch the software estimation arrive token
by token, with the same CAG system prompt as the HTTP endpoint. Run it from
this folder:

    uv run streamlit run streamlit_app.py

The API key is never written here: it comes from the environment, from
Streamlit secrets or from `.env`, in that order.
"""

import os
from typing import Any

import streamlit as st
from pydantic import ValidationError
from streamlit.delta_generator import DeltaGenerator
from streamlit.errors import StreamlitSecretNotFoundError

from app.config import Settings, configure_logging, get_settings
from app.context.examples import render_estimation, select_examples, total_hours
from app.services.llm_service import (
    GenerationOptions,
    LLMServiceError,
    stream_estimation,
    system_prompt_for,
)

# The same defaults as the HTTP endpoint; the sidebar shows what they produce.
OPTIONS = GenerationOptions()

st.set_page_config(page_title="Estimador CAG", page_icon="📐")


def export_secrets() -> None:
    """Copy the settings found in st.secrets to the environment, where Settings reads them.

    Variables already set in the environment win, and `.env` fills in the rest.
    """
    try:
        secrets = {
            name: str(st.secrets[name])
            for name in map(str.upper, Settings.model_fields)
            if name in st.secrets
        }
    except StreamlitSecretNotFoundError:  # no secrets.toml: .env or the environment decide
        return
    for name, value in secrets.items():
        os.environ.setdefault(name, value)


@st.cache_resource
def load_settings() -> Settings:
    """Validate the configuration and set up logging once per server process."""
    export_secrets()
    settings = get_settings()
    configure_logging()
    return settings


def show(message: dict[str, str]) -> None:
    content = message["content"]
    with st.chat_message(message["role"]):
        # Two trailing spaces keep the line breaks of a pasted transcription.
        st.markdown(content.replace("\n", "  \n") if message["role"] == "user" else content)


def thousands(number: int) -> str:
    return f"{number:,}".replace(",", ".")


def seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.1f} s".replace(".", ",")


def show_metrics(panel: DeltaGenerator, settings: Settings, call: dict[str, Any] | None) -> None:
    """Fill the metrics panel with the last call, or with what the next one will use."""
    with panel.container():
        st.subheader("Última llamada")
        if call is None:
            st.markdown(f"**Modelo:** `{OPTIONS.model or settings.llm_model}` ({settings.llm_provider})")
            st.caption("Todavía no se ha llamado al modelo en esta sesión.")
            return
        st.markdown(f"**Modelo:** `{call['model']}` ({call['provider']})")
        left, right = st.columns(2)
        left.metric("Tokens de entrada", thousands(call["input_tokens"]))
        right.metric("Tokens de salida", thousands(call["output_tokens"]))
        left.metric("Tiempo de respuesta", seconds(call["latency_ms"]))
        right.metric("Primer token", seconds(call["first_token_ms"]))
        if call["finish_reason"] not in ("stop", "end_turn"):
            st.warning(f"Respuesta incompleta: el modelo paró por `{call['finish_reason']}`.")


def show_context() -> None:
    """Read-only view of what every call sends before the conversation."""
    system_prompt = system_prompt_for(OPTIONS)
    examples = select_examples(OPTIONS.num_examples) if OPTIONS.use_examples else []

    st.subheader("Qué recibe el modelo")
    st.caption("Solo lectura. Es lo mismo que envía el endpoint `POST /api/v1/estimate`.")
    with st.expander(f"System prompt ({thousands(len(system_prompt))} caracteres)"):
        st.code(system_prompt, language=None, wrap_lines=True)
    with st.expander(f"Contexto estático inyectado ({len(examples)} estimaciones)"):
        st.caption(
            "Estimaciones anteriores de `app/context/examples.json`. "
            "Viajan dentro del system prompt en cada llamada: son el cache de CAG."
        )
        for number, example in enumerate(examples, start=1):
            st.markdown(f"**{number}. {example['title']}** · {total_hours(example)} horas")
            st.caption(example["meeting_summary"])
            st.code(render_estimation(example), language="markdown", wrap_lines=True)


def new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.last_call = None


st.title("Estimador CAG")
st.caption(
    "Pega la transcripción de una reunión con el cliente y recibe una estimación "
    "calibrada con proyectos anteriores. Después puedes seguir preguntando sobre ella."
)

try:
    settings = load_settings()
except ValidationError as error:
    st.error(f"Configuración incompleta: {error.errors()[0]['msg']}")
    st.stop()

# Streamlit reruns this script on every interaction; the conversation lives in
# the session state, which survives the reruns of one browser session.
messages: list[dict[str, str]] = st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_call", None)

with st.sidebar:
    # A placeholder, so the metrics can be replaced as soon as a stream ends.
    metrics_panel = st.empty()
    show_metrics(metrics_panel, settings, st.session_state.last_call)
    st.divider()
    show_context()
    st.divider()
    st.button("Nueva conversación", on_click=new_conversation)

for message in messages:
    show(message)

if prompt := st.chat_input("Pega aquí la transcripción de la reunión"):
    question = {"role": "user", "content": prompt}
    show(question)
    stream = stream_estimation([*messages, question], OPTIONS)
    with st.chat_message("assistant"):
        try:
            # Writes each delta as the provider sends it; the request starts here.
            st.write_stream(stream)
        except LLMServiceError as error:
            # The failed turn is not stored, so the history stays in user/assistant pairs.
            st.error(f"No se pudo generar la estimación: {error}")
        else:
            messages += [
                question,
                {"role": "assistant", "content": stream.result["estimation"]},
            ]
            st.session_state.last_call = {
                key: value
                for key, value in stream.result.items()
                if key not in ("estimation", "extracted_requirements")
            }
            show_metrics(metrics_panel, settings, st.session_state.last_call)
