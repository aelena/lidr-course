"""Chat interface for the estimator.

Paste a meeting transcription and get a software estimation, with the same
CAG system prompt as the HTTP endpoint. Run it from this folder:

    uv run streamlit run streamlit_app.py

The API key is never written here: it comes from the environment, from
Streamlit secrets or from `.env`, in that order.
"""

import os

import streamlit as st
from pydantic import ValidationError
from streamlit.errors import StreamlitSecretNotFoundError

from app.config import Settings, configure_logging, get_settings
from app.services.llm_service import LLMServiceError, stream_estimation

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

for message in messages:
    show(message)

if prompt := st.chat_input("Pega aquí la transcripción de la reunión"):
    question = {"role": "user", "content": prompt}
    show(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Estimando…"):
                stream = stream_estimation([*messages, question])
                for _ in stream:
                    pass
        except LLMServiceError as error:
            # The failed turn is not stored, so the history stays in user/assistant pairs.
            st.error(f"No se pudo generar la estimación: {error}")
        else:
            st.markdown(stream.result["estimation"])
            messages += [
                question,
                {"role": "assistant", "content": stream.result["estimation"]},
            ]
