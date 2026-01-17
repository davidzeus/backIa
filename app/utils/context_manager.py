# app/utils/context_manager.py
from contextvars import ContextVar
from typing import Optional

# Global context for the current request's Patient ID
# This prevents the LLM from handling (and hallucinating) the ID.
_params_patient_id: ContextVar[Optional[str]] = ContextVar("patient_id", default=None)

def set_patient_context(patient_id: str):
    """Sets the patient_id for the current context (thread/task)."""
    return _params_patient_id.set(patient_id)

def get_patient_context() -> str:
    """Gets the patient_id from the current context. Raises error if not set."""
    pid = _params_patient_id.get()
    if not pid:
        raise ValueError("PatientContext not set! Tool called outside of request context.")
    return pid


def clear_patient_context(token):
    """Resets the context (good practice, though API threads usually recycle)."""
    _params_patient_id.reset(token)

# 🆕 Contexto para pasar sources del Tool al Servicio (bypass del return string)
_params_agent_sources: ContextVar[Optional[list]] = ContextVar("agent_sources", default=None)

def set_agent_sources(sources: list):
    """Guarda las fuentes recuperadas por la tool para usarlas en la respuesta final."""
    return _params_agent_sources.set(sources)

def get_agent_sources() -> list:
    """Recupera las fuentes guardadas, o lista vacía si no hay."""
    return _params_agent_sources.get() or []

def clear_agent_sources(token):
    _params_agent_sources.reset(token)

