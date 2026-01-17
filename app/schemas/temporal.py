# app/schemas/temporal.py
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TemporalIntentType(str, Enum):
    DAY = "DAY"
    RANGE = "RANGE"
    LAST_N = "LAST_N"
    NONE = "NONE"


class TemporalWhich(str, Enum):
    last = "last"
    first = "first"


class TemporalWindow(BaseModel):
    start_iso: Optional[str] = Field(None, description="ISO-8601 inicio (tz-aware).")
    end_iso: Optional[str] = Field(
        None, description="ISO-8601 fin EXCLUSIVO (tz-aware)."
    )
    temporal_intent_type: TemporalIntentType = TemporalIntentType.NONE
    which: Optional[TemporalWhich] = None
    count: Optional[int] = 1


'''
class TemporalWindow(BaseModel):
    """
    Intención temporal normalizada por el LLM.
    - start_iso / end_iso en ISO-8601 (end_iso es LIMITE EXCLUSIVO).
    - which: 'last' | 'first' | None  (superlativo detectado)
    - count: cantidad pedida (por defecto 1)
    """
    start_iso: Optional[str] = Field(
        None, description="Inicio (ISO-8601). Ej: '2025-01-01 00:00:00'."
    )
    end_iso: Optional[str] = Field(
        None, description="Fin EXCLUSIVO (ISO-8601). Ej: '2025-02-01 00:00:00'."
    )
    temporal_intent_type: str = Field(
        "NONE", description="DAY | RANGE | LAST_N | NONE"
    )
    which: Optional[str] = Field(
        None, description="'last' | 'first' | None"
    )
    count: Optional[int] = Field(
        1, description="Cantidad solicitada si aplica."
    )

'''
