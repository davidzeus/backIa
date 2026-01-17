# app/utils/temporal_intent_utils.py
from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional, Tuple

from llama_index.core.llms import LLM
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import PromptTemplate

from app.config.time import HOSPITAL_TZ
from app.schemas.temporal import TemporalWindow

log = logging.getLogger(__name__)




# -----------------------------------------------------------------------------
# API pública del módulo (todo lo temporal + manejo de "año faltante")
# -----------------------------------------------------------------------------
__all__ = [
    # Parsing temporal
    "parse_temporal_intent",
    "intent_to_ts_window",
    "iso_day_to_midnight_ts",
    "to_iso_date",
    "day_bounds",
    "parse_iso_to_dt",
    # Año faltante
    "detect_missing_year",
    "handle_missing_year",
    "enrich_question_with_year",
    "assumed_year_note",
    "MissingYearPolicy",
]

# -----------------------------------------------------------------------------
# Políticas para año faltante
# -----------------------------------------------------------------------------
MissingYearPolicy = Literal["ASK", "ASSUME_CURRENT", "INFER_SINGLE_YEAR"]

_DEFAULT_POLICY: MissingYearPolicy = "INFER_SINGLE_YEAR"

def _infer_single_year_for_range(
    qdrant_client: Any,
    collection: str,
    paciente_id: str,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> Optional[int]:
    """
    Intenta inferir UN solo año para un rango corto sin año, ej. '30/11 al 03/12'.
    Estrategia:
      - prueba años (año_actual - 1, actual, +1)
      - cuenta si hay datos para *alguno* de los días del rango
      - si solo un año tiene datos → ese
    """
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range  # type: ignore
    except Exception:
        return None

    years_found = set()
    cand_years = (
        datetime.now().year - 1,
        datetime.now().year,
        datetime.now().year + 1,
    )
    for year in cand_years:
        # armamos dos ventanas: inicio y fin
        start_local = datetime(year, start_month, start_day, 0, 0, 0, tzinfo=HOSPITAL_TZ)
        end_local = datetime(year, end_month, end_day, 0, 0, 0, tzinfo=HOSPITAL_TZ) + timedelta(days=1)
        ts_start = int(start_local.astimezone(timezone.utc).timestamp())
        ts_end = int(end_local.astimezone(timezone.utc).timestamp())

        flt = Filter(
            must=[
                FieldCondition(key="paciente_id", match=MatchValue(value=str(paciente_id))),
                FieldCondition(key="fecha_ts", range=Range(gte=ts_start, lt=ts_end)),
            ]
        )
        try:
            cnt = qdrant_client.count(collection_name=collection, count_filter=flt, exact=False)
        except Exception:
            continue

        if getattr(cnt, "count", 0) > 0:
            years_found.add(year)

    if len(years_found) == 1:
        return list(years_found)[0]
    return None



def _infer_single_year_for_day(
    qdrant_client: Any,
    collection: str,
    paciente_id: str,
    month: int,
    day: int,
) -> Optional[int]:
    """
    Busca si existe un ÚNICO año con registros para ese paciente/mes/día (usa payload.fecha_ts).
    Escanea año actual ±1 (ajustable si se desea).
    """
    try:
        from qdrant_client.models import FieldCondition  # type: ignore
        from qdrant_client.models import Filter, MatchValue, Range
    except Exception:
        # Si no hay qdrant lib, no se puede inferir
        return None

    years_found = set()
    cand_years = (datetime.now().year - 1, datetime.now().year, datetime.now().year + 1)
    for year in cand_years:
        try:
            # Ventana [00:00, +1d) en TZ hospital → a epoch UTC
            local_start = datetime(year, month, day, 0, 0, 0, tzinfo=HOSPITAL_TZ)
            local_end = local_start + timedelta(days=1)
            ts_start = int(local_start.astimezone(timezone.utc).timestamp())
            ts_end = int(local_end.astimezone(timezone.utc).timestamp())

            flt = Filter(
                must=[
                    FieldCondition(
                        key="paciente_id", match=MatchValue(value=str(paciente_id))
                    ),
                    FieldCondition(
                        key="fecha_ts", range=Range(gte=ts_start, lt=ts_end)
                    ),
                ]
            )
            cnt = qdrant_client.count(
                collection_name=collection, count_filter=flt, exact=False
            )
            if getattr(cnt, "count", 0) > 0:
                years_found.add(year)
        except Exception:
            # Nunca rompemos por esto
            continue

    if len(years_found) == 1:
        return list(years_found)[0]
    return None


def handle_missing_year(
    question: str,
    paciente_id: str,
    policy: MissingYearPolicy = _DEFAULT_POLICY,
    *,
    qdrant_client: Any = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decide qué hacer cuando la pregunta trae fecha/mes sin año.
    Ahora es "auto": primero intenta inferir (rango o día), y recién si no puede, pregunta.
    """
    if not detect_missing_year(question):
        return {"action": "CONTINUE"}

    # Si no tenemos Qdrant o colección, no podemos inferir → pedimos
    if not qdrant_client or not collection:
        return {
            "action": "ASK",
            "message": "Por favor especificá el año (ej.: '2024-01-09' o '09/01/2024').",
        }

    qn = question.strip().lower()

    # 1) Intentar rango tipo "30/11 al 03/12" o "30/11 y 03/12"
    m_range = re.search(r"(\d{1,2})/(\d{1,2}).+?(\d{1,2})/(\d{1,2})", qn)
    if m_range:
        d1, m1, d2, m2 = int(m_range.group(1)), int(m_range.group(2)), int(m_range.group(3)), int(m_range.group(4))
        year = _infer_single_year_for_range(
            qdrant_client, collection, str(paciente_id), m1, d1, m2, d2
        )
        if year:
            return {"action": "ASSUME", "year": year}

    # 2) Si no era un rango, probar día único (tu función vieja)
    m_dm = re.search(r"\b(\d{1,2})/(\d{1,2})\b", qn)
    if m_dm:
        d, m = int(m_dm.group(1)), int(m_dm.group(2))
        year = _infer_single_year_for_day(
            qdrant_client, collection, str(paciente_id), m, d
        )
        if year:
            return {"action": "ASSUME", "year": year}

    # 3) Si era un mes por nombre → pedimos
    m_mon = re.search(
        r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b",
        qn,
        re.IGNORECASE,
    )
    if m_mon:
        return {
            "action": "ASK",
            "message": "Indicá el año para el mes solicitado (ej.: 'diciembre 2023').",
        }

    # 4) Último recurso
    return {
        "action": "ASK",
        "message": "Por favor especificá el año (ej.: '2024-01-09' o '09/01/2024').",
    }



# -----------------------------------------------------------------------------
# Utilidades base de tiempo
# -----------------------------------------------------------------------------
def to_iso_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def day_bounds(dt: datetime) -> Tuple[datetime, datetime]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=HOSPITAL_TZ)
    else:
        dt = dt.astimezone(HOSPITAL_TZ)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def iso_day_to_midnight_ts(iso_day: str) -> int:
    dt = datetime.fromisoformat(iso_day)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=HOSPITAL_TZ)
    else:
        dt = dt.astimezone(HOSPITAL_TZ)
    dt0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(dt0.astimezone(timezone.utc).timestamp())


# -----------------------------------------------------------------------------
# Detección de "año faltante"
# -----------------------------------------------------------------------------
MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

_MISSING_YEAR_RX = re.compile(
    r"""
    (?:\b\d{1,2}/\d{1,2}\b)                                   # dd/mm
    |(?:\b(enero|febrero|marzo|abril|mayo|junio|julio|
           agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b(?!\s+20\d{2}))
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_missing_year(q: str) -> bool:
    """Devuelve True si la pregunta menciona día/mes o mes sin especificar año."""
    if not q:
        return False
    qn = q.strip().lower()
    # Si ya hay ancla inequívoca, no pedimos año
    if any(
        k in qn
        for k in (
            "este mes",
            "este año",
            "ayer",
            "hoy",
            "última semana",
            "ultima semana",
        )
    ):
        return False
    return bool(_MISSING_YEAR_RX.search(qn))


def enrich_question_with_year(question: str, year: int) -> str:
    """Adjunta la aclaración de año para guiar al parser temporal (sin cambiar el texto visible al médico)."""
    question = (question or "").strip()
    # Si ya hay un año (20xx) no tocar.
    if re.search(r"\b20\d{2}\b", question):
        return question
    return f"{question} (año {year})"


def assumed_year_note(year: int) -> str:
    return f"Nota: se asumió el año {year}; especificá otro año si corresponde."


# -----------------------------------------------------------------------------
# LLM parsing de intención temporal
# -----------------------------------------------------------------------------
TEMPORAL_INTENT_PROMPT = PromptTemplate(
    """
Eres un Agente de Detección Temporal. Analiza la 'PREGUNTA' e identifica la ventana de tiempo y superlativos.

REGLAS:
1) start_iso y end_iso en ISO-8601 completos: 'YYYY-MM-DD HH:MM:SS'.
2) Usa la FECHA ACTUAL para relativas (ej: "última semana").
3) Si es un día/rango: start_iso = medianoche del inicio, end_iso = medianoche del día siguiente (límite EXCLUSIVO).
4) Si no hay intención temporal: start_iso = null, end_iso = null, temporal_intent_type = 'NONE'.
5) Si la pregunta pide "último(s) N" o "primero(s) N", completa:
   - which = 'last' o 'first'
   - count = N (por defecto 1 si no se indica)
6) Idioma: Español. Responde SOLO el JSON pedido.

FECHA ACTUAL: {now} (TZ: {tz_name})
PREGUNTA: {query_str}

{format_instructions}
"""
)


def _parse_iso_str(s: str | None) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # Completar formatos parciales
    if len(s) == 4 and s.isdigit():
        s = f"{s}-01-01 00:00:00"
    elif len(s) == 7 and s[4] == "-":
        s = f"{s}-01 00:00:00"
    elif len(s) == 10 and s[4] == "-" and s[7] == "-":
        s = f"{s} 00:00:00"

    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=HOSPITAL_TZ)
    else:
        dt = dt.astimezone(HOSPITAL_TZ)
    return dt


def _ensure_exclusive_end(
    start_dt: datetime, end_dt: datetime
) -> Tuple[datetime, datetime]:
    if end_dt <= start_dt:
        end_dt = start_dt.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
    return start_dt, end_dt


# -----------------------------------------------------------------------------
# HYBRID TEMPORAL PARSER (Rule-based + LLM)
# -----------------------------------------------------------------------------
class HybridTemporalParser:
    """
    Parsea intenciones temporales usando una estrategia híbrida:
    1. Determinística (Reglas/Regex): Para casos obvios ("último", "ayer"). Rápido y seguro.
    2. Probabilística (LLM): Para lenguaje natural complejo ("cuando vine por la gripe").
    """
    
    # Reglas estáticas (determinismo)
    KEYWORDS_LAST = ["ultimo", "ultima", "ultimos", "ultimas", "reciente", "recientes"]
    KEYWORDS_FIRST = ["primero", "primera", "primeros", "primeras", "antiguo", "antiguos"]
    
    def __init__(self, llm: LLM):
        self.llm = llm

    def parse(self, question: str, now: datetime, context_details: dict = None) -> Optional[Dict[str, Any]]:
        q_norm = self._normalize(question)
        
        # 1. Capa Determinística: Superlativos explícitos
        # Prioridad absoluta porque define el ordenamiento vs reranking
        det_super = self._try_deterministic_superlative(q_norm)
        if det_super:
            log.info(f"[HybridParser] Superlativo detectado por REGLA: {det_super['which']}")
            # Aún si detectamos esto, podríamos querer ver si hay fechas explicitas, 
            # pero generalmente "último" mata fecha salvo "último de 2023".
            # Por simplicidad, si es "último registro", devolvemos eso.
            return det_super

        # 2. Capa Probabilística: LLM
        # Si no hubo regla fuerte, delegamos al modelo
        return self._try_llm(question, now, context_details)

    def _normalize(self, text: str) -> str:
        if not text: 
            return ""
        # Quitamos acentos para matching simple
        try:
            return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
        except:
            return text.lower()

    def _try_deterministic_superlative(self, q_norm: str) -> Optional[Dict[str, Any]]:
        # Detector simple de "último/a"
        if any(w in q_norm for w in self.KEYWORDS_LAST):
            return {
                "type": "range", 
                "start": None, "end": None, 
                "which": "last", "count": 1, 
                "intent_type": "RELATIVE"
            }
        
        if any(w in q_norm for w in self.KEYWORDS_FIRST):
            return {
                "type": "range", 
                "start": None, "end": None, 
                "which": "first", "count": 1, 
                "intent_type": "RELATIVE"
            }
        return None

    def _try_llm(self, question: str, now: datetime, context: dict) -> Optional[Dict[str, Any]]:
        # Reutilizamos la lógica del prompt existente
        parser = PydanticOutputParser(TemporalWindow)
        format_instructions = parser.get_format_string()

        prompt = TEMPORAL_INTENT_PROMPT.partial_format(
            now=now.isoformat(),
            tz_name=HOSPITAL_TZ.tzname(now),
            format_instructions=format_instructions,
        )
        try:
            raw = self.llm.predict(prompt, query_str=question.strip())
            tw: TemporalWindow = parser.parse(raw)
            return self._convert_pydantic_to_dict(tw)
        except Exception as e:
            log.warning(f"[HybridParser] LLM parse error: {e}")
            return None

    def _convert_pydantic_to_dict(self, tw: TemporalWindow) -> Dict[str, Any]:
        start_dt = _parse_iso_str(tw.start_iso)
        end_dt = _parse_iso_str(tw.end_iso)
        
        which = (tw.which or "").strip().lower() or None
        if which not in ("last", "first"):
            which = None
            
        try:
            count = int(tw.count or 1)
        except:
            count = 1
        count = max(1, min(count, 100))

        if start_dt and end_dt:
            start_dt, end_dt = _ensure_exclusive_end(start_dt, end_dt)
        
        return {
            "type": "range",
            "start": start_dt,
            "end": end_dt,
            "which": which,
            "count": count,
            "intent_type": tw.temporal_intent_type,
        }

# -----------------------------------------------------------------------------
# Function Wrappers para mantener compatibilidad
# -----------------------------------------------------------------------------
def parse_temporal_intent(
    question: str,
    llm: LLM,
    now: datetime | None = None,
    *,
    paciente_id: str = "",
    qdrant_client: Any = None,
    collection: Optional[str] = None,
    missing_year_policy: MissingYearPolicy = _DEFAULT_POLICY,
) -> Optional[Dict[str, Any]]:
    
    if not question:
        return None
    if now is None:
        now = datetime.now(HOSPITAL_TZ)

    # 1. Manejo de Año Faltante (lógica existente preservada)
    my = handle_missing_year(
        question,
        paciente_id or "",
        policy=missing_year_policy,
        qdrant_client=qdrant_client,
        collection=collection,
    )
    if my.get("action") == "ASK":
        return {"type": "missing-year", "message": my.get("message") or "Especificá el año."}
    
    if my.get("action") == "ASSUME":
        year = int(my["year"])
        question = enrich_question_with_year(question, year)

    # 2. Delegar al Parser Híbrido
    import unicodedata # Asegurar import local si falta arriba
    parser = HybridTemporalParser(llm)
    return parser.parse(question, now)
    
# ... rest of utils ...

def intent_to_ts_window(intent: dict | None) -> tuple[int | None, int | None]:
    if not intent or intent.get("type") != "range":
        return None, None
    start_dt: datetime | None = intent.get("start")
    end_dt: datetime | None = intent.get("end")
    if not start_dt or not end_dt:
        return None, None
    ts_start = int(start_dt.astimezone(timezone.utc).timestamp())
    ts_end = int(end_dt.astimezone(timezone.utc).timestamp())
    return ts_start, ts_end


def parse_iso_to_dt(s: str | None):
    return _parse_iso_str(s)


def intent_to_ts_window(intent: dict | None) -> tuple[int | None, int | None]:
    if not intent or intent.get("type") != "range":
        return None, None
    start_dt: datetime | None = intent.get("start")
    end_dt: datetime | None = intent.get("end")
    if not start_dt or not end_dt:
        return None, None
    ts_start = int(start_dt.astimezone(timezone.utc).timestamp())
    ts_end = int(end_dt.astimezone(timezone.utc).timestamp())
    return ts_start, ts_end


def parse_iso_to_dt(s: str | None):
    """
    Convierte ISO a datetime con TZ del hospital.
    Acepta: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', '...Z' y 'T'.
    """
    return _parse_iso_str(s)


