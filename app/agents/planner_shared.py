# app/agents/planner_shared.py
from __future__ import annotations
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any

from app.helpers.service_matcher import detect_servicio
from app.helpers.query_helpers import mentions_internacion
from app.config.time import HOSPITAL_TZ
from app.schemas.ontology import QueryPlan
from app.schemas.esquema import ConsultaQdrantRequest

# 👇 IMPORTAMOS tus utilidades YA existentes
from app.utils.temporal_intent_utils import (
    parse_temporal_intent as llm_temporal_intent,   # LLM
    intent_to_ts_window,
)
from app.utils import temporal_parser as rules_temporal  # reglas puras

log = logging.getLogger(__name__)

SECTION_ALIASES: Dict[str, List[str]] = {
    "MEDICACIÓN": ["MEDICACIÓN", "INDICACIONES", "EVOLUCIÓN", "EPICRISIS"],
    "EVOLUCIÓN": ["EVOLUCIÓN", "INTERCONSULTA", "INDICACIONES"],
  
}

_NAME_PATTERNS: List[Tuple[str, str]] = [
    ("EVOLUCIÓN", r"\b(evoluc(?:ion|ión)(es)?|seguimiento|control(?:es)?)\b"),
    ("MOTIVO DE CONSULTA", r"\b(motivo\s+de\s+consulta(s)?|mc:?)\b"),
    ("INDICACIONES", r"\b(indicaci(?:on|ón|ones)|orden(es)?|prescrip(?:cion|ción)(es)?)\b"),
    ("INTERCONSULTA", r"\b(interconsulta(s)?|interc\.?on)\b"),
    ("EPICRISIS", r"\b(epicrisis|resumen\s+de\s+egreso)\b"),

]

# 👇 Solo para el último fallback literal dd/mm/yyyy ... dd/mm/yyyy
_DDMMYYYY_RX = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}).*?(\d{1,2}/\d{1,2}/\d{4})")


def detect_name_hint(question: str) -> Optional[str]:
    ql = (question or "").lower()
    for name, rx in _NAME_PATTERNS:
        try:
            if re.search(rx, ql, re.I):
                log.info(f"[planner_shared.name_hint] detectado '{name}' con patrón '{rx}'")
                return name
        except Exception:
            continue
    return None


def _ddmmyyyy_to_epoch(d: str, end: bool = False) -> Optional[int]:
    try:
        dt = datetime.strptime(d, "%d/%m/%Y").replace(tzinfo=HOSPITAL_TZ)
        if end:
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return int(dt.timestamp())
    except Exception:
        return None


def _fallback_argentine_range(q: str) -> tuple[Optional[int], Optional[int]]:
    m = _DDMMYYYY_RX.search(q)
    if not m:
        return None, None
    ts1 = _ddmmyyyy_to_epoch(m.group(1), end=False)
    ts2 = _ddmmyyyy_to_epoch(m.group(2), end=True)
    return ts1, ts2


def build_query_plan(
    llm,
    pregunta: str,
    payload: ConsultaQdrantRequest,
) -> QueryPlan:
    log.debug(f"[planner_shared] armando plan para: {pregunta!r}")

    # 1) servicio
    service_matches_raw = detect_servicio(pregunta, top_k=3) or []
    service_matches = [s for s in service_matches_raw if float(s.get("score", 0)) >= 0.7]
    log.debug(f"[planner] servicios detectados: {service_matches}")

    # 2) sección
    name_hint = detect_name_hint(pregunta)
    log.debug(f"[planner] name_hint: {name_hint}")
    section_aliases: List[str] = []
    if name_hint and name_hint in SECTION_ALIASES:
        section_aliases = SECTION_ALIASES[name_hint]

    # ------------------------------------------------------------------
    # 3) TIEMPO (orquestación)
    # ------------------------------------------------------------------
    ts_start: Optional[int] = None
    ts_end: Optional[int] = None
    which: Optional[str] = None
    count: int = 1

    # 3.a) primero LLM → usa TODO lo que ya tenés (año faltante, etc.)
    temporal_intent_llm = llm_temporal_intent(pregunta, llm=llm)  # puede devolver missing-year
    log.debug(f"[planner.time] intent LLM crudo: {temporal_intent_llm}")

    if temporal_intent_llm and temporal_intent_llm.get("type") == "range":
        # convierte a epoch con la función YA hecha
        ts_start, ts_end = intent_to_ts_window(temporal_intent_llm)
        which = temporal_intent_llm.get("which")
        count = int(temporal_intent_llm.get("count") or 1)

    # 3.b) si el LLM dijo “missing-year”, dejamos el mensaje pero tratamos de no romper
    elif temporal_intent_llm and temporal_intent_llm.get("type") == "missing-year":
        # acá NO seteamos ts_start/ts_end → lo verá la capa de arriba
        log.debug("[planner.time] LLM pidió año → no se aplica ventana todavía")

    # 3.c) si el LLM no detectó ventana → usamos TU parser por reglas
    if not ts_start and not ts_end:
        tw_rules = rules_temporal.parse_temporal_intent(pregunta)
        if tw_rules:
            # tw_rules es app.schemas.temporal.TemporalWindow
            # lo convertimos igual que el LLM
            if tw_rules.start_iso and tw_rules.end_iso:
                # reutilizamos parse_iso_to_dt o lo hacemos directo:
                dt_start = datetime.fromisoformat(tw_rules.start_iso).replace(tzinfo=HOSPITAL_TZ)
                dt_end = datetime.fromisoformat(tw_rules.end_iso).replace(tzinfo=HOSPITAL_TZ)
                ts_start = int(dt_start.timestamp())
                ts_end = int(dt_end.timestamp())
            which = (tw_rules.which or "").lower() or which
            count = int(tw_rules.count or count)
            log.debug(f"[planner.time] ventana por reglas: start={ts_start} end={ts_end}")

            # 👇 AJUSTE CLÍNICO: "entre 17/11/2023 y 18/11/2023" debe incluir los 2 días
            if ts_start and ts_end:
                # si la ventana es de exactamente 1 día...
                if (ts_end - ts_start) == 86400:
                    # ...y el texto tiene dos fechas completas con "entre|del ... y|al ..."
                    if re.search(
                        r"\b(entre|del)\s+\d{1,2}/\d{1,2}/\d{4}\s+(y|al)\s+\d{1,2}/\d{1,2}/\d{4}",
                        pregunta,
                        re.IGNORECASE,
                    ):
                        ts_end = ts_end + 86400  # +1 día → hace el rango inclusivo
                        log.info(
                            "[planner.time] rango dd/mm/yyyy-dd/mm/yyyy detectado → extendido 1 día para incluir la fecha final"
                        )

    # 3.d) último recurso → rango argentino literal
    if not ts_start and not ts_end:
        ts_start, ts_end = _fallback_argentine_range(pregunta.lower())
        if ts_start or ts_end:
            log.info(f"[planner_shared.fallback] detectado rango dd/mm/yyyy → start={ts_start} end={ts_end}")

    # ------------------------------------------------------------------
    # 4) internación
    # ------------------------------------------------------------------
    is_inpatient = mentions_internacion(pregunta)
    log.debug(f"[planner] is_inpatient={is_inpatient}")

    # ------------------------------------------------------------------
    # 5) armamos el plan
    #    👉 si tenemos epoch, los dejamos como INT, no como string
    # ------------------------------------------------------------------
    plan = QueryPlan(
        service_hints=service_matches,
        name_hint=name_hint,
        section_aliases=section_aliases,
        time_window={
            "start": str(ts_start) if ts_start else "",
            "end": str(ts_end) if ts_end else "",
        },
        is_inpatient=is_inpatient,
        which=which,
        count=count,
    )

    has_service = bool(service_matches)
    has_name = bool(name_hint)
    has_time = bool(ts_start or ts_end)
    has_inpatient = bool(is_inpatient)

    if not (has_service or has_name or has_time or has_inpatient):
        # plan sin señales → que el router sepa que no hay nada para aplicar
        return QueryPlan(
            service_hints=[],
            name_hint=None,
            section_aliases=[],
            time_window={"start": "", "end": ""},
            is_inpatient=False,
            which=None,
            count=1,
        )

    return plan
