# app/tools/planner_tool.py
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Importamos el parser de reglas (EL JUEZ) 
from app.utils import temporal_parser as rules_temporal


# Helpers existentes
from app.helpers.service_matcher import detect_servicio
from app.helpers.query_helpers import mentions_internacion
from app.config.time import HOSPITAL_TZ
from app.utils.context_manager import get_patient_context

log = logging.getLogger(__name__)

# --- Constantes ---
SECTION_ALIASES: Dict[str, List[str]] = {
    "MEDICACIÓN": ["MEDICACIÓN", "INDICACIONES", "EVOLUCIÓN", "EPICRISIS"],
    "EVOLUCIÓN": ["EVOLUCIÓN", "INTERCONSULTA", "INDICACIONES"],
    "MOTIVO DE CONSULTA": ["MOTIVO DE CONSULTA", "ANTECEDENTES", "INTERNACION"],
}

_NAME_PATTERNS: List[Tuple[str, str]] = [
    ("EVOLUCIÓN", r"\b(evoluc(?:ion|ión)(es)?|seguimiento|control(?:es)?)\b"),
    ("MOTIVO DE CONSULTA", r"\b(motivo\s+de\s+consulta(s)?|mc:?)\b"),
    ("INDICACIONES", r"\b(indicaci(?:on|ón|ones)|orden(es)?|prescrip(?:cion|ción)(es)?|dosis|mg|fármaco|farmaco|tratamiento)\b"),
    ("INTERCONSULTA", r"\b(interconsulta(s)?|interc\.?on)\b"),
    ("EPICRISIS", r"\b(epicrisis|resumen\s+de\s+egreso)\b"),
    ("ANTECEDENTES", r"\b(antecedente(?:s)?|historia)\b"),
]

# Regex para intención de RESUMEN
_SUMMARY_PATTERNS = r"\b(resumen|resum(?:e|í|a|as)|sinteti(?:z|c)(?:a|e|es)|sintesis|synthesize|summarize|triage)\b"


def detect_name_hint(question: str) -> Optional[str]:
    ql = (question or "").lower()
    for name, rx in _NAME_PATTERNS:
        try:
            if re.search(rx, ql, re.I):
                return name
        except Exception:
            continue
    return None


def get_query_plan_tools(pregunta: str) -> Dict[str, Any]:
    """
    Analiza la pregunta y genera un Plan de Consulta.
    ARQUITECTURA HÍBRIDA (PROPOSER-JUDGE) - VERSIÓN PRODUCTION READY
    """
    try:
        log.info(f"🚀 [PlannerTool] Iniciando análisis para: '{pregunta}'")
        
        # Inicialización de variables (Scope seguro)
        ts_start = None
        ts_end = None
        source_time = "none" # 'rules' | 'llm' | 'none'
        semantic_plan = {}   # Evitamos KeyError si no se llama al LLM
        which = None
        count_val = None     # Usamos variable temporal para count
        
        # --- PASO 1: ANÁLISIS TEMPORAL (Short-Circuit Logic) ---
        
        # A) Intentamos Reglas primero (Rápido y Determinista)
        tw_rules = rules_temporal.parse_temporal_intent(pregunta)
        
        if tw_rules and (tw_rules.start_iso or tw_rules.end_iso):
            log.info("⚡ [Planner] Fechas detectadas por REGLAS.")
            source_time = "rules"
            
            # Recuperamos metadatos ricos del parser (which, count)
            which = tw_rules.which
            count_val = tw_rules.count
            
            # Convertir ISO del parser a Timestamp
            if tw_rules.start_iso:
                dt = datetime.fromisoformat(tw_rules.start_iso).replace(tzinfo=HOSPITAL_TZ)
                ts_start = int(dt.timestamp())
            
            if tw_rules.end_iso:
                dt = datetime.fromisoformat(tw_rules.end_iso).replace(tzinfo=HOSPITAL_TZ)
                # NORMALIZACIÓN DE SEMÁNTICA:
                # El parser devuelve Fin Exclusivo (Start of Next Day).
                # Para consistencia con Qdrant (lte), restamos 1 segundo para obtener 23:59:59 del día objetivo.
                ts_end = int(dt.timestamp()) - 1

            # --- SMART BUFFER (Evolución diferida) ---
            # Si la ventana es pequeña (<= 24h), extendemos el fin 24h más.
            # Esto captura notas escritas el día siguiente (ej: "Ayer día 29...").
            if ts_start and ts_end and (ts_end - ts_start) <= 86400:
                log.info("🗓️ [Smart Buffer] Extendiendo búsqueda +24h para capturar evoluciones diferidas.")
                ts_end += 86400
        
        else:
            # B) Si las reglas fallan, llamamos al LLM (Inteligente y Flexible)
            log.info("🤔 [Planner] Reglas sin resultado. LLM Planner desactivado manualmente. Fallback a modo estricto.")
            # semantic_plan = get_smart_date_window(pregunta)
            
            # if semantic_plan.get("start_date_iso"):
            #     source_time = "llm"
            #     try:
            #         dt_start = datetime.fromisoformat(semantic_plan["start_date_iso"])
            #         if dt_start.tzinfo is None: dt_start = dt_start.replace(tzinfo=HOSPITAL_TZ)
            #         ts_start = int(dt_start.timestamp())
            #     except Exception as e: log.warning(f"⚠️ Planner LLM start err: {e}")

            # if semantic_plan.get("end_date_iso"):
            #     try:
            #         dt_end = datetime.fromisoformat(semantic_plan["end_date_iso"])
            #         if dt_end.tzinfo is None: dt_end = dt_end.replace(tzinfo=HOSPITAL_TZ)
            #         # El LLM devuelve YYYY-MM-DD (día inclusive).
            #         # Para consistencia, sumamos 23h 59m 59s.
            #         ts_end = int(dt_end.timestamp()) + 86399 
            #     except Exception as e: log.warning(f"⚠️ Planner LLM end err: {e}")


        # --- PASO 2: DETECCIÓN DE SERVICIOS ---
        service_matches_raw = detect_servicio(pregunta, top_k=3) or []
        service_matches = [s for s in service_matches_raw if float(s.get("score", 0)) >= 0.7]

        # --- PASO 3: INTENCIÓN DE RESUMEN Y DOCUMENTOS ---
        is_summary_regex = bool(re.search(_SUMMARY_PATTERNS, pregunta, re.IGNORECASE))
        is_summary_llm = (semantic_plan.get("intent") == "summary")
        is_summary = is_summary_regex or is_summary_llm

        # Definir límite de documentos (count)
        # Si las reglas trajeron un conteo específico ("últimas 3"), lo respetamos.
        # Si no, usamos defaults según si es resumen o puntual.
        if count_val and count_val > 1:
            final_count = count_val
        else:
            final_count = 35 if is_summary else 10

        # --- PASO 4: NAME HINTS & ALIASES ---
        name_hint = detect_name_hint(pregunta)
        section_aliases = SECTION_ALIASES.get(name_hint, []) if name_hint else []

        # --- PASO 5: ESTRATEGIA DE BÚSQUEDA (Strict Mode) ---
        # Regla: Si hay fechas explícitas (de reglas o LLM), somos estrictos.
        strict_mode = (ts_start is not None or ts_end is not None)
        
        # --- PASO 6: CONSTRUCCIÓN DEL PLAN FINAL ---
        plan = {
            "service_hints": service_matches,
            "name_hint": name_hint, 
            "section_aliases": section_aliases, # Ahora sí se usa 🚀
            "time_window": {
                "start": ts_start,
                "end": ts_end
            },
            "is_inpatient": mentions_internacion(pregunta),
            "is_summary": is_summary,
            "which": which,
            "count": final_count, 
            "strict": strict_mode,
            # Metadata informativa
            "_source_time": source_time,
            "_debug_rationale": semantic_plan.get("rationale")
        }
        
        log.info(f"✨ [PlannerTool] Plan Final ({source_time}): {plan}")
        return plan

    except Exception as e:
        log.error(f"❌ Error crítico en planner_tool: {e}", exc_info=True)
        return {"error": str(e), "strict": False}