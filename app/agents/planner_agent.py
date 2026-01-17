# app/agents/planner_agent.py
import logging
from typing import Optional

from llama_index.core.llms import LLM

from app.schemas.esquema import ConsultaQdrantRequest
from app.schemas.ontology import QueryPlan
from app.agents.planner_shared import build_query_plan, detect_name_hint  # 👈 uso central
from app.helpers.service_matcher import detect_servicio  # sigue expuesto por compatibilidad

log = logging.getLogger(__name__)

# exportamos lo que el resto del código espera
__all__ = [
    "plan_query_filters",
    "detect_servicio",
    "detect_name_hint",
]

def plan_query_filters(llm: LLM, pregunta: str, payload: ConsultaQdrantRequest) -> QueryPlan:
    """
    Wrapper histórico. Internamente usamos build_query_plan(...) para que el
    query_engine_manager_service y cualquier otro servicio no dupliquen lógica.
    """
    try:
        return build_query_plan(llm, pregunta, payload)
    except Exception as e:
        log.error(f"[planner_agent] fallo al planificar: {e}", exc_info=True)
        # fallback mínimo
        return QueryPlan(
            service_hints=[],
            name_hint=None,
            section_aliases=[],
            time_window={"start": "", "end": ""},
            is_inpatient=False,
            which=None,
            count=1,
        )
