# app/services/query_engine_manager_service.py
"""
Query Engine (seguro, simple y sin mezcla)

• Sin defaults globales: Settings.llm = None / Settings.embed_model = None.
• LLM y embedder por petición.
• SIN CACHÉ.
• Filtro Qdrant por paciente_id aplicado en CADA retrieval (vector_store_kwargs={"qdrant_filters": ...}).
• Modo estricto: si llega 1 nodo de otro paciente, aborta (raise).
• Auditoría preflight (count/scroll).
• Planner ACTIVADO por defecto (plan_query_filters).
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
import unicodedata
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Iterable, Generator

# 🔒 Evita defaults globales
try:
    from llama_index.core import Settings
    Settings.llm = None
    Settings.embed_model = None
except Exception:
    pass

from fastapi import Depends
from httpx import ReadTimeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pybreaker import CircuitBreaker

# Qdrant (http.models mantiene compatibilidad con el cliente HTTP)
from qdrant_client.http.models import (
    Filter, FieldCondition, Range, MatchValue, CountRequest
)

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank


from app.infra.ml_providers import get_llm, get_embedder
from app.schemas.esquema import ConsultaQdrantRequest

from app.utils.meta_to_context import MetaToContext
from app.utils.temporal_intent_utils import parse_temporal_intent, intent_to_ts_window
from app.config.time import HOSPITAL_TZ

# ⬇️ Helpers centralizados (NO dependemos de funciones que no existan)
from app.helpers.query_helpers import (
    COLLECTION, _qdrant,
    q_filter_paciente, q_filter_from_payload, ensure_payload_indexes,
    audit_patient_slice, detect_qdrant_text_key,
    SafeQdrantVectorStore, sanitize_nodes_for_llamaindex,
    serialize_response, sources_ok,
    sort_and_slice_by_temporal_super,
    get_latest_procedure_for_patient, mentions_internacion, 
    dump_qdrant_filter, dump_query_plan,
    safe_add  
)

# 🚀 Planner + señales (Top-K servicio, NAME clínico)
from app.agents.planner_agent import (
    plan_query_filters,         # ACTIVADO por defecto
    detect_servicio,
    detect_name_hint,
)

log = logging.getLogger(__name__)
_DEBUG_FLAG = os.getenv("HCI_DEBUG", os.getenv("HCI_DEBUG", "0"))
if not bool(int(str(_DEBUG_FLAG))):
    log.setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────
# Tunables con defaults seguros (si no hay ENV, usan estos)
# ─────────────────────────────────────────────────────────────
SIM_TOP_K = max(1, int(os.getenv("SIMILARITY_TOP_K", "12")))
SUMMARY_TOP_K = max(SIM_TOP_K * 4, int(os.getenv("SUMMARY_TOP_K", "50")))
LLAMA_VERBOSE = bool(int(os.getenv("LLAMA_VERBOSE", "0")))

STRICT_FILTER = True  # 🔒 abortar si llega algún nodo cruzado
breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

SERVICE_LLM_FILTERS = bool(int(os.getenv("SERVICE_LLM_FILTERS", "1")))
SERVICE_FILTER_MIN_SCORE = float(os.getenv("SERVICE_FILTER_MIN_SCORE", "0.60"))
SERVICE_FILTER_MODE = os.getenv("SERVICE_FILTER_MODE", "auto").lower()  # auto|must|should
SERVICE_FILTER_ZEROHITS_FALLBACK = bool(int(os.getenv("SERVICE_FILTER_ZEROHITS_FALLBACK", "1")))
SERVICE_FILTER_MAX_OR = max(1, int(os.getenv("SERVICE_FILTER_MAX_OR", "3")))

# ─────────────── Utilidades locales seguras ───────────────
def _strip_accents(s: str) -> str:
    """Quita tildes/acentos sin dependencias externas."""
    if not s:
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _wants_summary(q: str) -> bool:
    """
    Detección tolerante a typos y sinónimos para intención de 'resumen'.
    """
    norm = _strip_accents((q or "").lower())
    triggers = ["resumen", "reumen", "resumi", "resu ", "resu-", "sintesis", "sumario", "overview", "panorama", "entender"]
    if any(t in norm for t in triggers):
        return True
    if re.search(r"\bresu+\w*\b", norm):
        return True
    return False

def _scroll_sample_keys(sample_filter: Optional[Filter] = None, limit: int = 24) -> list[dict]:
    """Obtiene una muestra de payloads para inspeccionar claves disponibles en la colección."""
    try:
        pts, _ = _qdrant.scroll(collection_name=COLLECTION, scroll_filter=sample_filter, with_payload=True, limit=limit)
        return [getattr(p, "payload", {}) or {} for p in (pts or [])]
    except Exception:
        return []

def _field_exists_in_collection(field_key: str, sample_filter: Optional[Filter] = None) -> bool:
    """Heurística: true si en una muestra aparece la clave."""
    if not field_key:
        return False
    for pl in _scroll_sample_keys(sample_filter, limit=32):
        if field_key in (pl or {}):
            return True
    return False

def _safe_add(q_filter: Filter, condition: Any, field_key: Optional[str] = None,
              sample_filter: Optional[Filter] = None, must: bool = True) -> None:
    """
    Agrega la condición si el campo existe (o si no se especificó field_key).
    Evita 'matar' el recall por filtrar con un campo inexistente.
    """
    try:
        if (field_key is None) or _field_exists_in_collection(field_key, sample_filter):
            if must:
                q_filter.must.append(condition)
            else:
                q_filter.should.append(condition)
    except Exception:
        # En caso de duda, no agregamos nada para no reducir recall.
        pass


# ─────────────── QueryEngine ───────────────
class SimpleQueryEngine:
    def __init__(
        self,
        paciente_id: str,
        retriever,
        synthesizer,
        meta_to_context: Optional[MetaToContext] = None,
        temporal_superlative: Optional[Dict[str, Any]] = None,
        # 🆕 Añadimos parámetros para control de Reranking
        reranker_top_n: int = 10,
        use_reranker: bool = False,  # 🔴 DESACTIVADO: usar solo reranker global de rag_tool.py
        similarity_top_k_retrieve: int = 50, # Nueva cantidad a pedir a Qdrant
        text_key: str = "text", # 🆕 Recibimos la key detectada
    ):
        self._pid = str(paciente_id)
        self._text_key = text_key
        self._retriever = retriever
        self._synth = synthesizer
        self._meta_pp = meta_to_context
        # 'which': 'last' | 'first' | None ; 'count': int
        self._temporal_super = temporal_superlative or {"which": None, "count": 1}

        # 🆕 Configuración del Reranker (Modelo Open Source BGE)
        # 🔴 DESACTIVADO para evitar CUDA OOM - usar solo reranker global de rag_tool.py
        self._reranker = None
        if use_reranker:
            log.info(f"🔧 [reranker] Inicializando SentenceTransformerRerank con top_n={reranker_top_n}")
            self._reranker = SentenceTransformerRerank(
                    model="BAAI/bge-reranker-v2-m3", 
                    top_n=reranker_top_n, 
            )
        else:
            log.info(f"⚠️ [reranker] Reranker local DESACTIVADO - se usará reranker global de rag_tool.py")
        # 🆕 Guardamos el valor para control, aunque se pasa al retriever en el router
        self._retrieve_top_k = similarity_top_k_retrieve



    def query(self, pregunta: str):
        # 1. Retrieval (ahora pide 50 nodos)
        nodes = self._retriever.retrieve(pregunta or "")


        # 🆕 CORRECCIÓN BUG 2 y 3: Asegurar que node.text contenga el texto real (TEXT_KEY)
        # Esto es vital para que el Reranker y el LLM reciban el contexto completo
        for n in (nodes or []):
            node = getattr(n, "node", n)
            md: Dict[str, Any] = getattr(node, "metadata", {}) or {}
            if not getattr(node, "text", ""):
                text_content = md.get(self._text_key, "") # Usamos el TEXT_KEY inyectado
                node.text = text_content

        # 🆕 2. Reranking: Aplicar el Cross-Encoder para alta precisión
        # ⚠️ CRÍTICO: Si el usuario pide "el último" (superlativo temporal), el reranker semántico
        # puede descartar el registro más reciente si su texto no es "semánticamente perfecto".
        # En esos casos, confiamos más en la recuperación vectorial + ordenamiento por fecha.
        if not self._temporal_super.get("which") and self._reranker is not None:
            try:
                # Selecciona los top_n (10) más relevantes de los nodos recuperados.
                nodes = self._reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(pregunta))
                log.debug(f"[reranker] Reranking aplicado. Nodos finales: {len(nodes)}.")
            except Exception as e:
                log.warning(f"[reranker] Falló el reranking: {e}. Usando nodos sin rerank.")
        else:
            if self._temporal_super.get("which"):
                log.info(f"[reranker] SKIPPING Reranker porque hay superlativo temporal: {self._temporal_super}")
            else:
                log.info(f"[reranker] SKIPPING Reranker (desactivado) - se usará reranker global de rag_tool.py")

        # Modo estricto: no deberían venir cruzados si Qdrant filtró
        filtered, dropped = [], 0
        for n in (nodes or []):
            node = getattr(n, "node", n)
            md: Dict[str, Any] = getattr(node, "metadata", {}) or {}
            if str(md.get("paciente_id")) == self._pid:
                filtered.append(n)
            else:
                dropped += 1

        if dropped:
            msg = f"[filter] DESCARTE ILEGAL: llegaron {dropped} nodos de otro paciente (esperado={self._pid})"
            log.error(msg)
            if STRICT_FILTER:
                raise RuntimeError(msg)

        if not filtered:
            return "⚠️ No hay contexto del paciente solicitado en la base (o el filtro no encontró coincidencias)."

        # Enriquecimiento de contexto
        try:
            if self._meta_pp:
                filtered = self._meta_pp.postprocess_nodes(filtered)
        except Exception as e:
            log.warning(f"[meta_to_context] fallo postprocess_nodes: {e}")

        # 👇 Aplicar superlativo temporal (first/last N)
        try:
            filtered = sort_and_slice_by_temporal_super(
                filtered,
                temporal_super=self._temporal_super,
                top_n=self._temporal_super.get("count", 1)
            )
        except Exception as e:
            log.warning(f"[temporal_super] fallo orden/slice: {e}")

        # 🔒 Saneado final ANTES del sintetizador
        filtered = sanitize_nodes_for_llamaindex(filtered)
        if not filtered:
            return "⚠️ No hay contexto textual utilizable para esta consulta."

        return self._synth.synthesize(pregunta or "", nodes=filtered)

    def stream(self, pregunta: str) -> Generator[Dict[str, Any], None, None]:
        """Streaming con el sintetizador inyectado en _build_router."""
        try:
            # 1) Retrieval
            nodes = self._retriever.retrieve(pregunta or "")

            # 🆕 CORRECCIÓN BUG 2 y 3: Asegurar que node.text contenga el texto real (TEXT_KEY)
            for n in (nodes or []):
                node = getattr(n, "node", n)
                md: Dict[str, Any] = getattr(node, "metadata", {}) or {}
                if not getattr(node, "text", ""):
                    text_content = md.get(self._text_key, "") # Usamos el TEXT_KEY inyectado
                    node.text = text_content

            # 🆕 2. Reranking en streaming
            if not self._temporal_super.get("which") and self._reranker is not None:
                try:
                    nodes = self._reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(pregunta))
                    log.debug(f"[reranker-stream] Reranking aplicado. Nodos finales: {len(nodes)}.")
                except Exception as e:
                    log.warning(f"[reranker-stream] Falló el reranking: {e}. Usando nodos sin rerank.")
            else:
                if self._temporal_super.get("which"):
                    log.info(f"[reranker-stream] SKIPPING Reranker porque hay superlativo temporal: {self._temporal_super}")
                else:
                    log.info(f"[reranker-stream] SKIPPING Reranker (desactivado) - se usará reranker global de rag_tool.py")

            # 2) Guard-rail por paciente
            filtered, dropped = [], 0
            for n in (nodes or []):
                node = getattr(n, "node", n)
                md: Dict[str, Any] = getattr(node, "metadata", {}) or {}
                if str(md.get("paciente_id")) == self._pid:
                    filtered.append(n)
                else:
                    dropped += 1

            if dropped:
                msg = f"[filter] DESCARTE ILEGAL: llegaron {dropped} nodos de otro paciente (esperado={self._pid})"
                log.error(msg)
                if STRICT_FILTER:
                    raise RuntimeError(msg)

            if not filtered:
                final = {"answer": "⚠️ No hay contexto del paciente solicitado en la base (o el filtro no encontró coincidencias).",
                         "sources": []}
                yield {"type": "token", "text": final["answer"]}
                yield {"type": "end", "final": final}
                return

            # 3) Enriquecimiento
            try:
                if self._meta_pp:
                    filtered = self._meta_pp.postprocess_nodes(filtered)
            except Exception as e:
                log.warning(f"[meta_to_context] fallo postprocess_nodes: {e}")

            # 4) Superlativo temporal
            try:
                filtered = sort_and_slice_by_temporal_super(
                    filtered,
                    temporal_super=self._temporal_super,
                    top_n=self._temporal_super.get("count", 1)
                )
            except Exception as e:
                log.warning(f"[temporal_super] fallo orden/slice: {e}")

            # 5) Synth de streaming
            synth_stream = getattr(self, "_synth_stream", None)
            try:
                if synth_stream is None:
                    resp = self._synth.synthesize(pregunta or "", nodes=filtered, streaming=True)
                else:
                    resp = synth_stream.synthesize(pregunta or "", nodes=filtered)
            except NotImplementedError as nie:
                log.warning(f"[stream] LLM sin soporte de streaming: {nie}")
                normal = self.query(pregunta or "")
                final_payload = serialize_response(normal)
                yield {"type": "token", "text": "⚠️ Streaming no soportado por el LLM configurado; respondiendo de forma estándar."}
                yield {"type": "end", "final": final_payload}
                return

            # 6) Emitir tokens
            acc_tokens = []
            for chunk in getattr(resp, "response_gen", []) or []:
                if chunk:
                    s = str(chunk)
                    acc_tokens.append(s)
                    yield {"type": "token", "text": s}

            # 7) Final con fuentes
            final_resp = resp.get_response() if hasattr(resp, "get_response") else resp
            final_payload = serialize_response(final_resp)

            # Fallback: sin texto del LLM
            if not final_payload.get("answer") or not final_payload["answer"].strip() or final_payload["answer"].strip().startswith("⚠️"):
                joined = "".join(acc_tokens).strip()
                if joined:
                    final_payload["answer"] = joined

            yield {"type": "end", "final": final_payload}

        except NotImplementedError as nie:
            log.warning(f"[stream] LLM sin soporte de streaming: {nie}")
            normal = self.query(pregunta or "")
            final_payload = serialize_response(normal)
            yield {"type": "token", "text": "⚠️ Streaming no soportado por el LLM configurado; respondiendo de forma estándar."}
            yield {"type": "end", "final": final_payload}
        except Exception as e:
            log.exception("[stream] error en SimpleQueryEngine.stream")
            yield {"type": "error", "message": str(e)}

# ─────────────── Router ───────────────
class SimpleRouter:
    def __init__(self, paciente_id: str, qe_vector: SimpleQueryEngine, qe_summary: SimpleQueryEngine) -> None:
        self._paciente_id = str(paciente_id)
        self._qe_vector = qe_vector
        self._qe_summary = qe_summary

    def query(self, pregunta: str) -> Response:
        use_summary = _wants_summary(pregunta)
        engine = self._qe_summary if use_summary else self._qe_vector
        return engine.query(pregunta or "")

    def stream(self, pregunta: str):
        use_summary = _wants_summary(pregunta)
        engine = self._qe_summary if use_summary else self._qe_vector
        yield from engine.stream(pregunta or "")


# ─────────────── CONSTRUCTOR DEL ROUTER ───────────────
def _build_router(
    payload: ConsultaQdrantRequest,
    llm: LLM,
    embed_model: Any,
) -> Optional[SimpleRouter]:
    """
    AFINAR el filtro con lo que devuelva el planner (si existe).
    Si el planner no está o no trae señales → se comporta como producción.
    """
    # 0) Validación dura igual que prod
    if not (payload.paciente_id or payload.procedureNumber):
        raise ValueError("Debe especificarse al menos paciente_id o procedureNumber.")

    model_name = (
        getattr(llm, "model", None)
        or getattr(getattr(llm, "metadata", None), "model_name", None)
        or "llm"
    )
    log.debug(f"[router] construyendo para paciente={payload.paciente_id}, modelo={model_name}")

    # 1) Planner opcional (no debe romper)
    plan = None
    if plan_query_filters is not None and getattr(payload, "pregunta", None):
        try:
            plan = plan_query_filters(llm, payload.pregunta, payload)
            log.info("[planner] activado: plan generado para la consulta")
            log.info(f"[planner] PLAN={dump_query_plan(plan)}")
        except Exception as e:
            log.debug(f"[planner] ignorado por error: {e}")
            plan = None

    # Helper interno: ¿el plan trae algo realmente útil?
    def _plan_has_signals(p) -> bool:
        if not p:
            return False
        if getattr(p, "service_hints", None):
            return True
        if getattr(p, "name_hint", None):
            return True
        tw = getattr(p, "time_window", {}) or {}
        if tw.get("start") or tw.get("end"):
            return True
        if getattr(p, "is_inpatient", False):
            return True
        if getattr(p, "procedure_number", None):
            return True
        return False

    plan_has_signals = _plan_has_signals(plan)

    # 2) Índices + auditoría (como prod)
    ensure_payload_indexes()
    if payload.paciente_id:
        audit_patient_slice(payload.paciente_id)

    # 3) Detectar TEXT_KEY (como prod)
    sample_filter = q_filter_paciente(str(payload.paciente_id)) if payload.paciente_id else None
    TEXT_KEY = detect_qdrant_text_key(COLLECTION, sample_filter)
    log.warning(f"[qdrant] usando TEXT_KEY='{TEXT_KEY}' para la colección {COLLECTION}")

    # 4) Vector store / index (como prod, con SafeQdrantVectorStore)
    store = SafeQdrantVectorStore(
        client=_qdrant,
        collection_name=COLLECTION,
        text_key=TEXT_KEY,
    )
    vector_index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)

    # 5) Ventana temporal
    ts_start: Optional[int] = None
    ts_end: Optional[int] = None

    if plan_has_signals and plan and hasattr(plan, "time_window"):
        # usar la ventana que propuso el planner
        try:
            from datetime import timedelta  # por si no estaba arriba

            def _parse_plan_dt(v, end=False):
                if v is None or v == "":
                    return None
                # epoch
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, str) and v.isdigit():
                    return int(v)
                # ISO simple
                s = str(v)
                if len(s) == 10 and s.count("-") == 2:
                    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=HOSPITAL_TZ)
                    if end:
                        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                    return int(dt.timestamp())
                # ISO con hora
                dt = datetime.fromisoformat(s.split("Z")[0]).replace(tzinfo=HOSPITAL_TZ)
                return int(dt.timestamp())

            tw = getattr(plan, "time_window", {}) or {}
            ts_start = _parse_plan_dt(tw.get("start"))
            ts_end = _parse_plan_dt(tw.get("end"), end=True)
            log.info(f"[time] usando ventana del planner: start={ts_start}, end={ts_end}")
        except Exception as e:
            log.debug(f"[time] ignorando ventana del planner por error: {e}")
            ts_start = None
            ts_end = None

    # si el planner no ayudó, usamos el parser híbrido
    intent = None
    if ts_start is None and ts_end is None and getattr(payload, "pregunta", None):
        now = datetime.now(HOSPITAL_TZ)
        intent = parse_temporal_intent(payload.pregunta or "", now=now, llm=llm)
        ts_start, ts_end = intent_to_ts_window(intent)
        if ts_start or ts_end:
            log.debug(f"[time] ventana detectada por intent híbrido: ts_start={ts_start}, ts_end={ts_end}")

    # 6) Filtro base (como prod): paciente + procedure + grupo + ventana
    q_filter = q_filter_from_payload(payload, ts_start=ts_start, ts_end=ts_end)
    if q_filter is None:
        q_filter = Filter()
    elif isinstance(q_filter, dict):
        q_filter = Filter(
            must=q_filter.get("must") or [],
            must_not=q_filter.get("must_not") or [],
            should=q_filter.get("should") or [],
        )

    # asegurar listas mutables
    q_filter.must = list(getattr(q_filter, "must", []) or [])
    q_filter.must_not = list(getattr(q_filter, "must_not", []) or [])
    q_filter.should = list(getattr(q_filter, "should", []) or [])

    # 6.b) guard-rails de producción SIEMPRE
    # excluir marker técnico
    q_filter.must_not.append(
        FieldCondition(key="marker", match=MatchValue(value="snapshot"))
    )
    # exigir fuente clínica
    q_filter.must.append(
        FieldCondition(key="source", match=MatchValue(value="hci"))
    )
    # exigir texto útil
    q_filter.must.append(
        FieldCondition(key="texto_len", range=Range(gt=0))
    )

    # 7) Afinaciones del planner (todas opcionales, nunca rompen)
    # 7.a) Internación detectada por planner
    if plan_has_signals and bool(getattr(plan, "is_inpatient", False)):
        # esto ya lo hace tu fuente de datos, pero lo reforzamos
        q_filter.must.append(
            FieldCondition(key="procedureNumber", range=Range(gt=0))
        )
        log.debug("🏥 [planner] is_inpatient=True → procedureNumber > 0")

    # 7.b) procedure_number propuesto por el planner (caso consulta puntual)
    if (
        plan_has_signals
        and getattr(plan, "procedure_number", None)
        and _field_exists_in_collection("procedureNumber", sample_filter)
    ):
        q_filter.must.append(
            FieldCondition(
                key="procedureNumber",
                match=MatchValue(value=int(plan.procedure_number)),
            )
        )
        log.info(f"[planner] usando procedureNumber={plan.procedure_number}")

    # 7.c) service_hints → solo si el planner los trajo y el campo existe
    has_service_should = False
    if (
        plan_has_signals
        and getattr(plan, "service_hints", None)
        and _field_exists_in_collection("servicio_id", sample_filter)
    ):
        # tomamos los IDs con mejor score
        sids_scored = []
        for m in (plan.service_hints or []):
            if m.get("servicio_id"):
                sids_scored.append((int(m["servicio_id"]), float(m.get("score", 0.0))))
        sids_scored.sort(key=lambda x: x[1], reverse=True)

        if sids_scored:
            # regla simple: si el top tiene muy buen score → MUST, si no → SHOULD
            top_sid, top_sc = sids_scored[0]
            if top_sc >= 0.8:
                q_filter.must.append(
                    FieldCondition(key="servicio_id", match=MatchValue(value=top_sid))
                )
                log.info(f"[planner] servicio_id={top_sid} aplicado como MUST (score={top_sc:.2f})")
            else:
                for sid, _ in sids_scored[:5]:
                    q_filter.should.append(
                        FieldCondition(key="servicio_id", match=MatchValue(value=sid))
                    )
                has_service_should = True
                log.info(f"[planner] servicios aplicados como SHOULD: {[sid for sid,_ in sids_scored[:5]]}")

    # 7.d) name_hint (sección) → también opcional
    if (
        plan_has_signals
        and getattr(plan, "name_hint", None)
        and _field_exists_in_collection("name", sample_filter)
    ):
        q_filter.should.append(
            FieldCondition(key="name", match=MatchValue(value=plan.name_hint))
        )
        log.info(f"[planner] name_hint aplicado como SHOULD: {plan.name_hint}")

    # 7.e) si un filtro del planner dejó 0 hits, relajamos servicio (como tu lógica nueva)
    if has_service_should:
        try:
            cnt = _qdrant.count(
                collection_name=COLLECTION,
                count_filter=q_filter,
                exact=False,
            ).count or 0
            if cnt == 0:
                # quitamos servicios de should
                q_filter.should = [
                    c for c in q_filter.should
                    if not (isinstance(c, FieldCondition) and c.key == "servicio_id")
                ]
                has_service_should = False
                log.info("[planner] fallback: 0 hits → removiendo filtros de servicio")
        except Exception as e:
            log.debug(f"[planner] no se pudo hacer preflight de servicio: {e}")

    # LOG del filtro final
    log.info(f"[filter] FINAL={dump_qdrant_filter(q_filter)}")

    # 8) Prompts (mismos que prod)
    # Prompts clínicos
   
    QA_LANG_AWARE = PromptTemplate(
        "Eres un asistente clínico cuya única función es extraer información directamente del 'Contexto' proporcionado.\n"
        "Responde ÚNICAMENTE con información que figure literalmente en el CONTEXTO.\n"
        "No inventes ni infieras. No agregues ejemplos ni juicios clínicos. Parafrasea lo mínimo.\n"
        "Si NO hay datos pertinentes en el contexto, responde EXACTAMENTE:\n"
        "  'No hay registros sobre el tema consultado en la historia clínica disponible.'\n\n"
        "Formato (mismo idioma de la pregunta):\n"
        "Evita repetir y respeta unidades/nombres propios.\n\n"
        "CONTEXTO:\n{context_str}\n\n"
        "PREGUNTA:\n{query_str}\n\n"
        "RESPUESTA:"
    )
   



    SUM_LANG_AWARE = PromptTemplate(
        "Eres un experto clínico. Tu tarea es responder la SOLICITUD del usuario basándote en el CONTEXTO.\n"
        "Instrucciones de Razonamiento:\n"
        "1. NO generes la respuesta todavía. Primero, extrae mentalmente todos los hechos del CONTEXTO que coincidan con la SOLICITUD.\n"
        "2. Identifica relaciones Causa-Efecto (ej: 'Hemicolectomía' → implica 'Cáncer' o 'Patología grave').\n"
        "3. Si encuentras datos antiguos (ej: anamnèsis, antecedentes), SÍ inclúyelos.\n"
        "4. Sintetiza la respuesta final basándote en lo hallado.\n\n"
        "Contexto:\n{context_str}\n\n"
        "Solicitud:\n{query_str}\n\n"
        "Respuesta:"
    )

    meta_pp = MetaToContext()

    # 9) Retrievers con el filtro definitivo
    # ⚠️ IMPORTANTE: Aquí pedimos MÁS nodos (retrieve_top_k=30 o 50) para que el reranker pueda filtrar.
    # Antes intentábamos leerlo de qe_vector, pero qe_vector se crea DESPUÉS. Lo definimos aquí.
    retrieve_k_vector = 50
    log.debug(f"[router] Configurando retriever vectorial con similarity_top_k={retrieve_k_vector}")
    
    retriever_vector = vector_index.as_retriever(
        similarity_top_k=retrieve_k_vector,
        vector_store_kwargs={"qdrant_filters": q_filter},
    )
    retriever_sum = vector_index.as_retriever(
        similarity_top_k=(SUMMARY_TOP_K + 12) if has_service_should else SUMMARY_TOP_K,
        vector_store_kwargs={"qdrant_filters": q_filter},
    )

    # 10) Synthesizers (como prod)
    synth_qa = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        llm=llm,
        text_qa_template=QA_LANG_AWARE,
    )
    synth_sum = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=SUM_LANG_AWARE,
        llm=llm,
    )

    # 11) Engines
    # Calculamos valores finales de superlativo
    final_which = None
    final_count = 1
    
    # 1. Prioridad: Plan (que pudimos haber parchado con regex arriba)
    if plan and getattr(plan, "which", None):
        final_which = plan.which
        final_count = int(getattr(plan, "count", 1) or 1)
    else:
        # 2. Fallback: Si no hay plan, revisamos el intent que calculó el HybridParser (línea 460)
        # Esto es mucho más limpio que re-checkear con regex aquí.
        if intent and intent.get("which"):
            final_which = intent.get("which")
            final_count = int(intent.get("count") or 1)
            log.info(f"[router] Usando 'which' del HybridParser fallback: {final_which}")


    qe_vector = SimpleQueryEngine(
        paciente_id=str(payload.paciente_id or ""),
        retriever=retriever_vector,
        synthesizer=synth_qa,
        meta_to_context=meta_pp,
        temporal_superlative={"which": final_which, "count": final_count},
        reranker_top_n=10,
        use_reranker=False,  # 🔴 Desactivado - usar solo reranker global de rag_tool.py
        similarity_top_k_retrieve=50,
        text_key=TEXT_KEY,
    )

    qe_summary = SimpleQueryEngine(
        paciente_id=str(payload.paciente_id or ""),
        retriever=retriever_sum,
        synthesizer=synth_sum,
        meta_to_context=None,
        use_reranker=False,  # 🔴 Desactivado - usar solo reranker global de rag_tool.py
        temporal_superlative={"which": final_which, "count": final_count},
        reranker_top_n=10,
        similarity_top_k_retrieve=50,
        text_key=TEXT_KEY,
    )

    # 12) Router final
    router = SimpleRouter(
        paciente_id=str(payload.paciente_id or ""),
        qe_vector=qe_vector,
        qe_summary=qe_summary,
    )
    log.info(f"[router] Construido correctamente → paciente={payload.paciente_id}, tramite={payload.procedureNumber}")
    return router


# ─────────────── INYECCIÓN DE DEPENDENCIA ───────────────
def get_router(
    payload: ConsultaQdrantRequest,
    llm: LLM = Depends(get_llm),
    embed_model: Any = Depends(get_embedder),
) -> Optional[SimpleRouter]:
    return _build_router(payload=payload, llm=llm, embed_model=embed_model)

# ─────────────── Consulta segura ───────────────
TRANSIENTS = (TimeoutError, ReadTimeout,)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(TRANSIENTS),
    reraise=True,
)
def _safe_query(engine: SimpleRouter, pregunta: str) -> Response:
    resp: Response = engine.query(pregunta or "")
    expected = getattr(engine, "_paciente_id", None)
    try:
        ok = sources_ok(resp, expected) if expected else True
    except Exception:
        ok = True
    if expected and not ok:
        log.error(f"[guard] fuentes cruzadas detectadas; reintento vectorial (paciente_id={expected})")
        resp = engine._qe_vector.query(pregunta or "")
    return resp

def _safe_query_stream(engine: SimpleRouter, pregunta: str):
    """Stream seguro: emite tokens y cierra con 'end' (answer + sources)."""
    try:
        for ev in engine.stream(pregunta or ""):
            yield ev
    except Exception as e:
        log.exception("[stream] error")
        yield {"type": "error", "message": str(e)}

def query_hc_stream(pregunta: str, engine: Optional[SimpleRouter]):
    """Generador de eventos para el endpoint SSE/JSONL."""
    if engine is None:
        yield {"type": "token", "text": "⚠️ El paciente no tiene datos cargados."}
        yield {"type": "end", "final": {"answer": "", "sources": []}}
        return
    inicio = time.time()
    yield from _safe_query_stream(engine, pregunta or "")
    dur = time.time() - inicio
    log.debug(f"[query_hc_stream] Duración consulta: {dur:.2f}s")

# Alias por compatibilidad
def stream_hc(pregunta: str, engine: Optional[SimpleRouter]):
    return query_hc_stream(pregunta, engine)

@breaker
def query_hc(pregunta: str, engine: Optional[SimpleRouter]) -> Dict[str, Any]:
    if engine is None:
        return {"answer": "⚠️ El paciente no tiene datos cargados.", "sources": []}
    inicio = time.time()
    resp: Response = _safe_query(engine, pregunta or "")
    dur = time.time() - inicio
    log.debug(f"[query_hc] Duración consulta: {dur:.2f}s")
    return serialize_response(resp)
