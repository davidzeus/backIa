# app\helpers\query_helpers.py
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, Optional
from datetime import datetime
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType, Range
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from app.infra.qdrant_client_factory import get_qdrant_client
from app.schemas.esquema import ConsultaQdrantRequest

log = logging.getLogger(__name__)
_DEBUG_FLAG = os.getenv("HCI_DEBUG", os.getenv("HCI_DEBUG", "0"))
if not bool(int(str(_DEBUG_FLAG))):
    log.setLevel(logging.WARNING)

COLLECTION = os.getenv("QDRANT_COLLECTION", "hc_chat_db_hybrid")
_qdrant = get_qdrant_client()



def apply_name_hint_filter(q_filter: Filter, name_hint: Optional[str]) -> None:
    if not name_hint:
        return
    nh = (name_hint or "").strip().upper()

    if nh == "INTERCONSULTA":
        q_filter.must.append(FieldCondition(key="seccion_raiz", match=MatchValue(value="INTERCONSULTA")))
        q_filter.should.append(FieldCondition(key="name", match=MatchValue(value="INTERCONSULTA")))
        log.debug("🧭 name_hint=INTERCONSULTA → filtro aplicado (seccion_raiz/name)")
    elif nh == "EVOLUCIÓN":
        q_filter.must.append(FieldCondition(key="seccion_raiz", match=MatchValue(value="EVOLUCIÓN")))
        q_filter.should.append(FieldCondition(key="name", match=MatchValue(value="EVOLUCIÓN")))
    elif nh == "MOTIVO DE CONSULTA":
        q_filter.must.append(FieldCondition(key="seccion_raiz", match=MatchValue(value="MOTIVO DE CONSULTA")))
        q_filter.should.append(FieldCondition(key="name", match=MatchValue(value="MOTIVO DE CONSULTA")))

# ─────────────────────────────────────────────────────────────────────────────
# Mapeos planner → groups + reglas
# ─────────────────────────────────────────────────────────────────────────────
SECTION_TO_GROUPS = {
    "WARD/FLOOR":       ["internaciones", "evolucion", "diagnosticos", "laboratorio"],
    "ICU/UTI":          ["uci", "uti", "terapia_intensiva", "evolucion", "diagnosticos"],
    "EMERGENCY/ED":     ["guardia", "admisiones", "evolucion"],
    "OR/PROCEDURE":     ["procedimientos", "quirurgico", "hemiodinamia", "anestesia"],
    "OUTPATIENT/CLINIC":["ambulatorio", "consultorio"],
    "ORDERS":           ["indicaciones", "medicacion", "ordenes"],
    "DIAGNOSTICS":      ["imagenes", "laboratorio", "diagnosticos"],
    "ADMIN":            ["administrativo", "tramites", "censos"],
}

EVENT_TO_GROUPS = {
    "ADMISSION": ["internaciones", "admisiones"],
    "DISCHARGE": ["internaciones", "egresos"],
    "TRANSFER":  ["internaciones", "traslados"],
    "PROCEDURE": ["procedimientos", "quirurgico"],
    "EVOLUTION": ["evolucion"],
    "ORDER":     ["indicaciones", "ordenes", "medicacion"],
    "RESULT":    ["imagenes", "laboratorio", "diagnosticos"],
}

TEXTLEN_OPTIONAL_SECTIONS = {"WARD/FLOOR", "ADMISSION", "DISCHARGE", "TRANSFER"}

def mentions_internacion(q: str) -> bool:
    """
    Detecta si la pregunta del usuario se refiere a una internación
    usando una expresión regular robusta con sinónimos y plurales.
    """
    if not q:
        return False
    
    # Patrón mejorado que incluye más sinónimos y variaciones
    inpatient_pattern = re.compile(
        r"\b(internaci[oó]n(es)?|hospitalizaci[oó]n|ingresad[oa]|estuvo internad[oa]|pop|postoperatorio|ingreso hospitalario|egreso)\b",
        re.IGNORECASE
    )
    
    return bool(re.search(inpatient_pattern, q.lower()))
# ─────────────────────────────────────────────────────────────────────────────
# Qdrant helpers (filtros / índices / auditoría / text key / latest procedure)
# ─────────────────────────────────────────────────────────────────────────────
def q_filter_paciente(paciente_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="paciente_id", match=MatchValue(value=str(paciente_id)))])

def q_filter_from_payload(
    payload: ConsultaQdrantRequest,
    ts_start: int | None = None,
    ts_end: int | None = None,
) -> Filter:
    must_conditions = []

    tiene_pid = payload.paciente_id is not None and str(payload.paciente_id).strip() != ""
    tiene_tramite = payload.procedureNumber is not None
    if not (tiene_pid or tiene_tramite):
        raise ValueError("Debe especificarse al menos paciente_id o procedureNumber.")

    if tiene_pid:
        must_conditions.append(FieldCondition(key="paciente_id", match=MatchValue(value=str(payload.paciente_id))))

    if tiene_tramite:
        must_conditions.append(FieldCondition(key="procedureNumber", match=MatchValue(value=int(payload.procedureNumber))))

    if payload.group:
        if isinstance(payload.group, list):
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="group", match=MatchValue(value=g))
                    for g in payload.group if isinstance(g, str) and g.strip()
                ])
            )
        else:
            must_conditions.append(FieldCondition(key="group", match=MatchValue(value=str(payload.group))))

    if payload.healthHistoryGroup:
        must_conditions.append(FieldCondition(key="healthHistoryGroup", match=MatchValue(value=payload.healthHistoryGroup)))

    if payload.servicio_id:
        if isinstance(payload.servicio_id, list):
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="servicio_id", match=MatchValue(value=int(sid)))
                    for sid in payload.servicio_id
                ])
            )
        else:
            must_conditions.append(FieldCondition(key="servicio_id", match=MatchValue(value=int(payload.servicio_id))))

    if payload.servicio:
        must_conditions.append(FieldCondition(key="servicio", match=MatchValue(value=payload.servicio)))

    if payload.name:
        must_conditions.append(FieldCondition(key="name", match=MatchValue(value=payload.name)))

    if ts_start is not None or ts_end is not None:
        must_conditions.append(FieldCondition(key="fecha_ts", range=Range(gte=ts_start if ts_start is not None else None,
                                                                          lt=ts_end if ts_end is not None else None)))
    return Filter(must=must_conditions)

def ensure_payload_indexes():
    def _mk(field, schema):
        try:
            _qdrant.create_payload_index(collection_name=COLLECTION, field_name=field, field_schema=schema)
        except Exception as e:
            log.debug(f"[qdrant] create_payload_index {field}: {e}")

    _mk("paciente_id",        PayloadSchemaType.KEYWORD)
    _mk("procedureNumber",    PayloadSchemaType.INTEGER)
    _mk("fecha_ts",           PayloadSchemaType.INTEGER)
    _mk("servicio_id",        PayloadSchemaType.INTEGER)
    _mk("servicio",           PayloadSchemaType.KEYWORD)
    _mk("group",              PayloadSchemaType.KEYWORD)
    _mk("healthHistoryGroup", PayloadSchemaType.KEYWORD)
    _mk("name",               PayloadSchemaType.KEYWORD)

def audit_patient_slice(paciente_id: str, limit: int = 5):
    flt = q_filter_paciente(paciente_id)
    try:
        cnt = _qdrant.count(collection_name=COLLECTION, count_filter=flt, exact=True)
        log.info(f"[qdrant] count(pid={paciente_id})={cnt.count}")
        pts, _ = _qdrant.scroll(collection_name=COLLECTION, scroll_filter=flt, with_payload=True, limit=limit)
        bad = [p for p in pts if str(p.payload.get("paciente_id")) != str(paciente_id)]
        if bad:
            log.error(f"[qdrant] scroll halló payloads cruzados para pid={paciente_id}")
        else:
            log.info(f"[qdrant] scroll OK: {len(pts)} muestras pertenecen a pid={paciente_id}")
    except Exception as e:
        log.warning(f"[qdrant] auditoría falló: {e}")

def detect_qdrant_text_key(collection: str, sample_filter: Optional[Filter] = None) -> str:
    CANDIDATAS = ["text", "raw_text", "content", "page_content", "chunk", "body", "document", "doc", "texto", "_node_content"]
    try:
        pts, _ = _qdrant.scroll(collection_name=collection, scroll_filter=sample_filter, with_payload=True, limit=10)
        for p in pts or []:
            pl = (getattr(p, "payload", None) or {})
            node_content = pl.get("_node_content")
            if node_content:
                try:
                    node_data = json.loads(node_content)
                    if node_data.get("text") or node_data.get("content"):
                        return "_node_content"
                except Exception:
                    pass
            for k in CANDIDATAS:
                if k == "_node_content":
                    continue
                v = pl.get(k, None)
                if isinstance(v, str) and v.strip():
                    return k
    except Exception as e:
        log.warning(f"[detect_text_key] no se pudo scrollear: {e}")
    return "text"

def get_latest_procedure_for_patient(paciente_id: str) -> Optional[int]:
    """Devuelve el procedureNumber más reciente para un paciente dentro de 'internaciones'."""
    try:
        flt = Filter(must=[
            FieldCondition(key="paciente_id", match=MatchValue(value=str(paciente_id))),
            FieldCondition(key="group", match=MatchValue(value="internaciones")),
            FieldCondition(key="fecha_ts", range=Range(gt=0)),
        ])
        pts, _ = _qdrant.scroll(collection_name=COLLECTION, scroll_filter=flt, with_payload=True, limit=200)
        latest_ts = -1
        latest_proc = None
        for p in pts or []:
            md = getattr(p, "payload", {}) or {}
            ts = int(md.get("fecha_ts") or -1)
            proc = md.get("procedureNumber")
            if ts > latest_ts and proc is not None:
                latest_ts = ts
                latest_proc = int(proc)
        return latest_proc
    except Exception as e:
        log.warning(f"[latest_proc] fallo calculando último trámite: {e}")
        return None

def debug_patient_data_detailed(paciente_id: str):
    flt = q_filter_paciente(paciente_id)
    try:
        pts, _ = _qdrant.scroll(collection_name=COLLECTION, scroll_filter=flt, with_payload=True, limit=5)
        log.warning(f"[DEBUG_DETAILED] Analizando {len(pts)} puntos para paciente {paciente_id}")
        for i, p in enumerate(pts):
            payload = getattr(p, "payload", {})
            score = getattr(p, "score", None)
            log.warning(f"[DEBUG_DETAILED] Punto {i}:")
            log.warning(f"  - Score: {score} (tipo: {type(score)})")
            log.warning(f"  - Claves: {list(payload.keys())}")
            node_content = payload.get("_node_content")
            if node_content:
                try:
                    node_data = json.loads(node_content)
                    log.warning(f"  - _node_content keys: {list(node_data.keys())}")
                    for j, text in enumerate([node_data.get("text"), node_data.get("content"),
                                              node_data.get("text_"), node_data.get("content_")]):
                        if text and isinstance(text, str):
                            prev = text[:100] + "..." if len(text) > 100 else text
                            log.warning(f"  - Texto en ubicación {j}: '{prev}'")
                            break
                except Exception as e:
                    log.warning(f"  - Error parseando _node_content: {e}")
            else:
                log.warning("  - Sin _node_content")
    except Exception as e:
        log.error(f"[DEBUG_DETAILED] Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# SafeQdrantVectorStore + saneadores de nodos
# ─────────────────────────────────────────────────────────────────────────────
class SafeQdrantVectorStore(QdrantVectorStore):
    _CANDIDATES = ["text", "raw_text", "content", "page_content", "chunk", "body", "document", "doc", "texto"]

    def _extract_text_from_node_content(self, node_content: str) -> str | None:
        try:
            node_data = json.loads(node_content)
            text = (node_data.get("text") or node_data.get("content") or node_data.get("text_") or node_data.get("content_"))
            if isinstance(text, str) and text.strip():
                return text.strip()
            if "embedding" in node_data and "metadata" in node_data:
                meta = node_data.get("metadata", {})
                text = meta.get("text") or meta.get("content")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        except Exception as e:
            log.warning(f"[SafeQdrant] Error parseando _node_content: {e}")
        return None

    def _extract_text(self, payload: dict) -> str | None:
        for k in self._CANDIDATES:
            v = payload.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
            elif v is not None:
                try:
                    text = str(v).strip()
                    if text:
                        return text
                except Exception:
                    pass

        node_content = payload.get("_node_content")
        if node_content:
            text = self._extract_text_from_node_content(node_content)
            if text:
                return text

        excluded_fields = [
            "ingestion_ts", "_node_type", "doc_id", "ref_doc_id", "node_hash",
            "document_id", "source", "marker", "fecha", "fecha_ts", "paciente_id",
            "procedureNumber", "servicio_id", "name", "seccion"
        ]
        for key, value in payload.items():
            if (key not in excluded_fields and isinstance(value, str) and len(value.strip()) > 50):
                log.info(f"[SafeQdrant] Usando campo '{key}' como texto: {len(value)} chars")
                return value.strip()
        log.warning(f"[SafeQdrant] No se pudo extraer texto. Claves: {list(payload.keys())}")
        return None

    def parse_to_query_result(self, response) -> VectorStoreQueryResult:
        points = getattr(response, "scored_points", None) or getattr(response, "points", None) or response
        ids, nodes, sims = [], [], []
        valid_count = 0
        for p in points or []:
            payload = getattr(p, "payload", None) or (p.get("payload") if isinstance(p, dict) else {}) or {}
            score = getattr(p, "score", None) or (p.get("score") if isinstance(p, dict) else None)
            pid = getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None)

            text = self._extract_text(payload)
            if text is None or not isinstance(text, str) or not text.strip():
                log.warning(f"[SafeQdrant] Punto omitido - texto inválido. Payload keys: {list(payload.keys())}")
                continue

            try:
                node = TextNode(text=text, metadata=payload)
                nodes.append(node)
                sims.append(float(score) if isinstance(score, (int, float)) else 0.0)
                ids.append(str(pid) if pid is not None else None)
                valid_count += 1
            except Exception as e:
                log.warning(f"[SafeQdrant] Error creando TextNode: {e}")
                continue

        log.info(f"[SafeQdrant] Parseados {valid_count}/{len(points or [])} puntos válidos")
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)

# ─────────────────────────────────────────────────────────────────────────────
# Guard-rails + saneadores y serialización
# ─────────────────────────────────────────────────────────────────────────────
def sources_ok(resp: Any, expected_pid: str) -> bool:
    try:
        srcs: Iterable = getattr(resp, "source_nodes", []) or []
        for s in srcs:
            node = getattr(s, "node", None) or s
            md: Dict[str, Any] = getattr(node, "metadata", {}) or {}
            pid = str(md.get("paciente_id"))
            if pid != str(expected_pid):
                return False
        return True
    except Exception:
        return True

def node_has_text(n: Any) -> bool:
    try:
        node = getattr(n, "node", n)
        txt = ""
        if hasattr(node, "get_text"):
            txt = node.get_text() or ""
        else:
            txt = getattr(node, "text", "") or ""
        return isinstance(txt, str) and bool(txt.strip())
    except Exception:
        return False

def sanitize_nodes_for_llamaindex(nodes: Iterable[Any]) -> list[Any]:
    out = []
    for n in (nodes or []):
        node = getattr(n, "node", n)
        txt = None
        try:
            txt = getattr(node, "text", None)
            if txt is None and hasattr(node, "get_text"):
                txt = node.get_text()
        except Exception:
            txt = None
        if txt is None:
            continue
        if not isinstance(txt, str):
            try:
                txt = str(txt)
            except Exception:
                continue
        txt = txt.strip()
        if not txt:
            continue
        try:
            setattr(node, "text", txt)
        except Exception:
            continue
        out.append(n)
    return out

def node_text(node: Any) -> str:
    try:
        return node.get_text()
    except Exception:
        return getattr(node, "text", "") or ""

def serialize_nodes(nodes: Iterable[Any], limit: Optional[int] = 5) -> list[dict]:
    """
    Convierte una lista de nodos (NodeWithScore o TextNode) a la lista de dicts 'sources'.
    Reutilizable por el Agente y el RAG clásico.
    Args:
        nodes: Lista de nodos.
        limit: Cantidad máxima de nodos a retornar. None para retornar todos.
    """
    srcs_out = []
    for s in nodes or []:
        node = getattr(s, "node", None) or s
        score = getattr(s, "score", None)
        # Convert numpy floats to python floats for JSON serialization
        if score is not None:
            try:
                score = float(score)
            except Exception:
                pass
        md: Dict[str, Any] = getattr(node, "metadata", {}) or {}
        try:
            fecha_ts = int(md.get("fecha_ts")) if md.get("fecha_ts") is not None else -1
        except Exception:
            fecha_ts = -1
        srcs_out.append({
            "text": node_text(node),
            "score": score,
            "fecha_ts": fecha_ts,
            "metadata": {
                "paciente_id": str(md.get("paciente_id", "")) or "",
                "procedureNumber": md.get("procedureNumber") or md.get("internacion_tramite") or md.get("encounter_id") or "",
                "name": md.get("name"),
                "seccion": md.get("seccion") or md.get("section") or "",
                "seccion_raiz": md.get("seccion_raiz"),
                "group": md.get("group"),
                "healthHistoryGroup": md.get("healthHistoryGroup"),
                "servicio_id": md.get("servicio_id"),
                "servicio_desc": md.get("servicio") or md.get("servicio_desc") or "",
                "fecha": md.get("fecha") or md.get("fecha_evento") or "",
                "fecha_ts": fecha_ts,
                "profesional": md.get("profesional"),
            },
        })

    srcs_out.sort(key=lambda x: (-(x["score"] if x["score"] is not None else 0.0),
                                 -(x["fecha_ts"] if x["fecha_ts"] is not None else -1)))
    
    # Slice dinámico
    if limit is not None:
        return [dict(text=s["text"], score=s["score"], metadata=s["metadata"]) for s in srcs_out[:limit]]
    else:
        return [dict(text=s["text"], score=s["score"], metadata=s["metadata"]) for s in srcs_out]


def serialize_response(resp) -> Dict[str, Any]:
    if isinstance(resp, str):
        return {"answer": resp, "sources": []}

    answer = (getattr(resp, "response", "") or "").strip()
    
    # Delegamos la serialización de nodos
    top_sources = serialize_nodes(getattr(resp, "source_nodes", []) or [])
    
    return {"answer": answer or "⚠️ Sin contenido.", "sources": top_sources}




def sort_and_slice_by_temporal_super(nodes, temporal_super=None, top_n=None):
    """
    Ordena los nodos por fecha_ts DESC y aplica superlativos temporales ('last' / 'first').
    Garantiza orden determinista (fecha + id).
    """
    if not nodes:
        return nodes

    which = None
    count = 1
    if isinstance(temporal_super, dict):
        which = temporal_super.get("which")
        count = int(temporal_super.get("count") or 1)
    elif isinstance(temporal_super, str):
        which = temporal_super
        count = top_n or 1

    def _parse_fecha_ts(n):
        node = getattr(n, "node", n)
        md = getattr(node, "metadata", {}) or {}
        v = md.get("fecha_ts")
        if v is None:
            return None
        try:
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v)
            return datetime.fromisoformat(str(v))
        except Exception:
            return None

    with_ts = [(n, _parse_fecha_ts(n)) for n in nodes]
    with_ts = [(n, ts) for n, ts in with_ts if ts is not None]
    without_ts = [n for n, ts in [(n, _parse_fecha_ts(n)) for n in nodes] if ts is None]

    if not with_ts:
        return nodes

    # Orden descendente (más recientes primero)
    ordered = [n for n, _ in sorted(with_ts, key=lambda x: (x[1], id(x[0])), reverse=True)]
    ordered += without_ts

    if which in ("last", "latest", "recent"):
        return ordered[:count]
    elif which in ("first", "earliest", "oldest"):
        return list(reversed(ordered))[:count]
    return ordered

# --- Helpers para filtros tolerantes a esquema ---
def field_exists_in_collection(field: str, sample_filter: Optional[Filter] = None, probe: int = 32) -> bool:
    """
    Devuelve True si el 'field' aparece en algún payload de la colección (muestra pequeña).
    Evita romper cuando el índice no tiene ciertos campos.
    """
    try:
        pts, _ = _qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=sample_filter,
            with_payload=True,
            limit=probe,
        )
        for p in pts or []:
            payload = getattr(p, "payload", {}) or {}
            if field in payload:
                return True
        return False
    except Exception:
        return False

def safe_add(q_filter: Filter, *, must: bool = True, condition=None, field_key: Optional[str] = None, sample_filter: Optional[Filter] = None):
    """
    Agrega 'condition' al filtro SOLO si el campo existe (si se especifica 'field_key').
    Útil para colecciones sin 'texto_len', 'name_norm', 'servicio_id', etc.
    """
    if condition is None:
        return
    if field_key is not None and not field_exists_in_collection(field_key, sample_filter=sample_filter):
        return
    if must:
        q_filter.must.append(condition)
    else:
        q_filter.should.append(condition)

def dump_qdrant_filter(flt) -> dict:
    """Devuelve un dict legible del Filter (must/should/must_not -> [{key, match|range}])."""
    def _one(c):
        try:
            k = getattr(c, "key", None)
            mv = getattr(c, "match", None)
            rg = getattr(c, "range", None)
            if mv is not None and hasattr(mv, "value"):
                return {"key": k, "match": getattr(mv, "value", None)}
            if rg is not None:
                return {"key": k, "range": {
                    "gt": getattr(rg, "gt", None), "gte": getattr(rg, "gte", None),
                    "lt": getattr(rg, "lt", None), "lte": getattr(rg, "lte", None),
                }}
        except Exception:
            pass
        return str(c)
    try:
        return {
            "must":    [ _one(c) for c in (getattr(flt, "must", []) or []) ],
            "should":  [ _one(c) for c in (getattr(flt, "should", []) or []) ],
            "must_not":[ _one(c) for c in (getattr(flt, "must_not", []) or []) ],
        }
    except Exception:
        return {"raw": str(flt)}


def dump_query_plan(plan) -> dict:
    """
    Serializa un QueryPlan (o None) para logging sin romperse si faltan attrs.
    No asume que sea dict; usa getattr con defaults.
    """
    try:
        if plan is None:
            return {"plan": None}

        def _get(obj, name, default=None):
            try:
                return getattr(obj, name)
            except Exception:
                return default

        # Campos “core” del Pydantic QueryPlan
        out = {
            "need_sections": list(_get(plan, "need_sections", []) or []),
            "need_events":   list(_get(plan, "need_events", []) or []),
            "which":         _get(plan, "which", None),
            "count":         _get(plan, "count", None),
        }

        # Enriquecimientos que agregamos vía setattr (pueden no estar)
        # No fallar si no existen:
        extras = {
            "is_inpatient":      _get(plan, "is_inpatient", None),
            "name_hint":         _get(plan, "name_hint", None),
            "procedure_number":  _get(plan, "procedure_number", None),
            "time_window":       _get(plan, "time_window", None),
            "service_hints":     _get(plan, "service_hints", None),
        }
        out.update({k: v for k, v in extras.items() if v is not None})

        return out
    except Exception as e:
        return {"plan_dump_error": str(e)}

def strip_accents(s: str) -> str:
    try:
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    except Exception:
        return s



__all__ = [
    "COLLECTION",
    "_qdrant",
    # planner mapeos / reglas
    "SECTION_TO_GROUPS", "EVENT_TO_GROUPS", "TEXTLEN_OPTIONAL_SECTIONS", "TEXTLEN_OPTIONAL_GROUPS",
    "mentions_internacion",
    # qdrant
    "q_filter_paciente", "q_filter_from_payload", "ensure_payload_indexes",
    "audit_patient_slice", "detect_qdrant_text_key", "get_latest_procedure_for_patient",
    "debug_patient_data_detailed",
    # store y saneadores
    "SafeQdrantVectorStore", "sources_ok", "node_has_text", "sanitize_nodes_for_llamaindex",
    "node_text", "serialize_response", "serialize_nodes", "sort_and_slice_by_temporal_super",
    "field_exists_in_collection", "safe_add",
    "dump_qdrant_filter", "dump_query_plan","strip_accents"
]

