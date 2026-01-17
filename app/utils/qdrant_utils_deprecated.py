# app/utils/qdrant_utils.py
import hashlib
import json
import logging
import os
from typing import List, Optional

from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

log = logging.getLogger(__name__)

# ───────────────────────── Config ─────────────────────────
COLLECTION = os.getenv("QDRANT_COLLECTION", "hc_chat_db")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "10.0"))

_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
    # prefer_grpc=True,  # habilitar si tenés gRPC en Qdrant
)


# ───────────── Helpers internos ─────────────
def _extract_text(payload: dict) -> str:
    """Devuelve el texto del payload cualquiera sea el formato que llegue."""
    text = payload.get("text") or payload.get("chunk_text") or ""
    if not text and "_node_content" in payload:
        try:
            c = payload["_node_content"]
            if isinstance(c, str):
                c = json.loads(c)
            text = c.get("text", "")
        except Exception as e:
            log.warning("⚠️ _node_content parse error: %s", e)
    return text or ""


def _scroll_nodes(
    filter_: rest.Filter, page_size: int = 256, max_points: int = 2048
) -> List[TextNode]:
    """Itera con scroll (paginado) hasta max_points, sin vectores."""
    nodes: List[TextNode] = []
    offset = None

    while len(nodes) < max_points:
        # QdrantClient.scroll devuelve (points, next_page_offset)
        points, next_offset = _client.scroll(
            collection_name=COLLECTION,
            scroll_filter=filter_,
            limit=min(page_size, max_points - len(nodes)),
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )

        if not points:
            break

        for p in points:
            pl = p.payload or {}
            text = _extract_text(pl)
            if text:
                meta = {
                    k: v
                    for k, v in pl.items()
                    if k not in {"text", "chunk_text", "_node_content"}
                }
                nodes.append(TextNode(text=text, metadata=meta))

        if not next_offset:
            break
        offset = next_offset

    return nodes


# ───────────── Filtros (compat + solo paciente) ─────────────
def _flt(uid: str, pid: Optional[str]) -> rest.Filter:
    cond = [rest.FieldCondition(key="user_id", match=rest.MatchValue(value=str(uid)))]
    if pid:
        cond.append(
            rest.FieldCondition(
                key="paciente_id", match=rest.MatchValue(value=str(pid))
            )
        )
    return rest.Filter(must=cond)


def _flt_by_patient(pid: str) -> rest.Filter:
    return rest.Filter(
        must=[
            rest.FieldCondition(
                key="paciente_id", match=rest.MatchValue(value=str(pid))
            )
        ]
    )


# ───────────── API existente (compatibilidad) ─────────────
def get_nodes(uid: str, pid: Optional[str], limit: int = 2048) -> List[TextNode]:
    nodes = _scroll_nodes(_flt(uid, pid), max_points=limit)
    log.info("✅ %s nodos para user=%s paciente=%s", len(nodes), uid, pid)
    return nodes


def current_hash(uid: str, pid: Optional[str]) -> str:
    cnt = _client.count(collection_name=COLLECTION, count_filter=_flt(uid, pid)).count
    return hashlib.md5(str(cnt).encode()).hexdigest()


# ───────────── Nuevo (solo por paciente) ─────────────
def get_nodes_for_patient(pid: str, limit: int = 2048) -> List[TextNode]:
    nodes = _scroll_nodes(_flt_by_patient(pid), max_points=limit)
    log.info("✅ %s nodos para paciente=%s", len(nodes), pid)
    return nodes


def current_hash_for_patient(pid: str) -> str:
    cnt = _client.count(
        collection_name=COLLECTION, count_filter=_flt_by_patient(pid)
    ).count
    return hashlib.md5(str(cnt).encode()).hexdigest()
