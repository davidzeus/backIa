# app/utils/hc_vector_utils.py
"""
Utilidades para ingesta HC + Qdrant:
- Normalización y hashing
- IDs determinísticos por nodo
- Compatibilidad de cliente Qdrant (delete/scroll/count)
- Filtros por paciente/documento y escaneo de puntos
- Marcador de snapshot (recomendado en TOP-LEVEL con upsert_snapshot_marker)
"""

import hashlib
import inspect
import logging
import os
import re
import unicodedata
from datetime import datetime
from typing import List, Optional, Tuple

from llama_index.core import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Batch,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
)

# Logger del módulo (silenciado por defecto; habilitar con HCI_DEBUG=1 o INGESTA_DEBUG=1)
log = logging.getLogger(__name__)
if not bool(int(os.getenv("HCI_DEBUG", os.getenv("INGESTA_DEBUG", "0")))):
    log.setLevel(logging.WARNING)

# ======================= Normalización y hashing =======================


def normalize_text(t: Optional[str]) -> str:
    if t is None:
        return ""
    t = unicodedata.normalize("NFKC", t)
    lines = t.splitlines()
    out = []
    for line in lines:
        line = re.sub(r"[\x00-\x09\x0B-\x1F]+", " ", line)  # conserva \n
        line = line.replace("\u00a0", " ")  # NBSP → espacio
        line = re.sub(r"[ \t]+", " ", line).strip()
        out.append(line)
    return "\n".join(out).strip()


def node_hash_text(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def canon_fecha(fecha: Optional[str]) -> str:
    """Devuelve YYYY-MM-DD (tolerante a ISO con hora/Z)."""
    if not fecha:
        return ""
    s = str(fecha).replace("Z", "")
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d")
    except Exception:
        return s.split("T")[0] if "T" in s else s


def canon_seccion(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def snapshot_hash_from_nodes(nodes: List[Document]) -> str:
    """
    Hash estable del snapshot completo (ordena por fecha, seccion, name, hash_texto).
    Independiente del orden de entrada.
    """

    def _key(n: Document) -> Tuple[str, str, str, str]:
        fecha = str(n.metadata.get("fecha", ""))
        seccion = str(n.metadata.get("seccion", ""))
        name = str(n.metadata.get("name", ""))
        h = node_hash_text(n.text)
        return (fecha, seccion, name, h)

    nodes_sorted = sorted(nodes, key=_key)
    texto_total = "\n".join(normalize_text(n.text) for n in nodes_sorted)
    final_hash = hashlib.md5(texto_total.encode("utf-8")).hexdigest()
    log.debug(
        "HC_SNAPSHOT_HASH | nodes=%s | document_id=%s", len(nodes_sorted), final_hash
    )
    return final_hash


def point_id_for_node(patient_id: str, node_text: str, seccion: str, fecha: str) -> str:
    base = f"{patient_id}|{canon_seccion(seccion)}|{canon_fecha(fecha)}|{node_hash_text(node_text)}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


# ======================= Compat Qdrant client =======================


def client_count(client: QdrantClient, collection_name: str, flt: Filter) -> int:
    # Logueamos siempre el filtro que se usa
    try:
        flt_json = flt.model_dump_json(indent=2)
    except Exception:
        flt_json = str(flt)
    log.debug("QDRANT_COUNT | collection='%s' | filter=%s", collection_name, flt_json)

    sig = inspect.signature(client.count)
    if "count_filter" in sig.parameters:
        result = client.count(
            collection_name=collection_name, count_filter=flt, exact=True
        )
    else:
        result = client.count(collection_name=collection_name, filter=flt, exact=True)

    count = int(getattr(result, "count", 0))
    log.debug("QDRANT_COUNT_RESULT | count=%s", count)
    return count


def client_scroll(
    client: QdrantClient,
    collection_name: str,
    flt: Filter,
    *,
    limit=256,
    offset=None,
    with_payload=True,
    with_vectors=False,
):
    sig = inspect.signature(client.scroll)
    kw = dict(
        collection_name=collection_name,
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors,
        offset=offset,
    )
    if "scroll_filter" in sig.parameters:
        kw["scroll_filter"] = flt
    else:
        kw["filter"] = flt
    return client.scroll(**kw)


def delete_by_filter(client: QdrantClient, collection_name: str, flt: Filter):
    try:
        flt_json = flt.model_dump_json(indent=2)
    except Exception:
        flt_json = str(flt)
    log.debug("QDRANT_DELETE | collection='%s' | filter=%s", collection_name, flt_json)

    sig = inspect.signature(client.delete)
    if "filter" in sig.parameters:
        return client.delete(collection_name=collection_name, filter=flt, wait=True)
    return client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=flt),
        wait=True,
    )


# ======================= Filtros base (top-level o metadata.*) =======================


def make_filter_patient_source(patient_id: str, *, meta_prefix: bool) -> Filter:
    """Filtro paciente+source en una sola ubicación (top-level si meta_prefix=False; metadata.* si True)."""
    pkey = ("metadata." if meta_prefix else "") + "paciente_id"
    skey = ("metadata." if meta_prefix else "") + "source"
    return Filter(
        must=[
            FieldCondition(key=pkey, match=MatchValue(value=str(patient_id))),
            FieldCondition(key=skey, match=MatchValue(value="hci")),
        ]
    )


# ======================= Snapshot marker =======================


def _make_filter_snapshot_marker_meta(patient_id: str, document_id: str) -> Filter:
    # marker guardado bajo metadata.* (típico cuando se inserta con LlamaIndex)
    return Filter(
        must=[
            FieldCondition(
                key="metadata.paciente_id", match=MatchValue(value=str(patient_id))
            ),
            FieldCondition(
                key="metadata.document_id", match=MatchValue(value=str(document_id))
            ),
            FieldCondition(key="metadata.source", match=MatchValue(value="hci")),
            FieldCondition(key="metadata.marker", match=MatchValue(value="snapshot")),
        ]
    )


def _make_filter_snapshot_marker_top(patient_id: str, document_id: str) -> Filter:
    # marker guardado en top-level (recomendado con upsert_snapshot_marker)
    return Filter(
        must=[
            FieldCondition(key="paciente_id", match=MatchValue(value=str(patient_id))),
            FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            FieldCondition(key="source", match=MatchValue(value="hci")),
            FieldCondition(key="marker", match=MatchValue(value="snapshot")),
        ]
    )


def snapshot_marker_exists(
    client: QdrantClient, collection_name: str, patient_id: str, document_id: str
) -> bool:
    """Busca el marker primero en metadata.* y luego en top-level. Loguea ambos intentos."""
    log.debug(
        "HC_MARKER_LOOKUP | patient_id=%s | document_id=%s", patient_id, document_id
    )

    # 1) metadata.*
    flt_meta = _make_filter_snapshot_marker_meta(patient_id, document_id)
    c_meta = client_count(client, collection_name, flt_meta)
    log.debug("HC_MARKER_LOOKUP_RESULT | location=metadata.* | count=%s", c_meta)
    if c_meta > 0:
        return True

    # 2) top-level
    flt_top = _make_filter_snapshot_marker_top(patient_id, document_id)
    c_top = client_count(client, collection_name, flt_top)
    log.debug("HC_MARKER_LOOKUP_RESULT | location=top-level | count=%s", c_top)
    return c_top > 0


def upsert_snapshot_marker(
    client: QdrantClient,
    collection_name: str,
    patient_id: str,
    document_id: str,
    embedding_dim: int,
):
    """
    Inserta/actualiza 1 punto marker con payload TOP-LEVEL (robusto y simple).
    Recomendado para NO-OP determinista.
    """
    pid = hashlib.md5(
        f"marker|hci|{patient_id}|{document_id}".encode("utf-8")
    ).hexdigest()
    payload = {
        "paciente_id": str(patient_id),
        "document_id": str(document_id),
        "source": "hci",
        "marker": "snapshot",
        "ingestion_ts": datetime.utcnow().isoformat(),
    }
    zeros = [0.0] * int(embedding_dim)
    batch = Batch(ids=[pid], vectors=[zeros], payloads=[payload])
    log.debug("HC_MARKER_UPSERT | point_id=%s | collection=%s", pid, collection_name)
    client.upsert(collection_name=collection_name, points=batch, wait=True)


def build_snapshot_marker_node(patient_id: str, document_id: str) -> Document:
    """
    Alternativa: construir marker como Document (se guardará bajo metadata.* si se inserta con LlamaIndex).
    Preferí upsert_snapshot_marker para top-level.
    """
    text = f"snapshot|{patient_id}|{document_id}"
    meta = {
        "paciente_id": str(patient_id),
        "document_id": document_id,
        "source": "hci",
        "marker": "snapshot",
        "ingestion_ts": datetime.utcnow().isoformat(),
    }
    pid = hashlib.md5(text.encode("utf-8")).hexdigest()
    doc = Document(text=text, metadata=meta)
    try:
        doc.id_ = pid
    except Exception:
        doc.metadata["point_id"] = pid
    return doc


# Export público
__all__ = [
    "normalize_text",
    "node_hash_text",
    "canon_fecha",
    "canon_seccion",
    "snapshot_hash_from_nodes",
    "point_id_for_node",
    "client_count",
    "client_scroll",
    "delete_by_filter",
    "make_filter_patient_source",
    "snapshot_marker_exists",
    "upsert_snapshot_marker",
    "build_snapshot_marker_node",
]
