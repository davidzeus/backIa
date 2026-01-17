# app/services/ingesta_json_hci_completehealthhistory.py
"""
Ingesta HC -> Qdrant (simple y robusto con snapshot marker)
- Colección única: hc_chat_db
- Si snapshot (paciente_id + document_id) ya existe => NO-OP
- Si no existe => DELETE-ALL (paciente) + INSERT ALL + insertar marker (top-level)
"""

import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple
import re

from llama_index.core.schema import TextNode
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType

from app.infra.qdrant_client_factory import get_qdrant_client
from app.infra.ml_providers import get_embedder

from app.utils.agrupar_y_formatear_items_hc import (
    _canon_fecha_y_ts,
    _normalizar_grupo,
    generar_linea_de_tiempo_clinica,
)

# Helpers (ver app/utils/hc_vector_utils.py)
from app.utils.hc_vector_utils import (
    client_count as _client_count,
    delete_by_filter as _delete_by_filter,
    make_filter_patient_source as _make_filter_patient_source,
    node_hash_text as _node_hash_text,
    point_id_for_node as _point_id_for_node,
    snapshot_hash_from_nodes as _snapshot_hash_from_nodes,
    snapshot_marker_exists as _snapshot_marker_exists,
    upsert_snapshot_marker as _upsert_snapshot_marker,
)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hc_chat_db")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # usado por la colección
CHUNK_SIZE = int(os.getenv("HC_CHUNK_SIZE", "1000"))

_MIN_CHARS   = int(os.getenv("HC_MIN_CHARS", "5"))   # umbral mínimo de caracteres (bajo)
_MIN_TOKENS2 = int(os.getenv("HC_MIN_TOKENS2", "2")) # al menos 2 tokens si el texto es muy corto

# Logger de este módulo (silenciado por defecto: WARNING+)
log = logging.getLogger(__name__)
if not bool(int(os.getenv("INGESTA_DEBUG", "0"))):
    log.setLevel(logging.WARNING)


# === Helpers anti-vacío / saneo de nodos ===

_rx_space = re.compile(r"\s+")

def _sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = _rx_space.sub(" ", s)
    return s.strip()

def _is_nonempty_text(s: str) -> bool:
    """Texto no vacío y mínimamente útil (≥5 chars y ≈2 tokens si es corto)."""
    if not s:
        return False
    s = s.strip()
    if len(s) < _MIN_CHARS:
        return False
    # si es muy corto, exigir al menos 2 tokens
    if len(s) < 8 and len(s.split()) < _MIN_TOKENS2:
        return False
    return True
#---------------------------------------------------------------------------------
def _final_gate_nonempty(nodes: List[TextNode], paciente_id: str) -> Tuple[List[TextNode], int]:
    """Sanea, valida y completa metadatos mínimos por chunk; descarta vacíos."""
    kept, dropped = [], 0
    for n in nodes:
        txt = _sanitize_text(getattr(n, "text", "") or "")
        if not _is_nonempty_text(txt):
            dropped += 1
            continue

        md = dict(n.metadata or {})
        # mínimos garantizados
        md.setdefault("seccion", "SIN_SECCION")
        md.setdefault("seccion_raiz", "SIN_SECCION")
        md.setdefault("fecha", "")
        md.setdefault("fecha_ts", 0)
        md["paciente_id"] = str(paciente_id)
        md["texto_len"] = len(txt)  # longitud REAL del chunk

        n.text = txt
        n.metadata = md
        kept.append(n)
    return kept, dropped

# -----------------------------------------------------------------------------
# 1) Construcción de nodos desde la línea de tiempo
# -----------------------------------------------------------------------------
def agrupar_items_usando_linea_de_tiempo(
    json_clinico: Dict[str, Any],
    paciente_id: str,
) -> Dict[str, List[Document]]:
    linea_de_tiempo = generar_linea_de_tiempo_clinica(json_clinico)
    if not linea_de_tiempo:
        return {}

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    secciones: Dict[str, List[Document]] = defaultdict(list)

    for evento in linea_de_tiempo:
        texto = (evento.get("texto_formateado") or "").strip()
        if not texto:
            continue

        ruta = evento.get("ruta_seccion", "SIN_SECCION")
        nombre_nodo = ruta.split(" / ")[-1] if " / " in ruta else ruta
        seccion_raiz = evento.get("seccion_raiz", "SIN_SECCION")

        # ➜ Fecha clínica canonizada
        fecha_obj = evento.get("fecha_evento")
        fecha_iso, fecha_ts = _canon_fecha_y_ts(fecha_obj)

        servicio_id = evento.get("servicio_id")
        servicio_desc = evento.get("servicio") or "No especificado"
        doctor_full = evento.get("doctor_full") or "N/A"

        group_raw = evento.get("group")
        hhg = evento.get("healthHistoryGroup")
        proc_num = evento.get("procedureNumber")

        grupo_norm = _normalizar_grupo(group_raw, hhg)

        metadata = {
            "name": nombre_nodo,
            "seccion": ruta,
            "seccion_raiz": seccion_raiz,
            "servicio": servicio_desc,
            "doctor_full": doctor_full,
            "paciente_id": str(paciente_id),
            # fechas
            "fecha": fecha_iso,  # YYYY-MM-DD para match exacto
            "fecha_ts": fecha_ts,  # entero para rangos
            # HCI crudos
            "group": group_raw,
            "healthHistoryGroup": hhg,
            "procedureNumber": proc_num,
            # derivados
            "grupo": grupo_norm,
            "internacion_tramite": proc_num,
            "texto_len": len(texto),
        }

        if servicio_id is not None:
            try:
                metadata["servicio_id"] = int(servicio_id)
            except Exception:
                metadata["servicio_id"] = servicio_id

        doc = Document(text=texto, metadata=metadata)
        secciones[ruta].append(doc)

    all_chunks: Dict[str, List[Document]] = defaultdict(list)
    #===========================================================
    for seccion, docs in secciones.items():
        if not docs:
            continue
        nodes = splitter.get_nodes_from_documents(docs)

        # ⚠️ filtro ligero post-split (no crítico, la puerta final igual protege)
        _kept_local = []
        for node in nodes:
            node.text = _sanitize_text(getattr(node, "text", "") or "")
            if _is_nonempty_text(node.text):
                # texto_len por chunk (aquí o en la puerta final)
                md = dict(node.metadata or {})
                md["texto_len"] = len(node.text)
                node.metadata = md
                _kept_local.append(node)
        if _kept_local:
            all_chunks[seccion].extend(_kept_local)

    #===========================================================
    return all_chunks


#==Helpers====================================================================
def _sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = _rx_space.sub(" ", s)
    return s.strip()

def _is_nonempty_text(s: str) -> bool:
    """Texto no vacío y mínimamente útil (≥5 chars y ≈2 tokens si es corto)."""
    if not s:
        return False
    s = s.strip()
    if len(s) < _MIN_CHARS:
        return False
    # si es muy corto, exigir al menos 2 tokens
    if len(s) < 8 and len(s.split()) < _MIN_TOKENS2:
        return False
    return True

def _final_gate_nonempty(nodes: List[TextNode], paciente_id: str) -> (List[TextNode], int):
    """Sanea, valida y completa metadatos mínimos por chunk; descarta vacíos."""
    kept, dropped = [], 0
    for n in nodes:
        txt = _sanitize_text(getattr(n, "text", "") or "")
        if not _is_nonempty_text(txt):
            dropped += 1
            continue

        md = dict(n.metadata or {})
        # mínimos garantizados
        md.setdefault("seccion", "SIN_SECCION")
        md.setdefault("seccion_raiz", "SIN_SECCION")
        md.setdefault("fecha", "")
        md.setdefault("fecha_ts", 0)
        md["paciente_id"] = str(paciente_id)

        # longitud del CHUNK real (no del evento)
        md["texto_len"] = len(txt)

        n.text = txt
        n.metadata = md
        kept.append(n)
    return kept, dropped

# -----------------------------------------------------------------------------
# 2) Guardado en Qdrant (NO-OP o REPLACE + marker)
# -----------------------------------------------------------------------------
def _asegurar_coleccion_qdrant():
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        log.debug("📦 Creando colección '%s'", COLLECTION_NAME)
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    # Índices para filtros veloces (idempotentes; si existen, Qdrant lo avisa y seguimos)
    for field, schema in [
        ("paciente_id",        PayloadSchemaType.KEYWORD),
        ("procedureNumber",    PayloadSchemaType.INTEGER),
        ("fecha_ts",           PayloadSchemaType.INTEGER),
        ("servicio_id",        PayloadSchemaType.INTEGER),
        ("servicio",           PayloadSchemaType.KEYWORD),
        ("group",              PayloadSchemaType.KEYWORD),
        ("healthHistoryGroup", PayloadSchemaType.KEYWORD),
        ("name",               PayloadSchemaType.KEYWORD),
        ("marker",             PayloadSchemaType.KEYWORD),
        ("source",             PayloadSchemaType.KEYWORD),
        ("texto_len",          PayloadSchemaType.INTEGER),
        ("document_id",        PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        except Exception as e:
            log.debug("create_payload_index('%s') → %s", field, e)



def guardar_chunks(
    chunks: Dict[str, List[Document]],
    paciente_id: str,
) -> Dict[str, Any]:
    """
    Lógica determinista:
      1) Calcula document_id (hash del snapshot)
      2) Si existe snapshot marker => NO-OP
      3) Si no existe => DELETE-ALL (paciente) + INSERT ALL + insertar marker
    """
    try:
        client = get_qdrant_client()
        _asegurar_coleccion_qdrant()

        # Flatten
        all_nodes: List[Document] = [n for nodos in chunks.values() for n in nodos]

        # 🚪 Puerta final anti-vacío
        all_nodes, dropped = _final_gate_nonempty(all_nodes, paciente_id)
        if dropped:
            log.warning("⚠️ Nodos descartados por texto vacío/corto: %s", dropped)

        if not all_nodes:
            log.warning("No se generaron nodos válidos; no se insertará nada en Qdrant.")
            return {
                "mensaje": "No hay datos clínicos válidos para almacenar.",
                "paciente_id": paciente_id,
                "nodos_omitidos": dropped,
            }

        # IDs determinísticos y hash
        seen_hashes = set()
        dedup_nodes: List[Document] = []
        for n in all_nodes:
            # hash por contenido (dedupe)
            node_h = _node_hash_text(n.text)
            if node_h in seen_hashes:
                continue
            seen_hashes.add(node_h)

            seccion = str(n.metadata.get("seccion", ""))
            fecha   = str(n.metadata.get("fecha", ""))
            pid     = _point_id_for_node(str(paciente_id), n.text, seccion, fecha)
            try:
                n.id_ = pid  # TextNode.id_ (LlamaIndex)
            except Exception:
                n.metadata["point_id"] = pid

            n.metadata["node_hash"] = node_h
            dedup_nodes.append(n)

        if not dedup_nodes:
            return {
                "mensaje": "Todos los nodos eran duplicados o inválidos.",
                "paciente_id": paciente_id,
                "inserted": 0,
                "deleted": 0,
            }

        # Snapshot hash (estable)
        document_id = _snapshot_hash_from_nodes(dedup_nodes)
        log.debug(
            "HC_DOC_ID | patient_id=%s | document_id=%s | nodes=%s",
            paciente_id, document_id, len(dedup_nodes),
        )

        # ---- NO-OP si el snapshot ya existe ----
        if _snapshot_marker_exists(client, COLLECTION_NAME, paciente_id, document_id):
            log.debug("🔁 Snapshot existente (NO-OP) | patient=%s | doc=%s", paciente_id, document_id)
            return {
                "mensaje": "La historia clínica no ha cambiado y ya estaba almacenada.",
                "paciente_id": paciente_id,
                "document_id": document_id,
                "inserted": 0,
                "deleted": 0,
                "change_ratio": 0.0,
                "source": "hci",
            }

        # (Informativo) contar previos para métricas
        prev_total = 0
        try:
            flt_prev = _make_filter_patient_source(paciente_id, meta_prefix=True)
            prev_total = _client_count(client, COLLECTION_NAME, flt_prev)
            if prev_total == 0:
                flt_prev = _make_filter_patient_source(paciente_id, meta_prefix=False)
                prev_total = _client_count(client, COLLECTION_NAME, flt_prev)
        except Exception as e:
            log.warning("HC_PREV_COUNT_WARN | %s", e)
            prev_total = 0

        # Completar payload común
        now_iso = datetime.utcnow().isoformat()
        for n in dedup_nodes:
            n.metadata.update(
                {
                    "paciente_id": str(paciente_id),
                    "document_id": document_id,
                    "source": "hci",
                    "ingestion_ts": now_iso,
                }
            )

        # DELETE-ALL del paciente
        for meta in (True, False):
            try:
                flt_del = _make_filter_patient_source(paciente_id, meta_prefix=meta)
                _delete_by_filter(client, COLLECTION_NAME, flt_del)
            except Exception as e:
                log.warning("HC_DELETE_WARN | meta_prefix=%s | %s", meta, e)

        # INSERT ALL (con embedder explícito y storage contextualizado)
        embedder = get_embedder()
        store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=store)
        index = VectorStoreIndex.from_vector_store(store, storage_context=storage_context, embed_model=embedder)

        # Inserción (LlamaIndex hace batch interno; si tu versión soporta batch_size, podés pasarlo)
        index.insert_nodes(dedup_nodes)

        # INSERT marker en top-level (para futuros NO-OP)
        _upsert_snapshot_marker(client, COLLECTION_NAME, paciente_id, document_id, EMBEDDING_DIM)

        # (Opcional) verificar marker
        marker_ok = _snapshot_marker_exists(client, COLLECTION_NAME, paciente_id, document_id)
        log.debug("HC_MARKER_POSTCHECK | patient=%s | doc=%s | exists=%s", paciente_id, document_id, marker_ok)

        inserted = len(dedup_nodes) + 1  # +1 marker
        log.debug("✅ Replace completo | patient=%s | inserted=%s | deleted≈%s", paciente_id, inserted, prev_total)

        return {
            "mensaje": "Historia clínica actualizada con nuevos datos.",
            "paciente_id": paciente_id,
            "document_id": document_id,
            "inserted": inserted,
            "deleted": prev_total,
            "change_ratio": None,
            "source": "hci",
        }

    except UnexpectedResponse as ur:
        log.error("Error Qdrant: %s - %s", ur.status_code, ur.content)
        return {"error": "Error inesperado al comunicarse con Qdrant"}
    except Exception as exc:
        log.exception("Fallo general al guardar en Qdrant: %s", exc)
        return {"error": "Fallo general al guardar en Qdrant"}

