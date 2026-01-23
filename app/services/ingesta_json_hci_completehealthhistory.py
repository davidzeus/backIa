# app/services/ingesta_json_hci_completehealthhistory.py
"""
Ingesta HC -> Qdrant (Hybrid: Dense + Sparse SPLADE)
- Colección: hc_chat_db_hybrid
- Genera vectores densos (para contexto semántico) y dispersos (para keywords exactas).
- Si snapshot (paciente_id + document_id) ya existe => NO-OP
"""

import logging
import os
import re
import gc
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from llama_index.core.schema import TextNode
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType, SparseVectorParams, SparseIndexParams, PointStruct

from app.infra.qdrant_client_factory import get_qdrant_client
from app.infra.ml_providers import get_embedder, get_sparse_embedder

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
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hc_chat_db_hybrid") # 🟢 Nombre actualizado

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


# -----------------------------------------------------------------------------
# 2) Guardado en Qdrant (NO-OP o REPLACE + marker)
# -----------------------------------------------------------------------------
def _asegurar_coleccion_qdrant():
    client = get_qdrant_client()
    
    must_recreate = False
    if not client.collection_exists(COLLECTION_NAME):
        must_recreate = True
    else:
        # Validar esquema existente
        try:
            info = client.get_collection(COLLECTION_NAME)
            vect_cfg = info.config.params.vectors
            # Si no es diccionario (es vector default) o no tiene la clave 'text-dense'
            if not isinstance(vect_cfg, dict) or "text-dense" not in vect_cfg:
                log.warning(f"⚠️ Colección '{COLLECTION_NAME}' tiene esquema antiguo (sin 'text-dense'). Se recreará.")
                must_recreate = True
        except Exception as e:
            log.warning(f"No se pudo inspeccionar colección {COLLECTION_NAME}: {e}. Se intentará recrear.")
            must_recreate = True

    if must_recreate:
        log.info("📦 Creando colección HYBRID '%s' (text-dense + text-sparse)", COLLECTION_NAME)
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "text-dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False, 
                    )
                )
            }
        )
        
        # Índices de Payload
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
            ("node_hash",          PayloadSchemaType.KEYWORD),
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
    try:
        client = get_qdrant_client()
        _asegurar_coleccion_qdrant()

        # 1. Preparar Nodos (Saneamiento)
        all_nodes: List[Document] = [n for nodos in chunks.values() for n in nodos]
        all_nodes, dropped = _final_gate_nonempty(all_nodes, paciente_id)
        
        if not all_nodes:
            return {"mensaje": "Sin datos válidos.", "paciente_id": paciente_id, "inserted": 0, "nodos_omitidos": dropped}

        # 2. Check Snapshot (Idempotencia)
        document_id = _snapshot_hash_from_nodes(all_nodes)
        
        # ---- NO-OP si el snapshot ya existe ----
        if _snapshot_marker_exists(client, COLLECTION_NAME, paciente_id, document_id):
            log.info("🔁 Snapshot ya existente (NO-OP) | patient=%s | doc=%s", paciente_id, document_id)
            return {
                "mensaje": "La historia clínica no ha cambiado y ya estaba almacenada.",
                "paciente_id": paciente_id,
                "document_id": document_id,
                "inserted": 0,
                "change_ratio": 0.0,
                "source": "hci_hybrid"
            }

        # Limpieza previa de este paciente (Delete old versions)
        try:
            for meta in (True, False):
                flt_del = _make_filter_patient_source(paciente_id, meta_prefix=meta)
                _delete_by_filter(client, COLLECTION_NAME, flt_del)
        except Exception:
            pass

        # 3. PROCESAMIENTO HÍBRIDO POR LOTES (Manual Control)
        # BATCH_SIZE pequeño para evitar que SPLADE consuma toda la RAM
        BATCH_SIZE = 16 
        total_nodes = len(all_nodes)
        
        log.info(f"🚀 Procesando {total_nodes} nodos HÍBRIDOS en lotes de {BATCH_SIZE}...")

        dense_model = get_embedder()      # Tu modelo HuggingFace
        sparse_model = get_sparse_embedder() # Tu modelo FastEmbed (SPLADE)
        
        now_iso = datetime.utcnow().isoformat()
        inserted_count = 0

        for i in range(0, total_nodes, BATCH_SIZE):
            # A. Slice del lote
            batch_nodes = all_nodes[i : i + BATCH_SIZE]
            batch_texts = [n.text for n in batch_nodes]
            
            # B. Generación de Vectores (Aquí controlamos la memoria)
            # 1. Dense (Semántico)
            batch_dense = [dense_model.get_text_embedding(t) for t in batch_texts]
            
            # 2. Sparse (Keywords/SPLADE) - Convertimos generador a lista
            batch_sparse = list(sparse_model.embed(batch_texts))
            
            # C. Construcción de Puntos Qdrant
            points = []
            for j, node in enumerate(batch_nodes):
                # ID determinístico y Hash
                node_h = _node_hash_text(node.text)
                seccion = str(node.metadata.get("seccion", ""))
                fecha = str(node.metadata.get("fecha", ""))
                pid = _point_id_for_node(str(paciente_id), node.text, seccion, fecha)
                
                # Payload enriquecido
                payload = node.metadata.copy()
                payload.update({
                    "document_id": document_id,
                    "ingestion_ts": now_iso,
                    "node_hash": node_h,
                    "source": "hci",
                    # Guardamos el JSON del nodo para compatibilidad futura con LlamaIndex
                    "_node_content": node.json(exclude={"embedding"}) 
                })
                
                # Extraemos el vector sparse del objeto de FastEmbed
                sp_vec = batch_sparse[j]
                
                # Creamos el punto con AMBOS vectores (Named Vectors)
                points.append(PointStruct(
                    id=pid,
                    payload=payload,
                    vector={
                        "text-dense": batch_dense[j],
                        "text-sparse": {
                            "indices": sp_vec.indices.tolist(),
                            "values": sp_vec.values.tolist()
                        }
                    }
                ))
            
            # D. Subida Directa (Bypassing LlamaIndex insert)
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            inserted_count += len(points)
            
            # E. Limpieza de Memoria Inmediata
            del batch_dense
            del batch_sparse
            del points
            gc.collect() 
            
            log.info(f"   ✅ Lote {i//BATCH_SIZE + 1} subido ({min(i+BATCH_SIZE, total_nodes)}/{total_nodes})")

        # 4. Insertar Marker Final
        _upsert_snapshot_marker(client, COLLECTION_NAME, paciente_id, document_id, EMBEDDING_DIM)

        log.info(f"✅ Ingesta Híbrida completada. Insertados: {inserted_count}")

        return {
            "mensaje": "Historia clínica HÍBRIDA actualizada.", 
            "paciente_id": paciente_id,
            "document_id": document_id,
            "inserted": inserted_count, 
            "mode": "Hybrid (Dense + SPLADE)"
        }

    except UnexpectedResponse as ur:
        log.error("Error Qdrant: %s - %s", ur.status_code, ur.content)
        return {"error": "Error inesperado al comunicarse con Qdrant"}
    except Exception as exc:
        log.exception("❌ Error Crítico en Ingesta Híbrida: %s", exc)
        gc.collect() # Intento final de limpieza
        return {"error": str(exc)}
