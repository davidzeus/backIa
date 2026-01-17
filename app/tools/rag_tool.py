# app/tools/rag_tool.py
import logging
import time
import numpy as np
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import os

# --- Importaciones de Infraestructura IA ---
from app.infra.ml_providers import get_embedder
from app.helpers.query_helpers import (
    COLLECTION, _qdrant, detect_qdrant_text_key, SafeQdrantVectorStore, q_filter_paciente,
    serialize_nodes
)
from app.utils.context_manager import get_patient_context, set_agent_sources
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import NodeWithScore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    AutoTokenizer = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

log = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
SIM_TOP_K = 50  # Recuperación amplia Qdrant
RERANK_TOP_N =12
MAX_SAFE_LIMIT = 25  # <--- NUEVA VARIABLE
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" 
#RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


ENABLE_ONNX = True
CPU_THREADS = 12 # Ajustamos a un buen número para tu PC

# --- SINGLETON DEL RERANKER ---
_reranker_session = None
_reranker_model = None
_tokenizer = None
_use_onnx = False

def get_reranker_local():
    """
    Inicializa el Reranker en GPU o CPU según variable de entorno RERANKER_DEVICE.
    Singleton: solo carga UNA VEZ y permanece en memoria GPU.
    """
    global _reranker_session, _tokenizer, _use_onnx, _reranker_model

    if _reranker_session is not None or _reranker_model is not None:
        log.debug("✅ [RERANKER] Reutilizando instancia ya cargada en GPU.")
        return # Ya iniciado - NO recarga

    log.info("⚙️ [RERANKER] Inicializando motor local...")

    try:
        import torch
        
        # 🎮 LEER CONFIGURACIÓN DESDE .ENV
        device_config = os.getenv("RERANKER_DEVICE", "cpu").lower()
        
        # Validar que el dispositivo configurado esté disponible
        if device_config == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                log.info(f"🚀 [RERANKER] GPU detectada: {gpu_name}")
                log.info(f"💾 [RERANKER] VRAM disponible: {free_memory:.2f} GB / {total_memory:.2f} GB")
            else:
                log.warning("⚠️ [RERANKER] GPU configurada pero no disponible. Fallback a CPU.")
                device = "cpu"
        else:
            device = "cpu"
            log.info("🖥️ [RERANKER] Configurado para usar CPU")
        
        # Configuración de hilos CPU si corresponde
        if device == "cpu":
            torch.set_num_threads(CPU_THREADS)
            os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
            os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

        # Cargar modelo en el dispositivo configurado
        log.info(f"🔄 [RERANKER] Cargando {RERANKER_MODEL_NAME} en {device.upper()} (permanecerá en memoria)...")
        _reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=512,
            device=device
        )
        _use_onnx = False
        log.info(f"✅ [RERANKER] Modelo cargado en {device.upper()} y listo para inferencia.")
        
        # Warm-up
        _ = _reranker_model.predict([("warm up", "test")])
        
        # Mostrar uso de memoria si es GPU
        if device == "cuda":
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            log.info(f"📊 [RERANKER] VRAM usada: {vram_used:.2f} GB (modelo permanece en GPU)")

    except Exception as e:
        log.error(f"❌ Error iniciando Reranker local: {e}")
        raise e

def run_rerank_logic(query: str, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
    """
    Ejecuta la predicción (Score)
    """
    get_reranker_local() # Asegurar carga
    
    if not documents:
        return []

    pairs = [[query, doc] for doc in documents]

    # Usamos el modelo cargado (PyTorch CPU)
    scores = _reranker_model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )

    # Crear tuplas (indice, score) y ordenar
    results = [(idx, float(score)) for idx, score in enumerate(scores)]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


# --- FUNCIÓN MATEMÁTICA DE DECAIMIENTO ---
def calculate_recency_score(timestamp: float, reference_ts: float, half_life_days: int = 365) -> float:
    """
    Calcula un score de 0.0 a 1.0 basado en la antigüedad.
    Usa una curva de decaimiento exponencial.
    """
    if not timestamp or timestamp <= 0: return 0.0
    
    # Diferencia en días
    delta_seconds = reference_ts - timestamp
    # Si es futuro (error de data), lo tratamos como "hoy" (0 días)
    delta_days = max(0, delta_seconds / 86400)
    
    # Fórmula: Score = e^(-lambda * days)
    # Si delta_days = half_life_days, el score debería ser 0.5 aprox?
    # Usando lambda = ln(2) / half_life
    # Score = exp( - (ln(2) / half_life) * days )
    
    # Simplificación lineal-logarítmica robusta propuesta:
    # Score = exp( -1 * (days / half_life) )
    # Si days = half_life -> exp(-1) = 0.36
    score = math.exp(-1 * (delta_days / half_life_days))
    return score

def calculate_proximity_score(node_ts: float, target_ts: float, sigma_days: float = 2.0) -> float:
    """
    Calcula relevancia basada en proximidad a una FECHA OBJETIVO (Gaussian).
    Score = 1.0 si es el mismo momento.
    Score cae suavemente (campana de Gauss) a medida que nos alejamos.
    sigma_days controla el ancho de la campana (ej: 2 días de tolerancia alta).
    """
    if not target_ts or not node_ts: return 0.0
    
    diff_seconds = abs(node_ts - target_ts)
    diff_days = diff_seconds / 86400.0
    
    # Formula Gaussiana: exp( - (x^2) / (2 * sigma^2) )
    score = math.exp(- (diff_days**2) / (2 * (sigma_days**2)))
    return score

def apply_proximity_reranking(nodes, target_ts, alpha: float = 2.0):
    """Re-rankea nodos priorizando cercanía a la fecha target."""
    for n in nodes:
        node_obj = getattr(n, "node", n)
        sem_score = float(getattr(n, 'score', 0) or 0)
        
        # Obtener fecha del nodo
        try:
            val = node_obj.metadata.get("fecha_ts", 0)
            if isinstance(val, str): val = datetime.fromisoformat(val).timestamp()
            node_ts = float(val or 0)
        except:
            node_ts = 0
            
        prox_score = calculate_proximity_score(node_ts, target_ts, sigma_days=2.0)
        
        # Fusión: Score Semántico * (1 + Alpha * Proximidad)
        # Esto da un boost fuerte a lo cercano, sin matar lo semánticamente perfecto pero lejano
        n.score = sem_score * (1 + (alpha * prox_score))
        
        node_obj.metadata['debug_prox'] = round(prox_score, 4)

    nodes.sort(key=lambda x: x.score, reverse=True)
    return nodes


def apply_time_weighted_reranking(nodes, alpha: float = 0.5):
    """
    Fusiona el score semántico con el score temporal.
    Alpha controla cuánto pesa el tiempo (0.0 a 1.0+).
    """
    now_ts = datetime.now().timestamp()
    
    for n in nodes:
        node_obj = getattr(n, "node", n)
        
        # 1. Obtener Score Semántico
        # BAAI suele dar logits o scores no normalizados. Asumimos algo positivo.
        sem_score = float(getattr(n, 'score', 0) or 0)
        if sem_score < 0: sem_score = 0 # Safety clamp
        
        # 2. Obtener Score Temporal
        # Intentamos obtener fecha_ts del metadata
        md = getattr(node_obj, "metadata", {}) or {}
        try:
            val = md.get("fecha_ts", 0)
            if isinstance(val, str):
                 # Intenta parsear si es ISO, aunque suele ser int en Qdrant
                 try: val = datetime.fromisoformat(val).timestamp()
                 except: val = 0
            node_ts = float(val or 0)
        except:
            node_ts = 0
            
        time_score = calculate_recency_score(node_ts, now_ts, half_life_days=90) # 3 meses ~ vida media/relevancia alta
        
        # 3. FUSIÓN (Weighted Sum / Boosting)
        # Final = Semantico * (1 + (Alpha * Tiempo))
        # Si Alpha=2 y es Hoy (Time=1) -> Multiplica x3 el score semántico.
        # Si es Viejo (Time=0) -> Multiplica x1 (se mantiene igual).
        
        boosted_score = sem_score * (1 + (alpha * time_score))
        
        # Guardamos scores para debug en metadata del nodo
        node_obj.metadata['debug_sem'] = round(sem_score, 4)
        node_obj.metadata['debug_time'] = round(time_score, 4)
        node_obj.metadata['debug_final'] = round(boosted_score, 4)
        
        # Actualizamos el score principal del NodeWithScore
        n.score = boosted_score

    # Reordenar por el nuevo score
    nodes.sort(key=lambda x: x.score, reverse=True)
    return nodes



# --- FUNCIÓN PRINCIPAL DE LA HERRAMIENTA ---
def search_clinical_history(query: str, 
                            qdrant_filters: Optional[Dict[str, Any]] = None,
                            strict: bool = False,
                            limit: Optional[int] = None) -> str:
    """
    Busca en Qdrant y re-ordena LOCALMENTE usando el modelo BAAI.
    Incluye FALLBACK AUTOMÁTICO si los filtros del Planner son muy restrictivos.
    """
    try:
        start_total = time.perf_counter()
        target_proximity_ts = None 
        
        # 1. Configurar Límites
        requested_limit = limit if limit and limit > 0 else RERANK_TOP_N
        target_top_k = min(requested_limit, MAX_SAFE_LIMIT)
        dynamic_sim_k = max(SIM_TOP_K, target_top_k * 3) 

        patient_id = get_patient_context()
        cleaned_id = ''.join(filter(str.isdigit, str(patient_id)))
        
        if not cleaned_id:
            return "Error: ID de paciente inválido."

        # 2. Configurar Filtros Qdrant
        embed_model = get_embedder()
        
        # --- A. Filtro Base (Solo Paciente) ---
        patient_only_filter = q_filter_paciente(cleaned_id)
        if not isinstance(patient_only_filter, Filter):
            patient_only_filter = Filter(must=[patient_only_filter] if patient_only_filter else [])
        
        # --- B. Filtro Smart (Planner) ---
        # Clonamos del base para no ensuciarlo
        smart_filter = patient_only_filter.model_copy(deep=True)
        if smart_filter.must_not is None: smart_filter.must_not = []
        smart_filter.must_not.append(FieldCondition(key="marker", match=MatchValue(value="snapshot")))
        
        if smart_filter.must is None: smart_filter.must = []
        if smart_filter.should is None: smart_filter.should = []

        if qdrant_filters:
            is_strict = False # Mantener False por default según lógica previa
            
            # 1. Name Hints
            if 'name_hint' in qdrant_filters and qdrant_filters['name_hint']:
                hints = qdrant_filters['name_hint']
                if isinstance(hints, str): hints = [hints]
                group_matches = [FieldCondition(key="name", match=MatchValue(value=h)) for h in hints]
                if is_strict: smart_filter.must.append(Filter(should=group_matches))
                else: smart_filter.should.extend(group_matches)
            
            # 2. Time Window (Soft)
            tw = qdrant_filters.get('time_window') or {}
            start_ts = tw.get("start")
            end_ts = tw.get("end")

            if start_ts is not None or end_ts is not None:
                try:
                    range_params = {}
                    SOFT_WINDOW = 259200 # 3 días
                    
                    if start_ts is not None: 
                        range_params['gte'] = int(float(start_ts) - SOFT_WINDOW)
                        target_proximity_ts = float(start_ts)
                        
                    if end_ts is not None: 
                        range_params['lte'] = int(float(end_ts) + SOFT_WINDOW)
                        if not target_proximity_ts: target_proximity_ts = float(end_ts)

                    smart_filter.must.append(
                        FieldCondition(key="fecha_ts", range=Range(**range_params))
                    )
                    log.info(f"⏳ Filtro SOFT aplicado: GTE={range_params.get('gte')} | LTE={range_params.get('lte')}")
                except Exception as e:
                    log.error(f"❌ Error aplicando filtro temporal: {e}")
            
            # 3. Service Hints
            if 'service_hints' in qdrant_filters:
                sids = [int(s['servicio_id']) for s in qdrant_filters['service_hints'] if s.get('servicio_id')]
                if sids:
                    s_conds = [FieldCondition(key="servicio_id", match=MatchValue(value=sid)) for sid in sids]
                    if is_strict: smart_filter.must.append(Filter(should=s_conds))
                    else: smart_filter.should.extend(s_conds)

        # 3. Retrieval 1 (Smart)
        log.info(f"🔎 [RAG] Intento 1 (Smart): '{query}' | Target: {target_top_k}")
        
        text_key = detect_qdrant_text_key(COLLECTION, patient_only_filter)
        store = SafeQdrantVectorStore(client=_qdrant, collection_name=COLLECTION, text_key=text_key)
        index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)
        
        retriever = index.as_retriever(similarity_top_k=dynamic_sim_k, vector_store_kwargs={"qdrant_filters": smart_filter})
        nodes = retriever.retrieve(query)
        
        # --- 3.5 FALLBACK DE SEGURIDAD (BOTÓN DE PÁNICO) ---
        if not nodes:
            log.warning("⚠️ [RAG] Intento 1 falló (0 docs). Activando FALLBACK (Búsqueda Libre)...")
            
            # Creamos filtro fallback (Paciente + No Snapshot)
            fallback_filter = patient_only_filter.model_copy(deep=True)
            if fallback_filter.must_not is None: fallback_filter.must_not = []
            fallback_filter.must_not.append(FieldCondition(key="marker", match=MatchValue(value="snapshot")))

            retriever_fallback = index.as_retriever(similarity_top_k=dynamic_sim_k, vector_store_kwargs={"qdrant_filters": fallback_filter})
            nodes = retriever_fallback.retrieve(query)
            
            if nodes:
                log.info(f"✅ [RAG] Fallback exitoso: Recuperados {len(nodes)} docs.")
            else:
                log.warning("❌ [RAG] Fallback falló. La base de datos parece vacía para este paciente.")
                # Si falla también, retornamos vacío
        # ---------------------------------

        if not nodes:
            return "No se encontraron registros."

        # 4. Hidratación y Limpieza
        doc_texts = []
        valid_nodes = []
        for n in nodes:
            node = getattr(n, "node", n)
            txt = getattr(node, "text", "")
            if not txt:
                 md = getattr(node, "metadata", {}) or {}
                 txt = md.get(text_key, "")
            
            # Solo procesamos si hay texto
            if txt:
                node.text = txt
                doc_texts.append(txt)
                valid_nodes.append(n)
        
        # 3. RERANKING LOCAL (Cross-Encoder)
        final_nodes = []
        if doc_texts:
            log.info(f"🧠 [RERANK] Reordenando {len(doc_texts)} docs con {RERANKER_MODEL_NAME}...")
            start_rerank = time.perf_counter()
            
            try:
                # Intentamos reordenar
                ranked_indices = run_rerank_logic(query, doc_texts, top_k=target_top_k)
                
                for idx, score in ranked_indices:
                    node = valid_nodes[idx]
                    node.score = score
                    final_nodes.append(node)
                
                log.info(f"✅ [RERANK] Terminado en {time.perf_counter()-start_rerank:.4f}s")
                
            except Exception as e:
                # ⚠️ FALLBACK DE SEGURIDAD ⚠️
                log.error(f"❌ FALLÓ RERANKER: {e} -> Usando orden original de Qdrant.")
                final_nodes = nodes[:target_top_k] 
        else:
            final_nodes = nodes[:target_top_k]
            
        # 4. RERANKING TEMPORAL FINAL (Proximidad vs Relevancia)
        if target_proximity_ts:
             log.info(f"🎯 Aplicando Proximity Reranking hacia: {target_proximity_ts}")
             # Usamos final_nodes que ya tiene el score semántico base
             final_nodes = apply_proximity_reranking(final_nodes, target_proximity_ts, alpha=2.0)
        else:
             # Default: Recency si no hay target date específico
             final_nodes = apply_time_weighted_reranking(final_nodes, alpha=0.5) 


        # 3.5. RECENCY RANK FUSION (Decaimiento Temporal)
        # Si detectamos "último/reciente", aplicamos Time Boosting
        '''
        if qdrant_filters and qdrant_filters.get("which"):
             which_intent = str(qdrant_filters.get('which', '')).lower()
             
             if which_intent in ['last', 'latest', 'actual', 'reciente', 'ultimo', 'último']:
                # Alpha alto = El tiempo importa mucho.
                # doc empty (score 0.01) * boost -> sigue bajo
                # doc relevant (score 0.7) * boost -> sube mucho
                log.info(f"⏳ [RAG] Intención '{which_intent}' detectada. Aplicando Time Weighted Reranking (Alpha=2.0).")
                final_nodes = apply_time_weighted_reranking(final_nodes, alpha=2.0)
                
             # Si el count es pequeño (ej "ultimo 1"), cortamos AHORA, después del re-ranking híbrido.
             count = int(qdrant_filters.get('count') or 1)
             if count > 0:
                 final_nodes = final_nodes[:count]
        '''
        # 4. Salida
        # 4. Salida
        try:
            serialized = serialize_nodes(final_nodes, limit=None) # ✅ Guardar TODOS los docs usados
            log.info(f"💾 [RAG] Guardando {len(serialized)} sources en contexto.")
            set_agent_sources(serialized)
        except Exception as e:
            log.error(f"❌ Error guardando sources en contexto: {e}", exc_info=True)

        output_lines = [f"--- EVIDENCIA RECUPERADA ({len(final_nodes)} docs) ---"]
        
        # Contador visual para los docs que SI pasan el filtro
        doc_display_counter = 1
        
        for i, n in enumerate(final_nodes):
            node = getattr(n, "node", n)
            text = (node.text or "").strip()
            
            # --- 🛡️ FILTRO QUIRÚRGICO V2 (CORREGIDO) ---
            
            text_lower = text.lower()
            
            # 1. Frases que indican contenido vacío o burocrático
            noise_phrases = [
                "paciente en seguimiento por el servicio",
                "seguimiento por especialidad",
                "seguimiento por cirugia",
                "pase a sala",
                "sin particularidades",
                "paciente en seguimiento",
                "continuar con igual indicaciones",
                "en plan quirurgico", 
                "en plan quirúrgico"
            ]

            # 2. DETECCIÓN INTELIGENTE
            # Verificamos si alguna frase basura está DENTRO del texto.
            # Y ADEMÁS verificamos que el texto sea corto (< 150 chars).
            # Esto evita borrar una nota larga que empiece con "Paciente en seguimiento..." pero luego agregue datos nuevos.
            
            has_noise = any(phrase in text_lower for phrase in noise_phrases)
            is_short = len(text) < 150 
            
            # 3. Basura técnica (Textos vacíos, solo guiones, o metadatos sueltos)
            # Ej: "-", "...", o texto que parece solo una fecha/hora sin mensaje
            is_technical_garbage = len(text) < 5 or not any(c.isalnum() for c in text)

            # CONDICIÓN FINAL:
            # Si contiene la frase prohibida Y es corto -> ES BASURA -> BORRAR.
            # Si es basura técnica -> BORRAR.
            if (has_noise and is_short) or is_technical_garbage:
                continue 

            # -------------------------------------



            md = getattr(node, "metadata", {}) or {}
            fecha = md.get("fecha", "S/F")
            seccion = md.get("name") or md.get("seccion", "Nota")
            servicio = md.get("servicio_desc") or "General"
            
            # Usamos el score re-rankeado (boosted) si existe
            score_val = getattr(n, "score", 0.0)
            
            output_lines.append(f"📄 DOC #{doc_display_counter} (Score: {score_val:.2f}) | Fecha: {fecha} | Servicio: {servicio} | Tipo: {seccion}")
            output_lines.append(f"{text}\n")
            doc_display_counter += 1
            
        return "\n".join(output_lines)

    except Exception as e:
        log.error(f"❌ Error en RAG Local: {e}", exc_info=True)
        return f"Error interno: {str(e)}"
