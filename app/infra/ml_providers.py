# app/infra/ml_providers.py
"""
Proveedores de ML (Embeddings y LLM) para FastAPI / servicios.
- get_embedder(): singleton con LRU cache
- get_llm(): instancia nueva por request (thread-safe)
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

log = logging.getLogger(__name__)

# Config vía ENV (si quieres, centraliza en app/core/config.py)
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "16"))
EMBEDDER_DEVICE   = os.getenv("EMBEDDER_DEVICE", "cpu").lower()  # cuda o cpu
LLM_MODEL_QA      = os.getenv("LLM_MODEL_QA", "thewindmom/llama3-med42-8b:latest")
LLM_BASE_URL      = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT       = int(os.getenv("LLM_TIMEOUT", "1200"))

CACHE_DIR = Path.cwd() / "model_cache"
CACHE_DIR.mkdir(exist_ok=True)

@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbedding:
    """Embedder singleton (seguro de compartir en el proceso)."""
    log.info("🔧 [ml_providers] Loading embedder %s on %s ...", EMBEDDING_MODEL, EMBEDDER_DEVICE.upper())
    
    # 🎮 Configuración dinámica de dispositivo (cuda/cpu) desde .env
    # CUDA: ~500MB GPU, 10-20x más rápido en ingesta, 5x en consultas
    # CPU: 0MB GPU, suficiente para consultas individuales
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_folder=str(CACHE_DIR),
        embed_batch_size=EMBED_BATCH_SIZE,
        device=EMBEDDER_DEVICE,  # ✅ Configurable vía .env (EMBEDDER_DEVICE)
    )

def get_llm(model: Optional[str] = None) -> Ollama:
    """Instancia LLM por petición (evita estado global)."""
    return Ollama(
        model=model or LLM_MODEL_QA,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        request_timeout=LLM_TIMEOUT,
    )
