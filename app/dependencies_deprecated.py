"""
Dependencias compartidas (LLM y Embedder) para FastAPI
──────────────────────────────────────────────────────
• get_embedder()  → singleton con LRU‑cache
• get_llm()       → instancia nueva por request
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# ─────────────────────────────────────────────────────────────
# Configuración vía variables de entorno (.env)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL_QA = os.getenv("LLM_MODEL_QA")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
CACHE_DIR = Path.cwd() / "model_cache"
CACHE_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embedder():
    """Embedder singleton (seguro para compartir)."""
    log.info("🔧  (dependencies) Loading embedder %s …", EMBEDDING_MODEL)
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_folder=str(CACHE_DIR),
        embed_batch_size=16,
    )


def get_llm():
    """Crea una instancia Ollama por petición (request‑scoped)."""
    return Ollama(
        model=LLM_MODEL_QA,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        request_timeout=1200,
    )
