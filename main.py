# main.py
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from llama_index.core import Settings


# ── 1) .env primero ──────────────────────────────────────────────
load_dotenv(override=True)

# ── 2) logging global (usa LOG_LEVEL, HCI_DEBUG, DISABLE_ACCESS_LOG) ─
from app.config.logging_setup import setup_logging
setup_logging()

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Imports locales para evitar ciclos y cargar solo al inicio ---
    from app.infra.ml_providers import get_embedder
    from app.tools.rag_tool import get_reranker_local  # <--- NUEVO IMPORT
    
    # 1. Embedder global (seguro de compartir).
    Settings.embed_model = get_embedder()
    log.debug("✅ Embedder inicializado globalmente.")

    # 2. Reranker Warm-up (Pre-carga en RAM) <--- NUEVA LÓGICA
    log.info("🚀 [WARM-UP] Iniciando carga del Reranker (IA)...")
    try:
        # Al llamar a esta función, si la variable es None, carga el modelo.
        # Si ya existe, no hace nada. Esto "calienta" el motor.
        get_reranker_local() 
        log.info("✅ [WARM-UP] Reranker cargado y listo para inferencia.")
    except Exception as e:
        log.error(f"❌ Error crítico cargando Reranker: {e}")

    yield  # ← La app corre aquí
    
    log.debug("🧹 Lifespan finalizado correctamente.")


# ── App FastAPI ──────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Rutas ────────────────────────────────────────────────────────
from app.routes.hc_router import router as hc_router  
from app.routes.vision_router import router as vision_router

app.include_router(hc_router)

app.include_router(vision_router)


# ── Ejecutar app (solo dev) ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(
    "main:app",
    host=os.getenv("HOST","127.0.0.1"),
    port=int(os.getenv("PORT","8000")),
    reload=os.getenv("RELOAD","0") == "1",        # prod: 0
    log_level=os.getenv("LOG_LEVEL","error").lower(),  # default: error
    access_log=os.getenv("DISABLE_ACCESS_LOG","1") != "1",  # default: False
    log_config=None,  # ← evita que Uvicorn re-escriba logging
)
