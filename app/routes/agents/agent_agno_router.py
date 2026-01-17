# app/routes/agent_agno_router.py
from fastapi import APIRouter
from .appointment_router import router as appointment_router
from .patrimonio_router import router as patrimonio_router

# Crear un router principal con el prefijo /api/agente
router = APIRouter(prefix="/api/agent", tags=["Agentes IA"])

# Esto hace que las rutas estén disponibles bajo /api/agente/...
router.include_router(appointment_router, prefix="/appointment")
router.include_router(patrimonio_router, prefix="/patrimony")
