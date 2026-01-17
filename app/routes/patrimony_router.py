import os
import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
import logging

from typing import Optional

OLLAMA_HOST = os.getenv("LLM_BASE_URL", "http://localhost:11434")
MODELO_CEREBRO_IA = os.getenv("PAT_LLM_MODEL", "ministral-3:3b-instruct-2512-q8_0")

from app.services.patrimony_service import (
    GestorConsultasIA, 
    GestorHistorialRedis
)

logger = logging.getLogger(__name__)

# Crear un router principal con el prefijo /api/patrimony
router = APIRouter(prefix="/api/patrimony", tags=["Patrimonio"])

# API con FastAPI ---
@router.get("/chat-json", summary="Genera y ejecuta una consulta SQL")
async def consultar(
    message: str, 
    tabla_principal: str = "v_inventario",
    id_usuario: Optional[str] = None,
    id_session: Optional[str] = None,
    usar_historial: bool = False
):
    """
    Recibe una pregunta en lenguaje natural y devuelve un JSON con la respuesta.
    
    Parámetros:
    - message: La consulta en lenguaje natural
    - tabla_principal: Tabla a consultar (default: v_inventario)
    - id_usuario: ID del usuario (requerido si usar_historial=True)
    - id_session: ID de la sesión (requerido si usar_historial=True)
    - usar_historial: Si se debe usar el historial de Redis (default: False)
    """
    if usar_historial and (not id_usuario or not id_session):
        raise HTTPException(
            status_code=400, 
            detail="Se requieren id_usuario e id_session cuando usar_historial=True"
        )

    cerebro_ia = GestorConsultasIA(MODELO_CEREBRO_IA, OLLAMA_HOST)

    stream = cerebro_ia.traducir_pregunta_a_sql_stream(
        False, 
        message, 
        tabla_principal,
        id_usuario,
        id_session,
        usar_historial
    )

    respuesta = cerebro_ia.generate_response_no_stream(stream, id_usuario, id_session, usar_historial)

    return {"data": respuesta}


@router.get("/chat", summary="Genera una consulta, la ejecuta y la IA interpreta el resultado")
async def consultar_interpretado(
    message: str, 
    tabla_principal: str = "v_inventario",
    id_usuario: Optional[str] = None,
    id_session: Optional[str] = None,
    usar_historial: bool = False,
    stream: bool = False  
):
    """
    Recibe una pregunta, la traduce a SQL, ejecuta la consulta y devuelve una interpretación en lenguaje natural.
    Si stream=True, la respuesta se devuelve en tiempo real.
    """
    if usar_historial and (not id_usuario or not id_session):
        raise HTTPException(
            status_code=400, 
            detail="Se requieren id_usuario e id_session cuando usar_historial=True"
        )

    cerebro_ia = GestorConsultasIA(MODELO_CEREBRO_IA, OLLAMA_HOST)

    # 1. Generar la consulta SQL
    stream_sql = cerebro_ia.traducir_pregunta_a_sql_stream(
        False, 
        message, 
        tabla_principal,
        id_usuario,
        id_session,
        usar_historial
    )

    # 2. Ejecutar la consulta y obtener el resultado JSON
    try:
        resultado_json = cerebro_ia.generate_response_no_stream(stream_sql, id_usuario, id_session, usar_historial)
    except HTTPException as e:
        return {"error": e.detail, "status_code": e.status_code}
    except RuntimeError as e:
        return {"error": str(e), "status_code": 500}

    # 3. Interpretar el resultado con la IA
    historial_para_interpretar = []
    if usar_historial and id_usuario and id_session:
        historial_para_interpretar = cerebro_ia.gestor_historial.obtener_historial(id_usuario, id_session)

    try:
        response = cerebro_ia.interpretar_resultado_con_ia(
            pregunta_usuario=message,
            resultado_json=json.dumps(resultado_json, indent=2),
            historial=historial_para_interpretar,
            stream=stream
        )

        if stream:
            async def stream_generator():
                full_response = []
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        token = chunk['message']['content']
                        full_response.append(token)
                        yield json.dumps({"data": token}) + "\n"
                
                # Guardar la respuesta completa en el historial
                if usar_historial and id_usuario and id_session:
                    cerebro_ia.gestor_historial.guardar_mensaje(id_usuario, id_session, 'assistant', "".join(full_response))

            return EventSourceResponse(stream_generator())
        else:
            respuesta_natural = response['message']['content']
            # Guardar la respuesta final en el historial
            if usar_historial and id_usuario and id_session:
                cerebro_ia.gestor_historial.guardar_mensaje(id_usuario, id_session, 'assistant', respuesta_natural)

            return {"data": respuesta_natural}

    except RuntimeError as e:
        return {"error": str(e), "status_code": 500}

@router.get("/ver-historial", summary="Obtiene el historial de una sesión")
async def ver_historial(id_usuario: str, id_session: str, limite: int = 10):
    """
    Obtiene el historial de conversación de una sesión.
    """
    gestor = GestorHistorialRedis()
    historial = gestor.obtener_historial(id_usuario, id_session, limite)
    return {"historial": historial}

@router.delete("/limpiar-historial", summary="Limpia el historial de una sesión")
async def limpiar_historial(id_usuario: str, id_session: str):
    """
    Limpia el historial de conversación de una sesión específica.
    """
    gestor = GestorHistorialRedis()
    if gestor.limpiar_historial(id_usuario, id_session):
        return {"message": "Historial limpiado exitosamente"}
    else:
        raise HTTPException(status_code=500, detail="Error al limpiar el historial")
