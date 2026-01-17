# app/routers/chat_router.py
from fastapi import APIRouter, HTTPException, Request, Depends
from sse_starlette.sse import EventSourceResponse
from typing import Optional
import json
import logging

from ...agents.agno.core.config import Config
from ...agents.agno.services import ChatService

# Importar las herramientas que necesita el agente 
from  ...agents.agno.appointments.tools import get_all_appointments

logger = logging.getLogger(__name__)
router = APIRouter()

# --- DEPENDENCIAS ---

# 1. Inyector para la Configuración (Singleton)
# Solo se carga una vez al inicio.
_app_config = Config("./app/agents/agno/appointments/config.yaml") # Cargar configuración al inicio del módulo

def get_config() -> Config:
    return _app_config

# 2. Inyector para el Servicio
def get_chat_service(config: Config = Depends(get_config)) -> ChatService:
    return ChatService(config=config)

# --- ENDPOINT ---

@router.get("/chat",summary="Chatear con Agente que tiene acceso a la API Appointment",)
async def chat(
    request: Request,
    message: str,
    userchat_id: Optional[str] = None,
    sessionchat_id: Optional[str] = None,
    chat_service: ChatService = Depends(get_chat_service) # Inyecta el Servicio
):
    """Endpoint GET para chat con memoria persistente."""
    # Query string toma los parametros de la vista [API].[V_Appointment] de HCJS."
    # userchat_id y sessionchat_id se utilizan cuando se habilita la memoria persistente.

    try:
        
        # Construir la string query a partir de los parametros
        all_params = dict(request.query_params)
        del all_params['message']
        if 'userchat_id' in all_params: del all_params['userchat_id']
        if 'sessionchat_id' in all_params: del all_params['sessionchat_id']

        filter_message = "* Parametros y valores para armar la string query que necesita get_patient_appointments: " + str(all_params) + "\n "

        # Lista de Herramientas que podrá usar el agente
        tools=[get_all_appointments]
        # El Servicio devuelve un diccionario con el contenido y el agente
        response_data = chat_service.run_agent(
            filter_message + message,
            user_id=userchat_id, 
            session_id=sessionchat_id,
            tool_list=tools
        )
        
        # Extraemos el contenido y el agente
        content = response_data["content"]
        agent_instance = response_data["agent"]
        
        # Devolvemos la respuesta con información de sesión
        return {
            "data": content,
            "session_info": {
                "session_id": agent_instance.session_id,
                "user_id": agent_instance.user_id
            }
        }
    
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/stream",summary="Chatear con Agente y trasmitir contenido en tiempo real",)
async def chat_stream(
    request: Request,
    message: str,
    userchat_id: Optional[str] = None,
    sessionchat_id: Optional[str] = None,
    chat_service: ChatService = Depends(get_chat_service) # Inyecta el Servicio
):
    """Endpoint GET para chat con streaming y memoria persistente."""
    # Query string toma los parametros de la vista [API].[V_Appointment] de HCJS."
    # userchat_id y sessionchat_id se utilizan cuando se habilita la memoria persistente.

    try:
        
        # Construir la string query a partir de los parametros
        all_params = dict(request.query_params)
        del all_params['message']
        if 'userchat_id' in all_params: del all_params['userchat_id']
        if 'sessionchat_id' in all_params: del all_params['sessionchat_id']

        filter_message = "* Parametros y valores para armar la string query que necesita get_patient_appointments: " + str(all_params) + "\n "

        # El Servicio te devuelve un generador que contiene el agente (para IDs) 
        # y los chunks de respuesta.
        agent_generator = chat_service.run_agent_stream(
            filter_message + message,
            user_id=userchat_id, 
            session_id=sessionchat_id
        )
        
        #Debug
        # Obtenemos el objeto Agent para extraer sus IDs
        #agent_instance = await anext(agent_generator) 
        
        async def generate_response():
            async for content in agent_generator:
                # yield f"data: {content}\n\n"
                data = {"data": content,"is_complete": False}
                yield {"event": "message", "data":json.dumps(data)}
                # yield {"event": "message","data": json.dumps(data) }
            
            #Debug
            # Información de sesión al final
            # session_info = {
            #     "session_id": agent_instance.session_id,
            #     "user_id": agent_instance.user_id
            # }
            
            #yield { "event": "session_info","data": {"data": json.dumps(session_info)} }
            yield { "event": "end", "data":json.dumps({"data":"","is_complete": True})}
            
            
        return EventSourceResponse(generate_response())
    
    except Exception as e:
        logger.error(f"Error en chat stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))