# app/routers/chat_router.py
from fastapi import APIRouter, HTTPException, Request, Depends
from sse_starlette.sse import EventSourceResponse
from typing import Optional
import json
import logging

from ...agents.agno.core.config import Config
from ...agents.agno.services import ChatService

# Importar las herramientas que necesita el agente 
from  ...agents.agno.patrimonio.tools import sql_tool, consultar_patrimonio

logger = logging.getLogger(__name__)
router = APIRouter()

# --- DEPENDENCIAS ---

# 1. Inyector para la Configuración (Singleton)
# Solo se carga una vez al inicio.
_app_config = Config("./app/agents/agno/patrimonio/config.yaml") # Cargar configuración al inicio del módulo

def get_config() -> Config:
    return _app_config

# 2. Inyector para el Servicio
def get_chat_service(config: Config = Depends(get_config)) -> ChatService:
    return ChatService(config=config)

# --- ENDPOINT ---

@router.get("/chat",summary="Chatear con Agente que tiene acceso a la BD Patrimonio",)
async def chat(
    request: Request,
    message: str,
    userchat_id: Optional[str] = None,
    sessionchat_id: Optional[str] = None,
    chat_service: ChatService = Depends(get_chat_service) # Inyecta el Servicio
):
    """Endpoint GET para chat con memoria persistente."""
    try:
        # Construir la string query a partir de los parametros
        all_params = dict(request.query_params)
        del all_params['message']
        if 'userchat_id' in all_params: del all_params['userchat_id']
        if 'sessionchat_id' in all_params: del all_params['sessionchat_id']

        # Construir la string query a partir de los parametros si all_params no está vacío
        if all_params:
            filter_message = "* Campos para agregar al where: " + str(all_params) + "\n "
        else:
            filter_message = ""
            
        # Lista de Herramientas que podrá usar el agente
        tools=[sql_tool()] # accede a base sql

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

    try:
        # Lista de Herramientas que podrá usar el agente
        tools=[consultar_patrimonio]

        all_params = dict(request.query_params)
        del all_params['message']
        if 'userchat_id' in all_params: del all_params['userchat_id']
        if 'sessionchat_id' in all_params: del all_params['sessionchat_id']

        filter_message = "* Parametros y valores para armar la string query que necesita consultar_patrimonio: " + str(all_params) + "\n "

        # El Servicio te devuelve un generador que contiene el agente (para IDs) 
        # y los chunks de respuesta.
        agent_generator = chat_service.run_agent_stream(
            filter_message + message,
            user_id=userchat_id, 
            session_id=sessionchat_id,
            tool_list=tools
        )
               
        async def generate_response():
            async for content in agent_generator:
                data = {"data": content}
                yield {"event": "message", "data":json.dumps(data)}

            yield { "event": "end", "data":json.dumps({"data":""})}
            
        return EventSourceResponse(generate_response())
    
    except Exception as e:
        logger.error(f"Error en chat stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chatWithApi",summary="Chatear con Agente que tiene acceso a la BD Patrimonio",)
async def chat(
    request: Request,
    message: str,
    userchat_id: Optional[str] = None,
    sessionchat_id: Optional[str] = None,
    chat_service: ChatService = Depends(get_chat_service) # Inyecta el Servicio
):
    """Endpoint GET para chat con memoria persistente."""
    try:
        # Construir la string query a partir de los parametros
        all_params = dict(request.query_params)
        del all_params['message']
        if 'userchat_id' in all_params: del all_params['userchat_id']
        if 'sessionchat_id' in all_params: del all_params['sessionchat_id']

        # Construir la string query a partir de los parametros si all_params no está vacío
        if all_params:
            filter_message = "* Parametros y valores para armar la string query que necesita consultar_patrimonio: " + str(all_params) + "\n "
            
        # Lista de Herramientas que podrá usar el agente
        #tools=[sql_tool()] # accede a base sql
        tools=[consultar_patrimonio]
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