# app/core/factory.py
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.db.redis import RedisDb
from textwrap import dedent
from .config import Config
from sqlalchemy import create_engine
from agno.tools.sql import SQLTools
import os

AGENT_MODEL_HOST = os.getenv("LLM_BASE_URL", "localhost")
REDIS_URL = os.getenv("REDIS_URL")

import logging
logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory para crear el agente con la configuración"""
    
    @staticmethod
    def create_agent(config: Config, tool_list: list = None , user_id: str = None, session_id: str = None) -> Agent:
        """Crear agente basado en la configuración"""
        try:

            # Configurar modelo
            model_id = config.get('model.id')
            # model_host = config.get('model.host')
            model_temperature = config.get('model.temperature')
            model_timeout = config.get('model.request_timeout')

            model = Ollama(id=model_id,
                            host=AGENT_MODEL_HOST,
                            options={
                                "temperature": model_temperature,
                                "request_timeout": model_timeout
                            }
                            )

            # Configurar agente
            agent_config = config.get('agent')

            # Configurar Redis para memoria
            redis_url = REDIS_URL
            if not redis_url:
                # raise ValueError("URL de Redis no configurada")
                redis_db = None
                add_history_to_context=False
                enable_user_memories=False
                enable_session_summaries=False
            else:
                redis_db = RedisDb(db_url=redis_url)
                add_history_to_context=agent_config.get('add_history_to_context', False),
                enable_user_memories=agent_config.get('enable_user_memories', False),
                enable_session_summaries=agent_config.get('enable_session_summaries', False),
            
 
            agent = Agent(
                id=agent_config.get('id', 'api-agent'),
                name=agent_config.get('name', 'API Agent'),
                user_id=user_id or "default_user",
                session_id=session_id,
                model=model,
                reasoning=agent_config.get('reasoning', False),
                tools=tool_list,
                db=redis_db,  # Usar Redis para persistencia
                description=agent_config.get('description', ''),
                instructions=config.get('instructions', []),
                additional_context=dedent(config.get('additional_context', '')),
                debug_mode=agent_config.get('debug_mode', False),
                stream=agent_config.get('stream', True),
                stream_intermediate_steps=agent_config.get('stream_intermediate_steps', True),
                add_history_to_context=add_history_to_context,
                enable_user_memories=enable_user_memories,
                enable_session_summaries=enable_session_summaries,
                add_datetime_to_context=agent_config.get('add_datetime_to_context', False),
                markdown=agent_config.get('markdown', False)
            )
            
            logger.info(f"Agente creado exitosamente con ID: {agent.id}")
            return agent
            
        except Exception as e:
            logger.error(f"Error creando agente: {e}")
            raise