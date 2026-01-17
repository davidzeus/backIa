# app/agents/services.py
from typing import AsyncGenerator
from agno.agent import Agent
from .core.factory import AgentFactory
from .core.config import Config

import logging

logger = logging.getLogger(__name__)

class ChatService:
    """
    Servicio de Chat. Contiene la lógica de negocio para interactuar 
    con el Agente de Lenguaje Grande.
    """
    def __init__(self, config: Config):
        self._config = config
    
    def run_agent_stream(
        self, 
        message: str, 
        user_id: str = None, 
        session_id: str = None,
        tool_list: list = None,
    ) -> AsyncGenerator[Agent, None]:
        
        # # Lista de Herramientas que podrá usar el agente
        # tool_list=[get_all_appointments]
        
        # 1. Orquestación: Utiliza la Factory para obtener una instancia de Agent
        agent = AgentFactory.create_agent(
            self._config,
            tool_list=tool_list, 
            user_id=user_id, 
            session_id=session_id
        )

        def should_skip_content(content, inside_think_tags):
            if '<think>' in content:
                inside_think_tags[0] = True
            if '</think>' in content:
                inside_think_tags[0] = False
                return True
            return inside_think_tags[0]
        
        # 2. Ejecuta la lógica central y retorna el generador
        # La función de envoltura async sirve para manejar el generator de agent.run()
        async def agent_response_generator():
            #Debug
            #yield agent # Devolvemos el agente primero para obtener su ID de sesión

            use_streaming = self._config.get('agent').get('stream', True)
            hide_thinking = self._config.get('hidethinking', False)
            inside_think_tags = [False]  # Lista para permitir modificación en función auxiliar

            response = agent.run(message)
            if use_streaming:
                for chunk in response:
                    content = getattr(chunk, 'content', '')
                    eventtype = getattr(chunk, 'event', '')
                    if content and eventtype == 'RunContent':
                        if hide_thinking:
                            if should_skip_content(content, inside_think_tags): continue
                        yield content
            else:
                content = response.content
                if hide_thinking:
                    # Filtrar contenido think en modo no-streaming
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                    content = content.strip()
                yield content
        
        return agent_response_generator()
    
    def run_agent(
        self, 
        message: str, 
        user_id: str = None, 
        session_id: str = None,
        tool_list = None
    ):
        # Lista de Herramientas que podrá usar el agente
        # tools=[get_all_appointments]
        
        # 1. Orquestación: Utiliza la Factory para obtener una instancia de Agent
        agent = AgentFactory.create_agent(
            self._config,
            tool_list=tool_list, 
            user_id=user_id, 
            session_id=session_id
        )

        # Ejecutar agente
        response = agent.run(message)
        
        # Procesar la respuesta del generador para obtener el contenido completo
        content = ""
        hide_thinking = self._config.get('hidethinking', False)
        
        # Si es un generador, iteramos sobre él para obtener todo el contenido
        if hasattr(response, '__iter__') and not hasattr(response, 'content'):
            for chunk in response:
                chunk_content = getattr(chunk, 'content', '')
                if chunk_content:
                    content += chunk_content
        else:
            # Si no es un generador y tiene atributo content, lo usamos directamente
            content = getattr(response, 'content', str(response))
        
        # Filtrar contenido think si está configurado
        if hide_thinking:
            # Filtrar contenido think
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            # Limpiar líneas vacías adicionales que puedan quedar
            content = content.strip()
        
        # Devolver el contenido de la respuesta y el agente para acceder a sus IDs
        return {
            "content": content,
            "agent": agent
        }
