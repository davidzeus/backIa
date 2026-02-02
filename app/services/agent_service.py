import logging
import os
import time
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.db.redis import RedisDb

from app.schemas.esquema import ConsultaQdrantRequest
from app.utils.context_manager import (
    set_patient_context, clear_patient_context,
    get_agent_sources, set_agent_sources, clear_agent_sources
)
from app.tools.rag_tool import search_clinical_history
from app.tools.planner_tool import get_query_plan_tools

log = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE INFRAESTRUCTURA ---
AGENT_MODEL_NAME = os.getenv("LLM_MODEL_AGENT", "medgemma:4b")
AGENT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://10.10.0.48:6379/0")

def get_storage() -> RedisDb:
    return RedisDb(session_table="sessions", db_url=REDIS_URL)

# --- LÓGICA DEL AGENTE (CON RAZONAMIENTO NATIVO) ---

def get_clinical_agent(
    session_id: str,
    user_id: str,
    context_data: str,
    planner_info: str,
    intent: Dict[str, bool] 
) -> Agent:
    
    is_json = intent.get("is_json", False)
    is_technical = intent.get("is_technical", False)

    # 1. STOP SEQUENCES
    stop_seq = ["<end_of_turn>", "User:", "Observation:", "###"]
    if is_json:
        stop_seq += ["}\n\n", "```\n"]

    # 2. PROMPT BASE (LIMPIO PARA AGNO REASONING)
    # Nota: No le pedimos "Analiza paso a paso" manualmente, dejamos que Agno lo haga.
    instructions = [
        "<start_of_turn>user",
        "Eres SAMI, un Auditor Médico IA basado en evidencia.",
        "Tu tarea es procesar la === HISTORIA CLÍNICA === y responder con veracidad.",
        "Responde siempre en español.",
    ]

    if is_json:
        instructions.append("### MODO EXTRACCIÓN: Devuelve ÚNICAMENTE un objeto JSON válido.")
        instructions.append("No incluyas markdown (```json) ni texto introductorio.")
    elif is_technical:
        instructions.append("### MODO TÉCNICO: Describe el evento con detalle profesional.")
        instructions.append("CITA tus fuentes usando [DOC_ID: X].")
    else:
        instructions.append("### MODO CONSULTA: Responde de forma directa.")
        instructions.append("CITA tus fuentes usando [DOC_ID: X].")
    
    instructions += ["<end_of_turn>", "<start_of_turn>model"]

    return Agent(
        name="SAMI",
        model=Ollama(
            id=AGENT_MODEL_NAME,
            host=AGENT_BASE_URL,
            options={
                "temperature": 0.0 if is_json else 0.3,
                "num_ctx": 8192,
                "num_predict": 4096, # Ventana amplia para que quepa el <think> y la respuesta
                "top_p": 0.90, 
                "repeat_penalty": 1.1,
                "stop": stop_seq,
                "mirostat": 0,
            }
        ),
        db=get_storage(),
        session_id=session_id,
        user_id=user_id,
        instructions=instructions,
        additional_context=f"{planner_info}\n\n=== HISTORIA CLÍNICA ===\n{context_data}",
        
        # ✅ AHORA SÍ: ACTIVADO
        reasoning=True, 
        
        markdown=True,
    )

# --- PARSERS INTELIGENTES ---

def parse_medgemma_output(full_text: str) -> tuple[str, str]:
    """
    Separa el razonamiento nativo (<think>) de la respuesta.
    """
    reasoning = ""
    answer = full_text.strip()

    # 1. Soporte para etiquetas de razonamiento (DeepSeek/Agno style)
    if "<think>" in full_text:
        match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            answer = full_text.split("</think>")[-1].strip()
    
    # 2. Limpieza de Markdown JSON
    if "```json" in answer:
        answer = answer.replace("```json", "").replace("```", "").strip()
    
    return reasoning, answer

def _apply_semantic_watchdog(text: str, is_json: bool) -> str:
    if not is_json: return text
    try:
        # Busca el JSON más externo
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            data = json.loads(json_str)
            # Deduplicación simple
            for key in ["medications", "medicamentos", "diagnoses", "diagnosticos", "allergies"]:
                if key in data and isinstance(data[key], list):
                    unique = {str(i.get("name","")).upper(): i for i in data[key] if isinstance(i, dict)}
                    data[key] = list(unique.values())[:5]
            return json.dumps(data, indent=2, ensure_ascii=False)
    except: 
        pass
    return text

# --- FUNCIONES DE EJECUCIÓN ---

async def run_agent_consult(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    start_time = time.perf_counter()
    context = await _prepare_agent_execution(request)
    try:
        response = context["agent"].run(context["user_msg"])
        raw_ans = response.content if hasattr(response, 'content') else str(response)
        
        reasoning, answer = parse_medgemma_output(raw_ans)
        final_ans = _apply_semantic_watchdog(answer, context["is_json"])
        
        confidence = "ALTA"
        if not context["is_json"] and "[DOC_ID:" not in final_ans:
            confidence = "MEDIA (Falta Cita)"
        
        return {
            "respuesta": final_ans,
            "reasoning": reasoning, 
            "sources": get_agent_sources(),
            "confidence": confidence,
            "session_id": request.session_id,
            "meta": {"latency": round(time.perf_counter() - start_time, 2), "model": AGENT_MODEL_NAME}
        }
    except Exception as e:
        log.error(f"Error Agente: {e}", exc_info=True)
        return {"respuesta": f"Error: {str(e)}", "sources": []}
    finally:
        clear_patient_context(context["token_pid"])
        clear_agent_sources(context["token_sources"])

# ✅ FUNCIÓN STREAM AGREGADA 
async def run_agent_consult_stream(request: ConsultaQdrantRequest):
    context = await _prepare_agent_execution(request)
    buffer = ""
    try:
        async for chunk in context["agent"].arun(context["user_msg"], stream=True):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            buffer += content
            yield {"type": "token", "text": content}
        
        reasoning, answer = parse_medgemma_output(buffer)
        final = _apply_semantic_watchdog(answer, context["is_json"])
        yield {"type": "end", "final": {"respuesta": final, "reasoning": reasoning, "sources": get_agent_sources()}}
    finally:
        clear_patient_context(context["token_pid"])
        clear_agent_sources(context["token_sources"])

# --- PREPARACIÓN ---

async def _prepare_agent_execution(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    token_pid = set_patient_context(request.paciente_id)
    token_sources = set_agent_sources([])
    
    q_up = request.pregunta.upper()
    intent = {
        "is_json": any(x in q_up for x in ["JSON", "TRIAGE", "ESQUEMA"]),
        "is_technical": any(x in q_up for x in ["PROCEDIMIENTO", "CIRUGÍA", "TÉCNICA"])
    }
    
    plan = get_query_plan_tools(request.pregunta)
    doc_limit = 30 if (plan.get("is_summary") or intent["is_technical"]) else 15
    raw_results = search_clinical_history(request.pregunta, qdrant_filters=plan, limit=doc_limit)
    
    agent = get_clinical_agent(
        f"{request.session_id}::{request.paciente_id}",
        request.user_id, raw_results, _format_plan_msg(plan), intent
    )
    return {"agent": agent, "user_msg": request.pregunta, "token_pid": token_pid, "token_sources": token_sources, "is_json": intent["is_json"]}

def _format_plan_msg(plan: Dict[str, Any]) -> str:
    tw = plan.get("time_window", {}) or {}
    if tw.get("start") and tw.get("end"):
        return f"--- FOCO TEMPORAL: {datetime.fromtimestamp(tw['start']).strftime('%d-%m-%Y')} al {datetime.fromtimestamp(tw['end']).strftime('%d-%m-%Y')} ---\n"
    return ""

def delete_agent_session(session_id: str) -> int:
    try:
        get_storage().delete_session(session_id)
        return 1
    except: return -1
