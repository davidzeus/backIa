import logging
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.db.redis import RedisDb

from app.schemas.esquema import ConsultaQdrantRequest
from app.tools.rag_tool import search_clinical_history
from app.tools.planner_tool import get_query_plan_tools
from app.utils.context_manager import (
    set_patient_context, clear_patient_context,
    get_agent_sources, set_agent_sources, clear_agent_sources
)

log = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE INFRAESTRUCTURA ---
AGENT_MODEL_NAME = os.getenv("LLM_MODEL_AGENT", "medgemma:4b")
AGENT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://10.10.0.48:6379/0")

def get_storage() -> RedisDb:
    return RedisDb(session_table="sessions", db_url=REDIS_URL)

# --- LÓGICA DEL AGENTE ADAPTATIVO ---

def get_clinical_agent(
    session_id: str,
    user_id: str,
    context_data: str,
    planner_info: str,
    intent: Dict[str, bool] # Determina el modo operativo
) -> Agent:
    
    is_json = intent.get("is_json", False)
    is_technical = intent.get("is_technical", False)

    # 1. STOP SEQUENCES DINÁMICOS
    # Solo cortamos la generación en "}" si esperamos datos estructurados
    stop_seq = ["<end_of_turn>", "User:", "Observation:", "###"]
    if is_json:
        stop_seq += ["}]", "}\n"]

    # 2. PROMPT ADAPTATIVO (ANCLAJE LINGÜÍSTICO)
    # Evita que MedGemma derive al inglés en razonamientos complejos
    instructions = [
        "<start_of_turn>user",
        "Eres SAMI, un asistente médico de alta precisión.",
        "Tu tarea es procesar la === HISTORIA CLÍNICA === y responder con veracidad.",
        "**ESTABILIDAD LINGÜÍSTICA:** Responde siempre y únicamente en español.",
    ]

    # Mapeo genérico de instrucciones según intención
    if is_json:
        instructions.append("### MODO EXTRACCIÓN: Devuelve SOLO el objeto JSON solicitado sin preámbulos ni saludos.")
    elif is_technical:
        instructions.append("### MODO NARRATIVO TÉCNICO: Describe el evento con máximo detalle profesional. No generalices.")
    else:
        instructions.append("### MODO CONSULTA: Responde de forma directa, profesional y concisa.")
    
    instructions += ["<end_of_turn>", "<start_of_turn>model"]

    return Agent(
        name="SAMI",
        model=Ollama(
            id=AGENT_MODEL_NAME,
            host=AGENT_BASE_URL,
            options={
                # 0.0 para datos exactos, 0.2 para descripciones técnicas, 0.4 para charla médica fluida
                "temperature": 0.0 if is_json else (0.2 if is_technical else 0.3),
                "num_ctx": 8192,
                "num_predict": 1024, # Evita la saturación de ventana local de 1024 tokens
                "top_p": 0.90, 
                # PENALIZACIÓN QUIRÚRGICA: 1.1 para proteger unidades críticas como 'mg' o 'TA'
                "repeat_penalty": 1.1,
                "frequency_penalty": 0.5,
                "stop": stop_seq,
                "mirostat": 0,
            }
        ),
        db=get_storage(),
        session_id=session_id,
        user_id=user_id,
        instructions=instructions,
        additional_context=f"{planner_info}\n\n=== HISTORIA CLÍNICA ===\n{context_data}",
        markdown=True,
    )

# --- WATCHDOG SEMÁNTICO Y PARSERS ---

def _apply_semantic_watchdog(text: str, is_json: bool) -> str:
    """Detecta y colapsa duplicados en JSON usando el umbral 0.95"""
    if not is_json: return text
    try:
        start, end = text.find('{'), text.rfind('}') + 1
        if start == -1: return text
        data = json.loads(text[start:end])
        for key in ["medications", "medicamentos", "diagnoses", "diagnosticos", "allergies"]:
            if key in data and isinstance(data[key], list):
                unique = {}
                for item in data[key]:
                    if isinstance(item, dict) and "name" in item:
                        # Deduplicación semántica por nombre (ignora mayúsculas/espacios)
                        k = str(item["name"]).upper().strip().replace(".", "")
                        if k not in unique or item.get("date", "") > unique[k].get("date", ""):
                            unique[k] = item
                data[key] = list(unique.values())[:5] # Limita a 5 elementos por array
        return json.dumps(data, indent=2, ensure_ascii=False)
    except: return text

def parse_medgemma_output(full_text: str) -> tuple[str, str]:
    """Separa el razonamiento de la respuesta final."""
    reasoning, answer = "", full_text
    if "**ANÁLISIS CLÍNICO:**" in full_text and "**RESPUESTA FINAL:**" in full_text:
        parts = full_text.split("**RESPUESTA FINAL:**")
        reasoning = parts[0].replace("**ANÁLISIS CLÍNICO:**", "").strip()
        answer = parts[1].strip()
    return reasoning, answer

# --- FUNCIONES DE EJECUCIÓN (LISTAS PARA EL ROUTER) ---

async def run_agent_consult(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    start_time = time.perf_counter()
    context = await _prepare_agent_execution(request)
    try:
        response = context["agent"].run(context["user_msg"])
        raw_ans = response.content if hasattr(response, 'content') else str(response)
        
        reasoning, answer = parse_medgemma_output(raw_ans)
        final_ans = _apply_semantic_watchdog(answer, context["is_json"])
        
        return {
            "respuesta": final_ans,
            "reasoning": reasoning,
            "sources": get_agent_sources(),
            "session_id": request.session_id,
            "meta": {"latency": round(time.perf_counter() - start_time, 2), "model": AGENT_MODEL_NAME}
        }
    except Exception as e:
        log.error(f"Error en consulta: {e}")
        return {"respuesta": f"Error: {str(e)}", "sources": []}
    finally:
        clear_patient_context(context["token_pid"])
        clear_agent_sources(context["token_sources"])

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

# --- PREPARACIÓN DE CONTEXTO ---

async def _prepare_agent_execution(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    token_pid = set_patient_context(request.paciente_id)
    token_sources = set_agent_sources([])
    
    # DETECTOR GENÉRICO DE INTENCIÓN
    q_up = request.pregunta.upper()
    intent = {
        "is_json": any(x in q_up for x in ["JSON", "TRIAGE", "ESQUEMA"]),
        "is_technical": any(x in q_up for x in ["PROCEDIMIENTO", "CIRUGÍA", "TÉCNICA", "DETALLE", "EVOLUCIÓN"])
    }
    
    plan = get_query_plan_tools(request.pregunta)
    # Aumenta el límite de documentos si es resumen o análisis técnico profundo
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