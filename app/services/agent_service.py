# app/services/agent_service.py
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.db.redis import RedisDb

from app.schemas.esquema import ConsultaQdrantRequest
# ✅ Herramienta Manual probada
from app.tools.rag_tool import search_clinical_history
from app.tools.planner_tool import get_query_plan_tools
from app.utils.context_manager import (
    set_patient_context, clear_patient_context,
    get_agent_sources, set_agent_sources, clear_agent_sources
)

# --- CONFIGURACIÓN DE LOGGING ---
log = logging.getLogger(__name__)
_DEBUG_FLAG = os.getenv("HCI_DEBUG", "0")
if not bool(int(str(_DEBUG_FLAG))):
    log.setLevel(logging.WARNING)

# --- CONFIGURACIÓN DE AGENTE ---
# Usamos MedGemma como motor principal
AGENT_MODEL_NAME = os.getenv("LLM_MODEL_AGENT", "medgemma:4b")

AGENT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://10.10.0.48:6379/0")

def get_storage() -> RedisDb:
    return RedisDb(session_table="sessions", db_url=REDIS_URL)

# --- DEFINICIÓN DEL AGENTE (CON ESTRATEGIA MEDRAG) ---
# app/services/agent_service.py (Solo la función get_clinical_agent actualizada)

def get_clinical_agent(
    session_id: str,
    user_id: str,
    context_data: str,
    planner_info: str,
    is_summary: bool
) -> Agent:

    # --- 1. PROMPT BASE ---
    base_instructions = [
        "<start_of_turn>user",
        "Eres SAMI, un asistente médico eficiente.",
        "Tu tarea es leer la === HISTORIA CLÍNICA === y extraer datos veraces.",
        "NO inventes información. Si no hay datos, indica 'Sin registros'.",
        "**IDIOMA:** RESPONDE SIEMPRE Y ÚNICAMENTE EN ESPAÑOL. Traduce si es necesario.",
        "**CONCISIÓN:** Evita listas repetitivas. Si hay muchos elementos, agrúpalos.",
    ]

    # --- 2. INSTRUCCIONES ESPECÍFICAS (ANTI-BUCLES) ---
    if is_summary:
        specific_instructions = [
            "MODO RESUMEN: Sintetiza la información.",
            "Extrae hechos que coincidan con la solicitud.",
            "Identifica relaciones Causa-Efecto.",
            "Si se solicitan medicamentos, extrae TODOS (genéricos y marcas comerciales).",
            "REGLA DE ORO: Solo usa fechas explícitas en los registros.",
            "Si el dato no está, responde: 'No hay registros disponibles.'",
        ]
    else:
        specific_instructions = [
            "### TAREA: CONSULTA PUNTUAL",
            "Responde de forma directa y concisa.",
            "Usa negritas para resaltar hallazgos clave (ej: **E. coli**)."
        ]
    
    # --- 3. FORMATO DE CIERRE ---
    formatting_instructions = [
        "",
        #"Piensa paso a paso en **ANÁLISIS CLÍNICO** y luego da la **RESPUESTA FINAL**.",
        "<end_of_turn>",
        "<start_of_turn>model"
    ]

    full_instructions = base_instructions + specific_instructions + formatting_instructions

    return Agent(
        name="SAMI",
        model=Ollama(
            id=AGENT_MODEL_NAME,
            host=AGENT_BASE_URL,
            options={
                "temperature": 0.1, # Subir de 0.0 a 0.1 introduce leve "ruido" que rompe bucles
                "num_ctx": 8192,
                "num_predict": 1024,   # 2048 Reducimos para evitar verborragia infinita
                "top_k": 40,
                "top_p": 0.95, # Aumentar un poco para variedad
                
                # --- 🔧 AJUSTE CLAVE ANTI-BUCLES ---
                "repeat_penalty": 1.15, # 1.5
                "stop": ["<end_of_turn>", "User:", "Observation:", "```\n\n", "}]", "}\n"],
                
                # --- EXPERIMENTAL: Mirostat para evitar colapso repetitivo ---
                "mirostat": 2,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1
            }
        ),
        db=get_storage(),
        session_id=session_id,
        user_id=user_id,
        add_name_to_context=True,
        reasoning=True,
        debug_mode=bool(int(str(_DEBUG_FLAG))),
        description="Asistente Médico de IA",
        instructions=full_instructions,
        additional_context=f"{planner_info}\n\n=== HISTORIA CLÍNICA ===\n{context_data}",
        expected_output="Respuesta clínica profesional.",
        markdown=True,
    )

def format_plan_info(plan: Dict[str, Any]) -> str:
    """Extrae la inteligencia del Planner para el Prompt."""
    info_parts = []
    tw = plan.get("time_window", {}) or {}
    start = tw.get("start")
    end = tw.get("end")
    if start and end:
        dt_start = datetime.fromtimestamp(start).strftime('%d-%m-%Y')
        dt_end = datetime.fromtimestamp(end).strftime('%d-%m-%Y')
        info_parts.append(f"- VENTANA DE TIEMPO: {dt_start} al {dt_end}")
    
    services = plan.get("service_hints", [])
    if services:
        nombres = [s.get('desc', '') for s in services]
        info_parts.append(f"- ENFOQUE EN SERVICIOS: {', '.join(nombres)}")
        
    return "--- INTENCIÓN DE BÚSQUEDA ---\n" + "\n".join(info_parts) + "\n------------------------\n" if info_parts else ""

def is_internal_error(text: str) -> bool:
    if not text: return False
    return "error interno" in text.lower() or "nonetype" in text.lower()

# --- PREPARACIÓN DEL CONTEXTO ---
async def _prepare_agent_execution(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    token_pid = set_patient_context(request.paciente_id)
    token_sources = set_agent_sources([])
    internal_session_id = f"{request.session_id}::{request.paciente_id}"

    try:
        # 1. PLANNER
        plan = get_query_plan_tools(request.pregunta)
        is_summary = plan.get("is_summary", False)
        # Aumentamos el límite para resumen para capturar toda la medicación
        doc_limit = 30 if is_summary else 15
        
        # 2. RAG RETRIEVAL (Tu herramienta potente)
        log.info(f"🔎 [RAG] Buscando para '{request.paciente_id}' (Docs: {doc_limit})...")
        raw_results = search_clinical_history(
            request.pregunta,
            qdrant_filters=plan,
            strict=plan.get("strict", False),
            limit=doc_limit
        )

        # Fallback inteligente
        if raw_results and "No se encontraron registros" in raw_results and plan.get("service_hints"):
             log.warning("⚠️ Fallback: Quitando filtro de servicio estricto...")
             plan_loose = plan.copy()
             del plan_loose["service_hints"]
             raw_results_b = search_clinical_history(request.pregunta, qdrant_filters=plan_loose, strict=False, limit=doc_limit)
             if raw_results_b and "No se encontraron registros" not in raw_results_b:
                 raw_results = raw_results_b

        evidence_text = raw_results if not is_internal_error(raw_results) else "Sin datos clínicos disponibles."
        planner_info_msg = format_plan_info(plan)

        # 3. AGENTE
        agent = get_clinical_agent(
            session_id=internal_session_id,
            user_id=request.user_id,
            context_data=evidence_text,
            planner_info=planner_info_msg,
            is_summary=is_summary
        )

        return {
            "agent": agent,
            "user_msg": request.pregunta,
            "token_pid": token_pid,
            "token_sources": token_sources,
            "session_id": request.session_id,
            "meta_base": {"mode": "manual_rag", "summary": is_summary}
        }

    except Exception:
        if 'token_pid' in locals(): clear_patient_context(token_pid)
        if 'token_sources' in locals(): clear_agent_sources(token_sources)
        raise

def _extract_list_from_text_deprecated(text: str, key: str) -> list[str]:
    """Extrae y deduplica una lista JSON simple usando regex robusto."""
    import re
    # Busca patrones tipo "key": [ ... ]
    pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, text, re.DOTALL)
    
    clean_items = []
    if match:
        content = match.group(1)
        # Extraer items entre comillas dobles
        raw_items = re.findall(r'"([^"]*)"', content)
        
        # Deduplicar preservando orden (case insensitive para mayor seguridad)
        seen = set()
        for item in raw_items:
            # Limpieza básica
            i = item.strip()
            if not i: continue
            
            i_lower = i.lower()
            if i_lower not in seen:
                seen.add(i_lower)
                clean_items.append(i)
                
    return clean_items

def _extract_list_from_text(text: str, key: str) -> list[str]:
    """
    Extrae listas, normaliza y aplica deduplicación inteligente.
    Ej: Si existe 'Salbutamol 100mg', elimina 'Salbutamol' suelto.
    """
    import re
    
    # 1. Extracción Cruda (Regex)
    pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return []

    content = match.group(1)
    raw_items = re.findall(r'"([^"]*)"', content)
    
    # 2. Normalización Agresiva
    normalized_map = {}
    for item in raw_items:
        # Quitar espacios, puntos finales y poner minúsculas para comparar
        clean = item.strip().strip(".").strip()
        if len(clean) < 2: continue # Ignorar basura de 1 letra
        
        # Guardamos la versión original asociada a la versión "key" para comparar
        normalized_map[clean] = item.strip() # Preferimos la versión sin espacios extra

    # 3. Deduplicación por Contenencia (Smart Dedupe)
    # Ordenamos por longitud (del más largo al más corto)
    # Así, "Salbutamol 100mg" se procesa antes que "Salbutamol"
    sorted_keys = sorted(normalized_map.keys(), key=len, reverse=True)
    
    final_list = []
    
    for candidate in sorted_keys:
        # Verificamos si este candidato ya está "contenido" en algo que ya guardamos
        # Ej: "Salbutamol" está en "Salbutamol 100mg" -> Lo descartamos
        is_redundant = False
        for existing in final_list:
            # Chequeamos si el candidato es un substring del existente
            if candidate.lower() in existing.lower():
                is_redundant = True
                break
        
        if not is_redundant:
            # Si no es redundante, guardamos el texto original (bonito)
            final_list.append(normalized_map[candidate])

    return final_list

# --- PARSER ROBUSTO (CLAVE PARA MEDGEMMA) ---
def parse_medgemma_output(full_text: str) -> tuple[str, str]:
    """
    Intenta extraer el análisis y la respuesta.
    Si detecta intentos de JSON, usa regex para reconstruir un objeto válido y único,
    ignorando la basura repetitiva que genere el modelo al final.
    """
    reasoning = ""
    answer = full_text

    # 1. Separar Reasoning de Respuesta
    if "**ANÁLISIS CLÍNICO:**" in full_text and "**RESPUESTA FINAL:**" in full_text:
        parts = full_text.split("**RESPUESTA FINAL:**")
        reasoning = parts[0].replace("**ANÁLISIS CLÍNICO:**", "").strip()
        answer = parts[1].strip()
    
    # 2. Estrategia de Recuperación de JSON
    # Si parece que hay un JSON (corchetes o llaves), intentamos reconstruirlo manualmente
    if "medicamentos" in answer or "diagnosticos" in answer:
        try:
            import json
            
            # Extraemos listas conocidas usando regex (bypasseando el JSON roto)
            meds = _extract_list_from_text(answer, "medicamentos")
            dxs = _extract_list_from_text(answer, "diagnosticos")
            ants = _extract_list_from_text(answer, "antecedentes")
            
            if meds or dxs or ants:
                # Reconstruimos un JSON limpio
                clean_data = {
                    "medicamentos": meds,
                    "diagnosticos": dxs,
                    "antecedentes": ants
                }
                clean_json = json.dumps(clean_data, indent=2, ensure_ascii=False)
                
                # Devolvemos el JSON limpio envuelto en markdown
                answer = f"```json\n{clean_json}\n```"
                
        except Exception as e:
            # Si falla nuestra extracción manual, dejamos el texto original
            pass

    return reasoning, answer

# --- EJECUCIÓN STREAMING MEJORADA ---
async def run_agent_consult_stream(request: ConsultaQdrantRequest):
    try:
        context = await _prepare_agent_execution(request)
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    agent = context["agent"]
    token_pid = context["token_pid"]
    token_sources = context["token_sources"]

    buffer = ""
    # Usamos un parser más simple para el stream para evitar 'flickering'
    
    try:
        response_generator = agent.arun(context["user_msg"], stream=True)
        
        async for chunk in response_generator:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if not content: continue
            
            buffer += content
            
            # Streaming crudo pero efectivo (el parser final ordenará todo)
            # Detectamos si estamos en la fase de 'thinking' o 'answer' dinámicamente
            if "**RESPUESTA FINAL:**" in buffer:
                # Si acabamos de cruzar el umbral
                if "**RESPUESTA FINAL:**" in content:
                      yield {"type": "meta", "status": "thinking_end"}
                else:
                      # Estamos escribiendo la respuesta final
                      yield {"type": "token", "text": content}
            else:
                # Estamos en análisis
                yield {"type": "reasoning_token", "text": content}

        # POST-PROCESAMIENTO FINAL
        reasoning, answer = parse_medgemma_output(buffer)
        
        # Limpieza de Markdown de código JSON si existe
        if "```json" in answer:
            answer = answer.replace("```json", "").replace("```", "").strip()

        yield {
            "type": "end",
            "final": {
                "respuesta": answer,
                "reasoning": reasoning,
                "sources": get_agent_sources(),
                "agent_model": AGENT_MODEL_NAME,
                "session_id": context["session_id"]
            }
        }

    except Exception as e:
        log.error(f"❌ Error stream: {e}", exc_info=True)
        yield {"type": "error", "message": str(e)}

    finally:
        clear_patient_context(token_pid)
        clear_agent_sources(token_sources)

# La versión síncrona se actualiza usando la misma lógica de parseo
async def run_agent_consult(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    start_time = time.perf_counter()
    context = await _prepare_agent_execution(request)
    try:
        agent = context["agent"]
        start_gen = time.perf_counter()
        response = agent.run(context["user_msg"])
        gen_time = time.perf_counter() - start_gen
        
        final_answer = response.content if hasattr(response, 'content') else str(response)
        
        # Usamos el parser robusto
        reasoning, clean_ans = parse_medgemma_output(final_answer)
        
        # Limpieza JSON
        if "```json" in clean_ans:
            clean_ans = clean_ans.replace("```json", "").replace("```", "").strip()

        latency = round(time.perf_counter() - start_time, 2)
        
        return {
            "respuesta": clean_ans,
            "reasoning": reasoning,
            "sources": get_agent_sources(),
            "session_id": context["session_id"],
            "meta": {
                **context["meta_base"],
                "latency_seconds": latency,
                "llm_gen_seconds": round(gen_time, 2)
            },
            "agent_model": AGENT_MODEL_NAME
        }
    except Exception as e:
        log.error(f"❌ Error: {e}", exc_info=True)
        return {"respuesta": "Error.", "sources": [], "meta": {"error": str(e)}}
    finally:
        clear_patient_context(context["token_pid"])
        clear_agent_sources(context["token_sources"])

def delete_agent_session(session_id: str) -> int:
    try:
        get_storage().delete_session(session_id)
        return 1
    except Exception:
        return -1