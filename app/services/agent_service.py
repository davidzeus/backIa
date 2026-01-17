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
# ✅ Herramienta Manual probada (funciona mejor que Native en este setup)
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
#AGENT_MODEL_NAME = os.getenv("LLM_MODEL_AGENT", "qwen3:4b") 
AGENT_MODEL_NAME = os.getenv("LLM_MODEL_AGENT", "mistral") 

AGENT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://10.10.0.48:6379/0")

storage = RedisDb(session_table="sessions", db_url=REDIS_URL)

# --- DEFINICIÓN DEL AGENTE (SIN KNOWLEDGE NATIVO) ---
def get_clinical_agent(
    session_id: str, 
    user_id: str, 
    context_data: str, # <--- RAG Inyectado manualmente
    planner_info: str,
    is_summary: bool
) -> Agent:

    base_instructions = [
        "Piensa paso a paso antes de responder.",
        "Genera tu razonamiento interno dentro de etiquetas <think>...</think> antes de dar la respuesta final.",
        "Responde en ESPAÑOL profesional.",
    ]
    # base_instructions = [
    #     "Eres un experto analista médico.",
    #     "IMPORTANTE: Tu proceso de pensamiento debe ser explícito.",
    #     "1. PRIMERO: Analiza la información paso a paso dentro de etiquetas <think>...</think>.",
    #     "   - Chequea fechas.",
    #     "   - Verifica consistencia entre documentos.",
    #     "2. SEGUNDO: Entrega la respuesta final al usuario dentro de etiquetas <answer>...</answer>.",
    #     "NO entregues texto fuera de estas etiquetas.",
    # ]

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
            "MODO ESTRICTO: Extractor de datos literal.",
            "Responde ÚNICAMENTE con información del CONTEXTO CLÍNICO.",
            "Si se solicitan medicamentos, extrae TODOS.",
            "REGLA DE ORO: Solo usa fechas explícitas en los registros.",
            "EXHAUSTIVIDAD: Si la pregunta abarca un periodo, extrae TODO lo encontrado en CADA documento.",
            "SEMÁNTICA AMPLIA: Si preguntan por una zona (ej: 'Abdomen'), incluye 'Pared', 'Colección', 'Herida', etc. NO DESCARTES por tecnicismos.",
            "ANTE LA DUDA: Incluye el dato y aclara su origen (ej: 'En pared abdominal se halló X').",
            "Si el dato no está, responde: 'No hay registros disponibles.'",
        ]
    
    return Agent(
        name="SAMI",
        model=Ollama(
            id=AGENT_MODEL_NAME,  # ✅ Desde .env: LLM_MODEL_AGENT
            host=AGENT_BASE_URL,  # ✅ Desde .env: LLM_BASE_URL
            options={
                "temperature": 0.0, 
                "num_ctx": 8192, # Restored for safety
                "num_predict": 8192,
                "top_p": 0.9,
                "repeat_penalty": 1.1,  # Evita repeticiones
            }
        ),
        db=storage, 
        session_id=session_id,
        user_id=user_id,
        add_name_to_context=True,
        
        #  Desactivamos reasoning nativo porque causa bucles con Qwen
        reasoning=True, 
        debug_mode=bool(int(str(_DEBUG_FLAG))),

        description="Eres SAMI, un Asistente Médico Inteligente experto en análisis de Historias Clínicas. SIEMPRE respondes en ESPAÑOL.",
        instructions=[
            # --- 🌐 IDIOMA OBLIGATORIO ---
            "🇪🇸 INSTRUCCIÓN CRÍTICA DE IDIOMA: DEBES responder ÚNICAMENTE en ESPAÑOL.",
            "TODAS tus respuestas, análisis y razonamientos deben estar en ESPAÑOL.",
            "Nunca uses inglés, portugués u otro idioma. Solo ESPAÑOL.",
            "",
            # --- 🛡️ BLOQUE DE SEGURIDAD ANTI-ALUCINACIÓN (CRÍTICO) ---
            "¡ADVERTENCIA CRÍTICA DE SEGURIDAD!:",
            "1. TU PRIMERA PRIORIDAD es verificar si recibiste documentos en el contexto bajo '=== HISTORIA CLÍNICA ==='.",
            "2. Si el texto dice 'No se encontraron registros', 'No hay registros' o está vacío, DEBES RESPONDER EXACTAMENTE:",
            "   '⚠️ No dispongo de registros clínicos para el período solicitado en la base de datos.'",
            "3. IGNORA CUALQUIER INFORMACIÓN PREVIA del chat si el contexto actual ('=== HISTORIA CLÍNICA ===') dice que no hay registros.",
            "4. BAJO NINGUNA CIRCUNSTANCIA INVENTES DATOS ni uses fechas de mensajes anteriores si no están en el contexto actual.",
            "5. Si inventas información médica, pones en riesgo grave la vida del paciente.",
            "ANALIZA la información clínica paso a paso EN ESPAÑOL.",
            "GENERA TU RESPUESTA EN EL SIGUIENTE FORMATO EXACTO (TODO EN ESPAÑOL):",
            "<analysis>",
            "Aquí escribe tu razonamiento paso a paso EN ESPAÑOL, chequeando fechas y datos.",
            "</analysis>",
            "<answer>",
            "Aquí escribe la respuesta final al usuario EN ESPAÑOL profesional y claro.",
            "</answer>",
            "NO ESCRIBAS NADA FUERA DE ESTOS TAGS.",
            "NO REPITAS la respuesta. Sé conciso.",
            "NUNCA uses inglés. TODO EN ESPAÑOL.",
            "",
            "=== EJEMPLO DE SALIDA (EN ESPAÑOL) ===",
            "<analysis>",
            "Breve chequeo de datos: DOC #1 (15/10) TA 130/80. DOC #2 (14/10) TA 120/70.",
            "</analysis>",
            "<answer>",
            "El 15 de octubre la paciente tuvo presión arterial de 130/80 mmHg.",
            "</answer>",
            "========================="
        ] + specific_instructions,
        
        
        # ✅ RAG Inyectado (Prompt Engineering)
        additional_context=f"{planner_info}\n\n=== HISTORIA CLÍNICA ===\n{context_data}",
        
        expected_output="Respuesta COMPLETA en ESPAÑOL profesional, clara y concisa. NUNCA uses inglés.",
        add_datetime_to_context=True,
        add_history_to_context=True,
        num_history_runs=2,
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
        info_parts.append(f"- FILTRO TEMPORAL APLICADO: {dt_start} al {dt_end}")
    
    services = plan.get("service_hints", [])
    if services:
        nombres = [s.get('desc', '') for s in services]
        info_parts.append(f"- FOCO EN SERVICIOS: {', '.join(nombres)}")
        
    return "--- CONTEXTO TÉCNICO ---\n" + "\n".join(info_parts) + "\n------------------------\n" if info_parts else ""

def is_internal_error(text: str) -> bool:
    if not text: return False
    return "error interno" in text.lower() or "nonetype" in text.lower()

# --- PREPARACIÓN DEL CONTEXTO (MANUAL RAG) ---
async def _prepare_agent_execution(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    token_pid = set_patient_context(request.paciente_id)
    token_sources = set_agent_sources([]) 
    internal_session_id = f"{request.session_id}::{request.paciente_id}"

    try:
        # 1. PLANIFIER
        plan = get_query_plan_tools(request.pregunta)
        log.info(f"🧠 [PLANNER] Plan generado: {plan}")
        is_summary = plan.get("is_summary", False)
        doc_limit = 40 if is_summary else 10
        
        if is_summary: log.info("📚 MODO RESUMEN detectado.")
        else: log.info("🎯 MODO PUNTUAL detectado.")

        # 2. BÚSQUEDA MANUAL (usando rag_tool)
        log.info(f"🔎 [RAG] Buscando en '{request.paciente_id}'...")
        raw_results = search_clinical_history(
            request.pregunta, 
            qdrant_filters=plan, 
            strict=plan.get("strict", False), 
            limit=doc_limit
        )

        # Fallback de Servicios
        if raw_results and "No se encontraron registros" in raw_results and plan.get("service_hints"):
             log.warning("⚠️ Fallback: Quitando filtro de servicio...")
             plan_loose = plan.copy()
             del plan_loose["service_hints"]
             raw_results_b = search_clinical_history(request.pregunta, qdrant_filters=plan_loose, strict=False, limit=doc_limit)
             if raw_results_b and "No se encontraron registros" not in raw_results_b:
                 raw_results = raw_results_b

        evidence_text = raw_results if not is_internal_error(raw_results) else "Sin datos."
        planner_info_msg = format_plan_info(plan)

        # 3. CONSTRUCCIÓN DEL AGENTE
        agent = get_clinical_agent(
            session_id=internal_session_id, 
            user_id=request.user_id,
            context_data=evidence_text,      # <--- Inyección
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

# --- EJECUCIÓN SÍNCRONA ---
async def run_agent_consult(request: ConsultaQdrantRequest) -> Dict[str, Any]:
    start_time = time.perf_counter() # ⏱️ Medición de latencia
    context = await _prepare_agent_execution(request)
    try:
        agent = context["agent"]
        
        # ⏱️ Medir solo generación del LLM
        start_gen = time.perf_counter()
        response = agent.run(context["user_msg"], show_full_reasoning=True)
        gen_time = time.perf_counter() - start_gen
        log.info(f"🧠 [LLM] Generación completada en {gen_time:.2f}s")
        
        final_answer = response.content if hasattr(response, 'content') else str(response)
        
        # 🐛 DEBUG: Ver qué devuelve realmente el modelo
        log.info(f"🔎 [RAW AGENT OUTPUT] (First 200 chars): {final_answer[:200]}...")
        
        
        # ✅ PARSER XML (Strict Format)
        reasoning = None
        clean_ans = final_answer
        
        try:
            # 1. Extraer Analysis
            s_an = final_answer.find("<analysis>")
            e_an = final_answer.find("</analysis>")
            if s_an != -1 and e_an != -1:
                reasoning = final_answer[s_an+10:e_an].strip()
            
            # 2. Extraer Answer
            s_res = final_answer.find("<answer>")
            e_res = final_answer.find("</answer>")
            if s_res != -1 and e_res != -1:
                clean_ans = final_answer[s_res+8:e_res].strip()
            else:
                # Si falló el tag answer, intentamos limpiar lo que sobró del analysis
                if e_an != -1:
                     clean_ans = final_answer[e_an+11:].strip()
        except Exception:
            pass
        
        latency = round(time.perf_counter() - start_time, 2)
        
        return {
            "respuesta": clean_ans, 
            "reasoning": reasoning, 
            "sources": get_agent_sources(), 
            "session_id": context["session_id"],
            "meta": {
                **context["meta_base"],
                "latency_seconds": latency, # Total
                "llm_gen_seconds": round(gen_time, 2) # Solo LLM
            },
            "agent_model": AGENT_MODEL_NAME
        }
    except Exception as e:
        log.error(f"❌ Error: {e}", exc_info=True)
        return {"respuesta": "Error.", "sources": [], "meta": {"error": str(e)}}
    finally:
        clear_patient_context(context["token_pid"])
        clear_agent_sources(context["token_sources"])


# --- EJECUCIÓN STREAMING  ---
async def run_agent_consult_stream(request: ConsultaQdrantRequest):
    try:
        context = await _prepare_agent_execution(request)
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    agent = context["agent"]
    token_pid = context["token_pid"]
    token_sources = context["token_sources"]

    # Buffers para reconstruir la respuesta final limpia
    full_reasoning = []
    full_answer = []
    
    # Estado del Stream
    buffer = ""
    current_mode = "waiting" # waiting | analysis | answer
    
    try:
        # User msg es solo la pregunta (contexto ya inyectado en agent)
        # Nota: show_full_reasoning=True es importante para que el modelo envíe los tags
        
        response_generator = agent.arun(context["user_msg"], stream=True, show_full_reasoning=True)
        
        async for chunk in response_generator:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if not content:
                continue

            buffer += content

            # --- MÁQUINA DE ESTADOS PARA PARSEO DE XML EN STREAM ---
            
            # 1. DETECCIÓN DE APERTURA DE TAGS
            if current_mode == "waiting":
                if "<analysis>" in buffer:
                    # Se detectó inicio de razonamiento
                    current_mode = "analysis"
                    _, buffer = buffer.split("<analysis>", 1) # Limpiamos el tag del buffer
                    yield {"type": "meta", "status": "thinking_start"}
                
                elif "<think>" in buffer: # Compatibilidad con modelos deepseek/qwen puro
                    current_mode = "analysis"
                    _, buffer = buffer.split("<think>", 1)
                    yield {"type": "meta", "status": "thinking_start"}

                elif "<answer>" in buffer:
                    # Se detectó inicio de respuesta
                    current_mode = "answer"
                    _, buffer = buffer.split("<answer>", 1)
            
            # 2. PROCESAMIENTO SEGÚN MODO
            if current_mode == "analysis":
                # Buscamos cierre
                if "</analysis>" in buffer:
                    chunk_part, buffer = buffer.split("</analysis>", 1)
                    if chunk_part:
                        full_reasoning.append(chunk_part)
                        yield {"type": "reasoning_token", "text": chunk_part}
                    current_mode = "waiting"
                    yield {"type": "meta", "status": "thinking_end"}
                elif "</think>" in buffer:
                    chunk_part, buffer = buffer.split("</think>", 1)
                    if chunk_part:
                        full_reasoning.append(chunk_part)
                        yield {"type": "reasoning_token", "text": chunk_part}
                    current_mode = "waiting"
                    yield {"type": "meta", "status": "thinking_end"}
                else:
                    # Si no hay cierre, enviamos todo lo seguro (dejando un margen por si se corta un tag)
                    # Truco: Dejar los últimos caracteres en buffer por si viene un tag partido
                    if len(buffer) > 20: 
                        safe_chunk = buffer[:-15]
                        buffer = buffer[-15:]
                        full_reasoning.append(safe_chunk)
                        yield {"type": "reasoning_token", "text": safe_chunk}

            elif current_mode == "answer":
                # Buscamos cierre
                if "</answer>" in buffer:
                    chunk_part, buffer = buffer.split("</answer>", 1)
                    if chunk_part:
                        full_answer.append(chunk_part)
                        yield {"type": "token", "text": chunk_part}
                    current_mode = "waiting" # Terminó la respuesta
                else:
                    # Streaming fluido de la respuesta
                    if len(buffer) > 20:
                        safe_chunk = buffer[:-15]
                        buffer = buffer[-15:]
                        full_answer.append(safe_chunk)
                        yield {"type": "token", "text": safe_chunk}

        # --- LIMPIEZA FINAL (FLUSH) ---
        # Si quedó algo en el buffer que pertenece al modo actual
        if current_mode == "analysis" and buffer:
            clean_buf = buffer.replace("</analysis>", "").replace("</think>", "")
            full_reasoning.append(clean_buf)
            yield {"type": "reasoning_token", "text": clean_buf}
        
        elif current_mode == "answer" and buffer:
             clean_buf = buffer.replace("</answer>", "")
             full_answer.append(clean_buf)
             yield {"type": "token", "text": clean_buf}

        # Reconstruir strings finales
        final_reasoning_str = "".join(full_reasoning).strip()
        final_answer_str = "".join(full_answer).strip()

        # Fallback de seguridad: Si el modelo no usó tags <answer> y escupió texto plano al final
        if not final_answer_str and not final_reasoning_str and buffer:
             # Asumimos que todo fue respuesta si no hubo estructura
             final_answer_str = buffer
             yield {"type": "token", "text": buffer}

        # Evento Final con estructura limpia (igual que el síncrono)
        yield {
            "type": "end",
            "final": {
                "respuesta": final_answer_str,
                "reasoning": final_reasoning_str,
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




def delete_agent_session(session_id: str) -> int:
    """
    Elimina la sesión del Agente de Redis.
    Retorna:
      1: Eliminado OK (o no existía y no generó error)
     -1: Error
    """
    try:
        # Nota: RedisDb de agno no tiene método 'read' público confiable.
        # delete_session suele ser idempotente (si no existe, no falla).
        storage.delete_session(session_id)
        log.info(f"🗑️ Sesión eliminada de Redis: {session_id}")
        return 1
    except Exception as e:
        log.error(f"❌ Error eliminando sesión {session_id}: {e}")
        return -1

