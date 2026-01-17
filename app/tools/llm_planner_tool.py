# app/tools/llm_planner.py
import json
import logging
import os
from datetime import datetime
from agno.models.ollama import Ollama

log = logging.getLogger(__name__)

# --- CONFIGURACIÓN DEL MODELO DE PLANIFICACIÓN ---
# ✅ Usa variable de entorno LLM_MODEL_AGENT del .env
PLANNER_MODEL_NAME = os.getenv("LLM_MODEL_AGENT", "mistral")
PLANNER_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")

planner_llm = Ollama(
    id=PLANNER_MODEL_NAME,
    host=PLANNER_BASE_URL,
    options={
        "temperature": 0.0,   # Determinista: siempre la misma respuesta para la misma fecha
        "num_ctx": 4096,
        "json": True          # Forzamos al modelo a responder SOLO JSON válido
    }
)

def get_smart_date_window(query: str) -> dict:
    """
    Usa un LLM para entender fechas relativas ("semana pasada", "enero")
    y devolver un rango ISO exacto.
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    weekday = now.strftime("%A")

    # Prompt diseñado para resolver ambigüedades temporales
    prompt = f"""
    Eres un experto en extracción de entidades temporales para búsquedas médicas.
    HOY es: {today_str} ({weekday}).

    Tu misión es analizar la consulta del usuario y generar parámetros de búsqueda estructurados.

    REGLAS DE TIEMPO (CRÍTICO):
    1. Si menciona un MES sin año (ej: "enero"):
       - Si el mes ya pasó este año, usa el año actual.
       - Si el mes es futuro respecto a hoy, usa el año anterior.
       - Si especifica año ("enero 2024"), úsalo.
    2. "Última semana" / "últimos 7 días": calcula el rango desde (hoy - 7) hasta hoy.
    3. "Ayer": hoy - 1 día.
    4. Si NO hay mención temporal: devuelve null en las fechas.

    REGLAS DE INTENCIÓN (intent):
    - "resumen", "evolución", "historia", "synthesize", "contame" -> "summary"
    - "¿cuándo...?", "¿qué gérmenes...?", "dato puntual", "valores" -> "specific"
    - "medicación", "droga", "receta", "indicación", "antibiótico" -> "medication"

    Consulta del usuario: "{query}"

    Responde ÚNICAMENTE con este objeto JSON:
    {{
        "start_date_iso": "YYYY-MM-DD" | null,
        "end_date_iso": "YYYY-MM-DD" | null,
        "intent": "summary" | "specific" | "medication",
        "rationale": "Breve explicación de por qué elegiste esas fechas"
    }}
    """

    try:
        # Ejecutamos el LLM
        response = planner_llm.run(prompt)
        
        # Extracción y limpieza defensiva
        content = response.content if hasattr(response, 'content') else str(response)
        clean_content = content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(clean_content)
        log.info(f"🧠 [LLM Planner] Inferencia: {data}")
        return data

    except Exception as e:
        log.error(f"⚠️ Error en LLM Planner: {e}")
        # Fallback seguro por si falla el modelo o el JSON
        return {
            "start_date_iso": None, 
            "end_date_iso": None, 
            "intent": "specific",
            "rationale": "Error fallback"
        }