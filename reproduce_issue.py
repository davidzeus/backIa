
import asyncio
import json
import os
from app.services.agent_service import run_agent_consult
from app.schemas.esquema import ConsultaQdrantRequest

# Mock settings if needed
os.environ["LLM_MODEL_AGENT"] = "medgemma:4b"

async def reproduction():
    payload = {
      "user_id": "maria",
      "session_id": "turno-123",
      "paciente_id": "163234",
      "pregunta": "RESUMEN TRIAGE - Extrae SOLO los elementos críticos y devuelve JSON puro (sin texto).\n\n**INFORMACIÓN DEL PACIENTE:**\n• Nombre: MARIA ESTHER RAMIREZ\n• Fecha Nacimiento: 1963-09-28\n• Sexo: No especificado\n• Nombre: MARIA ESTHER\n• Apellido: RAMIREZ\n\n**ACCESO A HISTORIA CLÍNICA COMPLETA:**\n• Incluye: consultas previas, diagnósticos, tratamientos, laboratorios, estudios, medicación, alergias\n• Fecha de preparación de datos: 17/1/2026, 15:38:59\n\nFormato (arrays vacíos [] si no hay datos):\n{\n  \"allergies\": [{\"name\":\"\",\"severity\":\"low|medium|high|critical\",\"date\":\"DD-MM-AAAA\"}],\n  \"medications\": [{\"name\":\"\",\"dose\":\"\",\"active\":true,\"date\":\"DD-MM-AAAA\"}],\n  \"diagnoses\": [{\"name\":\"\",\"status\":\"active|chronic\",\"date\":\"DD-MM-AAAA\"}],\n  \"medicalHistory\": [\"antecedente relevante con fecha si está disponible\"]\n}\n\nReglas:\n1) SOLO JSON, sin explicaciones\n2) Máximo 5 items por array\n3) Solo datos activos/críticos\n4) Incluir fechas en formato DD-MM-AAAA cuando estén disponibles"
    }
    
    request = ConsultaQdrantRequest(**payload)
    
    print("--- INICIANDO PRUEBA DE REPRODUCCIÓN ---")
    try:
        # Nota: Esto intentará conectar a tu Redis y Ollama locales.
        # Asegúrate de tenerlos corriendo o que el código maneje fallos de conexión gracefulmente si solo queremos probar lógica interna (aunque para ver el loop necesitamos el LLM).
        result = await run_agent_consult(request)
        print("\n--- RESPUESTA DEL AGENTE ---")
        print(result["respuesta"])
        print("----------------------------")
    except Exception as e:
        print(f"Error ejecutando agent: {e}")

if __name__ == "__main__":
    asyncio.run(reproduction())
