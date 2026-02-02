import base64
import logging
import os
import json
import requests
from fastapi import HTTPException
from PIL import Image
import io

log = logging.getLogger(__name__)

async def analyze_medical_image(image_bytes: bytes, prompt: str = "Describe esta imagen médica detalladamente.") -> str:
    """
    Analiza una imagen médica llamando directamente a la API de Ollama.
    Esto evita problemas de validación de Pydantic en versiones recientes de LlamaIndex.
    """
    try:
        # 1. Validar integridad de imagen
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

        # 2. Configuración desde variables de entorno
        # Nota: LLM_BASE_URL suele ser 'http://ip:11434'
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        # Usamos gemma3:4b para cumplir reglas de Kaggle y cuidar la VRAM (pesa ~3GB)
        model_name = os.getenv("LLM_MODEL_VISION", "gemma3:4b")
        
        # 3. Preparar payload para Ollama API
        # Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        
        # Codificar a Base64
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        url = f"{base_url}/api/generate"
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False  # Respuesta completa de una vez
        }
        
        log.info(f"👁️ Enviando solicitud directa a Ollama ({url}). Modelo: {model_name}")
        
        # 4. Enviar request
        # Usamos timeout generoso porque visión puede tardar
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code != 200:
            log.error(f"❌ Error Ollama API: {response.status_code} - {response.text}")
            raise HTTPException(status_code=502, detail=f"Error del modelo de IA: {response.text}")
            
        # 5. Procesar respuesta
        data = response.json()
        model_response = data.get("response", "")
        
        return model_response

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="El modelo tardó demasiado en responder.")
    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"❌ Error crítico en análisis de imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
