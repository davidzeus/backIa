import os
import logging
from llama_index.multi_modal_llms.ollama import OllamaMultiModal

log = logging.getLogger(__name__)

_vision_model_instance = None

def get_vision_model() -> OllamaMultiModal:
    """
    Retorna (o crea) la instancia del modelo multimodal de Ollama.
    Usa la variable de entorno LLM_MODEL_AGENT (def: medgemma:4b) 
    y LLM_BASE_URL (def: http://localhost:11434).
    """
    global _vision_model_instance
    
    if _vision_model_instance is not None:
        return _vision_model_instance

    model_name = os.getenv("LLM_MODEL_AGENT", "medgemma:4b")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    
    log.info(f"👁️ Inicializando modelo de Visión (OllamaMultiModal): {model_name} en {base_url}")
    
    try:
        _vision_model_instance = OllamaMultiModal(
            model=model_name,
            base_url=base_url,
            temperature=0.0  # Para análisis más determinista
        )
        return _vision_model_instance
    except Exception as e:
        log.error(f"❌ Error fatal iniciando modelo de visión: {e}")
        raise e
