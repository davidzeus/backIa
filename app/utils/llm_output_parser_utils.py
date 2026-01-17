import json
import logging

from jsonschema import validate
from llama_index.core.selectors import SelectionOutputParser

logger = logging.getLogger(__name__)

# ✅ Esquema corregido
ROUTER_SCHEMA = {
    "type": "object",
    "properties": {"index": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["index", "reason"],
}


def validate_llm_output(output: str) -> dict:
    """Valida y parsea salida JSON del LLM con esquema estricto"""
    try:
        parsed = json.loads(output)
        validate(instance=parsed, schema=ROUTER_SCHEMA)
        return parsed
    except Exception as e:
        logger.error(f"Invalid LLM output: {output}")
        raise ValueError(f"Formato inválido: {str(e)}")


class CustomOutputParser(SelectionOutputParser):
    def parse(self, output: str):
        try:
            return validate_llm_output(output)
        except ValueError:
            cleaned = output.replace("'", '"').replace("choice", "index")
            return validate_llm_output(cleaned)
