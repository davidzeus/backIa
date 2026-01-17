# app/tools/snomed_tool.py

from typing import Any, Dict

from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine import BaseQueryEngine

from app.services.snomed_service import buscar_conceptos_snomed


class SNOMEDQueryEngine(BaseQueryEngine):
    def __init__(self, ecl="< 404684003>"):
        # Corregido → pasamos el callback_manager requerido
        super().__init__(callback_manager=CallbackManager([]))
        self.ecl = ecl

    def _query(self, query_str: str, **kwargs) -> Response:
        conceptos = buscar_conceptos_snomed(query_str, self.ecl)

        if not conceptos:
            result_text = f"No se encontraron conceptos SNOMED para: '{query_str}'."
        else:
            result_text = f"Conceptos SNOMED para '{query_str}':\n\n"
            for c in conceptos:
                result_text += (
                    f"- {c['conceptId']} - {c['preferredTerm']} - {c['fsn']}\n"
                )

        return Response(response=result_text)

    async def _aquery(self, query_str: str, **kwargs) -> Response:
        return self._query(query_str, **kwargs)

    def _get_prompt_modules(self) -> Dict[str, Any]:
        return {}
