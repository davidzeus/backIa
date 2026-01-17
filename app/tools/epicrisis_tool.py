# app/tools/epicrisis_tool.py
from typing import Any, Dict, List

from llama_index.core import SummaryIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import TextNode

from app.config.llm_config import get_llm


class EpicrisisQueryEngine(BaseQueryEngine):
    """Genera una epicrisis estructurada a partir de nodos de HC."""

    def __init__(self, nodes: List[TextNode], llm=None):
        if not nodes:
            raise ValueError("❌ No se recibieron nodos para la epicrisis.")

        # motor interno: SummaryIndex + LLM “summary”
        internal = SummaryIndex(nodes).as_query_engine(
            llm=llm or get_llm("summary"),
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            use_async=False,
            verbose=True,
        )

        # --- _solo_ callback_manager para el padre ---
        super().__init__(callback_manager=internal.callback_manager)

        # luego copiamos los componentes privados
        self._retriever = getattr(internal, "_retriever", None)
        self._response_synthesizer = getattr(internal, "_response_synthesizer", None)
        self._engine = internal
        self.streaming = False  # evita que BaseQueryEngine busque .llm

    # ------------ métodos obligatorios ------------
    def _query(self, query_str: str, **kwargs) -> Response:
        return self._engine.query(self._build_prompt(query_str))

    async def _aquery(self, query_str: str, **kwargs) -> Response:
        return await self._engine.aquery(self._build_prompt(query_str))

    def _get_prompt_modules(self) -> Dict[str, Any]:
        return {}

    # ------------ helper ------------
    @staticmethod
    def _build_prompt(historia: str) -> str:
        return (
            "Generá una epicrisis médica con las siguientes secciones:\n\n"
            "1. Datos del paciente\n2. Motivo de ingreso\n3. Diagnóstico de ingreso\n"
            "4. Antecedentes médicos\n5. Estudios realizados\n6. Tratamiento\n"
            "7. Evolución\n8. Diagnóstico de egreso\n9. Plan de alta\n10. Pronóstico\n\n"
            "Sos un médico experto. Usá lenguaje claro y profesional.\n\n"
            f"Historia clínica:\n{historia}"
        )
