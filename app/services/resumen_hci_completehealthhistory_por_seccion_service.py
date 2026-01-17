# resumen_hci_completehealthhistory_por_seccion_service.py
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict

import tiktoken
from llama_index.core import Document, Settings, SummaryIndex
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode

from app.utils.agrupar_y_formatear_items_hc import generar_linea_de_tiempo_clinica

logging.basicConfig(level=logging.INFO)

CONTEXTO_MAXIMO_TOKENS = 8192

PROMPT_PASE_UNICO = (
    "Eres un asistente médico experto. Tu tarea es generar un resumen narrativo, profesional y en español "
    "de la siguiente sección clínica, cuyas notas están ordenadas cronológicamente.\n\n"
    "**Instrucciones de Síntesis Críticas:**\n"
    "1. **Crea una narrativa unificada y coherente.** Conecta los eventos y muestra la evolución del paciente.\n"
    "2. **Elimina toda la redundancia.** Si un evento se menciona varias veces, sintetízalo en una sola descripción.\n"
    "3. **No infieras información.** Usa solo lo que esté explícitamente escrito.\n"
    "4. **Respeta las abreviaturas:** 'PHP' es 'Plan de Hidratación Parenteral', 'SF' es 'Solución Fisiológica', 'EV' es 'Endovenoso'.\n"
    "5. **No añadas frases de cierre genéricas.** Termina cuando se acabe la información clínica.\n"
    "6. Responde solo en español y con tono formal.\n\n"
    "**Contexto Clínico Ordenado:**\n"
    "---------------------\n"
    "{contexto_ordenado}\n"
    "---------------------\n\n"
    "**Resumen Narrativo Final:**"
)

MAP_PROMPT_STR = (
    "Eres un asistente médico experto. A continuación se presenta un fragmento de una historia clínica en español. "
    "Extrae los eventos clínicos clave, diagnósticos, procedimientos, medicaciones y resultados relevantes "
    "de este fragmento, de forma concisa y profesional. No inventes ni repitas datos. "
    "Responde solo en español y respeta las abreviaturas: 'PHP', 'SF', 'EV'.\n"
    "---------------------\n"
    "Fragmento: {context_str}\n"
    "---------------------\n"
    "Eventos clínicos clave en español:"
)

REDUCE_PROMPT_STR = (
    "Te proporciono varios resúmenes parciales en español, ordenados cronológicamente. "
    "Combina esta información en un único resumen narrativo profesional en español. "
    "Elimina repeticiones y crea una historia fluida y coherente. "
    "No agregues información nueva ni infieras datos. "
    "Responde solo en español y respeta abreviaturas.\n"
    "---------------------\n"
    "Resúmenes parciales: {context_str}\n"
    "---------------------\n"
    "Resumen narrativo final en español:"
)

map_prompt = PromptTemplate(MAP_PROMPT_STR)
reduce_prompt = PromptTemplate(REDUCE_PROMPT_STR)

# ────────────────── Settings context helper ──────────────────
if hasattr(Settings, "context"):  # LlamaIndex ≥ 0.11.3
    _settings_ctx = Settings.context
else:  # fallback manual

    @contextmanager
    def _settings_ctx(**kwargs):
        saved = {k: Settings.__dict__.get(k) for k in kwargs}
        try:
            for k, v in kwargs.items():
                setattr(Settings, k, v)
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    Settings.__dict__.pop(k, None)
                else:
                    setattr(Settings, k, v)


# ────────────────── FUNCIÓN PRINCIPAL ────────────────────────
async def resumir_hc_por_seccion(
    json_clinico: Dict,
    llm: BaseLLM,  # ← llm aislado por request
) -> Dict[str, str]:
    """
    Devuelve un dict {ruta_seccion: resumen}.
    * Usa pase único si el texto entra en contexto.
    * Usa TREE_SUMMARIZE (map‑reduce) cuando la sección supera ~7 k tokens.
    """
    linea_de_tiempo = generar_linea_de_tiempo_clinica(json_clinico)
    if not linea_de_tiempo:
        return {"Error": "No se pudieron extraer eventos clínicos válidos."}

    secciones = defaultdict(list)
    for ev in linea_de_tiempo:
        secciones[ev["ruta_seccion"]].append(ev["texto_formateado"])

    # ⬇️ Para contar tokens
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    resultados: Dict[str, str] = {}

    # Usamos Settings.context(llm=llm) para que TODO lo interno vea el LLM correcto
    with _settings_ctx(llm=llm):
        for ruta, textos in secciones.items():
            contexto_ordenado = "\n\n---\n\n".join(textos)

            # Texto muy corto → se devuelve tal cual
            if len(contexto_ordenado.split()) < 20:
                resultados[ruta] = contexto_ordenado.strip()
                logging.info(f"✅ Sección '{ruta}' devuelta sin resumir (texto corto).")
                continue

            prompt = PROMPT_PASE_UNICO.format(contexto_ordenado=contexto_ordenado)
            n_tok = len(enc.encode(prompt))
            t0 = time.perf_counter()

            # Pase único (< 90 % del contexto máximo)
            if n_tok < CONTEXTO_MAXIMO_TOKENS * 0.9:
                logging.info(f"✅ Procesando '{ruta}' ({n_tok} tokens) en pase único.")
                try:
                    rsp = await llm.acomplete(prompt)
                    resultados[ruta] = str(rsp).strip()
                    logging.info(
                        f"⏱️ Tiempo '{ruta}': {time.perf_counter() - t0:.2f} s."
                    )
                except Exception as e:
                    logging.error(
                        f"❌ Error en pase único (‘{ruta}’): {e}", exc_info=True
                    )
                    resultados[ruta] = f"[Error al resumir sección: {e}]"
                continue  # → siguiente sección

            # TREE_SUMMARIZE (map‑reduce) para textos grandes
            logging.warning(
                f"⚠️ '{ruta}' ({n_tok} tokens) excede límite → TREE_SUMMARIZE."
            )
            try:
                parser = SimpleNodeParser(chunk_size=2000, chunk_overlap=200)
                nodes = parser.get_nodes_from_documents(
                    [Document(text=contexto_ordenado)]
                )

                summary_index = SummaryIndex(nodes, llm=llm)  # llm inyectado
                qe = summary_index.as_query_engine(
                    response_mode=ResponseMode.TREE_SUMMARIZE,
                    use_async=True,
                    summary_template=reduce_prompt,
                    text_qa_template=map_prompt,
                )
                rsp = await qe.aquery("Resume la historia clínica completa en español.")
                resultados[ruta] = str(rsp).strip()
                logging.info(
                    f"✅ Resumen TREE_SUM para '{ruta}' en {time.perf_counter() - t0:.2f} s."
                )
            except Exception as e:
                logging.error(f"❌ Error TREE_SUMMARIZE (‘{ruta}’): {e}", exc_info=True)
                resultados[ruta] = f"[Error en TREE_SUMMARIZE: {e}]"

    return resultados
