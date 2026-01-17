# app/utils/meta_to_context.py 

import unicodedata
import logging # Asegúrate de que logging está importado
from typing import List

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

# Configura el logger para este archivo
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _strip_accents(s: str) -> str:
    try:
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    except Exception:
        return s


def _make_meta_header(md: dict) -> str:
    """
    Encabezado compacto con metadata relevante del nodo, incluyendo
    un "hecho" explícito sobre internaciones para el LLM.
    """
    svc = str(md.get("servicio", "") or md.get("servicio_desc", "")).strip()
    sec = str(md.get("seccion", "") or "").strip()
    raiz = str(md.get("seccion_raiz", "") or "").strip()
    fecha = str(md.get("fecha", "") or "").strip()
    doc = str(md.get("profesional", "") or md.get("doctor_full", "")).strip()
    pid = str(md.get("paciente_id", "") or "").strip()
    
    # --- LÓGICA DE DEPURACIÓN Y ENRIQUECIMIENTO ---
    procedure_number = md.get("procedureNumber")
    log.debug(f"Analizando metadatos: procedureNumber = '{procedure_number}' (Tipo: {type(procedure_number)})")

    parts = []

    if procedure_number and str(procedure_number).strip():
        log.debug(f"CONDICIÓN CUMPLIDA para procedureNumber '{procedure_number}': Añadiendo 'TIPO_REGISTRO: INTERNACION'")
        parts.append(f"TIPO_REGISTRO: INTERNACION (TRÁMITE {procedure_number})")
    # --- FIN DE LA LÓGICA ---

    if svc and svc.lower() != 'no especificado':
        parts.append(f"servicio: {svc}")
    if sec:
        parts.append(f"seccion: {sec}")
    if raiz and raiz != sec:
        parts.append(f"seccion_raiz: {raiz}")
    if fecha:
        parts.append(f"fecha: {fecha}")
    if doc:
        parts.append(f"profesional: {doc}")
    if pid:
        parts.append(f"paciente_id: {pid}")

    return ("### METADATA\n" + " | ".join(parts) + "\n\n") if parts else ""


class MetaToContext(BaseNodePostprocessor):
    """Preprende metadata legible al contenido del nodo (si no estaba)."""

    def _apply(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        log.debug(f"--- Ejecutando MetaToContext Postprocessor en {len(nodes)} nodos ---")
        out: List[NodeWithScore] = []
        for n in nodes:
            try:
                md = getattr(n.node, "metadata", {}) or {}
                header = _make_meta_header(md)
                txt = n.node.get_text() or ""
                # El header ya tiene un \n\n al final
                if header and not txt.startswith("### METADATA"):
                    n.node.text = f"{header}{txt}"
            except Exception:
                pass
            out.append(n)
        return out

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle=None
    ) -> List[NodeWithScore]:
        return self._apply(nodes)

    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle=None
    ) -> List[NodeWithScore]:
        return self._apply(nodes)