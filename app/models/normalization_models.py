from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SnomedConceptOutput(BaseModel):
    """
    Encapsula toda la información relevante de un único concepto SNOMED CT.
    Es un sub-modelo para mantener la estructura limpia.
    """

    conceptId: str = Field(..., description="El ID del concepto SNOMED CT.")
    term: str = Field(
        ..., description="El término preferido en español para el concepto."
    )
    fsn: str = Field(
        ...,
        description="El Fully Specified Name (Nombre Completo del Sistema) para desambiguación.",
    )


class NormalizedFinding(BaseModel):
    """
    Representa un único hallazgo clínico procesado, con su mapeo y justificación.
    """

    original_text: str = Field(
        ...,
        description="El fragmento de texto o concepto original extraído de la epicrisis.",
    )
    is_relevant: bool = Field(
        ...,
        description="Indica si el LLM considera este hallazgo clínicamente relevante para ser codificado.",
    )
    # Usamos 'Optional' que en Python se traduce a 'Union[SnomedConceptOutput, None]'.
    # Esto fuerza a que el valor sea o un objeto SnomedConceptOutput completo o None. No hay ambigüedad.
    snomed_concept: Optional[SnomedConceptOutput] = Field(
        None,
        description="El concepto SNOMED CT más apropiado. Es 'null' si no se encontró o si is_relevant es false.",
    )
    justification: str = Field(
        ...,
        description="Explicación concisa del LLM sobre la elección del código o la razón para no codificarlo. Clave para la auditoría.",
    )


class NormalizationResult(BaseModel):
    """
    El modelo de salida completo y estructurado para una solicitud de normalización.
    """

    # Usamos el tipo datetime de Pydantic para una validación automática.
    request_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Fecha y hora UTC de la solicitud de normalización.",
    )
    model_used: str = Field(
        ...,
        description="El nombre del modelo o pipeline utilizado para la normalización.",
    )
    normalized_findings: List[NormalizedFinding] = Field(
        ...,
        description="La lista de todos los hallazgos clínicos procesados del texto.",
    )


class NormalizationRequest(BaseModel):
    text: str = Field(
        ..., description="Texto clínico completo que se desea normalizar."
    )
