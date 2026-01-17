# app/schemas.py
from typing import List, Optional

from pydantic import BaseModel, Field




class ConsultaQdrantRequest(BaseModel):
    """DTO para consultas HC en Qdrant (con filtros crudos opcionales)."""

    # IDENTIFICACIÓN (obligatorios para memoria/particionado)
    user_id: str = Field(
        ...,
        min_length=1,
        description="Identificador del usuario que consulta (particiona la memoria).",
        examples=["maria", "operador-42"],
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Identificador de la sesión de chat (aísla o persiste memoria según clave).",
        examples=["turno-123", "sesion-abc"],
    )

    # obligatorios del dominio
    paciente_id: str = Field(
        ...,
        min_length=1,
        description="ID del paciente (string).",
        examples=["168249"],
    )
    pregunta: str = Field(
        ...,
        min_length=3,
        description="Pregunta sobre la HC del paciente.",
        examples=["¿Cuál fue el motivo de ingreso?"],
    )

    # filtros crudos opcionales (match exacto)
    group: Optional[List[str]] = Field(
        None,
        description='Grupo HCI. Ej: "Ej: ["CM;HISTORIA CLINICA","INDICACIONES FARMACOLÓGICAS"] o "CM;HISTORIA CLINICA"',
        examples=[["CM;HISTORIA CLINICA", "INDICACIONES FARMACOLÓGICAS"]],
    )

    healthHistoryGroup: str | None = Field(
        None, description='Subgrupo HCI. Ej: "CM;EVOLUCIÓN"', examples=["CM;EVOLUCIÓN"]
    )
    procedureNumber: int | None = Field(
        None, description="N° de trámite de internación (si aplica).", examples=[211143]
    )
    servicio_id: List[int] | None = Field(
        None, description="ID(s) de servicio. Ej: [26] o [26,92].", examples=[[26]]
    )
    servicio: str | None = Field(
        None,
        description='Descripción exacta del servicio. Ej: "MEDICINA (5º CATEDRA)"',
        examples=["MEDICINA (5º CATEDRA)"],
    )
    name: str | None = Field(
        None,
        description='Nombre de estructura/nodo. Ej: "MOTIVO CONSULTA INICIAL"',
        examples=["MOTIVO CONSULTA INICIAL"],
    )

 # 👇 NUEVO: el plan del planner (se inyecta en _build_router si no vino)

    class Config:
        extra = "allow"   # aceptar claves nuevas sin romper