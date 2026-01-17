# app/schemas/ontology.py
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, ConfigDict

class QueryPlan(BaseModel):
    # Campos “core” del planner
    need_sections: List[str] = []
    need_events: List[str] = []
    which: Optional[Literal["first", "last"]] = None
    count: int = 1

    # Enriquecimientos opcionales (planner y reglas)
    is_inpatient: Optional[bool] = None
    name_hint: Optional[str] = None
    procedure_number: Optional[int] = None
    time_window: Optional[Dict[str, str]] = None  # ej: {"start":"2025-05-01","end":"2025-05-31"}
    service_hints: Optional[List[Dict[str, Any]]] = None

    # Aceptar extras por si en el futuro agregamos más señales
    model_config = ConfigDict(extra="allow")




