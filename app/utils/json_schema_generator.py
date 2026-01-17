from typing import Any, Dict, Union

DESCRIPCIONES_CLINICAS = {
    "estructura": "Ruta jerárquica del ítem clínico, por ejemplo 'INDICACIONES ; INDICACIÓN FARMACOLÓGICA'.",
    "valor": "Valor clínico registrado, puede ser texto libre o codificado.",
    "observacion": "Comentario adicional o aclaración del profesional.",
    "fecha_inicio": "Fecha en la que inicia el evento clínico.",
    "fecha_fin": "Fecha en la que finaliza el evento clínico (si aplica).",
    "servicio": "Servicio hospitalario que generó el registro (Ej: Clínica Médica, Terapia Intensiva).",
    "especialidad": "Especialidad médica relacionada al evento (Ej: Cardiología, Nefrología).",
    "profesional": "Nombre del profesional o usuario que registró el dato.",
    "numero_de_orden": "Número que indica el orden del ítem en la historia.",
    "confidencialidad": "Nivel de confidencialidad del registro clínico.",
    "nombre_item": "Nombre del ítem clínico específico (Ej: Tensión arterial, Hemoglobina glicosilada).",
    "hc_id": "Identificador único de la historia clínica.",
    "paciente_id": "ID del paciente al que corresponde el registro.",
    "resultado_id": "ID del resultado clínico registrado.",
}


def inferir_tipo(valor: Any) -> Union[str, Dict[str, Any]]:
    """
    Infiera el tipo JSON y estructura recursiva si es necesario.
    """
    if isinstance(valor, str):
        return "string"
    elif isinstance(valor, bool):
        return "boolean"
    elif isinstance(valor, int):
        return "integer"
    elif isinstance(valor, float):
        return "number"
    elif isinstance(valor, list):
        if valor and isinstance(valor[0], dict):
            return {"type": "array", "items": inferir_schema(valor[0])}
        else:
            return {
                "type": "array",
                "items": {"type": inferir_tipo(valor[0]) if valor else "string"},
            }
    elif isinstance(valor, dict):
        return inferir_schema(valor)
    else:
        return "string"


def inferir_schema(objeto: Dict[str, Any]) -> Dict[str, Any]:
    schema = {"type": "object", "properties": {}, "required": []}

    for clave, valor in objeto.items():
        tipo = inferir_tipo(valor)
        descripcion = DESCRIPCIONES_CLINICAS.get(clave, f"Campo clínico '{clave}'")

        schema["properties"][clave] = {
            "type": tipo["type"] if isinstance(tipo, dict) else tipo,
            "description": descripcion,
        }

        if isinstance(tipo, dict) and tipo.get("type") in ["object", "array"]:
            schema["properties"][clave].update(tipo)

        schema["required"].append(clave)

    return schema


def generar_schema_desde_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera el esquema JSON completo a partir del JSON de historia clínica.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "description": "Historia clínica estructurada para RAG médico",
        **inferir_schema(json_data),
    }
