import ast
import re
from datetime import datetime
from typing import Any, Optional


def hc_json_to_markdown(data_original: dict) -> str:
    """
    Función de producción DEFINITIVA. Convierte un JSON de HC dinámico en un
    documento Markdown estructurado, enriquecido y ordenado, listo para RAG.
    Fusiona la lógica de dominio específica con una arquitectura robusta.
    """
    print("\n[PROCESADOR DEFINITIVO] --- Iniciando conversión ---")

    # 1. Desenvolver el JSON hasta encontrar el núcleo de datos.
    data_a_procesar = _extraer_contenido_hc(data_original)
    if isinstance(data_a_procesar, str):  # Si la extracción devolvió un error
        return data_a_procesar

    # 2. Procesar todos los 'entrys' para extraer eventos enriquecidos.
    eventos_enriquecidos = []
    print(
        f"[PROCESADOR DEFINITIVO] Se encontraron {len(data_a_procesar.get('entrys', []))} 'entrys' para procesar."
    )
    for i, entry in enumerate(data_a_procesar.get("entrys", [])):
        nuevos_eventos = _procesar_entry(entry, i)
        if nuevos_eventos:
            eventos_enriquecidos.extend(nuevos_eventos)

    print(
        f"[PROCESADOR DEFINITIVO] Se extrajeron un total de {len(eventos_enriquecidos)} eventos clínicos válidos."
    )

    # 3. Ordenar todos los eventos de la historia clínica por fecha.
    eventos_enriquecidos.sort(
        key=lambda x: x.get("fecha_evento", datetime.min), reverse=True
    )

    # 4. Construir el documento Markdown final a partir de los eventos ordenados.
    return _construir_documento_markdown(eventos_enriquecidos)


def _extraer_contenido_hc(data_original: dict) -> dict | str:
    """Función robusta para desenvolver el JSON anidado."""
    data = data_original
    for i in range(5):
        if isinstance(data, dict) and data.get("ok") and "entrys" in data:
            print(
                f"[PROCESADOR DEFINITIVO] Núcleo de datos encontrado en el intento {i+1}."
            )
            return data
        if isinstance(data, dict) and len(data) == 1:
            data = list(data.values())[0]
            if isinstance(data, str):
                try:
                    data = ast.literal_eval(
                        data.replace("null", "None")
                        .replace("true", "True")
                        .replace("false", "False")
                    )
                except (ValueError, SyntaxError):
                    return "Error: El string JSON interno no es válido."
        else:
            break
    return "Error: No se encontró la estructura de HC válida con 'ok' y 'entrys'."


def _procesar_entry(entry: dict, entry_index: int) -> list:
    """Procesa un único 'entry' y devuelve una lista de eventos planos."""
    eventos_del_entry = []
    fecha_entry = _parse_fecha(entry.get("dateHour", entry.get("date")))

    if not fecha_entry:
        print(
            f"[ADVERTENCIA] Omitiendo entry #{entry_index+1} (ID: {entry.get('id')}) por no poder parsear su fecha principal."
        )
        return []

    contexto = {
        "fecha_entry": fecha_entry,
        "servicio": entry.get("service", {}).get("description", "N/A"),
        "doctor": (
            f"{entry.get('doctor', {}).get('lastName', '')}, {entry.get('doctor', {}).get('firstName', '')}"
            if entry.get("doctor")
            else "N/A"
        ),
    }

    for struct in entry.get("structureWithRecords", []):
        eventos_del_entry.extend(_procesar_nodo_recursivo(struct, contexto, []))

    return eventos_del_entry


def _procesar_nodo_recursivo(nodo: dict, contexto: dict, path: list) -> list:
    """Recorre recursivamente la estructura de la HC, construyendo el path y formateando los datos."""
    eventos = []
    nombre_nodo = nodo.get("name", "Sección sin nombre")
    path_actual = path + [nombre_nodo]

    for rec in nodo.get("records", []):
        fecha_evento = _parse_fecha(rec.get("dateTime")) or contexto["fecha_entry"]
        descripcion = _formatear_descripcion(rec, path_actual)

        if descripcion:  # Solo añadir si la descripción no está vacía
            eventos.append(
                {
                    "path": path_actual,
                    "descripcion": descripcion,
                    "fecha_evento": fecha_evento,
                    **contexto,  # Añade todo el contexto (servicio, doctor, etc.)
                }
            )

    for child in nodo.get("childs", []):
        eventos.extend(_procesar_nodo_recursivo(child, contexto, path_actual))

    return eventos


def _construir_documento_markdown(eventos: list) -> str:
    """Construye el string final de Markdown."""
    if not eventos:
        return "# Historia Clínica del Paciente\n\n(No se encontraron eventos clínicos con contenido para mostrar)."

    md_parts = ["# Historia Clínica del Paciente\n"]
    fecha_actual_imprimida = None

    for ev in eventos:
        fecha_evento = ev["fecha_evento"]
        if fecha_evento.date() != fecha_actual_imprimida:
            fecha_actual_imprimida = fecha_evento.date()
            md_parts.append(
                f"\n## --- Día: {fecha_actual_imprimida.strftime('%A, %d de %B de %Y')} ---\n"
            )

        path_str = " / ".join(ev["path"])
        hora_str = fecha_evento.strftime("%H:%M")

        md_parts.append(f"\n### {path_str}\n")
        md_parts.append(
            f"**Fecha y Hora:** {hora_str} | **Servicio:** {ev['servicio']} | **Profesional:** {ev['doctor']}\n"
        )
        md_parts.append(f"> {ev['descripcion']}\n")

    return "".join(md_parts)


def _formatear_descripcion(rec: dict, path: list) -> Optional[str]:
    """Usa tu lógica de formateo inteligente."""
    name = path[-1].upper()
    path[0].upper() if path else ""
    valor_crudo = rec.get("value")
    obs = limpiar_texto(rec.get("observation", ""))

    if valor_crudo is None:
        return None

    if "SIGNOS VITALES" in [p.upper() for p in path]:
        unidad = {
            "TA": "mmHg",
            "FC": "LPM",
            "FR": "RPM",
            "T° AX": "°C",
            "SAT 0,21%": "%",
        }.get(path[-1], "")
        valor_formateado = limpiar_texto(valor_crudo)
        return (
            f"Registro de {path[-1]}: **{valor_formateado} {unidad}**."
            if valor_formateado
            else None
        )

    if name == "INDICACIÓN FARMACOLÓGICA" and isinstance(valor_crudo, dict):
        via = {1: "Oral", 3: "Subcutánea", 5: "Endovenosa", 9: "Inhalatoria"}.get(
            valor_crudo.get("VIA"), f"Vía ID {valor_crudo.get('VIA')}"
        )
        narrativa = f"Se indicó **{valor_crudo.get('MEDICAMENTO', 'N/A')}**. Dosis: {valor_crudo.get('DOSIS', 'N/A')}. Frecuencia: {valor_crudo.get('FRECUENCIA', 'N/A')}. Vía: {via}."
        if obs:
            narrativa += f" *Observación: {obs}*"
        return narrativa

    texto_limpio = limpiar_texto(valor_crudo)
    if not texto_limpio or texto_limpio == "..":
        return None

    return texto_limpio


def _parse_fecha(txt: Any) -> Optional[datetime]:
    """Versión más tolerante para parsear fechas."""
    if not isinstance(txt, str) or len(txt.strip()) < 8:
        return None
    txt = txt.strip()

    # Manejo de zonas horarias comunes como la tuya "-03:00"
    if ":" == txt[-3:-2]:
        txt = txt[:-3] + txt[-2:]

    formatos = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%d-%m-%Y %H:%M",
    ]
    for fmt in formatos:
        try:
            return datetime.strptime(txt, fmt)
        except (ValueError, TypeError):
            continue

    print(f"[ADVERTENCIA DE PARSEO] No se pudo parsear la fecha: '{txt}'")
    return None


def limpiar_texto(valor: any) -> str:
    """Limpia el texto de un registro."""
    if valor is None:
        return ""
    texto = " / ".join(map(str, valor)) if isinstance(valor, list) else str(valor)
    texto = re.sub(r"<[^>]+>", "", texto)
    texto = texto.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(texto.split()).strip()
