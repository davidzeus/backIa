import json
import logging  # Para mejor manejo de errores
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generar_texto_clinico_agrupado(json_data: dict) -> str:
    """
    Genera un texto clínico estructurado a partir de datos JSON,
    agrupando la información por 'group' y 'name' de la estructura
    y ordenando los eventos cronológicamente dentro de cada sección.
    """
    if not isinstance(json_data, dict) or "entrys" not in json_data:
        logging.error("Formato de JSON inválido o sin 'entrys'.")
        return "Error: Datos de entrada inválidos."

    # --- Extracción de Nombre/ID Paciente ---
    paciente_info = "Paciente desconocido"
    if json_data["entrys"]:
        first_entry = json_data["entrys"][0]
        patient_id = first_entry.get("patientId")
        # Intenta obtener nombre del doctor como placeholder si no hay más info
        doctor = first_entry.get("doctor", {})
        first_name = doctor.get("firstName", "").strip()
        last_name = doctor.get("lastName", "").strip()

        if first_name or last_name:
            # Podríamos querer un campo de paciente real, pero usamos doctor como fallback
            # paciente_info = f"Historia Clínica (Médico: {first_name} {last_name})"
            paciente_info = f"Paciente ID {patient_id}"  # Es más preciso usar el ID
        elif patient_id:
            paciente_info = f"Paciente ID {patient_id}"
    # --- Fin Extracción Paciente ---

    eventos = []
    for entry in json_data.get("entrys", []):
        # Usa la fecha del entry como fallback si el record no tiene una específica
        fecha_entry_fallback_str = entry.get("date", "")
        try:
            # Intenta parsear la fecha del entry para usarla si es necesario
            fecha_entry_fallback_dt = parsear_fecha(
                fecha_entry_fallback_str, is_entry_date=True
            )
        except ValueError:
            fecha_entry_fallback_dt = datetime.min  # O manejar de otra forma

        estructuras = entry.get("structureWithRecords", [])
        for estructura in estructuras:
            # Pasa la fecha parseada del entry como fallback
            eventos.extend(
                extraer_eventos_recursivo(estructura, fecha_entry_fallback_dt)
            )

    if not eventos:
        return f"{paciente_info}\n\nNo se encontraron eventos clínicos registrados."

    # Ordenar todos los eventos por fecha (usando datetime.min para errores)
    eventos.sort(key=lambda x: x["fecha"])

    # Agrupar por grupo y luego por nombre
    # Usamos una tupla ('Grupo', 'Nombre') como clave para mantener la asociación
    # O podemos anidar los defaultdict como hiciste
    grupos = defaultdict(lambda: defaultdict(list))
    for evento in eventos:
        grupo_key = evento["group"] if evento["group"] else "Sin Grupo Específico"
        nombre_key = evento["name"]
        # Formatear la descripción incluyendo la fecha del evento específico
        fecha_str = (
            evento["fecha"].strftime("%d/%m/%Y %H:%M")
            if evento["fecha"] != datetime.min
            else "Fecha desconocida"
        )
        grupos[grupo_key][nombre_key].append(f"- {evento['descripcion']} [{fecha_str}]")

    # --- Armado final del texto ---
    texto_final = f"{paciente_info}\n=======================================\n\n"

    # Ordenar grupos (opcional, pero puede mejorar legibilidad)
    # Se puede definir un orden específico si se desea
    grupos_ordenados = sorted(grupos.items())

    for group, secciones in grupos_ordenados:
        texto_final += f"--- Grupo: {group} ---\n"
        secciones_ordenadas = sorted(
            secciones.items()
        )  # Ordenar nombres dentro del grupo

        for name, items in secciones_ordenadas:
            texto_final += f"\n  {name}:\n"
            # Asegurarse de que cada item esté en una nueva línea
            texto_final += "\n".join(f"    {item}" for item in items) + "\n"

        texto_final += "\n"  # Espacio extra entre grupos

    return texto_final.strip()


def extraer_eventos_recursivo(estructura: dict, fecha_fallback: datetime) -> list:
    """
    Función recursiva para extraer eventos de una estructura y sus hijos.
    Intenta dar formato específico a la descripción según el 'name'.
    """
    eventos = []
    group = estructura.get("group")
    name = estructura.get("name", "Sin Nombre").strip()
    structure_id = estructura.get("id")  # Para depuración

    # Procesar records directos de esta estructura
    records = estructura.get("records", [])
    for record in records:
        # Prioriza la fecha del record, si no, usa el fallback (fecha del entry)
        fecha_registro_str = record.get("dateTime")
        try:
            fecha_evento = (
                parsear_fecha(fecha_registro_str)
                if fecha_registro_str
                else fecha_fallback
            )
        except ValueError as e:
            logging.warning(
                f"Error parseando fecha '{fecha_registro_str}' para record ID {record.get('id')} en estructura '{name}' (ID: {structure_id}). Usando fallback. Error: {e}"
            )
            fecha_evento = fecha_fallback  # O datetime.min si prefieres

        # Extrae la descripción de forma más inteligente
        descripcion = formatear_descripcion_evento(record, name)

        eventos.append(
            {
                "group": group,
                "name": name,
                "descripcion": descripcion,
                "fecha": fecha_evento,
                "structure_id": structure_id,  # Para referencia
                "record_id": record.get("id"),
            }
        )

    # Procesar childs recursivamente
    childs = estructura.get("childs", [])
    for child_struct in childs:
        # Los hijos usan la misma fecha de fallback que el padre (la del entry)
        eventos.extend(extraer_eventos_recursivo(child_struct, fecha_fallback))

    return eventos


def formatear_descripcion_evento(record: dict, structure_name: str) -> str:
    """
    Formatea la descripción del evento basándose en el nombre de la estructura
    y la información disponible en el record.
    """
    value = record.get("value")
    value_id = record.get("valueId")
    observation = record.get("observation", "")
    resource = record.get("resource")

    # Obtén el valor crudo de tableRef (puede ser string o None)
    table_ref_raw = record.get("tableRef")

    # Conviértelo a minúsculas SOLO si es un string, de lo contrario usa ''
    table_ref = table_ref_raw.lower() if isinstance(table_ref_raw, str) else ""
    # table_ref = record.get('tableRef', '').lower()

    descripcion = "N/D"  # Valor por defecto

    try:
        # --- Lógica específica por Nombre de Estructura ---
        if "DIAGNÓSTICO" in structure_name.upper():
            diag_text = value
            code_system = "ID"  # Default si no hay tabla o resource
            code = value_id if value_id else "N/A"

            if isinstance(resource, list) and resource:  # SNOMED puede venir en lista
                resource = resource[0]

            if isinstance(resource, dict):
                diag_text = resource.get(
                    "description", value
                )  # Prioriza descripción del resource
                if "snomed" in table_ref:
                    code_system = "SNOMED-CT"
                    code = resource.get("id", value_id)
                elif "cie10" in table_ref:
                    code_system = "CIE-10"
                    code = resource.get("code", value_id)  # CIE10 usa 'code'
                else:
                    code = resource.get("id", code)  # Otro tipo de resource

            elif isinstance(value, str):  # Si no hay resource, usa el value
                diag_text = value

            descripcion = f"{diag_text} ({code_system}: {code})"

        elif "MEDICAMENTO" in structure_name.upper():
            med_desc = "Medicamento no especificado"
            med_cant = ""
            med_pres = ""
            med_detail = ""

            # Caso 1: Value es un diccionario (formato nuevo o viejo)
            if isinstance(value, dict):
                med_desc = value.get("DESCRIPCION", "N/A")
                med_cant = f"Cantidad: {value.get('CANTIDAD', 'N/A')}"
                med_pres = f"Presentación: {value.get('PRESENTACIÓN', 'N/A')}"
                # A veces la descripción ya incluye detalles, resource puede añadir más
                if isinstance(resource, list) and resource:
                    resource = resource[0]
                if isinstance(resource, dict):
                    med_desc = resource.get(
                        "description", med_desc
                    )  # Prioriza resource si mejora
                    # Podríamos añadir 'tradeName' si es útil: resource.get('tradeName')

            # Caso 2: Value es string (podría ser nombre simple)
            elif isinstance(value, str):
                med_desc = value
                if isinstance(resource, list) and resource:
                    resource = resource[0]
                if isinstance(resource, dict):
                    med_desc = resource.get(
                        "description", med_desc
                    )  # Mejora con resource
                    med_pres = f"Forma: {resource.get('pharmaceuticalForm', 'N/A')}"
                    # Cantidad no está clara en este caso, podría estar en obs?

            med_detail = f"{med_cant} {med_pres}".strip()
            descripcion = f"{med_desc}"
            if med_detail:
                descripcion += f" ({med_detail})"

        elif "PRACTICA" in structure_name.upper():  # Órdenes Médicas
            prac_text = value
            code_system = "ID"
            code = value_id if value_id else "N/A"

            if isinstance(resource, list) and resource:
                resource = resource[0]
            if isinstance(resource, dict):
                prac_text = resource.get("description", value)
                if "snomed" in table_ref:
                    code_system = "SNOMED-CT"
                    code = resource.get("id", value_id)
                else:  # Otro código
                    code = resource.get("id", code)
            elif isinstance(value, str):
                prac_text = value

            descripcion = f"{prac_text} ({code_system}: {code})"

        elif "PRESCRIPCIÓN" in structure_name.upper():
            descripcion = (
                f"Indicación: {value}" if value else "Indicación no especificada"
            )

        elif "ESPECIALIDAD" in structure_name.upper():
            descripcion = f"{value}"

        elif "FECHA DE EMISIÓN" in structure_name.upper():
            descripcion = f"Emitido: {value}"  # La fecha ya se añade fuera

        # --- Fallback Genérico ---
        else:
            if isinstance(value, dict):
                # Intenta claves comunes o vuelca el JSON
                descripcion = (
                    value.get("DESCRIPCION")
                    or value.get("description")
                    or json.dumps(value, ensure_ascii=False, indent=2)
                )
            elif value is not None:
                descripcion = str(value)
            else:
                descripcion = "Valor no proporcionado"

        # Añadir observación si existe y no es redundante
        if observation and observation.strip():
            # Evita añadir 'Obs: ' si la descripción ya es la observación (caso raro)
            if str(observation) != descripcion:
                descripcion += f" (Obs: {observation})"

    except Exception as e:
        logging.error(
            f"Error formateando descripción para record ID {record.get('id')} en estructura '{structure_name}': {e}",
            exc_info=True,
        )
        descripcion = f"Error al procesar: {value}"  # Indica error en la salida

    # Asegurarse de que la descripción no sea excesivamente larga (opcional)
    # max_len = 200
    # if len(descripcion) > max_len:
    #     descripcion = descripcion[:max_len-3] + "..."

    return descripcion.strip()


def parsear_fecha(fecha_str: str, is_entry_date=False) -> datetime:
    """Parsea una cadena de fecha en varios formatos comunes."""
    if not fecha_str:
        raise ValueError("Cadena de fecha vacía")

    formatos_posibles = [
        "%Y-%m-%dT%H:%M:%S.%f%z",  # Formato ISO 8601 con offset y microsegundos
        "%Y-%m-%dT%H:%M:%S%z",  # Formato ISO 8601 con offset sin microsegundos
        "%Y-%m-%dT%H:%M:%S",  # Formato ISO 8601 sin offset ni microsegundos
        "%d-%m-%Y %H:%M:%S",  # Formato de tu JSON 'entry.date'
        "%d/%m/%Y",  # Formato común de fecha (si viene solo fecha)
        # Añade más formatos si son necesarios
    ]

    # A veces el offset tiene ':' que Python < 3.7 no maneja bien, o es 'Z'
    fecha_str_norm = fecha_str.replace("Z", "+00:00")
    if (
        "." in fecha_str_norm and "+" in fecha_str_norm.split(".")[-1]
    ):  # Maneja microsegundos antes de offset
        parts = fecha_str_norm.split(".")
        if len(parts) == 2:
            fecha_str_norm = (
                parts[0]
                + "."
                + parts[1][:6]
                + fecha_str_norm[len(parts[0]) + len(parts[1]) + 1 :]
            )  # Limita a 6 digitos microseg

    for fmt in formatos_posibles:
        try:
            # Manejo especial para offsets con ':' si es necesario (Python >= 3.7 lo maneja bien)
            return datetime.strptime(fecha_str_norm, fmt)
        except ValueError:
            continue

    # Si es la fecha del entry y falla, quizás no tenga hora
    if is_entry_date:
        try:
            return datetime.strptime(fecha_str_norm, "%d-%m-%Y")  # Intenta solo fecha
        except ValueError:
            pass  # Falla si no es solo fecha

    raise ValueError(f"Formato de fecha no reconocido: {fecha_str}")


# --- Para probar (descomenta y ajusta la ruta) ---
# if __name__ == "__main__":
#     json_file_path = 'historia_paciente.json' # Asegúrate que exista
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             datos_hc = json.load(f)
#         print("📄 JSON cargado.")
#         texto_resultado = generar_texto_clinico_agrupado(datos_hc)
#         print("\n--- Texto Clínico Agrupado Generado ---")
#         print(texto_resultado)
#         print("--------------------------------------")
#     except FileNotFoundError:
#         print(f"❌ Error: Archivo no encontrado en {json_file_path}")
#     except json.JSONDecodeError:
#         print(f"❌ Error: El archivo {json_file_path} no es un JSON válido.")
#     except Exception as e:
#         print(f"❌ Ocurrió un error inesperado: {e}")
