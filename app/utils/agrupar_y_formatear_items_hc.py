# app/utils/agrupar_y_formatear_items_hc.py
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytz

from app.config.time import HOSPITAL_TZ

logging.basicConfig(level=logging.INFO)


# Helpers nuevos #################################################################################
def _first_non_empty(d: dict, keys: list):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", []):
            return v
    return None


def _normalizar_grupo(group_raw: str, hhg: str):
    def norm(g):
        if not g:
            return None
        g = g.strip().lower()
        if g.startswith("cm;"):  # “CM;HISTORIA CLINICA”, “CM;EVOLUCIÓN”, etc.
            return "camas_cm"
        return g.replace(" ", "_")

    return norm(group_raw) or norm(hhg)


"""
def _inferir_tipo_atencion(grupo_norm: str, procedure_number):
    if grupo_norm == "camas_cm" or procedure_number:
        return "internacion"
    return "ambulatorio"

"""

###################################################################################


# --- SIN CAMBIOS EN ESTAS FUNCIONES ---
def generar_linea_de_tiempo_clinica(json_data: dict) -> List[Dict[str, Any]]:
    # ... (código sin cambios)
    try:
        if not isinstance(json_data, dict) or "entrys" not in json_data:
            logging.warning("JSON inválido o sin clave 'entrys'.")
            return []
        items_list = _extraer_estructura_plana(json_data.get("entrys", []))
        arbol = _construir_arbol_desde_lista_plana(items_list)
        eventos = []
        for nodo_raiz in arbol:
            _extraer_eventos_recursivo(nodo_raiz, [], eventos)
        eventos.sort(key=lambda x: x.get("fecha_evento", datetime.min))
        eventos_unicos = _deduplicar_eventos(eventos)
        return eventos_unicos
    except Exception:
        logging.exception("Fallo al generar la línea de tiempo clínica")
        return []


def _extraer_estructura_plana(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    elementos = []
    for entry in entries:
        fallback_fecha = _parse_fecha(entry.get("date"))
        doctor = entry.get("doctor") or {}
        servicio = entry.get("service") or {}
        group_raw = entry.get("group")
        hhg = entry.get("healthHistoryGroup")
        for struct in entry.get("structureWithRecords", []):
            struct["_doctor"] = doctor
            struct["_servicio"] = servicio
            struct["_fecha"] = fallback_fecha
            struct["_group"] = group_raw
            struct["_hh_group"] = hhg
            elementos.append(struct)
    return elementos


def _construir_arbol_desde_lista_plana(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    mapa = {item["id"]: item for item in items if "id" in item}
    arbol = []
    for item in items:
        item.setdefault("hijos", [])
        padre_id = item.get("fatherId")
        if padre_id and padre_id in mapa:
            mapa[padre_id].setdefault("hijos", []).append(item)
        else:
            arbol.append(item)
    return arbol


# --- FIN DE SECCIÓN SIN CAMBIOS ---


def _extraer_eventos_recursivo(
    nodo: Dict[str, Any],
    ruta_actual: List[str],
    lista_eventos: List[Dict[str, Any]],
    servicio_padre: Optional[Dict[str, Any]] = None,
    doctor_padre: Optional[Dict[str, Any]] = None,
):
    # ➜ Heredamos servicio/doctor si el nodo no lo trae
    servicio_actual = nodo.get("_servicio") or servicio_padre
    doctor_actual = nodo.get("_doctor") or doctor_padre
    group_raw = nodo.get("_group")
    hhg = nodo.get("_hh_group")
    group_raw_entry = nodo.get("_group")

    nueva_ruta = ruta_actual + [nodo.get("name", "SIN_NOMBRE")]

    if nodo.get("records"):
        for rec in nodo["records"]:
            fecha_record = _parse_fecha(rec.get("dateTime")) or nodo.get("_fecha")
            if not fecha_record:
                continue

            texto_record = _formatear_record_individual(nodo, rec)
            if not texto_record:
                continue

            hash_contenido = hashlib.md5(
                (str(rec.get("value", "")) + str(rec.get("observation", ""))).encode()
            ).hexdigest()

            seccion_completa = " / ".join(nueva_ruta)
            seccion_raiz = nueva_ruta[0] if nueva_ruta else "SIN_SECCION"
            procedure_number = rec.get("procedureNumber")

            # 🔻 Fallback de group: entry -> struct -> record.estructure.group
            group_raw_struct = nodo.get("group")
            group_raw_record = (rec.get("estructure") or {}).get("group")
            group_raw = group_raw_entry or group_raw_struct or group_raw_record

            evento = {
                "ruta_seccion": seccion_completa,
                "seccion_raiz": seccion_raiz,
                "fecha_evento": fecha_record,
                "texto_formateado": texto_record,
                "hash_contenido": hash_contenido,
                "servicio_id": servicio_actual.get("id") if servicio_actual else None,
                "servicio": (
                    servicio_actual.get("description") if servicio_actual else None
                ),
                "doctor_full": (
                    f"{doctor_actual.get('lastName', '')}, {doctor_actual.get('FirstName', '') or doctor_actual.get('firstName', '')}".strip(
                        ", "
                    )
                    if doctor_actual
                    else None
                ),
                "group": group_raw,  #
                "healthHistoryGroup": hhg,
                "procedureNumber": procedure_number,
            }
            lista_eventos.append(evento)


# --------------------
def _deduplicar_eventos(eventos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hashes_vistos = set()
    eventos_unicos = []
    for ev in eventos:
        if ev["hash_contenido"] not in hashes_vistos:
            eventos_unicos.append(ev)
            hashes_vistos.add(ev["hash_contenido"])
    return eventos_unicos


# Diccionario para decodificar valores. ¡Esto es clave para la calidad!
VIA_DECODE = {
    1: "Vía Oral (VO)",
    3: "Subcutánea (SC)",
    5: "Endovenosa (EV)",
    9: "Inhalatoria",
}


def _limpiar_texto_deprecated(texto: Any) -> str:
    """Elimina etiquetas HTML/XML y normaliza espacios. Acepta cualquier tipo para ser más robusto."""
    if texto is None:
        return ""
    texto_str = str(texto)
    # Eliminar etiquetas como <DESCRIPCION>
    texto_limpio = re.sub(r"<[^>]+>", "", texto_str)
    # Reemplazar saltos de línea y tabulaciones por un espacio y eliminar espacios múltiples
    texto_limpio = re.sub(r"[\r\n\t]+", " ", texto_limpio)
    return re.sub(r"\s+", " ", texto_limpio).strip()

def _limpiar_texto(texto: Any) -> str:
    """
    Elimina HTML pero intenta preservar estructura visual básica (saltos de línea).
    """
    if texto is None:
        return ""
    texto_str = str(texto)
    
    # 1. Reemplazar etiquetas de bloque (<br>, </p>, </div>) por saltos de línea reales
    # Esto ayuda al LLM a entender que son items distintos.
    texto_str = re.sub(r"<(br|p|div|tr)[^>]*>", "\n", texto_str, flags=re.IGNORECASE)
    texto_str = re.sub(r"</(p|div|tr)>", "\n", texto_str, flags=re.IGNORECASE)

    # 2. Eliminar el resto de etiquetas HTML (<span...>, <PRE...>, <b...>)
    texto_limpio = re.sub(r"<[^>]+>", "", texto_str)
    
    # 3. Normalizar espacios, pero respetando el salto de línea (\n)
    # Quitamos espacios repetidos horizonalmente, y limitamos saltos verticales a máx 2.
    lines = []
    for line in texto_limpio.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
            
    return "\n".join(lines)


def _formatear_record_individual(nodo: Dict[str, Any], rec: Dict[str, Any]) -> str:
    """
    Función refinada para formatear un registro individual, con lógica específica para diferentes tipos de datos.
    """
    name = nodo.get("name", "")
    val = rec.get("value")
    obs = _limpiar_texto(rec.get("observation", ""))

    texto_final = ""

    # ✅ MEJORA DE DEPURACIÓN: Lógica específica y más inteligente para cada tipo de dato.
    if isinstance(val, dict):
        # Lógica especial para INDICACIONES FARMACOLÓGICAS
        if "MEDICAMENTO" in val:
            partes_texto = [f"{val['MEDICAMENTO']}"]
            if val.get("DOSIS") and str(val["DOSIS"]).strip() != ".":
                partes_texto.append(f"Dosis: {_limpiar_texto(val['DOSIS'])}")
            if val.get("FRECUENCIA") and str(val["FRECUENCIA"]).strip() != ".":
                partes_texto.append(f"Frecuencia: {_limpiar_texto(val['FRECUENCIA'])}")
            if val.get("VIA"):
                via_texto = VIA_DECODE.get(val["VIA"], f"Código Vía {val['VIA']}")
                partes_texto.append(f"Vía: {via_texto}")
            texto_final = f"- {name}: " + ", ".join(partes_texto)
        else:  # Formato genérico para otros diccionarios (como INDICACION GENERAL)
            # dict_items = [f"{v}" for k, v in val.items() if v and str(v).strip()]
            # texto_final = f"- {name}: " + ", ".join(dict_items)
            # --- INICIO DEL CAMBIO ---
            dict_items = []
            for k, v in val.items():
                # Verificamos que tenga contenido
                if v and str(v).strip():
                    # ¡AQUÍ ESTÁ LA CLAVE! Pasamos 'v' por _limpiar_texto
                    val_limpio = _limpiar_texto(v)
                    if val_limpio:
                        dict_items.append(val_limpio)
            
            texto_final = f"- {name}: " + ", ".join(dict_items)
            
    elif isinstance(val, list):
        # Limpia y une los elementos de la lista
        val_limpio = " / ".join(
            _limpiar_texto(item) for item in val if _limpiar_texto(item)
        )
        if not val_limpio:
            return ""
        texto_final = f"- {name}: {val_limpio}"

    else:
        # Lógica para texto plano
        val_limpio = _limpiar_texto(val)
        if not val_limpio:
            return ""  # Ignora registros con valor vacío
        texto_final = f"- {name}: {val_limpio}"

    if obs and obs.lower() not in [
        "-",
        "s",
        "1",
    ]:  # Filtra observaciones no informativas
        texto_final += f" (Obs: {obs})"

    # ✅ MEJORA DE DEPURACIÓN: Lógica de fechas más limpia y robusta.
    start_date_str = rec.get("startDate")
    end_date_str = rec.get("endDate")
    if start_date_str:
        dt_inicio = _parse_fecha(start_date_str)
        fecha_inicio_fmt = (
            dt_inicio.strftime("%d-%m-%Y") if dt_inicio else start_date_str
        )

        periodo_texto = " (ACTIVA)"  # Por defecto, la indicación está activa
        if end_date_str:
            dt_fin = _parse_fecha(end_date_str)
            if dt_fin:
                fecha_fin_fmt = dt_fin.strftime("%d-%m-%Y")
                periodo_texto = f", Finalizada: {fecha_fin_fmt}"
            else:
                periodo_texto = f", Finalizada: {end_date_str}"

        texto_final += f" [Inició: {fecha_inicio_fmt}{periodo_texto}]"

    return texto_final.strip()


def _parse_fecha(txt: Any) -> Optional[datetime]:
    # ... (código sin cambios)
    if not txt or not isinstance(txt, str):
        return None
    txt = txt.strip()
    formatos = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]
    for fmt in formatos:
        try:
            dt = datetime.strptime(txt, fmt)
            if dt.tzinfo is not None:
                dt = dt.astimezone(pytz.UTC).replace(tzinfo=None)
            return dt
        except (ValueError, TypeError):
            continue
    logging.debug(f"No se pudo parsear la fecha: '{txt}'")
    return None


###########################
def _canon_fecha_y_ts(fecha_obj) -> tuple[str, int]:
    """
    Devuelve:
      - fecha_iso: 'YYYY-MM-DD' **en TZ del hospital** (día local)
      - fecha_ts : epoch **de la medianoche local** convertida a UTC
    Acepta datetime o string; usa _parse_fecha(...) para parseo.
    """
    dt = None
    if hasattr(fecha_obj, "year"):
        dt = fecha_obj  # datetime ya parseado
    elif isinstance(fecha_obj, str):
        dt = _parse_fecha(fecha_obj)

    if dt is None:
        dt = datetime.utcnow()  # fallback

    # 1) Asegurar que 'dt' esté en TZ del hospital (si viene naive, lo localizamos)
    if dt.tzinfo is None:
        dt_local = dt.replace(tzinfo=HOSPITAL_TZ)
    else:
        dt_local = dt.astimezone(HOSPITAL_TZ)

    # 2) Truncar a medianoche **local**
    dt_local_mid = dt_local.replace(hour=0, minute=0, second=0, microsecond=0)

    # 3) Calcular epoch de esa medianoche local convertido a UTC
    dt_utc = dt_local_mid.astimezone(timezone.utc)
    fecha_ts = int(dt_utc.timestamp())

    # 4) La 'fecha' visible debe reflejar el día local del hospital
    fecha_iso = dt_local_mid.strftime("%Y-%m-%d")
    return fecha_iso, fecha_ts


###############################
def inspeccionar_linea_de_tiempo_clinica(json_data: dict):
    """
    Función de depuración que imprime la línea de tiempo formateada en consola.
    """
    linea_de_tiempo = generar_linea_de_tiempo_clinica(json_data)

    print("\n" + "=" * 40)
    print("|  INSPECCIÓN DE LÍNEA DE TIEMPO CLÍNICA |")
    print(f"|  Eventos encontrados: {len(linea_de_tiempo)}             |")
    print("=" * 40 + "\n")

    if not linea_de_tiempo:
        print("No se generaron eventos para la línea de tiempo.")
        return

    for i, evento in enumerate(linea_de_tiempo):
        print(f"--- [Evento {i+1} de {len(linea_de_tiempo)}] ---")
        print(f"  [Fecha]:  {evento['fecha_evento']}")
        print(f"  [Ruta]:   {evento['ruta_seccion']}")
        print(f"  [Doctor]: {evento.get('doctor', 'N/A')}")
        print(f"  [Texto Formateado]: {evento['texto_formateado']}")
        print("-" * (22 + len(str(i + 1)) + len(str(len(linea_de_tiempo)))))

    print("\n" + "=" * 40)
    print("|          FIN DE LA INSPECCIÓN          |")
    print("=" * 40 + "\n")
