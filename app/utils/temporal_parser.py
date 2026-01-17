# app/utils/temporal_parser.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

from app.config.time import HOSPITAL_TZ
from app.schemas.temporal import TemporalWindow

# --- números en español (básico) ---
SPANISH_NUM = {
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "dieciséis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21, "veintiún": 21, "veintidos": 22, "veintidós": 22,
    "veintitres": 23, "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintiséis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29, "treinta": 30, "treinta y uno": 31, "treinta y un": 31,
    "treinta y dos": 32, "treinta y tres": 33, "treinta y cuatro": 34,
    "treinta y cinco": 35, "treinta y seis": 36, "treinta y siete": 37,
    "treinta y ocho": 38, "treinta y nueve": 39, "cuarenta": 40,
    "cuarenta y uno": 41, "cuarenta y dos": 42, "cuarenta y tres": 43,
    "cuarenta y cuatro": 44, "cuarenta y cinco": 45, "cuarenta y seis": 46,
    "cuarenta y siete": 47, "cuarenta y ocho": 48, "cuarenta y nueve": 49,
    "cincuenta": 50, "cincuenta y uno": 51, "cincuenta y dos": 52,
    "cincuenta y tres": 53, "cincuenta y cuatro": 54, "cincuenta y cinco": 55,
    "cincuenta y seis": 56, "cincuenta y siete": 57, "cincuenta y ocho": 58,
    "cincuenta y nueve": 59, "sesenta": 60, "sesenta y uno": 61,
    "sesenta y dos": 62, "sesenta y tres": 63, "sesenta y cuatro": 64,
    "sesenta y cinco": 65, "sesenta y seis": 66, "sesenta y siete": 67,
    "sesenta y ocho": 68, "sesenta y nueve": 69, "setenta": 70,
    "setenta y uno": 71, "setenta y dos": 72, "setenta y tres": 73,
    "setenta y cuatro": 74, "setenta y cinco": 75, "setenta y seis": 76,
    "setenta y siete": 77, "setenta y ocho": 78, "setenta y nueve": 79,
    "ochenta": 80, "ochenta y uno": 81, "ochenta y dos": 82,
    "ochenta y tres": 83, "ochenta y cuatro": 84, "ochenta y cinco": 85,
    "ochenta y seis": 86, "ochenta y siete": 87, "ochenta y ocho": 88,
    "ochenta y nueve": 89, "noventa": 90, "noventa y uno": 91,
    "noventa y dos": 92, "noventa y tres": 93, "noventa y cuatro": 94,
    "noventa y cinco": 95, "noventa y seis": 96, "noventa y siete": 97,
    "noventa y ocho": 98, "noventa y nueve": 99, "cien": 100,
}


def parse_spanish_int(s: str) -> Optional[int]:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    return SPANISH_NUM.get(s)


# --- meses en español ---
MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def _ensure_tz(dt: datetime) -> datetime:
    return (
        dt.replace(tzinfo=HOSPITAL_TZ)
        if dt.tzinfo is None
        else dt.astimezone(HOSPITAL_TZ)
    )


def _day_bounds(dt: datetime) -> Tuple[datetime, datetime]:
    dt = _ensure_tz(dt)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)  # fin exclusivo


def _year_bounds(year: int) -> Tuple[datetime, datetime]:
    start = _ensure_tz(datetime(year, 1, 1))
    return start, _ensure_tz(datetime(year + 1, 1, 1))


def _month_bounds(year: int, month: int) -> Tuple[datetime, datetime]:
    start = _ensure_tz(datetime(year, month, 1))
    if month == 12:
        end = _ensure_tz(datetime(year + 1, 1, 1))
    else:
        end = _ensure_tz(datetime(year, month + 1, 1))
    return start, end


def _to_iso(dt: datetime) -> str:
    return _ensure_tz(dt).strftime("%Y-%m-%d %H:%M:%S")


############
def _resolve_partial_day_month(day: int, month: int, now: datetime) -> datetime:
    """
    Resuelve dd/mm. Prioriza el PASADO inmediato para contextos clínicos.
    
    Lógica:
    1. Crea la fecha con el año actual.
    2. Si esa fecha cae en el FUTURO respecto a 'now', asume que se refiere 
       al año anterior (historia clínica).
    
    Ejemplo (Hoy: 20/12/2025):
    - Input "05/01" -> 05/01/2025 (Pasado, OK)
    - Input "25/12" -> 25/12/2024 (El de 2025 es futuro, devolvemos 2024)
    """
    try:
        # Intentamos con el año actual
        candidate = _ensure_tz(datetime(now.year, month, day))
    except ValueError:
        # Manejo especial bisiestos: Si piden 29/02 y este año no es bisiesto
        # probamos el año anterior (o el actual, fallará igual si no lo es)
        # Para simplificar, si falla construcción, asumimos año pasado directo si fuera válido
        # o retornamos error. Aquí un fallback simple:
        return _ensure_tz(datetime(now.year - 1, month, day))

    # Si la fecha construida es mayor a 'now' (futuro), restamos un año.
    # En medicina, rara vez buscamos registros del futuro.
    if candidate > now:
        return _ensure_tz(datetime(now.year - 1, month, day))
    
    return candidate

def _resolve_partial_day_month_deprecated(day: int, month: int, now: datetime) -> datetime:
    """
    Resuelve año para dd/mm sin año: toma el año cuya fecha esté más cercana a 'now'
    (year_now, year_now-1, year_now+1), minimizando distancia absoluta en días.
    Sin acceso a densidad de eventos (eso lo puede mejorar el servicio consultando Qdrant).
    """
    candidates = [
        _ensure_tz(datetime(now.year - 1, month, day)),
        _ensure_tz(datetime(now.year, month, day)),
        _ensure_tz(datetime(now.year + 1, month, day)),
    ]
    return min(candidates, key=lambda d: abs((d - now).days))


def _week_bounds(dt: datetime) -> Tuple[datetime, datetime]:
    dt = _ensure_tz(dt)
    # Lunes como inicio de semana
    start = (dt - timedelta(days=(dt.weekday()))).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start, start + timedelta(days=7)


##############
def _parse_generic_date(date_str: str, now: datetime) -> Optional[datetime]:
    """
    Parsea fechas DD/MM/YYYY o DD/MM (usando lógica inteligente de año).
    """
    # Intenta DD/MM/YYYY
    match_full = re.match(r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})", date_str)
    if match_full:
        return _ensure_tz(datetime(int(match_full.group(3)), int(match_full.group(2)), int(match_full.group(1))))
    
    # Intenta DD/MM (usa lógica de año inteligente)
    match_short = re.match(r"(\d{1,2})[\/\-](\d{1,2})", date_str)
    if match_short:
        return _resolve_partial_day_month(int(match_short.group(1)), int(match_short.group(2)), now)
        
    return None



@dataclass
class Parsed:
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    which: Optional[str] = None  # 'last'|'first'|None
    count: Optional[int] = 1
    intent_type: str = "NONE"  # DAY|RANGE|LAST_N|NONE


def parse_temporal_intent(
    question: str, now: Optional[datetime] = None, tz: Optional[str] = None
) -> Optional[TemporalWindow]:
    """
    Parser por reglas (sin LLM).
    Soporta:
      - “este año/mes”, “año 2025”, “2025”, “julio 2025”, “12/03/2025”, “2025-03-12”
      - “entre el 5 y el 20 de mayo (de 2025)”
      - superlativos: “los 3 últimos”, “las dos primeras”
    Devuelve TemporalWindow con fin exclusivo.
    """

# --- Regex Patterns for Dates (Reusable) ---
# DD/MM/YYYY or DD-MM-YYYY
DATE_DMY_REGEX = r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b"
# YYYY-MM-DD
DATE_YMD_REGEX = r"\b(\d{4})[\/\-](\d{2})[\/\-](\d{2})\b"
# DD/MM (current year inferred later, but regex catches it)
DATE_DM_REGEX = r"\b(\d{1,2})[\/\-](\d{1,2})\b"
# Written months: "17 de noviembre de 2023", "17 de noviembre", "noviembre 2023"
# Note: Complex to put in single regex, handled by existing logic or specific new ones if needed.
# For relative strict parsing, we will focus on numeric dates first or clear "day de month" patterns.

def _extract_date_from_match(text_after_keyword: str, now: datetime) -> Optional[datetime]:
    """
    Intenta extraer una feha estricta del texto que sigue a una keyword relativa.
    Soporta: DD/MM/YYYY, YYYY-MM-DD.
    """
    text = text_after_keyword.strip()
    
    # 1. DD/MM/YYYY
    m = re.match(DATE_DMY_REGEX, text)
    if m:
        d, m_, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _ensure_tz(datetime(y, m_, d))
        
    # 2. YYYY-MM-DD
    m = re.match(DATE_YMD_REGEX, text)
    if m:
        y, m_, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _ensure_tz(datetime(y, m_, d))

    # 3. DD/MM (asume año actual o gestión de año faltante si se integrara, aquí usaremos lógica simple local o reusamos _resolve)
    # Para ser estricto en "tras...", mejor pedimos año o asumimos actual con cuidado. 
    # Usaremos _resolve_partial_day_month si matchea.
    m = re.match(DATE_DM_REGEX, text)
    if m:
        d, m_ = int(m.group(1)), int(m.group(2))
        return _resolve_partial_day_month(d, m_, now)

    return None

def parse_temporal_intent(
    question: str, now: Optional[datetime] = None, tz: Optional[str] = None
) -> Optional[TemporalWindow]:
    """
    Parser por reglas (sin LLM).
    Soporta:
      - “este año/mes”, “año 2025”, “2025”, “julio 2025”, “12/03/2025”, “2025-03-12”
      - “entre el 5 y el 20 de mayo (de 2025)”
      - superlativos: “los 3 últimos”, “las dos primeras”
      - Relativos: "tras/después del 17/11/2023", "antes del 01/01/2024"
    Devuelve TemporalWindow con fin exclusivo.
    """
    if not question:
        return None
    q = question.lower().strip()
    now = _ensure_tz(now or datetime.now(HOSPITAL_TZ))
    year_now, month_now = now.year, now.month

    p = Parsed()

    # --- superlativos: últimos/primeros N ---
    m = re.search(
        r"\b(últim\w+|primer\w+)\s+(?:los|las|el|la)?\s*([0-9]+|[a-záéíóú]+)?", q
    )
    if m:
        which_span = m.group(1)
        n_span = m.group(2) or ""
        p.which = "last" if which_span.startswith("últim") else "first"
        n = parse_spanish_int(n_span) if n_span else 1
        p.count = max(1, n or 1)
        p.intent_type = "LAST_N"
        
    # --- RELATIVOS: DESPUES / ANTES ---
    # Estrategia: Buscar keyword y VERIFICAR si lo que sigue es una fecha.
    # Si no es fecha, IGNORAR (para no romper "tras el evento X").
    
    # AFTER / DESPUES
    # "tras ...", "despues de ...", "posterior a ...", "a partir de ..."
    # Regex captura lo que sigue para validarlo
    if p.start is None and p.end is None and not p.which:
        after_keywords = r"(?:tras|despu[eé]s\s+(?:del?|de\s+la)|posterior\s+a(?:l|(?:\s+la))?|a\s+partir\s+(?:del?|de\s+la)|luego\s+(?:del?|de\s+la))\s+(.*)"
        m_after = re.search(after_keywords, q)
        if m_after:
            potential_date_str = m_after.group(1)
            # Intentar extraer fecha del inicio del string capturado
            extracted_dt = _extract_date_from_match(potential_date_str, now)
            if extracted_dt:
                # Caso "tras 17/11/2023" -> Start: 17/11/2023, End: NOW (o futuro)
                # Asumimos hasta 'now' para contexto histórico, o un poco más si se desea.
                # Generalmente "qué pasó tras X" implica hasta el presente.
                p.start = extracted_dt
                p.end = now + timedelta(days=1) # Un poco de margen futuro o 'now' exacto. Usaremos NOW+buffer pequeo o NOW.
                # Si queremos "hasta hoy inclusive", end debe ser mañana 00:00 o NOW. 
                # Usemos NOW para ser seguros, o mejor, hasta el final del día actual.
                p.end = _ensure_tz(datetime(now.year, now.month, now.day)) + timedelta(days=1)
                p.intent_type = "RANGE"

    # BEFORE / ANTES
    # "antes de ...", "previo a ..."
    if p.start is None and p.end is None and not p.which:
        before_keywords = r"(?:antes\s+(?:del?|de\s+la)|previo\s+a(?:l|(?:\s+la))?|anterior\s+a(?:l|(?:\s+la))?)\s+(.*)"
        m_before = re.search(before_keywords, q)
        if m_before:
            potential_date_str = m_before.group(1)
            extracted_dt = _extract_date_from_match(potential_date_str, now)
            if extracted_dt:
                # Caso "antes del 17/11/2023" -> Start: None (past), End: 17/11/2023
                # Ojo: "antes de" suele excluir la fecha límite o incluirla?
                # "antes de navidad" -> hasta el momento previo.
                # Tomaremos extracted_dt como el END exclusivo.
                p.start = None # Open start
                p.end = extracted_dt
                p.intent_type = "RANGE"

    # --- deícticos ---
    # --- “este año/mes”, “mes pasado”, “hoy”, “ayer”, “última semana” ---
    if p.start is None and p.end is None and not p.which:
        if "este año" in q:
            p.start, p.end = _year_bounds(year_now)
            p.intent_type = "RANGE"
        elif "este mes" in q:
            p.start, p.end = _month_bounds(year_now, month_now)
            p.intent_type = "RANGE"
        elif "mes pasado" in q:
            prev_y, prev_m = (
                (year_now - 1, 12) if month_now == 1 else (year_now, month_now - 1)
            )
            p.start, p.end = _month_bounds(prev_y, prev_m)
            p.intent_type = "RANGE"
        elif "esta semana" in q:  # NEW
            p.start, p.end = _week_bounds(now)
            p.intent_type = "RANGE"
        elif "semana pasada" in q:  # NEW
            start, end = _week_bounds(now - timedelta(days=7))
            p.start, p.end, p.intent_type = start, end, "RANGE"
        elif "hoy" in q:
            p.start, p.end = _day_bounds(now)
            p.intent_type = "DAY"
        elif "ayer" in q:
            p.start, p.end = _day_bounds(now - timedelta(days=1))
            p.intent_type = "DAY"
        elif "anteayer" in q or "antes de ayer" in q or "ante-ayer" in q:  # NEW
            p.start, p.end = _day_bounds(now - timedelta(days=2))
            p.intent_type = "DAY"
        elif "última semana" in q or "ultima semana" in q:
            end = _ensure_tz(datetime(now.year, now.month, now.day))  # hoy 00:00
            start = end - timedelta(days=7)
            p.start, p.end = start, end
            p.intent_type = "RANGE"

    # --- relativas: "hace N días/semanas/meses" (punto en el tiempo o rango 1d) ---  # NEW
    if p.start is None and p.end is None and not p.which:
        m = re.search(
            r"hace\s+(?P<n>\d+|[a-záéíóú]+)\s+(?P<u>d[ií]as|semanas|meses?)", q
        )
        if m:
            n = parse_spanish_int(m.group("n")) or 1
            u = m.group("u")
            delta = (
                timedelta(days=n)
                if "día" in u or "dia" in u or "días" in u or "dias" in u
                else timedelta(weeks=n) if "semana" in u else None
            )
            if delta is None:
                # meses aprox por 30 días
                delta = timedelta(days=30 * n)
            target = now - delta
            p.start, p.end = _day_bounds(target)
            p.intent_type = "DAY"
#---------------------------------------------------------------------------------------------
    # [NUEVO BLOQUE] Rango explícito con fechas numéricas: "entre el 16/11 y el 18/11" o "del 16-11 al 18-11"
    if p.start is None and p.end is None and not p.which:
        # Regex captura dos fechas completas (con / o -) separadas por conectores
        range_full_date_pattern = (
            r"(?:entre|de|desde)\s+(?:el\s+)?(?P<d1>\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})"  # Fecha 1
            r"\s+(?:y|a|al|hasta)\s+(?:el\s+)?(?P<d2>\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})"  # Fecha 2
        )
        m_range = re.search(range_full_date_pattern, q)
        
        if m_range:
            try:
                dt1 = _parse_generic_date(m_range.group("d1"), now)
                dt2 = _parse_generic_date(m_range.group("d2"), now)
                
                if dt1 and dt2:
                    # Ordenamos por si el usuario dice "entre el 18 y el 16"
                    start_dt = min(dt1, dt2)
                    end_dt = max(dt1, dt2)
                    
                    p.start, _ = _day_bounds(start_dt) # Inicio del día 1 (00:00)
                    _, p.end = _day_bounds(end_dt)     # Fin del día 2 (exclusive, día siguiente 00:00)
                    
                    p.intent_type = "RANGE"
            except Exception:
                pass # Si falla, dejamos que sigan las otras reglas


    

    # --- entre el X y el Y de <mes> (de <año>) ---
    if p.start is None and p.end is None and not p.which:
        m = re.search(
            r"entre\s+el\s+(\d{1,2})\s+y\s+el\s+(\d{1,2})(?:\s+de\s+([a-záéíóú]+))?(?:\s+de\s+(\d{4}))?",  # NEW: mes opcional
            q,
        )
        if m:
            d1, d2 = int(m.group(1)), int(m.group(2))
            mon_name = (m.group(3) or "").strip()
            y = int(m.group(4)) if m.group(4) else year_now
            if mon_name:
                mnum = MONTHS.get(mon_name, None)
                if mnum:
                    start = _ensure_tz(datetime(y, mnum, min(d1, d2)))
                    end = _ensure_tz(datetime(y, mnum, max(d1, d2))) + timedelta(days=1)
                    p.start, p.end, p.intent_type = start, end, "RANGE"
            else:
                # Sin mes: asumimos mismo mes actual (fallback benigno)
                start = _ensure_tz(datetime(y, month_now, min(d1, d2)))
                end = _ensure_tz(datetime(y, month_now, max(d1, d2))) + timedelta(
                    days=1
                )
                p.start, p.end, p.intent_type = start, end, "RANGE"

    # --- fecha puntual dd/mm/yyyy o yyyy-mm-dd (soporta espacios) ---
    if p.start is None and p.end is None and not p.which:
        m = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})\b", q)
        if m:
            d, m_, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start, end = _day_bounds(_ensure_tz(datetime(y, m_, d)))
            p.start, p.end, p.intent_type = start, end, "DAY"
    if p.start is None and p.end is None and not p.which:
        m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", q)
        if m:
            y, m_, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start, end = _day_bounds(_ensure_tz(datetime(y, m_, d)))
            p.start, p.end, p.intent_type = start, end, "DAY"

    # --- fecha "29 de diciembre de 2023" (Español explícito) --- # NEW
    if p.start is None and p.end is None and not p.which:
        m = re.search(
            r"\b(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})\b", q
        )
        if m:
            d = int(m.group(1))
            mon_name = m.group(2)
            y = int(m.group(3))
            mnum = MONTHS.get(mon_name)
            if mnum:
                start, end = _day_bounds(_ensure_tz(datetime(y, mnum, d)))
                p.start, p.end, p.intent_type = start, end, "DAY"

    # --- fecha puntual dd/mm (sin año) (soporta espacios) ---  # NEW
    if p.start is None and p.end is None and not p.which:
        m = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b", q)
        if m:
            d, m_ = int(m.group(1)), int(m.group(2))
            base = _resolve_partial_day_month(d, m_, now)
            p.start, p.end, p.intent_type = _day_bounds(base) + ("DAY",)

    # --- “julio 2025”, “mayo” (asume año actual), “2025” ---
    if p.start is None and p.end is None and not p.which:
        m = re.search(r"\b([a-záéíóú]+)\s+(20\d{2})\b", q)
        if m:
            mon_name, y = m.group(1), int(m.group(2))
            mnum = MONTHS.get(mon_name, None)
            if mnum:
                p.start, p.end = _month_bounds(y, mnum)
                p.intent_type = "RANGE"
    if p.start is None and p.end is None and not p.which:
        m = re.search(r"\b(20\d{2})\b", q)
        if m:
            y = int(m.group(1))
            p.start, p.end = _year_bounds(y)
            p.intent_type = "RANGE"
    if p.start is None and p.end is None and not p.which:
        for name, mnum in MONTHS.items():
            if re.search(rf"\b{name}\b", q):
                p.start, p.end = _month_bounds(year_now, mnum)
                p.intent_type = "RANGE"
                break

    # Si no detectamos nada y tampoco superlativo: no hay intención
    if p.start is None and p.end is None and not p.which:
        return None

    # Construir TemporalWindow
    tw = TemporalWindow(
        start_iso=_to_iso(p.start) if p.start else None,
        end_iso=_to_iso(p.end) if p.end else None,
        temporal_intent_type=p.intent_type,
        which=p.which,
        count=p.count or 1,
    )
    return tw

