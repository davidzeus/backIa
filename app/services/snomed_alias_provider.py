#snomed_alias_provider
import os, json, unicodedata, re, hashlib, time
from functools import lru_cache
from typing import Dict, List, Optional

try:
    import redis  # pip install redis
except Exception:
    redis = None

from app.services.snomed_service import buscar_conceptos_snomed, obtener_sinonimos

CACHE_PATH = os.getenv("SNOMED_SECTIONS_CACHE", "service_aliases_snomed.json")
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_TTL = int(os.getenv("SNOMED_REDIS_TTL", "0") or "0")  # 0 = sin TTL
# Cambiá la versión si cambiás los SEEDS/formato para invalidar caché viejo.
CACHE_VERSION = "v1"

SEEDS = {
    "ICU/UTI": "Unidad de terapia intensiva",
    "EMERGENCY/ED": "Servicio de urgencias",
    "WARD/FLOOR": "Internación hospitalaria",
    "OR/PROCEDURE": "Quirófano",
    "OUTPATIENT/CLINIC": "Atención ambulatoria",
    "DIAGNOSTICS": "Diagnóstico por imágenes",
    "LAB": "Laboratorio clínico",
}

def _nfd_lower(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()

def _redis_client() -> Optional["redis.Redis"]:
    if not REDIS_URL or not redis:
        return None
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        # ping rápido para asegurar disponibilidad
        r.ping()
        return r
    except Exception:
        return None

def _redis_key() -> str:
    # clave estable que cambia si cambian los SEEDS
    seeds_fingerprint = hashlib.sha1(json.dumps(SEEDS, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"snomed:sections:aliases:{CACHE_VERSION}:{seeds_fingerprint}"

def _save_file_cache(data: Dict[str, List[str]]) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _load_file_cache() -> Optional[Dict[str, List[str]]]:
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        return json.loads(open(CACHE_PATH, "r", encoding="utf-8").read())
    except Exception:
        return None

def _build_aliases_from_snomed() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for sec, seed in SEEDS.items():
        conceptos = buscar_conceptos_snomed(seed, limit=3) or []
        bag = set([_nfd_lower(seed)])
        for c in conceptos:
            cid = str(c.get("conceptId") or "").strip()
            if not cid:
                continue
            for term in obtener_sinonimos(cid):
                if term and len(term) >= 3:
                    bag.add(_nfd_lower(term))
        out[sec] = sorted(bag)
    return out

@lru_cache(maxsize=1)
def load_section_aliases() -> Dict[str, List[str]]:
    """
    Orden de carga:
      1) Redis (si está disponible)
      2) Archivo en disco (JSON)
      3) Construcción desde SNOMED + persistencia (Redis y disco)
    """
    r = _redis_client()
    key = _redis_key()

    # 1) Redis
    if r:
        try:
            raw = r.get(key)
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict) and data:
                    return data
        except Exception:
            pass

    # 2) Archivo
    data = _load_file_cache()
    if isinstance(data, dict) and data:
        # hidratar Redis si está disponible
        if r:
            try:
                payload = json.dumps(data, ensure_ascii=False)
                if REDIS_TTL > 0:
                    r.setex(key, REDIS_TTL, payload)
                else:
                    r.set(key, payload)
            except Exception:
                pass
        return data

    # 3) Construir desde SNOMED
    data = _build_aliases_from_snomed()
    # persistir a disco
    _save_file_cache(data)
    # persistir a Redis
    if r:
        try:
            payload = json.dumps(data, ensure_ascii=False)
            if REDIS_TTL > 0:
                r.setex(key, REDIS_TTL, payload)
            else:
                r.set(key, payload)
        except Exception:
            pass
    return data

def compile_section_regexes() -> Dict[str, re.Pattern]:
    aliases = load_section_aliases()
    rx: Dict[str, re.Pattern] = {}
    for sec, terms in aliases.items():
        pats = []
        for t in set(terms or []):
            esc = re.escape(t).replace(r"\ ", r"[ \-_/]+")
            pats.append(rf"(?:^|[^A-Za-zÁÉÍÓÚÑáéíóúñ]){esc}(?:$|[^A-Za-zÁÉÍÓÚÑáéíóúñ])")
        if pats:
            rx[sec] = re.compile("|".join(pats), re.IGNORECASE)
    return rx

def refresh_aliases(force_rebuild: bool = False) -> Dict[str, List[str]]:
    """
    Fuerza recálculo desde SNOMED (o rehidratación si force_rebuild=False y ya hay Redis/disco).
    Útil para tarea de mantenimiento o al actualizar la release de SNOMED.
    """
    if not force_rebuild:
        # devuelve lo que haya (Redis/disco o construye si no hay)
        return load_section_aliases()
    data = _build_aliases_from_snomed()
    _save_file_cache(data)
    r = _redis_client()
    if r:
        key = _redis_key()
        payload = json.dumps(data, ensure_ascii=False)
        try:
            if REDIS_TTL > 0:
                r.setex(key, REDIS_TTL, payload)
            else:
                r.set(key, payload)
        except Exception:
            pass
    # invalidar memoization de lru_cache
    try:
        load_section_aliases.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    return data
