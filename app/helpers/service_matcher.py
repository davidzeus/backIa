# app/helpers/service_matcher.py
from __future__ import annotations
import json, os, re, logging
from unicodedata import normalize
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_DEBUG_FLAG = os.getenv("HCI_DEBUG", "0")
if bool(int(str(_DEBUG_FLAG))):
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.WARNING)

# Carga los JSON desde el mismo directorio que este script (app/helpers/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELPERS_DIR = os.path.join(BASE_DIR, '..', 'helpers')

def _load_json(filename: str, default):
    path = os.path.join(HELPERS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # El log ahora mostrará la ruta correcta que intentó cargar
        log.warning(f"[planner] no pude cargar el archivo de servicio {path}: {e}")
        return default

# Carga los archivos desde la ruta correcta
SERVICE_CATALOG = _load_json("service_catalog.json", [])
SERVICE_REGEX_MAP = _load_json("service_regex_map.json", {})
SERVICE_ALIASES = _load_json("service_aliases.json", {})

#_WORD_BOUND = r"(?:^|[^A-Za-zÁÉÍÓÚÑáéíóúñ])" 

def _norm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = normalize("NFD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).lower().strip()
    return s

def _compile_alias_to_regex(term: str) -> str:
    esc = re.escape(_norm(term))
    return rf"\b{esc}\b"

def _compile_union(patterns: List[str]) -> Optional[re.Pattern]:
    pats = [p for p in patterns if p and isinstance(p, str)]
    if not pats: return None
    try:
        return re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    except Exception as e:
        log.warning(f"[service_matcher] fallo compilando union: {e}")
        return None

def _build_index():
    """Precompila señales por servicio: regex legacy, exactos, comunes y reglas."""
    by_sid: Dict[int, Dict] = {}

    id_by_name = {}
    for s in SERVICE_CATALOG:
        try:
            sid = int(s["id"])
            name = str(s.get("name", "")).strip()
            if not name: continue
            id_by_name[_norm(name)] = sid
            by_sid.setdefault(sid, {"name": name})
        except Exception:
            continue

    legacy_compiled: Dict[int, re.Pattern] = {}
    for k, pat in SERVICE_REGEX_MAP.items():
        sid = id_by_name.get(_norm(k))
        if sid is None: continue
        try:
            legacy_compiled[sid] = re.compile(pat, re.IGNORECASE)
            by_sid.setdefault(sid, {"name": by_sid.get(sid, {}).get("name", k)})
        except Exception as e:
            log.warning(f"[service_matcher] regex legacy inválido para {k}: {e}")

    compiled_exact, compiled_common = {}, {}
    rules: Dict[int, Dict[str, List[str]]] = {}

    for sid_str, block in SERVICE_ALIASES.items():
        try:
            sid = int(sid_str)
        except Exception:
            continue
        
        alias_exact = block.get("alias_exact", []) or []
        alias_common = block.get("alias_common", []) or []
        require_any = block.get("require_any", []) or []
        deny = block.get("deny", []) or []

        exact_pats = [_compile_alias_to_regex(t) for t in alias_exact if t]
        common_pats = [_compile_alias_to_regex(t) for t in alias_common if t]

        ce = _compile_union(exact_pats)
        cc = _compile_union(common_pats)

        if ce: compiled_exact[sid] = ce
        if cc: compiled_common[sid] = cc
        rules[sid] = {
            "require_any": [_norm(x) for x in require_any if x],
            "deny": [_norm(x) for x in deny if x],
        }
        by_sid.setdefault(sid, {"name": block.get("name") or by_sid.get(sid, {}).get("name")})

    return {
        "meta": by_sid,
        "legacy": legacy_compiled,
        "exact": compiled_exact,
        "common": compiled_common,
        "rules": rules,
    }

_INDEX = _build_index()

def _contains_any(text_norm: str, needles: List[str]) -> bool:
    return any(n and n in text_norm for n in needles)

def detect_servicio(text: str, top_k: int = 3) -> List[Dict]:
    if not text: return []
    text_norm = _norm(text)
    log.debug(f"[service_matcher] Texto a analizar: '{text}', Normalizado: '{text_norm}'")

    out: List[Tuple[int, float, str]] = []
    for sid, meta in _INDEX["meta"].items():
        



        deny = _INDEX["rules"].get(sid, {}).get("deny", [])
        if _contains_any(text_norm, deny):
            continue

        # <-- CORRECCIÓN: Usar text_norm para TODAS las búsquedas regex
        
        # 1. alias_exact (Score 1.0)
        ce = _INDEX["exact"].get(sid)
        if ce and ce.search(text_norm): # <-- CORRECCIÓN
            out.append((sid, 1.00, "alias_exact"))
            continue

        # 2. legacy regex (Score 0.90)
        legacy = _INDEX["legacy"].get(sid)
        if legacy and legacy.search(text_norm): # <-- CORRECCIÓN
            out.append((sid, 0.90, "legacy_regex"))

        # 3. alias_common (Score 0.80 con require_any, 0.60 sin)
        cc = _INDEX["common"].get(sid)
        if cc and cc.search(text_norm): # <-- CORRECCIÓN
            req = _INDEX["rules"].get(sid, {}).get("require_any", [])
            if not req or _contains_any(text_norm, req):
                out.append((sid, 0.80, "alias_common+require_any"))
            else:
                out.append((sid, 0.60, "alias_common"))

    # Agrupar por servicio y quedarse con el mejor score
    best: Dict[int, Tuple[float, str]] = {}
    for sid, sc, how in out:
        if sid not in best or sc > best[sid][0]:
            best[sid] = (sc, how)

    ranked = sorted(
        ({"servicio_id": sid, "score": sc, "method": how, "name": _INDEX["meta"].get(sid, {}).get("name", str(sid))}
         for sid, (sc, how) in best.items()),
        key=lambda x: (-x["score"], x["name"])
    )
    
    if ranked:
        log.info(f"[service_matcher] Servicios detectados: {ranked[:max(1, int(top_k))]}")
    
    return ranked[:max(1, int(top_k))]

def detect_servicio_top1(text: str) -> Optional[int]:
    matches = detect_servicio(text, top_k=1)
    return int(matches[0]["servicio_id"]) if matches else None

def get_service_name(sid: int) -> str:
    return _INDEX["meta"].get(int(sid), {}).get("name", "Desconocido")