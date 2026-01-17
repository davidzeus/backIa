from typing import Dict, Any, Optional, Tuple
from qdrant_client.models import Filter, FieldCondition, MatchValue

ALLOWED_FIELDS = {"seccion_raiz", "name", "seccion_raiz_norm", "name_norm"}
ALLOWED_OPS = {"eq"}

def compile_llm_filter(suggest: Dict[str, Any]) -> Tuple[Optional[str], Filter]:
    """
    Convierte la sugerencia del LLM (JSON) en un Filter seguro para Qdrant.
    Ignora campos/ops fuera de allow-list. Devuelve (name_hint, Filter).
    """
    qf = Filter(must=[], should=[])
    name_hint: Optional[str] = None

    if not isinstance(suggest, dict):
        return None, qf

    nh = str(suggest.get("name_hint") or "").strip()
    name_hint = nh if nh else None

    fs = suggest.get("filters") or {}
    for bucket in ("must", "should"):
        for cond in (fs.get(bucket) or []):
            field = str(cond.get("field") or "")
            op    = str(cond.get("op") or "")
            val   = cond.get("value")
            if field in ALLOWED_FIELDS and op in ALLOWED_OPS and isinstance(val, str) and val:
                (qf.must if bucket == "must" else qf.should).append(
                    FieldCondition(key=field, match=MatchValue(value=val))
                )

    return name_hint, qf
