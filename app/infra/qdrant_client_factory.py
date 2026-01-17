# app/infra/qdrant_client_factory.py
from functools import lru_cache
from qdrant_client import QdrantClient
import os


def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    v = str(val).strip().lower()
    return v in ("1", "true", "t", "yes", "y")

def _normalize_url(u: str | None) -> str:
    # Fallback razonable al dashboard que mostraste (HTTP:6333)
    if not u or not u.strip():
        return "http://10.10.0.48:6333"
    u = u.strip()
    # Si no trae esquema, asumimos http
    if not (u.startswith("http://") or u.startswith("https://") or u.startswith("grpc://")):
        u = "http://" + u
    return u

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    url = _normalize_url(os.getenv("QDRANT_URL"))
    prefer_grpc = _as_bool(os.getenv("QDRANT_PREFER_GRPC"), default=False)  # ⬅️ Default HTTP
    timeout = int(os.getenv("QDRANT_TIMEOUT", "60"))
    api_key = os.getenv("QDRANT_API_KEY") or None

    return QdrantClient(
        url=url,
        prefer_grpc=prefer_grpc,
        timeout=timeout,
        api_key=api_key,
    )
