# app/services/memory.py
from __future__ import annotations
import os
import json
import time
from typing import List, Tuple, Optional, Protocol

# ---------- Interfaz ----------
class ChatMemory(Protocol):
    def append(self, key: str, role: str, text: str) -> None: ...
    def last_pairs(self, key: str, max_pairs: int = 8) -> List[Tuple[str, str]]: ...
    def summarize_into(self, key: str, text: str) -> None: ...
    def get_summary(self, key: str) -> str: ...

# ---------- Implementación en proceso ----------
class LocalMemory:
    def __init__(self) -> None:
        self._store = {}      # key -> list[{"role": "user"/"assistant", "text": "..."}]
        self._summary = {}    # key -> str

    def append(self, key: str, role: str, text: str) -> None:
        self._store.setdefault(key, []).append({"role": role, "text": text, "ts": time.time()})

    def last_pairs(self, key: str, max_pairs: int = 8) -> List[Tuple[str, str]]:
        msgs = self._store.get(key, [])
        pairs: List[Tuple[str, str]] = []
        curr = []
        for m in reversed(msgs):
            curr.append(m)
            if len(curr) == 2 and curr[0]["role"] == "assistant" and curr[1]["role"] == "user":
                pairs.append((curr[1]["text"], curr[0]["text"]))
                curr = []
        pairs.reverse()
        return pairs[-max_pairs:]

    def summarize_into(self, key: str, text: str) -> None:
        prev = self._summary.get(key, "")
        merged = (prev + "\n" + text).strip()
        self._summary[key] = (merged[:1200] + "…") if len(merged) > 1200 else merged

    def get_summary(self, key: str) -> str:
        return self._summary.get(key, "")

# ---------- Implementación Redis (opcional) ----------
class RedisMemory:
    def __init__(self) -> None:
        import redis  # pip install redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._r = redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        # valida conexión (si falla, dejamos que la factory haga fallback)
        self._r.ping()

    def _msgs_key(self, key: str) -> str: return f"hc:mem:{key}:msgs"
    def _sum_key(self, key: str) -> str:  return f"hc:mem:{key}:sum"

    def append(self, key: str, role: str, text: str) -> None:
        item = json.dumps({"role": role, "text": text, "ts": time.time()})
        self._r.rpush(self._msgs_key(key), item)

    def last_pairs(self, key: str, max_pairs: int = 8) -> List[Tuple[str, str]]:
        raw = self._r.lrange(self._msgs_key(key), -2*max_pairs*4, -1)
        msgs = [json.loads(x) for x in raw]
        pairs: List[Tuple[str, str]] = []
        curr = []
        for m in reversed(msgs):
            curr.append(m)
            if len(curr) == 2 and curr[0]["role"] == "assistant" and curr[1]["role"] == "user":
                pairs.append((curr[1]["text"], curr[0]["text"]))
                curr = []
        pairs.reverse()
        return pairs[-max_pairs:]

    def summarize_into(self, key: str, text: str) -> None:
        prev = self._r.get(self._sum_key(key)) or ""
        merged = (prev + "\n" + text).strip()
        merged = (merged[:1200] + "…") if len(merged) > 1200 else merged
        self._r.set(self._sum_key(key), merged)

    def get_summary(self, key: str) -> str:
        return self._r.get(self._sum_key(key)) or ""

# ---------- Factory ----------
def get_memory_backend() -> ChatMemory:
    backend = (os.getenv("MEMORY_BACKEND") or "local").lower()
    if backend == "redis":
        try:
            return RedisMemory()
        except Exception as e:
            # Fallback suave si Redis no está disponible
            import logging
            logging.getLogger("app.services.memory").warning(
                f"[memory] Redis no disponible, uso LocalMemory. Motivo: {e}"
            )
            return LocalMemory()
    return LocalMemory()

# ---------- Helper de formato ----------
def format_history_for_context(pairs: List[Tuple[str, str]], summary: str = "") -> str:
    lines = []
    if summary:
        lines.append(f"[RESUMEN HISTORIAL]\n{summary}\n")
    for i, (u, a) in enumerate(pairs, 1):
        lines.append(f"[Turno {i}]\nUsuario: {u}\nAsistente: {a}\n")
    return "\n".join(lines).strip()
