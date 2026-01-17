# app/cache/engine_cache.py
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

CACHE_TTL_S = int(os.getenv("ROUTER_CACHE_TTL", "900"))  # 15 min
CACHE_MAX = int(os.getenv("ROUTER_CACHE_MAX", "128"))


class EngineCacheManager:
    def __init__(self) -> None:
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._locks: Dict[str, threading.Lock] = {}

    # ---------- keys ----------
    @staticmethod
    def make_key(paciente_id: str, *segments: str) -> str:
        seg = ":".join(s for s in segments if s)
        return f"engine:pac:{paciente_id}{(':'+seg) if seg else ''}"

    # ---------- locks ----------
    def get_lock(self, key: str) -> threading.Lock:
        if key not in self._locks:
            self._locks[key] = threading.Lock()
        return self._locks[key]

    # ---------- LRU+TTL ----------
    def put(self, key: str, payload: Dict[str, Any]) -> None:
        payload = {**payload, "ts": time.time()}
        self._store[key] = payload
        self._store.move_to_end(key)
        # trim LRU
        while len(self._store) > CACHE_MAX:
            self._store.popitem(last=False)

    def get(
        self, key: str, expect_hash: Optional[str] = None, ttl_s: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        entry = self._store.get(key)
        if not entry:
            return None
        # hash mismatch → miss
        if expect_hash is not None and entry.get("hash") != expect_hash:
            return None
        # TTL
        ttl = CACHE_TTL_S if ttl_s is None else ttl_s
        if ttl > 0 and (time.time() - entry.get("ts", 0) > ttl):
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry

    # ---------- invalidación ----------
    def invalidate(self, predicate: Callable[[str], bool]) -> int:
        to_delete = [k for k in self._store.keys() if predicate(k)]
        for k in to_delete:
            self._store.pop(k, None)
            self._locks.pop(k, None)
        if to_delete:
            log.info("🧹 Engine cache: borradas %s claves", len(to_delete))
        return len(to_delete)

    def invalidate_for_patient(self, paciente_id: str) -> int:
        prefix = f"engine:pac:{paciente_id}"
        return self.invalidate(lambda k: k.startswith(prefix))


# instancia global
cache_manager = EngineCacheManager()


# helper para endpoints
def invalidate_engine_cache_for_patient(paciente_id: str) -> int:
    return cache_manager.invalidate_for_patient(paciente_id)
