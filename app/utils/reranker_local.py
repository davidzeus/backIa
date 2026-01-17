# app/utils/reranker_local.py
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

WORD_RX = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9%°]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in WORD_RX.findall(text)]


def _uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _date_score(
    fecha_ts: Optional[int], now_ts: Optional[int] = None, half_life_days: float = 90.0
) -> float:
    """
    Boost suave por recencia (sigmoide / exponencial).
    half_life_days: a los ~90 días el boost se reduce a ~0.5.
    """
    if not isinstance(fecha_ts, (int, float)):
        return 0.0
    if now_ts is None:
        now_ts = int(datetime.utcnow().timestamp())
    age_days = max(0.0, (now_ts - int(fecha_ts)) / 86400.0)
    # decay exponencial
    return math.exp(-math.log(2) * (age_days / half_life_days))


def _is_consultation_like(md: Dict[str, Any], head: str) -> bool:
    bag = " ".join(
        str(md.get(k) or "")
        for k in (
            "seccion",
            "seccion_raiz",
            "group",
            "name",
            "servicio",
            "servicio_desc",
        )
    ).lower()
    blob = (bag + " " + (head or "")).lower()
    for tok in (
        "consulta",
        "interconsulta",
        "evolucion",
        "evolución",
        "guardia",
        "seguimiento",
        "motivo de consulta",
    ):
        if tok in blob:
            return True
    return False


class LocalHeuristicReranker:
    """
    Reranker heurístico ligero:
      score = α * overlap(query, text) + β * recency(fecha_ts) + γ * bonus_consulta
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.25,
        gamma: float = 0.15,
        top_n: Optional[int] = None,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.top_n = top_n  # si None, conserva K original

    def postprocess_nodes(self, nodes: List[Any], query: str) -> List[Any]:
        if not nodes:
            return nodes
        q_tokens = _uniq(_tokenize(query))

        # precompute idf-like weights (sin corpus global: usamos 1/log(1+df))
        df: Dict[str, int] = {}
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        heads: List[str] = []
        fechas: List[Optional[int]] = []

        for n in nodes:
            node = getattr(n, "node", n)
            text = getattr(node, "text", "") or ""
            texts.append(text)
            md = getattr(node, "metadata", {}) or {}
            metas.append(md)
            fechas.append(md.get("fecha_ts"))
            head = text.split("\n", 1)[0]
            heads.append(head)
            # df para tokens de este documento
            seen = set(_tokenize(text))
            for t in seen:
                df[t] = df.get(t, 0) + 1

        max(1, len(nodes))
        idf = {t: 1.0 / math.log(2.0 + dfc) for t, dfc in df.items()}

        def overlap_score(qtoks, text):
            ttoks = _tokenize(text)
            if not ttoks or not qtoks:
                return 0.0
            tset = set(ttoks)
            score = 0.0
            for qt in qtoks:
                if qt in tset:
                    score += idf.get(qt, 1.0)
            # normalización suave por longitud
            return score / (1.0 + math.log(10.0 + len(ttoks)))

        scored = []
        for i, n in enumerate(nodes):
            ov = overlap_score(q_tokens, texts[i])
            rec = _date_score(fechas[i])
            bonus = 1.0 if _is_consultation_like(metas[i], heads[i]) else 0.0
            s = self.alpha * ov + self.beta * rec + self.gamma * bonus
            scored.append((s, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        if self.top_n is not None and self.top_n > 0:
            scored = scored[: self.top_n]
        return [n for _, n in scored]
