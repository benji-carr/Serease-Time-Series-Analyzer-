from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


def infer_seasonal_period_from_freq(freq: Optional[str]) -> Optional[int]:
    if freq is None:
        return None
    f = freq.upper()
    if f in {"H"}:
        return 24
    if f in {"D"}:
        return 7
    if f in {"W"}:
        return 52
    if f in {"M", "MS"}:
        return 12
    if f in {"Q", "QS"}:
        return 4
    return None


def contiguous_nan_blocks(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    blocks: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    if mask.empty:
        return blocks

    idx = mask.index
    in_block = False
    start = None
    length = 0

    for t, is_nan in mask.items():
        if is_nan and not in_block:
            in_block = True
            start = t
            length = 1
        elif is_nan and in_block:
            length += 1
        elif (not is_nan) and in_block:
            end = prev
            blocks.append((start, end, length))
            in_block = False
            start = None
            length = 0
        prev = t

    if in_block and start is not None:
        blocks.append((start, idx[-1], length))

    return blocks


def to_numpy_1d(s: pd.Series) -> np.ndarray:
    return np.asarray(s.values, dtype=float)

def compute_fft_periodogram_candidates(s: pd.Series, top_k: int = 8) -> List[Dict[str, Any]]:
    x = s.dropna().values.astype(float)
    x = x - np.mean(x)

    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0)

    df = pd.DataFrame({"freq": freqs, "power": power})
    df = df[df["freq"] > 0].sort_values("power", ascending=False).head(int(top_k)).copy()
    df["period_steps"] = (1.0 / df["freq"]).replace([np.inf], np.nan)

    out = []
    for _, r in df.iterrows():
        out.append({
            "freq": float(r["freq"]),
            "power": float(r["power"]),
            "period_steps": float(r["period_steps"]) if pd.notna(r["period_steps"]) else None,
        })
    return out


def choose_period_from_candidates(period_res) -> Optional[int]:
    if period_res is None or not getattr(period_res, "artifacts", None):
        return None
    cand = None
    for art in period_res.artifacts:
        if art.name == "period_candidates":
            cand = art.payload
            break
    if cand is None or getattr(cand, "empty", True):
        return None

    df = cand.copy()
    df = df[df["period_steps"].notna()]
    if df.empty:
        return None

    df["period_int"] = df["period_steps"].round().astype(int)
    df = df[df["period_int"] >= 2]
    if df.empty:
        return None

    df = df.sort_values("power", ascending=False)
    return int(df.iloc[0]["period_int"])


