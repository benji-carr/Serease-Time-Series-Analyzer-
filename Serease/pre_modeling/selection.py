# serease/pre_modeling/selection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .containers import TransformedSeries


@dataclass(frozen=True)
class StationarityPolicy:
    """
    Decision policy for selecting a stationary candidate.

    Statistical interpretation:
      - ADF: H0 = unit root (non-stationary). We want p <= alpha_adf.
      - KPSS: H0 = stationary. We want p >= alpha_kpss.
      - Overdifferencing guard: strong negative lag-1 autocorrelation suggests overdifferencing.
    """
    alpha_adf: float = 0.05
    alpha_kpss: float = 0.05

    # Overdifferencing guard thresholds on lag-1 autocorrelation (acf1)
    acf1_soft: float = -0.30   # soft penalty threshold
    acf1_hard: float = -0.50   # hard exclusion threshold

    # Hard caps (should match TransformPlan.max_* in practice)
    max_d: int = 2
    max_D: int = 1


# -----------------------------
# Utilities
# -----------------------------
def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _pass_stationarity(row: Dict[str, Any], policy: StationarityPolicy) -> bool:
    """
    Pass criteria:
      - ADF p-value <= alpha_adf
      - KPSS p-value >= alpha_kpss

    If either p-value is missing/unparseable, returns False.
    """
    adf_p = _as_float(row.get("adf_pvalue"))
    kpss_p = _as_float(row.get("kpss_pvalue"))
    if adf_p is None or kpss_p is None:
        return False
    return (adf_p <= policy.alpha_adf) and (kpss_p >= policy.alpha_kpss)


def _overdiff_class(row: Dict[str, Any], policy: StationarityPolicy) -> str:
    """
    Classifies overdifferencing risk.

    Returns:
      - "none": no indication of overdifferencing
      - "soft": possible overdifferencing (penalize but allowed)
      - "hard": strong overdifferencing (avoid if possible)

    We prefer the explicit 'overdiff_flag' if diagnostics provided it,
    otherwise we infer from acf1.
    """
    flag = row.get("overdiff_flag")
    if flag in ("soft", "hard"):
        return flag

    acf1 = _as_float(row.get("acf1"))
    if acf1 is None:
        return "none"
    if acf1 <= policy.acf1_hard:
        return "hard"
    if acf1 <= policy.acf1_soft:
        return "soft"
    return "none"


def _complexity_key(D: int, d: int) -> Tuple[int, int, int, int, int]:
    """
    Deterministic tie-breaking key enforcing your preference:

      Primary: minimize total differencing (D + d)
      Tie 1 : prefer seasonal differencing over regular differencing -> (-D)
              (so D=1,d=0 beats D=0,d=1 when both have total=1)
      Tie 2 : minimize regular differencing d
      Final : stable ordering by (D,d)

    Return is a tuple used for sorting ascending.
    """
    return (D + d, -D, d, D, d)


# -----------------------------
# Main selector
# -----------------------------
def select_stationary_candidate(
    candidates: List[TransformedSeries],
    stationarity_rows: List[Dict[str, Any]],
    policy: StationarityPolicy,
) -> Tuple[Optional[TransformedSeries], Dict[str, Any]]:
    """
    Select the best stationary candidate given diagnostics evidence.

    Inputs:
      - candidates: list of TransformedSeries (each with plan.D, plan.d, plan.seasonal_period_m)
      - stationarity_rows: diagnostic rows produced by DiagnosticsEngine.stationarity_tests()
      - policy: thresholds + caps + overdifferencing guard parameters

    Output:
      - best_candidate: the selected TransformedSeries (or None if no candidates)
      - meta: a compact explanation payload for reporting/debugging

    Selection rules (high-level):
      1) Prefer candidates that PASS stationarity (ADF+KPSS).
      2) If any pass, avoid "hard" overdifferenced candidates when possible.
      3) Among remaining, choose least differenced using your tie-break rule:
            minimize (D+d), then prefer D over d on ties.
      4) If none pass, still choose a candidate (so pipeline can proceed):
            avoid "hard" overdifferencing if possible, then least differenced.
    """
    # Build a lookup from (D,d,m) -> row. Include a fallback on (D,d,None) when m isn't present.
    row_map: Dict[Tuple[int, int, Optional[int]], Dict[str, Any]] = {}
    for r in stationarity_rows:
        D = int(r.get("D", 0))
        d = int(r.get("d", 0))
        m_raw = r.get("m", None)
        try:
            m = int(m_raw) if m_raw is not None else None
        except Exception:
            m = None
        row_map[(D, d, m)] = r

    scored: List[Dict[str, Any]] = []
    for i, ts in enumerate(candidates):
        D = int(ts.plan.D)
        d = int(ts.plan.d)
        m = ts.plan.seasonal_period_m
        m_key = int(m) if m is not None else None

        # Enforce caps
        if D > policy.max_D or d > policy.max_d:
            continue

        # Find diagnostics row; fall back to (D,d,None) if m-matched row is missing
        row = row_map.get((D, d, m_key)) or row_map.get((D, d, None)) or {}

        # Prefer explicit orchestrator-added pass flag if present
        if "pass_stationarity" in row:
            passed = bool(row["pass_stationarity"])
        else:
            passed = _pass_stationarity(row, policy)

        overdiff = _overdiff_class(row, policy)
        complexity = _complexity_key(D, d)

        scored.append(
            {
                "idx": i,
                "ts": ts,
                "D": D,
                "d": d,
                "m": m_key,
                "passed": passed,
                "overdiff": overdiff,
                "complexity": complexity,
                "adf_pvalue": row.get("adf_pvalue"),
                "kpss_pvalue": row.get("kpss_pvalue"),
                "acf1": row.get("acf1"),
            }
        )

    meta: Dict[str, Any] = {
        "policy": {
            "alpha_adf": policy.alpha_adf,
            "alpha_kpss": policy.alpha_kpss,
            "acf1_soft": policy.acf1_soft,
            "acf1_hard": policy.acf1_hard,
            "max_d": policy.max_d,
            "max_D": policy.max_D,
        },
        "any_candidates": bool(scored),
        "any_passed": False,
        "passed": None,   # whether the selected candidate passed stationarity
        "selected": None, # selected D,d,m,overdiff
        "reason": None,
    }

    if not scored:
        meta["reason"] = "no_candidates_scored"
        return None, meta

    # Split into passed / non-passed pools
    passed_pool = [s for s in scored if s["passed"]]
    meta["any_passed"] = bool(passed_pool)

    # Sort helper: overdifferencing severity first, then your complexity rule
    def sort_key(s: Dict[str, Any]) -> Tuple[int, Tuple[int, int, int, int, int]]:
        overdiff_rank = {"none": 0, "soft": 1, "hard": 2}.get(s["overdiff"], 1)
        return (overdiff_rank, s["complexity"])

    if passed_pool:
        best = sorted(passed_pool, key=sort_key)[0]
        meta["passed"] = True
        meta["selected"] = {"D": best["D"], "d": best["d"], "m": best["m"], "overdiff": best["overdiff"]}
        meta["reason"] = "selected_from_passed_pool"
        return best["ts"], meta

    # None passed: avoid hard overdifferencing if any alternative exists,
    # then take the least differenced candidate by your complexity rule.
    non_hard = [s for s in scored if s["overdiff"] != "hard"]
    pool = non_hard if non_hard else scored

    best = sorted(pool, key=sort_key)[0]
    meta["passed"] = False
    meta["selected"] = {"D": best["D"], "d": best["d"], "m": best["m"], "overdiff": best["overdiff"]}
    meta["reason"] = "selected_from_nonpassed_pool"
    return best["ts"], meta
