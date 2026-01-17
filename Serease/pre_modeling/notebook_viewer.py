from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    display = print


def _to_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, (list, tuple, np.ndarray)):
        return pd.Series(y)
    raise TypeError(f"Expected pd.Series or array-like, got {type(y)}")


def _seasonal_prototype_repeat(y: pd.Series, m: int) -> pd.Series:
    """
    Build a length-n series by:
      1) computing mean-by-phase over positions (t mod m)
      2) repeating that m-length prototype across the full series length

    This gives a clean 'seasonal component' proxy useful for resemblance plots.
    """
    s = y.astype(float).copy()
    n = len(s)
    if m <= 1 or n < m:
        return pd.Series([np.nan] * n, index=s.index)

    # Use positional phase (robust to non-datetime indexes)
    phase = np.arange(n) % m
    df = pd.DataFrame({"y": s.values, "phase": phase})
    proto = df.groupby("phase")["y"].mean()

    repeated = np.take(proto.values, phase)
    return pd.Series(repeated, index=s.index, name=f"seasonal_proto_m{m}")


def plot_time_series(y: pd.Series, title: str = "Time plot") -> None:
    y = _to_series(y)
    plt.figure()
    plt.plot(y.index, y.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(y.name or "y")
    plt.tight_layout()
    plt.show()


def plot_period_resemblance(
    y: pd.Series,
    m_values: Sequence[int],
    max_candidates: int = 5,
    use_first_n: Optional[int] = None,
) -> None:
    """
    For each candidate m:
      - top: seasonal prototype repeated over time
      - bottom: the actual series

    This mirrors the idea of your example image (clean seasonal signal vs noisy series).
    """
    y = _to_series(y).dropna()
    if use_first_n is not None:
        y = y.iloc[: int(use_first_n)]

    m_list = [int(m) for m in m_values][: int(max_candidates)]

    for m in m_list:
        proto = _seasonal_prototype_repeat(y, m)

        fig = plt.figure()
        # Two stacked axes
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(proto.index, proto.values)
        ax1.set_title(f"Candidate m={m}: seasonal prototype (mean-by-phase, repeated)")
        ax1.set_ylabel("prototype")
        ax1.tick_params(labelbottom=False)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(y.index, y.values)
        ax2.set_title("Series")
        ax2.set_xlabel("Time")
        ax2.set_ylabel(y.name or "y")

        plt.tight_layout()
        plt.show()


def _plot_stem_with_confint(
    values: Sequence[float],
    confint: Optional[Sequence[Sequence[float]]] = None,
    title: str = "",
    xlabel: str = "Lag",
    ylabel: str = "",
) -> None:
    """
    Stem plot with per-lag confidence intervals rendered as a band.

    Expected shapes:
      values: length nlags+1 (including lag 0)
      confint: shape (nlags+1, 2) where each row is [low, high]
    """
    vals = np.asarray(values, dtype=float)
    lags = np.arange(len(vals))

    plt.figure()
    markerline, stemlines, baseline = plt.stem(lags, vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if confint is not None:
        ci = np.asarray(confint, dtype=float)
        if ci.shape[0] == len(vals) and ci.shape[1] == 2:
            lo = ci[:, 0]
            hi = ci[:, 1]
            # band across lags (matplotlib chooses default color)
            plt.fill_between(lags, lo, hi, alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_acf_pacf_from_payload(acf_pacf_payload: Dict[str, Any]) -> None:
    """
    Produces TWO separate plots:
      - ACF stem + conf band
      - PACF stem + conf band
    """
    acf_vals = acf_pacf_payload.get("acf")
    pacf_vals = acf_pacf_payload.get("pacf")
    acf_ci = acf_pacf_payload.get("acf_confint")
    pacf_ci = acf_pacf_payload.get("pacf_confint")

    if acf_vals is not None:
        _plot_stem_with_confint(
            values=acf_vals,
            confint=acf_ci,
            title="ACF",
            ylabel="ACF",
        )

    if pacf_vals is not None:
        _plot_stem_with_confint(
            values=pacf_vals,
            confint=pacf_ci,
            title="PACF",
            ylabel="PACF",
        )


def show_stationarity_table(rows: List[Dict[str, Any]], max_rows: int = 50) -> None:
    """
    Opt-in display of stationarity table.
    """
    if not rows:
        print("No stationarity rows.")
        return

    df = pd.DataFrame(rows).copy()
    # friendly ordering if present
    preferred_cols = [
        "D", "d", "m", "n",
        "adf_pvalue", "kpss_pvalue", "acf1",
        "pass_stationarity",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    display(df.head(int(max_rows)))


class NotebookPreModelViewer:
    """
    Notebook viewer for your pre-modeling output.
    Works best with PreModelState.
    """

    def __init__(self, show_stationarity: bool = False):
        self.show_stationarity = show_stationarity

    def render_from_state(self, state) -> None:
        # Basic time plot (raw)
        y_raw = state.bundle.y_raw
        plot_time_series(y_raw, title="Time plot (raw)")

        # Period resemblance plots (from period_candidates)
        cand = state.diagnostics.period_candidates or {}
        candidates = cand.get("candidates", [])
        m_values = [c.get("m") for c in candidates if c.get("m") is not None]
        if m_values:
            plot_period_resemblance(y_raw, m_values=m_values, max_candidates=5)

        # Stationarity table (opt-in)
        if self.show_stationarity:
            show_stationarity_table(state.diagnostics.stationarity_table)

        # ACF/PACF (final)
        payload = state.diagnostics.acf_pacf_payload or {}
        if payload:
            plot_acf_pacf_from_payload(payload)

    def render_from_report(self, report: Dict[str, Any], y: Optional[pd.Series] = None) -> None:
        """
        If your report doesn't include raw y, pass y explicitly.
        """
        artifacts = report.get("artifacts", {})
        if y is None:
            raise ValueError("report does not include y; pass y=... (pd.Series)")

        plot_time_series(y, title="Time plot (raw)")

        pc = artifacts.get("period_candidates", {})
        m_values = [c.get("m") for c in pc.get("candidates", []) if c.get("m") is not None]
        if m_values:
            plot_period_resemblance(y, m_values=m_values, max_candidates=5)

        if self.show_stationarity:
            show_stationarity_table(artifacts.get("stationarity_table", []))

        payload = artifacts.get("acf_pacf_payload", {})
        if payload:
            plot_acf_pacf_from_payload(payload)
