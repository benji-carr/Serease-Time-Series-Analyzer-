from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, List, Tuple

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotTheme:
    """
    Minimal theme configuration for stable matplotlib output.

    Keep simple in MVP to avoid seaborn/matplotlib style drift.
    """
    figsize: Tuple[int, int] = (10, 4)
    title_size: int = 12
    label_size: int = 10


def apply_theme(theme: Optional[PlotTheme] = None) -> PlotTheme:
    """
    Return a PlotTheme to be used by all plot helpers.

    This function is a hook point to centralize style decisions later.
    """
    return theme or PlotTheme()


def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64-encoded PNG string suitable for embedding in HTML.

    Returns
    -------
    str
        Base64 string without data URI prefix (reporter will add 'data:image/png;base64,').
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------
# Stable plotting helpers (must never crash the reporter)
# ---------------------------------------------------------------------
def lineplot_series(s: Optional[pd.Series], title: str = "", theme: Optional[PlotTheme] = None) -> plt.Figure:
    """
    Plot a single time series line.

    Notes:
      - Used by the reporter; must never crash on empty data.
      - Preserves index order as provided.
    """
    th = apply_theme(theme)
    fig = plt.figure(figsize=th.figsize)
    ax = fig.add_subplot(111)

    if not isinstance(s, pd.Series) or len(s) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ss = s.astype("float64").replace([np.inf, -np.inf], np.nan)

    if ss.dropna().empty:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.plot(ss.index, ss.values)
    ax.set_title(title, fontsize=th.title_size)
    ax.set_xlabel("Time", fontsize=th.label_size)
    ax.set_ylabel("Value", fontsize=th.label_size)
    return fig


def periodogram_plot(
    period_candidates: Sequence[Dict[str, Any]],
    title: str = "Period Candidates",
    theme: Optional[PlotTheme] = None,
) -> plt.Figure:
    """
    Plot period candidate scores (simple line plot).

    Expected payload items (best-effort):
      {"m": int, "score": float, "confidence": float, ...}

    Robustness:
      - Skips rows that cannot be parsed into (m, score).
      - Never crashes on malformed payload.
    """
    th = apply_theme(theme)
    fig = plt.figure(figsize=th.figsize)
    ax = fig.add_subplot(111)

    if not period_candidates:
        ax.text(0.5, 0.5, "No period candidates", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ms: List[int] = []
    scores: List[float] = []

    for d in period_candidates:
        try:
            m_raw = d.get("m", None)
            score_raw = d.get("score", None)
            if m_raw is None or score_raw is None:
                continue
            m = int(m_raw)
            score = float(score_raw)
            if not np.isfinite(score):
                continue
            ms.append(m)
            scores.append(score)
        except Exception:
            continue

    if not ms:
        ax.text(0.5, 0.5, "No valid (m, score) pairs", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Sort by m for readability
    order = np.argsort(ms)
    ms_sorted = [ms[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    ax.plot(ms_sorted, scores_sorted, marker="o")
    ax.set_title(title, fontsize=th.title_size)
    ax.set_xlabel("Period (m)", fontsize=th.label_size)
    ax.set_ylabel("Score", fontsize=th.label_size)
    return fig


def stationarity_scatter_plot(
    rows: Sequence[Dict[str, Any]],
    title: str = "ADF vs KPSS",
    theme: Optional[PlotTheme] = None,
) -> plt.Figure:
    """
    Plot a stationarity scatter: ADF p-value vs KPSS p-value.

    Expected row keys (best-effort):
      {"variant": str, "adf_p": float|None, "kpss_p": float|None}

    Robustness:
      - Skips rows with missing/non-finite p-values.
      - Never crashes on malformed payload.
    """
    th = apply_theme(theme)
    fig = plt.figure(figsize=th.figsize)
    ax = fig.add_subplot(111)

    if not rows:
        ax.text(0.5, 0.5, "No stationarity results", ha="center", va="center")
        ax.set_axis_off()
        return fig

    xs: List[float] = []
    ys: List[float] = []

    for r in rows:
        try:
            a = r.get("adf_p", None)
            k = r.get("kpss_p", None)
            if a is None or k is None:
                continue
            af = float(a)
            kf = float(k)
            if not (np.isfinite(af) and np.isfinite(kf)):
                continue
            xs.append(af)
            ys.append(kf)
        except Exception:
            continue

    if not xs:
        ax.text(0.5, 0.5, "No finite p-values", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.scatter(xs, ys)
    ax.set_title(title, fontsize=th.title_size)
    ax.set_xlabel("ADF p-value (lower is better)", fontsize=th.label_size)
    ax.set_ylabel("KPSS p-value (higher is better)", fontsize=th.label_size)
    return fig


def acf_plot_from_payload(
    payload: Dict[str, Any],
    title: str = "ACF",
    theme: Optional[PlotTheme] = None,
) -> plt.Figure:
    """
    Plot ACF from a payload dict as a stem plot with optional confidence bands.

    Expected payload keys
    ---------------------
    lags : list[int]
    acf  : list[float]
    conf : float (optional)  # e.g., 1.96/sqrt(n) or another band height

    Rendering guarantees
    --------------------
    - Never raises due to missing/empty payload.
    - Truncates lags/values to the same length if inconsistent.
    """
    th = apply_theme(theme)
    fig = plt.figure(figsize=th.figsize)
    ax = fig.add_subplot(111)

    try:
        lags_raw = payload.get("lags") or []
        vals_raw = payload.get("acf") or []
        conf = payload.get("conf", None)

        lags = list(lags_raw)
        vals = list(vals_raw)

        if len(lags) == 0 or len(vals) == 0:
            ax.text(0.5, 0.5, "ACF not available", ha="center", va="center")
            ax.set_axis_off()
            return fig

        n = min(len(lags), len(vals))
        lags = [int(x) for x in lags[:n]]
        vals = [float(x) for x in vals[:n]]

        # Drop non-finite pairs
        pairs = [(L, V) for (L, V) in zip(lags, vals) if np.isfinite(V)]
        if not pairs:
            ax.text(0.5, 0.5, "No finite ACF values", ha="center", va="center")
            ax.set_axis_off()
            return fig

        lags2, vals2 = zip(*pairs)

        ax.axhline(0.0, linewidth=1.0)
        ax.stem(lags2, vals2)  # no use_line_collection for compatibility

        if conf is not None:
            c = float(conf)
            ax.axhline(+c, linestyle="--")
            ax.axhline(-c, linestyle="--")

        ax.set_title(title, fontsize=th.title_size)
        ax.set_xlabel("Lag", fontsize=th.label_size)
        ax.set_ylabel("ACF", fontsize=th.label_size)
        ax.set_xlim(min(lags2) - 0.5, max(lags2) + 0.5)
        return fig

    except Exception as e:
        ax.text(0.5, 0.5, f"ACF render error: {e}", ha="center", va="center")
        ax.set_axis_off()
        return fig


def pacf_plot_from_payload(
    payload: Dict[str, Any],
    title: str = "PACF",
    theme: Optional[PlotTheme] = None,
) -> plt.Figure:
    """
    Plot PACF from a payload dict as a stem plot with optional confidence bands.

    Expected payload keys
    ---------------------
    lags : list[int]
    pacf : list[float]
    conf : float (optional)

    Rendering guarantees
    --------------------
    - Never raises due to missing/empty payload.
    - Truncates lags/values to the same length if inconsistent.
    """
    th = apply_theme(theme)
    fig = plt.figure(figsize=th.figsize)
    ax = fig.add_subplot(111)

    try:
        lags_raw = payload.get("lags") or []
        vals_raw = payload.get("pacf") or []
        conf = payload.get("conf", None)

        lags = list(lags_raw)
        vals = list(vals_raw)

        if len(lags) == 0 or len(vals) == 0:
            ax.text(0.5, 0.5, "PACF not available", ha="center", va="center")
            ax.set_axis_off()
            return fig

        n = min(len(lags), len(vals))
        lags = [int(x) for x in lags[:n]]
        vals = [float(x) for x in vals[:n]]

        pairs = [(L, V) for (L, V) in zip(lags, vals) if np.isfinite(V)]
        if not pairs:
            ax.text(0.5, 0.5, "No finite PACF values", ha="center", va="center")
            ax.set_axis_off()
            return fig

        lags2, vals2 = zip(*pairs)

        ax.axhline(0.0, linewidth=1.0)
        ax.stem(lags2, vals2)

        if conf is not None:
            c = float(conf)
            ax.axhline(+c, linestyle="--")
            ax.axhline(-c, linestyle="--")

        ax.set_title(title, fontsize=th.title_size)
        ax.set_xlabel("Lag", fontsize=th.label_size)
        ax.set_ylabel("PACF", fontsize=th.label_size)
        ax.set_xlim(min(lags2) - 0.5, max(lags2) + 0.5)
        return fig

    except Exception as e:
        ax.text(0.5, 0.5, f"PACF render error: {e}", ha="center", va="center")
        ax.set_axis_off()
        return fig


def stl_components_plot(
    components: Dict[str, Any],
    title: str = "STL Components",
    theme: Optional[PlotTheme] = None,
) -> plt.Figure:
    """
    Plot STL decomposition components.

    Expected payload:
      {"trend": Series|None, "seasonal": Series|None, "resid": Series|None, "m": int|None}

    Notes:
      - Uses 3 stacked axes in one figure.
      - Never crashes if components are missing.
    """
    th = apply_theme(theme)
    fig = plt.figure(figsize=(th.figsize[0], th.figsize[1] * 1.6))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    trend = components.get("trend")
    seasonal = components.get("seasonal")
    resid = components.get("resid")

    ax1.set_title(title, fontsize=th.title_size)

    if isinstance(trend, pd.Series) and len(trend) > 0:
        tt = trend.astype("float64").replace([np.inf, -np.inf], np.nan)
        ax1.plot(tt.index, tt.values)
        ax1.set_ylabel("Trend")
    else:
        ax1.text(0.5, 0.5, "Trend not available", ha="center", va="center")
        ax1.set_axis_off()

    if isinstance(seasonal, pd.Series) and len(seasonal) > 0:
        ss = seasonal.astype("float64").replace([np.inf, -np.inf], np.nan)
        ax2.plot(ss.index, ss.values)
        ax2.set_ylabel("Seasonal")
    else:
        ax2.text(0.5, 0.5, "Seasonal not available", ha="center", va="center")
        ax2.set_axis_off()

    if isinstance(resid, pd.Series) and len(resid) > 0:
        rr = resid.astype("float64").replace([np.inf, -np.inf], np.nan)
        ax3.plot(rr.index, rr.values)
        ax3.set_ylabel("Residual")
    else:
        ax3.text(0.5, 0.5, "Residual not available", ha="center", va="center")
        ax3.set_axis_off()

    return fig
