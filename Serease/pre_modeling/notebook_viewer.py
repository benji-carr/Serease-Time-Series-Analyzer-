# Serease/pre_modeling/notebook_viewer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    display = print


# ----------------------------
# Core helpers
# ----------------------------
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

    phase = np.arange(n) % m
    df = pd.DataFrame({"y": s.values, "phase": phase})
    proto = df.groupby("phase")["y"].mean()

    repeated = np.take(proto.values, phase)
    return pd.Series(repeated, index=s.index, name=f"seasonal_proto_m{m}")


# ----------------------------
# Plotters: time + period resemblance
# ----------------------------
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

        plt.figure()
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


# ----------------------------
# Stationarity summary (chosen plan)
# ----------------------------
def print_chosen_stationarity(state) -> None:
    """
    Always-on compact summary for the FINAL selected (m,D,d), without requiring the full table.
    """
    if getattr(state, "final", None) is None:
        print("No final stationary series selected.")
        return

    D = int(getattr(state.plan, "D", 0))
    d = int(getattr(state.plan, "d", 0))
    m = getattr(state.plan, "seasonal_period_m", None)
    m_key = int(m) if m is not None else None

    rows = getattr(state.diagnostics, "stationarity_table", None) or []
    match = None
    for r in rows:
        try:
            rD = int(r.get("D", -999))
            rd = int(r.get("d", -999))
            rm_raw = r.get("m", None)
            rm = int(rm_raw) if rm_raw is not None else None
        except Exception:
            continue

        if rD == D and rd == d and rm == m_key:
            match = r
            break

    print("\n=== CHOSEN STATIONARITY TESTS ===")
    print(f"Selected: m={m_key}, D={D}, d={d}")

    if match is None:
        print("No matching row found in stationarity_table.")
        return

    print("ADF p-value :", match.get("adf_pvalue"))
    print("KPSS p-value:", match.get("kpss_pvalue"))
    print("acf1        :", match.get("acf1"))
    print("pass        :", match.get("pass_stationarity"))


# ----------------------------
# ACF/PACF stem + confidence bands
# ----------------------------
def _plot_stem_with_confint(
    values,
    confint=None,
    title="",
    xlabel="Lag",
    ylabel="",
    band_mode: str = "constant",   # "constant" (±1.96/sqrt(n)) or "per_lag" (statsmodels confint)
    n: int | None = None,
    z: float = 1.96,
    skip_lag0: bool = False,
) -> None:
    """
    Stem plot with either:
      - constant band: ± z/sqrt(n)  (what most people expect)
      - per-lag band: confint returned by statsmodels (often widens with lag)

    NOTE: older matplotlib doesn't support use_line_collection; we handle both.
    """
    vals = np.asarray(values, dtype=float)
    lags = np.arange(len(vals))

    if skip_lag0 and len(vals) > 1:
        vals = vals[1:]
        lags = lags[1:]
        if confint is not None:
            try:
                confint = np.asarray(confint)[1:]
            except Exception:
                confint = None

    plt.figure()
    try:
        plt.stem(lags, vals, use_line_collection=True)
    except TypeError:
        plt.stem(lags, vals)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if band_mode == "constant":
        if n is not None and n > 1:
            bound = z / np.sqrt(float(n))
            plt.axhline(bound, linestyle="--")
            plt.axhline(-bound, linestyle="--")
    elif band_mode == "per_lag":
        if confint is not None:
            ci = np.asarray(confint, dtype=float)
            if ci.shape[0] == len(vals) and ci.shape[1] == 2:
                lo = ci[:, 0]
                hi = ci[:, 1]
                plt.fill_between(lags, lo, hi, alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_acf_pacf_from_payload(
    acf_pacf_payload: Dict[str, Any],
    band_mode: str = "constant",
    skip_lag0: bool = True,
) -> None:
    """
    Produces TWO separate plots:
      - ACF stem + band
      - PACF stem + band

    Expects payload keys (from DiagnosticsEngine.acf_pacf):
      - n
      - acf, pacf
      - acf_confint, pacf_confint  (optional)
    """
    acf_vals = acf_pacf_payload.get("acf")
    pacf_vals = acf_pacf_payload.get("pacf")
    acf_ci = acf_pacf_payload.get("acf_confint")
    pacf_ci = acf_pacf_payload.get("pacf_confint")
    n = acf_pacf_payload.get("n")

    if acf_vals is not None:
        _plot_stem_with_confint(
            values=acf_vals,
            confint=acf_ci,
            title="ACF",
            ylabel="ACF",
            band_mode=band_mode,
            n=n,
            skip_lag0=skip_lag0,
        )

    if pacf_vals is not None:
        _plot_stem_with_confint(
            values=pacf_vals,
            confint=pacf_ci,
            title="PACF",
            ylabel="PACF",
            band_mode=band_mode,
            n=n,
            skip_lag0=skip_lag0,
        )


# ----------------------------
# STL plotting
# ----------------------------
def plot_stl_components(stl_components: Dict[str, Any], title_prefix: str = "STL") -> None:
    """
    Three separate plots: trend, seasonal, resid.
    STL output currently stores lists without an index; we plot by position.
    """
    if not stl_components or "trend" not in stl_components:
        print("STL components not available.")
        return

    trend = stl_components.get("trend", [])
    seasonal = stl_components.get("seasonal", [])
    resid = stl_components.get("resid", [])

    x = np.arange(len(trend))

    plt.figure()
    plt.plot(x, trend)
    plt.title(f"{title_prefix}: Trend")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(x, seasonal)
    plt.title(f"{title_prefix}: Seasonal")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(x, resid)
    plt.title(f"{title_prefix}: Residual")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Stationarity table (opt-in)
# ----------------------------
def show_stationarity_table(rows: List[Dict[str, Any]], max_rows: int = 50) -> None:
    """
    Opt-in display of stationarity table.
    """
    if not rows:
        print("No stationarity rows.")
        return

    df = pd.DataFrame(rows).copy()
    preferred_cols = [
        "D", "d", "m", "n",
        "adf_pvalue", "kpss_pvalue", "acf1",
        "pass_stationarity",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    display(df.head(int(max_rows)))


# ----------------------------
# Main viewer
# ----------------------------
class NotebookPreModelViewer:
    """
    Notebook viewer for your pre-modeling output.
    Works best with PreModelState.

    Defaults:
      - stationarity table is opt-in
      - ACF/PACF use constant band ±1.96/sqrt(n) (typical "significance" band)
      - ACF/PACF skip lag 0 by default (cleaner)
    """

    def __init__(
        self,
        show_stationarity: bool = False,
        acf_band_mode: str = "constant",   # "constant" or "per_lag"
        skip_lag0: bool = True,
    ):
        self.show_stationarity = show_stationarity
        self.acf_band_mode = acf_band_mode
        self.skip_lag0 = skip_lag0

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

        # Always show chosen ADF/KPSS summary (without forcing the whole table)
        print_chosen_stationarity(state)

        # Stationarity table (opt-in)
        if self.show_stationarity:
            show_stationarity_table(state.diagnostics.stationarity_table)

        # ACF/PACF (final)
        payload = state.diagnostics.acf_pacf_payload or {}
        if payload:
            plot_acf_pacf_from_payload(payload, band_mode=self.acf_band_mode, skip_lag0=self.skip_lag0)
        else:
            print("ACF/PACF payload not available (final series may not have been selected).")

        # STL
        stl = state.diagnostics.stl_components or {}
        if stl:
            plot_stl_components(stl, title_prefix="STL")
        else:
            if state.plan.seasonal_period_m is None:
                print("STL skipped: seasonal_period_m not set.")
            else:
                print("STL components not available (STL may have failed or not run).")

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
            plot_acf_pacf_from_payload(payload, band_mode=self.acf_band_mode, skip_lag0=self.skip_lag0)

        stl = artifacts.get("stl_components", {})
        if stl:
            plot_stl_components(stl, title_prefix="STL")
