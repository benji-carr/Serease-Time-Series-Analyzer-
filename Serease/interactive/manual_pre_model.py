# Serease/interactive/manual_pre_model.py
from __future__ import annotations

import sys
from typing import Optional

import pandas as pd

from Serease.pre_processing import (
    DataIngestor,
    SchemaDetector,
    TimeSeriesCleaner,
)

from Serease.pre_modeling.containers import SeriesBundle, TransformPlan
from Serease.pre_modeling.orchestrator import PreModelOrchestrator
from Serease.pre_modeling.notebook_viewer import (
    plot_time_series,
    plot_rolling_mean_var,
    plot_period_resemblance,
    plot_acf_pacf_from_payload,
    plot_stl_like_statsmodels,
    print_chosen_stationarity,
)


# -------------------------------------------------
# Utilities (pure helpers)
# -------------------------------------------------
def _prompt(msg: str) -> str:
    return input(msg).strip().lower()


# -------------------------------------------------
# Manual controller
# -------------------------------------------------
def run_manual_pre_model(
    csv_path: str,
    user_target_col: Optional[str] = None,
):
    """
    Human-in-the-loop pre-modeling workbench.
    Intended for notebooks / REPL use.
    """

    # ----------------------------
    # Load + clean
    # ----------------------------
    ingestor = DataIngestor(csv_path)
    df = ingestor.load()
    meta = ingestor.get_metadata()

    schema = SchemaDetector(
        df=df,
        ingestion_meta=meta,
        user_target_col=user_target_col,
    ).detect()

    cleaned_df, ts_meta = TimeSeriesCleaner(df, schema).clean()

    y = cleaned_df[schema.target_col].copy()
    if not isinstance(y.index, pd.DatetimeIndex):
        y.index = pd.to_datetime(y.index)
    y.name = schema.target_col

    # Mutable working series
    y_working = y.copy()
    freq_hint = ts_meta.freq

    # Current manual decisions
    plan = TransformPlan()

    print("\n=== MANUAL PRE-MODEL WORKBENCH ===")
    print(f"Series: {schema.target_col}")
    print(f"Original freq: {freq_hint}")
    print("---------------------------------\n")

    # ----------------------------
    # Main menu loop
    # ----------------------------
    while True:
        print("\nMenu:")
        print(" 1) View time plot")
        print(" 2) Resample frequency")
        print(" 3) Rolling mean / variance")
        print(" 4) Periodogram + candidate m")
        print(" 5) Test seasonal period m (STL)")
        print(" 6) Test variance transform")
        print(" 7) Run ADF / KPSS on current plan")
        print(" 8) Run full pre-model pipeline")
        print(" q) Quit")

        cmd = _prompt("Choose option: ")

        # ----------------------------
        if cmd == "1":
            plot_time_series(y_working, title="Time plot")

        # ----------------------------
        elif cmd == "2":
            rule = _prompt("Enter resample rule (e.g. H, D, W) or blank to cancel: ")
            if rule:
                how = _prompt("Aggregation [sum/mean]: ") or "sum"
                if how == "sum":
                    y_working = y_working.resample(rule).sum(min_count=1).dropna()
                else:
                    y_working = y_working.resample(rule).mean().dropna()
                freq_hint = rule
                plot_time_series(y_working, title=f"Resampled ({rule}, {how})")

        # ----------------------------
        elif cmd == "3":
            # requires running raw diagnostics once
            bundle = SeriesBundle(y_working, freq=freq_hint, name=y_working.name)
            state = PreModelOrchestrator().run(bundle)
            plot_rolling_mean_var(state.diagnostics.raw)

        # ----------------------------
        elif cmd == "4":
            bundle = SeriesBundle(y_working, freq=freq_hint, name=y_working.name)
            state = PreModelOrchestrator().run(bundle)
            cands = state.diagnostics.period_candidates or {}
            m_vals = [c["m"] for c in cands.get("candidates", []) if "m" in c]
            plot_period_resemblance(y_working, m_vals)

        # ----------------------------
        elif cmd == "5":
            m = _prompt("Enter seasonal period m (integer): ")
            try:
                m = int(m)
                plot_stl_like_statsmodels(y_working, m=m, robust=True, title=f"STL (m={m})")
                plan.seasonal_period_m = m
            except Exception:
                print("Invalid m.")

        # ----------------------------
        elif cmd == "6":
            vt = _prompt("Variance transform [none/log]: ")
            if vt in ("none", "log"):
                plan.variance_transform = vt
                print(f"Variance transform set to {vt}")

        # ----------------------------
        elif cmd == "7":
            bundle = SeriesBundle(y_working, freq=freq_hint, name=y_working.name)
            state = PreModelOrchestrator().run(bundle)
            print_chosen_stationarity(state)

        # ----------------------------
        elif cmd == "8":
            bundle = SeriesBundle(y_working, freq=freq_hint, name=y_working.name)
            state = PreModelOrchestrator().run(bundle)
            print("\nFinal plan:")
            print(state.plan)
            print_chosen_stationarity(state)
            plot_acf_pacf_from_payload(state.diagnostics.acf_pacf_payload)

        # ----------------------------
        elif cmd in ("q", "quit", "exit"):
            print("Exiting manual pre-model workbench.")
            break

        else:
            print("Unknown option.")
