# Serease/interactive/manual_pre_model.py
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import pandas as pd

from Serease.pre_processing import DataIngestor, SchemaDetector, TimeSeriesCleaner
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


# ----------------------------
# Helpers
# ----------------------------
def _prompt(msg: str) -> str:
    return input(msg).strip()


def _prompt_int(msg: str, default: Optional[int] = None) -> Optional[int]:
    s = _prompt(msg)
    if s == "" and default is not None:
        return default
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        return None


def _prompt_float(msg: str, default: Optional[float] = None) -> Optional[float]:
    s = _prompt(msg)
    if s == "" and default is not None:
        return default
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def print_plan(plan: TransformPlan) -> None:
    print("\n=== CURRENT TRANSFORM PLAN ===")
    print("variance_transform      :", plan.variance_transform)
    print("boxcox_lambda           :", plan.boxcox_lambda)
    print("boxcox_offset           :", plan.boxcox_offset)
    print("detrend_for_periodogram :", plan.detrend_for_periodogram)
    print("seasonal_period_m       :", plan.seasonal_period_m)
    print("D (seasonal diff order) :", plan.D)
    print("d (regular diff order)  :", plan.d)
    print("max_D                   :", plan.max_D)
    print("max_d                   :", plan.max_d)
    print("================================\n")


# ----------------------------
# Main workbench
# ----------------------------
def run_manual_pre_model(csv_path: str, user_target_col: Optional[str] = None):
    # ----------------------------
    # Load + schema + clean
    # ----------------------------
    ingestor = DataIngestor(csv_path)
    df = ingestor.load()
    ing_meta = ingestor.get_metadata()

    schema = SchemaDetector(df=df, ingestion_meta=ing_meta, user_target_col=user_target_col).detect()
    cleaned_df, ts_meta = TimeSeriesCleaner(df=df, schema=schema).clean()

    y = cleaned_df[schema.target_col].copy()
    if not isinstance(y.index, pd.DatetimeIndex):
        y.index = pd.to_datetime(y.index)
    y.name = schema.target_col

    # Working series (resampling changes this)
    y_working = y.copy()
    freq_hint = getattr(ts_meta, "freq", None)

    # Manual plan + save slot
    plan = TransformPlan()
    saved_plan = replace(plan)

    # Lock toggles (default to True so manual settings stick)
    lock_m = True
    lock_variance = True
    lock_diffs = True

    orch = PreModelOrchestrator()

    print("\n=== MANUAL PRE-MODEL WORKBENCH ===")
    print(f"Series: {schema.target_col}")
    print(f"Original freq: {freq_hint}")
    print("----------------------------------")

    while True:
        print("\nMenu:")
        print(" 0) Show current TransformPlan + locks")
        print(" 1) View time plot")
        print(" 2) Resample frequency (preview)")
        print(" 3) Rolling mean / variance (raw diagnostics)")
        print(" 4) Periodogram candidates + resemblance plots")
        print(" 5) Set seasonal period m + STL preview")
        print(" 6) Set variance transform + offset")
        print(" 7) Set differencing orders (D, d)")
        print(" 8) Set detrend_for_periodogram")
        print(" 9) Save plan")
        print("10) Revert to saved plan")
        print("11) Reset plan to defaults")
        print("12) Toggle locks (m/variance/diffs)")
        print("13) Run pipeline using current plan (with locks)")
        print(" q) Quit")

        cmd = _prompt("Choose option: ").strip().lower()

        if cmd in ("q", "quit", "exit"):
            print("Exiting workbench.")
            break

        # ----------------------------
        if cmd == "0":
            print_plan(plan)
            print("Locks:")
            print("  lock_m       :", lock_m)
            print("  lock_variance:", lock_variance)
            print("  lock_diffs   :", lock_diffs)

        # ----------------------------
        elif cmd == "1":
            plot_time_series(y_working, title="Time plot (working series)")

        # ----------------------------
        elif cmd == "2":
            rule = _prompt("Enter resample rule (e.g. H, D, W, MS) or blank to cancel: ").strip()
            if not rule:
                continue
            how = _prompt("Aggregation [sum/mean] (default sum): ").strip().lower() or "sum"

            if how == "mean":
                y_working = y_working.resample(rule).mean().dropna()
            else:
                y_working = y_working.resample(rule).sum(min_count=1).dropna()

            freq_hint = rule
            print(f"Resampled to {rule} using {how}. n={len(y_working)}")
            plot_time_series(y_working, title=f"Resampled ({rule}, {how})")

        # ----------------------------
        elif cmd == "3":
            # Run pipeline once to populate raw diagnostics (plan doesn't matter much here)
            bundle = SeriesBundle(y_raw=y_working, exog_raw=None, freq=freq_hint, name=y_working.name)
            state = orch.run(bundle, manual_plan=plan, lock_m=lock_m, lock_variance=lock_variance, lock_diffs=lock_diffs)
            plot_rolling_mean_var(state.diagnostics.raw)

        # ----------------------------
        elif cmd == "4":
            bundle = SeriesBundle(y_raw=y_working, exog_raw=None, freq=freq_hint, name=y_working.name)
            state = orch.run(bundle, manual_plan=plan, lock_m=lock_m, lock_variance=lock_variance, lock_diffs=lock_diffs)

            cands = state.diagnostics.period_candidates or {}
            m_vals = [c.get("m") for c in cands.get("candidates", []) if c.get("m") is not None]

            if not m_vals:
                print("No period candidates available.")
            else:
                print("Candidate m values:", m_vals)
                plot_period_resemblance(y_working, m_vals, max_candidates=5)

        # ----------------------------
        elif cmd == "5":
            m = _prompt_int("Enter seasonal period m (integer), blank to cancel: ")
            if m is None:
                continue
            plan.seasonal_period_m = int(m)
            print(f"Plan seasonal_period_m set to {plan.seasonal_period_m}")

            # STL preview (use working series)
            plot_stl_like_statsmodels(y_working, m=int(m), robust=True, title=f"STL preview (m={m})")

        # ----------------------------
        elif cmd == "6":
            vt = _prompt("Variance transform [none/log] (blank to cancel): ").strip().lower()
            if vt == "":
                continue
            if vt not in ("none", "log"):
                print("Unsupported. Use 'none' or 'log'.")
                continue
            plan.variance_transform = vt

            off = _prompt_float("boxcox_offset (for log positivity), default keep current: ", default=plan.boxcox_offset)
            if off is not None:
                plan.boxcox_offset = float(off)

            print(f"Plan variance_transform={plan.variance_transform}, offset={plan.boxcox_offset}")

        # ----------------------------
        elif cmd == "7":
            D = _prompt_int("Set D (seasonal diff order, 0/1/..): ", default=plan.D)
            d = _prompt_int("Set d (regular diff order, 0/1/2..): ", default=plan.d)
            if D is not None:
                plan.D = int(D)
            if d is not None:
                plan.d = int(d)
            print(f"Plan differencing set: D={plan.D}, d={plan.d}")

        # ----------------------------
        elif cmd == "8":
            dt = _prompt("detrend_for_periodogram [none/linear/diff1] (blank to cancel): ").strip().lower()
            if dt == "":
                continue
            if dt not in ("none", "linear", "diff1"):
                print("Invalid. Use none/linear/diff1.")
                continue
            plan.detrend_for_periodogram = dt
            print(f"Plan detrend_for_periodogram set to {plan.detrend_for_periodogram}")

        # ----------------------------
        elif cmd == "9":
            saved_plan = replace(plan)
            print("Saved current plan.")

        # ----------------------------
        elif cmd == "10":
            plan = replace(saved_plan)
            print("Reverted to saved plan.")
            print_plan(plan)

        # ----------------------------
        elif cmd == "11":
            plan = TransformPlan()
            print("Plan reset to defaults.")
            print_plan(plan)

        # ----------------------------
        elif cmd == "12":
            print("\nToggle locks (current):")
            print(" 1) lock_m        :", lock_m)
            print(" 2) lock_variance :", lock_variance)
            print(" 3) lock_diffs    :", lock_diffs)
            which = _prompt("Toggle which [1/2/3] (blank cancel): ").strip()
            if which == "1":
                lock_m = not lock_m
            elif which == "2":
                lock_variance = not lock_variance
            elif which == "3":
                lock_diffs = not lock_diffs
            print("Updated locks:", lock_m, lock_variance, lock_diffs)

        # ----------------------------
        elif cmd == "13":
            bundle = SeriesBundle(y_raw=y_working, exog_raw=None, freq=freq_hint, name=y_working.name)
            state = orch.run(bundle, manual_plan=plan, lock_m=lock_m, lock_variance=lock_variance, lock_diffs=lock_diffs)

            print_plan(state.plan)
            print_chosen_stationarity(state)

            if state.diagnostics.acf_pacf_payload:
                plot_acf_pacf_from_payload(state.diagnostics.acf_pacf_payload)

            # STL preview using final m (if any)
            if state.plan.seasonal_period_m is not None:
                plot_stl_like_statsmodels(y_working, m=int(state.plan.seasonal_period_m), robust=True,
                                          title=f"STL (m={state.plan.seasonal_period_m})")

            print("\nNotes:")
            for n in state.diagnostics.notes:
                print(" -", n)

        else:
            print("Unknown option.")
