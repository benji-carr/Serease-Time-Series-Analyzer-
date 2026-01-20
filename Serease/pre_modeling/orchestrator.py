# serease/pre_modeling/orchestrator.py
from __future__ import annotations

from .containers import PreModelState, SeriesBundle
from .diagnostics import DiagnosticsEngine
from .transforms import TransformEngine
from .views import ViewBuilder
from .steps import build_step_registry
from .selection import StationarityPolicy, select_stationary_candidate


class PreModelOrchestrator:
    def __init__(self, diag: DiagnosticsEngine | None = None, xf: TransformEngine | None = None):
        self.diag = diag or DiagnosticsEngine()
        self.xf = xf or TransformEngine()
        self.views = ViewBuilder(self.xf)

    def run(
            self,
            bundle: SeriesBundle,
            manual_plan: "TransformPlan | None" = None,
            lock_m: bool = False,
            lock_variance: bool = False,
            lock_diffs: bool = False,
    ) -> PreModelState:
        """
        Run the pre-model pipeline.

        manual_plan:
          - If provided, state.plan is initialized from this plan (copied).
        lock_*:
          - If True, the orchestrator will NOT overwrite that part of the plan
            from diagnostic-driven suggestions/selection.
        """
        from dataclasses import replace

        state = PreModelState(bundle=bundle)
        registry = build_step_registry()

        # Initialize from manual plan if provided (copy to avoid external mutation)
        if manual_plan is not None:
            state.plan = replace(manual_plan)

        # -------------------------
        # Step 1: RAW diagnostics
        # -------------------------
        raw_spec = registry[0].spec
        raw_view = self.views.get_view(state, raw_spec)
        state.diagnostics.raw = self.diag.raw_diagnostics(raw_view, freq_hint=bundle.freq)

        # -------------------------
        # Step 2: Variance assessment (RAW)
        # -------------------------
        var_spec = registry[1].spec
        var_view = self.views.get_view(state, var_spec)
        state.diagnostics.variance = self.diag.variance_assessment(var_view, freq_hint=bundle.freq)

        if not lock_variance:
            suggested = state.diagnostics.variance.get("suggested_transform", "none")
            offset = float(state.diagnostics.variance.get("recommended_offset", 0.0) or 0.0)

            if suggested in ("log", "boxcox"):
                # Preference: log over boxcox (interpretability)
                state.plan.variance_transform = "log" if suggested == "boxcox" else suggested
                state.plan.boxcox_offset = offset
                state.diagnostics.notes.append(
                    f"Variance transform set to {state.plan.variance_transform} with offset={state.plan.boxcox_offset:.6g}."
                )
            else:
                state.plan.variance_transform = "none"
                state.plan.boxcox_offset = 0.0
        else:
            state.diagnostics.notes.append(
                f"Manual lock: variance preserved (variance_transform={state.plan.variance_transform}, "
                f"offset={state.plan.boxcox_offset:.6g})."
            )

        # -------------------------
        # Step 3: Periodogram (FOR_PERIODOGRAM view)
        # -------------------------
        pg_spec = registry[2].spec
        pg_view = self.views.get_view(state, pg_spec)
        state.diagnostics.period_candidates = self.diag.periodogram_candidates(pg_view)

        if not lock_m:
            chosen_m = state.diagnostics.period_candidates.get("chosen_m")
            if chosen_m:
                state.plan.seasonal_period_m = int(chosen_m)
                state.diagnostics.notes.append(f"Seasonal period m set to {state.plan.seasonal_period_m}.")
        else:
            state.diagnostics.notes.append(
                f"Manual lock: seasonal_period_m preserved (m={state.plan.seasonal_period_m})."
            )

        # -------------------------
        # Step 4: Stationarity sweep + selection
        # -------------------------
        state.candidates = self.views.build_stationarity_candidates(state)

        alpha_adf = 0.05
        alpha_kpss = 0.05

        stationarity_rows = []
        for ts in state.candidates:
            row = self.diag.stationarity_tests(ts)

            row.setdefault("D", ts.plan.D)
            row.setdefault("d", ts.plan.d)
            row.setdefault("m", ts.plan.seasonal_period_m)

            adf_p = row.get("adf_pvalue")
            kpss_p = row.get("kpss_pvalue")

            passed = False
            if (adf_p is not None) and (kpss_p is not None):
                try:
                    passed = (float(adf_p) <= alpha_adf) and (float(kpss_p) >= alpha_kpss)
                except Exception:
                    passed = False

            row["pass_stationarity"] = passed
            stationarity_rows.append(row)

        state.diagnostics.stationarity_table = stationarity_rows

        policy = StationarityPolicy(
            alpha_adf=0.05,
            alpha_kpss=0.05,
            acf1_soft=-0.30,
            acf1_hard=-0.50,
            max_d=state.plan.max_d,
            max_D=state.plan.max_D,
        )

        best, meta = select_stationary_candidate(
            candidates=state.candidates,
            stationarity_rows=state.diagnostics.stationarity_table,
            policy=policy,
        )

        if best is not None:
            state.final = best

            if not lock_diffs:
                # adopt the chosen stationary candidate plan (includes D/d and carries variance + m too)
                state.plan = best.plan
            else:
                state.diagnostics.notes.append(
                    f"Manual lock: differencing preserved (D={state.plan.D}, d={state.plan.d}). "
                    f"Selected candidate would have been (D={best.plan.D}, d={best.plan.d})."
                )

            state.diagnostics.notes.append(
                f"Selected stationary transform: variance={state.plan.variance_transform}, "
                f"m={state.plan.seasonal_period_m}, D={state.plan.D}, d={state.plan.d}. "
                f"passed={meta.get('passed')}, any_passed={meta.get('any_passed')}."
            )
            state.diagnostics.stationarity_selection = meta
        else:
            state.diagnostics.notes.append("No stationarity candidates were generated/selected.")

        # -------------------------
        # Step 5: ACF/PACF on FINAL_STATIONARY view
        # -------------------------
        if state.final is not None:
            acf_spec = registry[3].spec
            final_view = self.views.get_view(state, acf_spec)
            payload = self.diag.acf_pacf(final_view)

            # Add seasonal lag summary using plan.m if available
            m = state.plan.seasonal_period_m
            if m is not None and payload.get("acf_confint") is not None:
                # DiagnosticsEngine.acf_pacf currently returns no "nlags" key.
                # Use len(acf)-1 as max lag.
                acf_vals = payload.get("acf") or []
                conf = payload.get("acf_confint") or []
                max_lag = len(acf_vals) - 1

                seasonal = []
                for k in [m, 2 * m, 3 * m]:
                    if 0 <= k <= max_lag and k < len(conf):
                        lo, hi = conf[k]
                        seasonal.append(
                            {
                                "lag": int(k),
                                "acf": float(acf_vals[k]),
                                "ci_low": float(lo),
                                "ci_high": float(hi),
                                "significant": (lo > 0) or (hi < 0),
                            }
                        )
                payload["seasonal_lag_summary"] = seasonal

            state.diagnostics.acf_pacf_payload = payload

        # -------------------------
        # Step 6: STL (variance-stable, interpretability view)
        # -------------------------
        if state.plan.seasonal_period_m is not None:
            stl_spec = registry[4].spec
            stl_view = self.views.get_view(state, stl_spec)
            state.diagnostics.stl_components = self.diag.stl(stl_view, m=state.plan.seasonal_period_m)

        return state
