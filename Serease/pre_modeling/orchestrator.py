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

    def run(self, bundle: SeriesBundle) -> PreModelState:
        state = PreModelState(bundle=bundle)
        registry = build_step_registry()

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

        # Update plan based on variance evidence
        suggested = state.diagnostics.variance.get("suggested_transform", "none")
        offset = float(state.diagnostics.variance.get("recommended_offset", 0.0) or 0.0)
        # Your stated preference: use log for interpretability if stabilization is needed.
        if suggested in ("log", "boxcox"):
            # your preference: log over boxcox
            state.plan.variance_transform = "log" if suggested == "boxcox" else suggested
            state.plan.boxcox_offset = offset
            # optionally store lambda if you ever want to use it
            # state.plan.boxcox_lambda = state.diagnostics.variance.get("boxcox_lambda_mle")
            state.diagnostics.notes.append(
                f"Variance transform set to {state.plan.variance_transform} with offset={state.plan.boxcox_offset:.6g}."
            )
        else:
            state.plan.variance_transform = "none"
            state.plan.boxcox_offset = 0.0

        # -------------------------
        # Step 3: Periodogram (FOR_PERIODOGRAM view)
        # -------------------------
        pg_spec = registry[2].spec
        pg_view = self.views.get_view(state, pg_spec)
        state.diagnostics.period_candidates = self.diag.periodogram_candidates(pg_view)

        # Update plan.m if chosen_m exists
        chosen_m = state.diagnostics.period_candidates.get("chosen_m")
        if chosen_m:
            state.plan.seasonal_period_m = int(chosen_m)
            state.diagnostics.notes.append(f"Seasonal period m set to {state.plan.seasonal_period_m}.")

        # -------------------------
        # Step 4: Stationarity sweep + selection
        #   - build candidate transforms (in canonical preference order)
        #   - run stationarity tests (ADF/KPSS) + compute acf1 for overdiff guard (once implemented)
        #   - select: least-differenced passing candidate + guardrails
        #     with seasonal preference: (D=1,d=0) preferred over (D=0,d=1) on ties
        # -------------------------
        state.candidates = self.views.build_stationarity_candidates(state)

        # thresholds used for the explicit pass/fail annotation
        alpha_adf = 0.05
        alpha_kpss = 0.05

        stationarity_rows = []
        for ts in state.candidates:
            row = self.diag.stationarity_tests(ts)

            # Ensure these keys exist for selection indexing & reporting
            row.setdefault("D", ts.plan.D)
            row.setdefault("d", ts.plan.d)
            row.setdefault("m", ts.plan.seasonal_period_m)

            # Explicit pass/fail (for reporting/debugging)
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
            state.plan = best.plan

            state.diagnostics.notes.append(
                f"Selected stationary transform: variance={state.plan.variance_transform}, "
                f"m={state.plan.seasonal_period_m}, D={state.plan.D}, d={state.plan.d}. "
                f"passed={meta.get('passed')}, any_passed={meta.get('any_passed')}."
            )

            # Store selection metadata for reporting/debugging
            # (You can move this to its own packet field later)
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
                max_lag = payload["nlags"]
                acf = payload["acf"]
                conf = payload["acf_confint"]

                seasonal = []
                for k in [m, 2 * m, 3 * m]:
                    if k <= max_lag:
                        lo, hi = conf[k]
                        seasonal.append({
                            "lag": k,
                            "acf": acf[k],
                            "ci_low": lo,
                            "ci_high": hi,
                            "significant": (lo > 0) or (hi < 0),
                        })
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
