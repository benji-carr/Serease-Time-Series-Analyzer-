# serease/pre_modeling/transforms.py
from __future__ import annotations

from typing import Optional, Tuple, List, Callable
import numpy as np
import pandas as pd

from .containers import (
    SeriesBundle,
    TransformPlan,
    SeriesView,
    TransformStepRecord,
    TransformedSeries,
)


class TransformEngine:
    """Owns transformations and provenance. No diagnostics live here."""

    def _align_exog(self, y: pd.Series, exog: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Align exogenous variables to the transformed y index.
        Current behavior: reindex exog to y's index.
        """
        if exog is None:
            return None
        return exog.reindex(y.index)

    def apply_variance_transform(
        self, y: pd.Series, plan: TransformPlan
    ) -> Tuple[pd.Series, List[TransformStepRecord], Optional[Callable[[pd.Series], pd.Series]]]:
        steps: List[TransformStepRecord] = []
        inverse_fn: Optional[Callable[[pd.Series], pd.Series]] = None

        if plan.variance_transform == "none":
            return y, steps, inverse_fn

        if plan.variance_transform == "log":
            off = float(plan.boxcox_offset)

            # Ensure positivity for log; if not, raise clearly (pipeline should have set offset)
            y_shifted = y.astype(float) + off
            bad = y_shifted <= 0
            if bool(np.any(bad.dropna())):
                min_bad = float(y_shifted.min())
                raise ValueError(
                    f"log transform requires y+offset > 0 everywhere. "
                    f"Found non-positive values (min(y+offset)={min_bad}). "
                    f"Increase boxcox_offset."
                )

            y2 = pd.Series(np.log(y_shifted.values), index=y.index, name=y.name)
            steps.append(TransformStepRecord(name="log", params={"offset": off}, invertible=True))

            def _inv(s: pd.Series) -> pd.Series:
                return pd.Series(np.exp(s.astype(float).values) - off, index=s.index, name=s.name)

            inverse_fn = _inv
            return y2, steps, inverse_fn

        if plan.variance_transform == "boxcox":
            lam = plan.boxcox_lambda
            if lam is None:
                raise ValueError("boxcox_lambda must be set when variance_transform='boxcox'")

            steps.append(TransformStepRecord(name="boxcox", params={"lambda": lam}, invertible=True))
            raise NotImplementedError("Box-Cox transform not implemented yet (policy prefers log).")

        raise ValueError(f"Unknown variance_transform: {plan.variance_transform}")

    def detrend_for_periodogram(self, y: pd.Series, plan: TransformPlan) -> Tuple[pd.Series, List[TransformStepRecord]]:
        steps: List[TransformStepRecord] = []

        if plan.detrend_for_periodogram == "none":
            return y, steps

        if plan.detrend_for_periodogram == "linear":
            # Remove least-squares line on finite points only
            vals = y.astype(float).values
            mask = np.isfinite(vals)
            if int(mask.sum()) < 3:
                raise ValueError("Not enough finite points to fit linear detrend for periodogram.")

            x = np.arange(len(y), dtype=float)
            coef = np.polyfit(x[mask], vals[mask], 1)
            trend = coef[0] * x + coef[1]
            y2 = pd.Series(vals - trend, index=y.index, name=y.name)

            steps.append(
                TransformStepRecord(
                    name="detrend_linear",
                    params={"slope": float(coef[0]), "intercept": float(coef[1])},
                    invertible=False,
                )
            )
            return y2, steps

        if plan.detrend_for_periodogram == "diff1":
            y2 = y.diff().dropna()
            steps.append(TransformStepRecord(name="diff1_for_periodogram", params={}, invertible=False))
            return y2, steps

        raise ValueError(f"Unknown detrend_for_periodogram: {plan.detrend_for_periodogram}")

    def apply_differences(self, y: pd.Series, plan: TransformPlan) -> Tuple[pd.Series, List[TransformStepRecord]]:
        steps: List[TransformStepRecord] = []
        y2 = y

        # Seasonal difference first (canonical SARIMA)
        if plan.D > 0:
            m = plan.seasonal_period_m
            if m is None or m <= 1:
                raise ValueError("seasonal_period_m must be set (>1) before seasonal differencing.")
            for k in range(plan.D):
                y2 = y2.diff(m).dropna()
                steps.append(
                    TransformStepRecord(
                        name="seasonal_diff",
                        params={"m": int(m), "order": int(k + 1)},
                        invertible=False,
                    )
                )

        # Then non-seasonal differencing
        if plan.d > 0:
            for k in range(plan.d):
                y2 = y2.diff().dropna()
                steps.append(
                    TransformStepRecord(
                        name="diff",
                        params={"order": int(k + 1)},
                        invertible=False,
                    )
                )

        return y2, steps

    # ---------------------------
    # View builders
    # ---------------------------
    def build_raw_view(self, bundle: SeriesBundle) -> SeriesView:
        return SeriesView(
            view_id="raw",
            y=bundle.y_raw,
            exog=bundle.exog_raw,
            applied_steps=[],
        )

    def build_var_stable_view(self, bundle: SeriesBundle, plan: TransformPlan) -> SeriesView:
        y, steps, _inv = self.apply_variance_transform(bundle.y_raw, plan)
        exog = self._align_exog(y, bundle.exog_raw)
        return SeriesView(view_id="var_stable", y=y, exog=exog, applied_steps=steps)

    def build_for_periodogram_view(self, bundle: SeriesBundle, plan: TransformPlan) -> SeriesView:
        y1, steps1, _inv = self.apply_variance_transform(bundle.y_raw, plan)
        y2, steps2 = self.detrend_for_periodogram(y1, plan)
        exog = self._align_exog(y2, bundle.exog_raw)
        return SeriesView(view_id="for_periodogram", y=y2, exog=exog, applied_steps=steps1 + steps2)

    def build_stationary_view(self, bundle: SeriesBundle, plan: TransformPlan) -> TransformedSeries:
        # full: variance transform + (D,d)
        y1, steps1, inv = self.apply_variance_transform(bundle.y_raw, plan)
        y2, steps2 = self.apply_differences(y1, plan)
        exog = self._align_exog(y2, bundle.exog_raw)

        return TransformedSeries(
            y=y2,
            exog=exog,
            plan=plan,
            applied_steps=steps1 + steps2,
            inverse_fn=inv,
        )

    def build_stl_interpretation_view(self, bundle: SeriesBundle, plan: TransformPlan) -> SeriesView:
        # Typically: variance stabilized only (no differencing)
        y, steps, _inv = self.apply_variance_transform(bundle.y_raw, plan)
        exog = self._align_exog(y, bundle.exog_raw)
        return SeriesView(view_id="stl_interpretation", y=y, exog=exog, applied_steps=steps)
