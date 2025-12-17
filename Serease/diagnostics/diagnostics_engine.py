from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, List, Tuple

import numpy as np
import pandas as pd

from .report_types import DiagnosticsReport, StepResult


@dataclass
class DiagnosticsConfig:
    """
    Configuration for DiagnosticsEngine.

    Keep this conservative for MVP: prefer stability and minimal surprise.
    """
    dataset_name: str = "dataset"

    # Display / size limits (reporter may also impose its own).
    max_rows_table: int = 200
    max_variants_stationarity_table: int = 25
    max_lags_acf_pacf: int = 48

    # Step toggles (engine side). Reporter has its own config for display.
    enable_exog_step: bool = True

    # ACF/PACF confidence band: 95% default (two-sided)
    acf_pacf_z: float = 1.96


class DiagnosticsEngine:
    """
    Computes diagnostics artifacts and returns a DiagnosticsReport.

    Contract:
      - Step names must remain stable:
          "overview", "missingness", "seasonality_assessment", "variant_selection",
          "stationarity_sweep", "acf_pacf", "stl", "exog" (optional)

      - Artifact names must remain stable:
          missingness:            "missing_blocks"
          seasonality_assessment: "period_candidates"
          variant_selection:      "selection_ranked"
          stationarity_sweep:     "stationarity_table"
          acf_pacf:               "acf_pacf_payload"
          stl:                    "stl_components"
    """

    def __init__(self, config: Optional[DiagnosticsConfig] = None) -> None:
        """Create an engine with an optional config."""
        self.config = config or DiagnosticsConfig()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(
        self,
        cleaned_df: pd.DataFrame,
        schema_meta: Optional[Any] = None,
        ts_meta: Optional[Any] = None,
        transform_bundle: Optional[Any] = None,
        df_raw: Optional[pd.DataFrame] = None,
        ingestion_meta: Optional[Any] = None,
    ) -> DiagnosticsReport:
        """
        Run diagnostics and return a DiagnosticsReport.

        Parameters
        ----------
        cleaned_df:
            Cleaned time series dataframe with DateTimeIndex.
        schema_meta:
            Object describing target/date/exog columns (shape defined upstream).
        ts_meta:
            Object with time-series metadata (freq, missingness, etc.).
        transform_bundle:
            Output from TimeSeriesTransformer.fit; used for stationarity sweep + variants.
        df_raw, ingestion_meta:
            Optional debugging context.

        Returns
        -------
        DiagnosticsReport
            Contains StepResult entries for all stable step names.
        """
        report = DiagnosticsReport(dataset_name=self.config.dataset_name)

        # Always include all stable steps (even if empty).
        report.add(self._step_overview(cleaned_df, schema_meta, ts_meta))
        report.add(self._step_missingness(cleaned_df, ts_meta))
        report.add(self._step_seasonality_assessment(cleaned_df, ts_meta))
        report.add(self._step_variant_selection(transform_bundle))
        report.add(self._step_stationarity_sweep(transform_bundle))
        report.add(self._step_acf_pacf(cleaned_df, schema_meta, ts_meta))
        report.add(self._step_stl(cleaned_df, ts_meta))

        if self.config.enable_exog_step:
            report.add(self._step_exog(cleaned_df, schema_meta))

        report.meta.update(
            {
                "has_transform_bundle": transform_bundle is not None,
                "n_rows": int(len(cleaned_df)),
                "n_cols": int(cleaned_df.shape[1]),
            }
        )
        return report

    # ---------------------------------------------------------------------
    # Internal helpers (small + stable)
    # ---------------------------------------------------------------------
    def _get_target_col(self, cleaned_df: pd.DataFrame, schema_meta: Optional[Any]) -> str:
        """
        Determine target column:
          1) schema_meta.target_col if present
          2) fallback: first numeric column
          3) fallback: first column
        """
        if schema_meta is not None:
            tc = getattr(schema_meta, "target_col", None)
            if isinstance(tc, str) and tc in cleaned_df.columns:
                return tc

        numeric_cols = [c for c in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[c])]
        if numeric_cols:
            return str(numeric_cols[0])

        # last resort: first col
        return str(cleaned_df.columns[0])

    def _safe_series(self, cleaned_df: pd.DataFrame, col: str) -> pd.Series:
        """Return numeric float series, preserving index; coercing non-numeric to NaN."""
        s = cleaned_df[col]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        return s.astype("float64")

    def _contiguous_nan_blocks(self, s: pd.Series) -> List[Dict[str, Any]]:
        """
        Identify contiguous NaN runs in a series.

        Returns list of dict rows:
          {"col": str, "start": timestamp, "end": timestamp, "n": int}
        """
        is_na = s.isna().to_numpy()
        if is_na.size == 0 or not is_na.any():
            return []

        idx = s.index
        blocks: List[Dict[str, Any]] = []
        start_i: Optional[int] = None

        for i, flag in enumerate(is_na):
            if flag and start_i is None:
                start_i = i
            elif (not flag) and start_i is not None:
                end_i = i - 1
                blocks.append(
                    {
                        "start": str(idx[start_i]),
                        "end": str(idx[end_i]),
                        "n": int(end_i - start_i + 1),
                    }
                )
                start_i = None

        if start_i is not None:
            end_i = len(is_na) - 1
            blocks.append(
                {"start": str(idx[start_i]), "end": str(idx[end_i]), "n": int(end_i - start_i + 1)}
            )

        return blocks

    def _get_seasonal_period_trusted(self, ts_meta: Optional[Any]) -> Optional[int]:
        """
        Conservative seasonal-period retrieval.

        We do NOT infer m here. We only use:
          - ts_meta.seasonal_period (if present and >=2)
          - ts_meta.m (if present and >=2)
        Otherwise return None.
        """
        if ts_meta is None:
            return None
        for attr in ("seasonal_period", "m"):
            v = getattr(ts_meta, attr, None)
            if v is None:
                continue
            try:
                mi = int(v)
                if mi >= 2:
                    return mi
            except Exception:
                continue
        return None

    # ---------------------------------------------------------------------
    # Step implementations
    # ---------------------------------------------------------------------
    def _step_overview(
        self,
        cleaned_df: pd.DataFrame,
        schema_meta: Optional[Any],
        ts_meta: Optional[Any],
    ) -> StepResult:
        """
        Step: "overview"

        Purpose:
          - Provide basic dataset/time-range stats that contextualize the report.
        """
        step = StepResult(step_name="overview")

        try:
            step.summary["n_obs"] = int(len(cleaned_df))
            step.summary["n_cols"] = int(cleaned_df.shape[1])

            if isinstance(cleaned_df.index, pd.DatetimeIndex) and len(cleaned_df) > 0:
                step.summary["start"] = str(cleaned_df.index.min())
                step.summary["end"] = str(cleaned_df.index.max())
                step.summary["index_is_monotonic"] = bool(cleaned_df.index.is_monotonic_increasing)
        except Exception as e:
            step.warnings.append(f"overview failed to compute basic stats: {e}")

        if schema_meta is not None:
            step.summary["schema_target_col"] = getattr(schema_meta, "target_col", None)
            step.summary["schema_date_col"] = getattr(schema_meta, "date_col", None)
            step.summary["schema_exog_cols_n"] = (
                len(getattr(schema_meta, "exog_cols", []) or []) if schema_meta is not None else 0
            )

        if ts_meta is not None:
            step.summary["ts_freq"] = getattr(ts_meta, "freq", None)
            step.summary["ts_n_obs"] = getattr(ts_meta, "n_obs", None)
            step.summary["ts_missing_dates"] = getattr(ts_meta, "n_missing_timestamps", None)

        return step

    def _step_missingness(self, cleaned_df: pd.DataFrame, ts_meta: Optional[Any]) -> StepResult:
        """
        Step: "missingness"
        Artifact: "missing_blocks"

        Computes contiguous NaN blocks per column (bounded table size).
        """
        step = StepResult(step_name="missingness")

        blocks: List[Dict[str, Any]] = []
        total_missing = 0

        try:
            for col in cleaned_df.columns:
                s = cleaned_df[col]
                nmiss = int(pd.isna(s).sum())
                total_missing += nmiss
                if nmiss == 0:
                    continue

                col_blocks = self._contiguous_nan_blocks(s)
                for b in col_blocks:
                    b["col"] = str(col)
                blocks.extend(col_blocks)

            # Truncate for safety
            blocks = blocks[: self.config.max_rows_table]

            step.summary["n_missing_total"] = int(total_missing)
            step.summary["n_missing_blocks"] = int(len(blocks))
            step.add_artifact("missing_blocks", payload=blocks)

            if total_missing == 0:
                step.notes.append("No missing values detected in cleaned_df.")
            else:
                step.notes.append("Missingness blocks computed as contiguous NaN runs per column.")

        except Exception as e:
            step.warnings.append(f"missingness step failed: {e}")
            step.add_artifact("missing_blocks", payload=[])

        return step

    def _step_seasonality_assessment(self, cleaned_df: pd.DataFrame, ts_meta: Optional[Any]) -> StepResult:
        """
        Step: "seasonality_assessment"
        Artifact: "period_candidates"

        Conservative MVP:
          - Do NOT infer m.
          - If ts_meta includes period candidates (precomputed upstream), pass them through.
          - Else, return empty list and a note.
        """
        step = StepResult(step_name="seasonality_assessment")

        candidates: List[Dict[str, Any]] = []
        try:
            if ts_meta is not None:
                # If upstream already computed candidates, pass through
                pc = getattr(ts_meta, "period_candidates", None)
                if isinstance(pc, list):
                    candidates = pc[: self.config.max_rows_table]

            step.add_artifact("period_candidates", payload=candidates)

            if candidates:
                step.summary["n_candidates"] = int(len(candidates))
                step.notes.append("Period candidates were provided by ts_meta (no inference performed here).")
            else:
                step.summary["n_candidates"] = 0
                step.notes.append("No period candidates available. MVP does not infer seasonal period automatically.")

        except Exception as e:
            step.warnings.append(f"seasonality assessment failed: {e}")
            step.add_artifact("period_candidates", payload=[])

        return step

    def _step_stationarity_sweep(self, transform_bundle: Optional[Any]) -> StepResult:
        """
        Step: "stationarity_sweep"
        Artifact: "stationarity_table"

        Uses TransformBundle.metas (no recomputation). Rows are designed to be readable.
        """
        step = StepResult(step_name="stationarity_sweep")

        rows: List[Dict[str, Any]] = []
        if transform_bundle is None:
            step.notes.append("No TransformBundle provided; stationarity sweep unavailable.")
            step.add_artifact("stationarity_table", payload=[])
            return step

        try:
            metas = getattr(transform_bundle, "metas", {}) or {}
            # Stable order: insertion order if dict-preserved; otherwise sort by name
            items = list(metas.items())
            if not isinstance(metas, dict):
                items = []

            # Build rows
            for name, meta in items:
                rows.append(
                    {
                        "variant": str(getattr(meta, "name", name)),
                        "adf_p": getattr(meta, "adf_pvalue", None),
                        "kpss_p": getattr(meta, "kpss_pvalue", None),
                        "is_stationary": getattr(meta, "is_stationary", None),
                        "seasonal_period": getattr(meta, "seasonal_period", None),
                        "recommended_for": ",".join(getattr(meta, "recommended_for", []) or []),
                    }
                )

            # Truncate (readability)
            rows = rows[: self.config.max_variants_stationarity_table]

            step.add_artifact("stationarity_table", payload=rows)
            step.summary["n_variants_listed"] = int(len(rows))
            step.notes.append("Stationarity sweep built from TransformBundle.metas (no tests computed in reporter).")

        except Exception as e:
            step.warnings.append(f"stationarity sweep failed: {e}")
            step.add_artifact("stationarity_table", payload=[])

        return step

    def _step_variant_selection(self, transform_bundle: Optional[Any]) -> StepResult:
        """
        Step: "variant_selection"
        Artifact: "selection_ranked"

        MVP ranking logic (interpretable + deterministic):
          1) Prefer meta.is_stationary == True
          2) Prefer lower ADF p-value
          3) Prefer higher KPSS p-value
          4) Prefer fewer operations (simpler transform chain)

        This is not "model performance" selection; it is a diagnostics-first heuristic.
        """
        step = StepResult(step_name="variant_selection")
        ranked: List[Dict[str, Any]] = []

        if transform_bundle is None:
            step.notes.append("No TransformBundle provided; cannot rank variants.")
            step.add_artifact("selection_ranked", payload=[])
            return step

        try:
            metas = getattr(transform_bundle, "metas", {}) or {}
            items = list(metas.items())

            def _score(meta: Any) -> Tuple[int, float, float, int]:
                # Stationary first (0 is best)
                is_stat = getattr(meta, "is_stationary", None)
                stat_rank = 0 if is_stat is True else (1 if is_stat is None else 2)

                adf = getattr(meta, "adf_pvalue", None)
                kpss = getattr(meta, "kpss_pvalue", None)

                # If missing, treat as worst
                adf_val = float(adf) if adf is not None else 1.0
                kpss_val = float(kpss) if kpss is not None else 0.0

                ops = getattr(meta, "operations", []) or []
                complexity = int(len(ops))

                # lower tuple is better, except KPSS we want higher (so use -kpss)
                return (stat_rank, adf_val, -kpss_val, complexity)

            items_sorted = sorted(items, key=lambda kv: _score(kv[1]))

            for rank, (name, meta) in enumerate(items_sorted[: self.config.max_rows_table], start=1):
                is_stat = getattr(meta, "is_stationary", None)
                adf = getattr(meta, "adf_pvalue", None)
                kpss = getattr(meta, "kpss_pvalue", None)
                ops = getattr(meta, "operations", []) or []

                reason_parts = []
                if is_stat is True:
                    reason_parts.append("stationary")
                elif is_stat is False:
                    reason_parts.append("non-stationary")
                else:
                    reason_parts.append("ambiguous")

                if adf is not None:
                    reason_parts.append(f"adf_p={float(adf):.4g}")
                if kpss is not None:
                    reason_parts.append(f"kpss_p={float(kpss):.4g}")
                reason_parts.append(f"ops={len(ops)}")

                ranked.append(
                    {
                        "rank": int(rank),
                        "variant": str(getattr(meta, "name", name)),
                        "score": str(_score(meta)),
                        "reason": "; ".join(reason_parts),
                    }
                )

            step.add_artifact("selection_ranked", payload=ranked)
            step.summary["n_ranked"] = int(len(ranked))
            step.notes.append("Variant ranking is a diagnostics heuristic (not forecasting performance).")

        except Exception as e:
            step.warnings.append(f"variant selection failed: {e}")
            step.add_artifact("selection_ranked", payload=[])

        return step

    def _step_acf_pacf(
        self,
        cleaned_df: pd.DataFrame,
        schema_meta: Optional[Any],
        ts_meta: Optional[Any],
    ) -> StepResult:
        """
        Step: "acf_pacf"
        Artifact: "acf_pacf_payload"

        Payload:
          {"lags":[...], "acf":[...], "pacf":[...], "conf": float, "n": int}

        Notes on logic:
          - Uses target series (schema_meta.target_col if available, else first numeric column).
          - Drops NaNs and inf values before computing.
          - Caps lags to min(config.max_lags_acf_pacf, n//2 - 1).
          - Confidence band uses 95% (z=1.96): conf = z / sqrt(n).
          - Computes with statsmodels if available; otherwise falls back to a simple autocorr estimate for ACF and
            uses a conservative placeholder for PACF (zeros) to avoid misleading partial correlation.
        """
        step = StepResult(step_name="acf_pacf")

        payload = {"lags": [], "acf": [], "pacf": [], "conf": None, "n": None}
        step.add_artifact("acf_pacf_payload", payload=payload)

        try:
            target_col = self._get_target_col(cleaned_df, schema_meta)
            y = self._safe_series(cleaned_df, target_col).replace([np.inf, -np.inf], np.nan).dropna()

            n = int(len(y))
            payload["n"] = n
            step.summary["target_col"] = target_col
            step.summary["n_used"] = n

            if n < 5:
                step.notes.append("Too few observations to compute meaningful ACF/PACF.")
                return step

            max_lags = int(self.config.max_lags_acf_pacf)
            # Keep lags well-defined relative to sample size
            lag_cap = max(1, min(max_lags, (n // 2) - 1))
            lags = list(range(0, lag_cap + 1))

            # 95% two-sided confidence band
            conf = float(self.config.acf_pacf_z) / float(np.sqrt(n))
            payload["conf"] = conf

            # Prefer statsmodels implementations (standard in time-series)
            try:
                from statsmodels.tsa.stattools import acf as sm_acf  # type: ignore
                from statsmodels.tsa.stattools import pacf as sm_pacf  # type: ignore

                acf_vals = sm_acf(y.values, nlags=lag_cap, fft=True)
                pacf_vals = sm_pacf(y.values, nlags=lag_cap, method="ywmle")

                payload["lags"] = lags
                payload["acf"] = [float(v) for v in acf_vals[: len(lags)]]
                payload["pacf"] = [float(v) for v in pacf_vals[: len(lags)]]
                step.notes.append(f"ACF/PACF computed with statsmodels up to lag={lag_cap}.")
                return step

            except Exception as e:
                step.warnings.append(f"statsmodels acf/pacf failed; using fallback ACF only: {e}")

            # Fallback ACF (normalized autocorrelation)
            x = y.values.astype("float64")
            x = x - np.mean(x)
            denom = np.sum(x * x)
            if denom <= 0:
                step.notes.append("Series variance is zero; ACF/PACF not defined.")
                return step

            acf_vals = [1.0]
            for k in range(1, lag_cap + 1):
                num = float(np.sum(x[k:] * x[:-k]))
                acf_vals.append(num / float(denom))

            payload["lags"] = lags
            payload["acf"] = [float(v) for v in acf_vals]
            payload["pacf"] = [0.0 for _ in lags]  # conservative: avoid fake PACF
            step.notes.append("Fallback ACF computed; PACF unavailable without statsmodels (set to zeros).")
            return step

        except Exception as e:
            step.warnings.append(f"acf/pacf step failed: {e}")
            return step

    def _step_stl(self, cleaned_df: pd.DataFrame, ts_meta: Optional[Any]) -> StepResult:
        """
        Step: "stl"
        Artifact: "stl_components"

        Conservative MVP:
          - Only runs STL if a trusted seasonal period m is provided by ts_meta.
          - If m is unavailable, return empty components with a note (do not guess m).
        """
        step = StepResult(step_name="stl")

        payload = {"trend": None, "seasonal": None, "resid": None, "m": None}
        step.add_artifact("stl_components", payload=payload)

        m = self._get_seasonal_period_trusted(ts_meta)
        if m is None:
            step.notes.append("No trusted seasonal period (m) available; STL skipped in MVP.")
            return step

        # Choose a target series: first numeric column (or target if schema meta were present upstream)
        try:
            # No schema_meta passed to stl step in run() right now; keep conservative
            numeric_cols = [c for c in cleaned_df.columns if pd.api.types.is_numeric_dtype(cleaned_df[c])]
            if not numeric_cols:
                step.notes.append("No numeric series available for STL.")
                return step

            col = str(numeric_cols[0])
            y = self._safe_series(cleaned_df, col).replace([np.inf, -np.inf], np.nan).dropna()

            if len(y) < (2 * m + 5):
                step.notes.append(f"Not enough data for STL with m={m} (n={len(y)}).")
                payload["m"] = int(m)
                return step

            try:
                from statsmodels.tsa.seasonal import STL  # type: ignore

                stl = STL(y, period=int(m), robust=True)
                res = stl.fit()

                payload["trend"] = res.trend
                payload["seasonal"] = res.seasonal
                payload["resid"] = res.resid
                payload["m"] = int(m)

                step.summary["stl_col"] = col
                step.summary["m"] = int(m)
                step.notes.append("STL computed with statsmodels (robust=True).")
                return step

            except Exception as e:
                step.warnings.append(f"STL failed (statsmodels missing or error): {e}")
                payload["m"] = int(m)
                return step

        except Exception as e:
            step.warnings.append(f"stl step failed: {e}")
            return step

    def _step_exog(self, cleaned_df: pd.DataFrame, schema_meta: Optional[Any]) -> StepResult:
        """
        Step: "exog" (optional)

        MVP: basic sanity summary of exogenous columns (if known):
          - counts
          - missingness rate per exog
          - simple correlation with target (if numeric + aligned)

        No stable artifact contract defined for exog yet, so we keep it in summary/notes only.
        """
        step = StepResult(step_name="exog")

        exog_cols = []
        if schema_meta is not None:
            exog_cols = getattr(schema_meta, "exog_cols", []) or []

        if not exog_cols:
            step.notes.append("No exog columns configured (schema_meta.exog_cols empty/missing).")
            return step

        try:
            target_col = self._get_target_col(cleaned_df, schema_meta)
            y = self._safe_series(cleaned_df, target_col)

            step.summary["n_exog"] = int(len(exog_cols))
            shown = 0

            for c in exog_cols:
                if c not in cleaned_df.columns:
                    step.warnings.append(f"exog col '{c}' not found in cleaned_df.")
                    continue

                x = cleaned_df[c]
                miss_rate = float(pd.isna(x).mean())
                step.summary[f"exog_missing_rate:{c}"] = round(miss_rate, 4)

                # Correlation only if numeric and enough overlap
                if pd.api.types.is_numeric_dtype(x):
                    xx = pd.to_numeric(x, errors="coerce").astype("float64")
                    joined = pd.concat([y, xx], axis=1).dropna()
                    if len(joined) >= 10:
                        corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
                        step.summary[f"exog_corr:{c}"] = round(corr, 4)

                shown += 1
                if shown >= 12:
                    step.notes.append("Exog summary truncated (MVP).")
                    break

            step.notes.append("Exog step is MVP: missingness + simple correlation summaries only.")
            return step

        except Exception as e:
            step.warnings.append(f"exog step failed: {e}")
            return step
