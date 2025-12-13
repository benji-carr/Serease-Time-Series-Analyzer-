from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .report_types import Artifact, DiagnosticsReport, StepResult


@dataclass
class DiagnosticsConfig:
    max_lags: int = 40
    enable_exog: bool = False
    enable_tables: bool = False
    alpha: float = 0.05
    top_n_candidates: int = 15


class DiagnosticsEngine:
    """
    Diagnostics orchestrator.

    Inputs are expected to be downstream of ingestion/schema/cleaning/transform:
    - cleaned_df has DateTimeIndex
    - schema_meta has target_col/date_col/exog_cols
    - ts_meta has freq/n_obs/etc.
    - transform_bundle contains transformed variants (optional but recommended)

    This engine produces a DiagnosticsReport containing StepResult entries.
    """

    def __init__(
        self,
        cleaned_df: pd.DataFrame,
        schema_meta: Any,
        ts_meta: Any,
        transform_bundle: Optional[Any] = None,
        transformer: Optional[Any] = None,
        df_raw: Optional[pd.DataFrame] = None,
        ingestion_meta: Optional[Any] = None,
        config: Optional[DiagnosticsConfig] = None,
    ) -> None:
        self.cleaned_df = cleaned_df
        self.schema_meta = schema_meta
        self.ts_meta = ts_meta
        self.bundle = transform_bundle
        self.transformer = transformer
        self.df_raw = df_raw
        self.ingestion_meta = ingestion_meta
        self.config = config or DiagnosticsConfig()

    def run(self, enable_exog: Optional[bool] = None) -> DiagnosticsReport:
        cfg = self.config
        if enable_exog is not None:
            cfg.enable_exog = bool(enable_exog)

        report = DiagnosticsReport()

        report.add(self._step_overview())
        report.add(self._step_missingness())
        report.add(self._step_seasonality_phase1())
        report.add(self._step_variant_selection())
        report.add(self._step_stationarity_sweep())
        report.add(self._step_acf_pacf())
        report.add(self._step_stl_decomposition())

        if cfg.enable_exog:
            report.add(self._step_exog_placeholder())

        return report

    def _target_series(self) -> pd.Series:
        tc = getattr(self.schema_meta, "target_col", None)
        if tc is None or tc not in self.cleaned_df.columns:
            raise ValueError("schema_meta.target_col not found in cleaned_df.")
        return self.cleaned_df[tc]

    def _step_overview(self) -> StepResult:
        y = self._target_series()
        s = y.dropna()

        summary: Dict[str, Any] = {
            "n_obs_total": int(len(y)),
            "n_obs_non_missing": int(len(s)),
            "start": str(y.index.min()) if len(y) else None,
            "end": str(y.index.max()) if len(y) else None,
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
            "std": float(s.std()) if len(s) else None,
        }

        return StepResult(step="overview", summary=summary)

    def _step_missingness(self) -> StepResult:
        y = self._target_series()
        n_missing = int(y.isna().sum())
        frac = float(n_missing / len(y)) if len(y) else 0.0

        summary: Dict[str, Any] = {
            "n_missing": n_missing,
            "missing_frac": round(frac, 6),
        }

        artifacts = [
            Artifact(name="missing_blocks", payload=None)
        ]

        return StepResult(step="missingness", summary=summary, artifacts=artifacts)

    def _step_seasonality_phase1(self) -> StepResult:
        summary: Dict[str, Any] = {
            "seasonal_period": None,
            "method": "placeholder",
        }
        notes = [
            "Seasonal period detection not implemented in skeleton."
        ]
        artifacts = [
            Artifact(name="period_candidates", payload=None)
        ]
        return StepResult(step="seasonality_assessment", summary=summary, notes=notes, artifacts=artifacts)

    def _step_variant_selection(self) -> StepResult:
        summary: Dict[str, Any] = {
            "base_variant": "raw",
            "selected_stationary_variant": None,
        }
        notes = [
            "Variant selection not implemented in skeleton."
        ]
        artifacts = [
            Artifact(name="selection_ranked", payload=None)
        ]
        return StepResult(step="variant_selection", summary=summary, notes=notes, artifacts=artifacts)

    def _step_stationarity_sweep(self) -> StepResult:
        summary: Dict[str, Any] = {
            "tested_variants": [],
            "criteria": "placeholder",
        }
        notes = [
            "Stationarity sweep (ADF/KPSS) not implemented in skeleton."
        ]
        artifacts = [
            Artifact(name="stationarity_table", payload=None)
        ]
        return StepResult(step="stationarity_sweep", summary=summary, notes=notes, artifacts=artifacts)

    def _step_acf_pacf(self) -> StepResult:
        summary: Dict[str, Any] = {
            "variant": None,
            "max_lags": int(self.config.max_lags),
            "alpha": float(self.config.alpha),
        }
        notes = [
            "ACF/PACF computation not implemented in skeleton."
        ]
        artifacts = [
            Artifact(name="acf_pacf_payload", payload=None)
        ]
        return StepResult(step="acf_pacf", summary=summary, notes=notes, artifacts=artifacts)

    def _step_stl_decomposition(self) -> StepResult:
        summary: Dict[str, Any] = {
            "seasonal_period": None,
            "status": "skipped",
        }
        notes = [
            "STL decomposition not implemented in skeleton."
        ]
        artifacts = [
            Artifact(name="stl_components", payload=None)
        ]
        return StepResult(step="stl", summary=summary, notes=notes, artifacts=artifacts)

    def _step_exog_placeholder(self) -> StepResult:
        summary: Dict[str, Any] = {
            "enabled": True,
            "status": "skipped",
        }
        notes = [
            "Exogenous diagnostics not implemented in skeleton."
        ]
        return StepResult(step="exog", summary=summary, notes=notes)
