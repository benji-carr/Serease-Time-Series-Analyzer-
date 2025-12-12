from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL

from .report_types import DiagnosticsReport, DiagnosticResult, DiagnosticArtifact
from .utils import (
    compute_fft_periodogram_candidates,
    choose_period_from_candidates,
    infer_seasonal_period_from_freq,
)


@dataclass
class VariantSelectionPolicy:
    adf_alpha: float = 0.05
    kpss_alpha: float = 0.05
    min_n_for_tests: int = 30
    max_lags: int = 60
    oversdiff_neg_acf1_threshold: float = -0.6
    prefer_transform_order: Tuple[str, ...] = ("log", "raw", "log1p")


@dataclass
class SeasonalityPolicy:
    periodogram_top_k: int = 8
    max_period_frac: float = 0.33
    min_peak_ratio: float = 6.0
    acf_gate: bool = True
    acf_min_peak: float = 0.25
    min_obs_for_periodogram: int = 40


class DiagnosticsEngine:
    def __init__(
        self,
        cleaned_df: pd.DataFrame,
        schema_meta: Any,
        ts_meta: Any,
        transform_bundle: Any,
        transformer: Optional[Any] = None,
        df_raw: Optional[pd.DataFrame] = None,
        ingestion_meta: Optional[Any] = None,
    ) -> None:
        self.cleaned_df = cleaned_df
        self.schema = schema_meta
        self.ts_meta = ts_meta
        self.bundle = transform_bundle
        self.transformer = transformer
        self.df_raw = df_raw
        self.ingestion_meta = ingestion_meta

        if getattr(self.schema, "target_col", None) is None:
            raise ValueError("schema_meta.target_col is required.")
        if getattr(self.schema, "date_col", None) is None:
            raise ValueError("schema_meta.date_col is required.")
        if not isinstance(self.cleaned_df.index, pd.DatetimeIndex):
            raise ValueError("cleaned_df must have a DateTimeIndex.")
        if self.cleaned_df.index.has_duplicates:
            raise ValueError("cleaned_df index has duplicates; expected output from TimeSeriesCleaner.")
        if not self.cleaned_df.index.is_monotonic_increasing:
            raise ValueError("cleaned_df index must be sorted ascending.")

        self.target_col = self.schema.target_col
        self.date_col = self.schema.date_col
        self.exog_cols = list(getattr(self.schema, "exog_cols", []) or [])
        self.freq = getattr(self.ts_meta, "freq", None)

    def run(
        self,
        seasonal_period: Optional[int] = None,
        enable_exog: bool = False,
        fourier_K: int = 5,
        ccf_max_lag: int = 30,
        variant_policy: Optional[VariantSelectionPolicy] = None,
        seasonality_policy: Optional[SeasonalityPolicy] = None,
        stl_robust: bool = True,
    ) -> DiagnosticsReport:
        vpol = variant_policy or VariantSelectionPolicy()
        spol = seasonality_policy or SeasonalityPolicy()

        report = DiagnosticsReport(
            target_col=self.target_col,
            date_col=self.date_col,
            freq=self.freq,
            n_obs=int(len(self.cleaned_df)),
            start=self.cleaned_df.index.min(),
            end=self.cleaned_df.index.max(),
        )

        report.add(self._overview())
        report.add(self._missingness())

        base_variant = self._choose_base_variant(vpol)
        report.add(DiagnosticResult(step="base_variant", ok=True, summary={"base_variant": base_variant}))

        m = seasonal_period
        seasonality = self._detect_seasonality(base_variant=base_variant, policy=spol)
        report.add(seasonality)

        if m is None:
            m = seasonality.summary.get("selected_m", None)

        if m is None:
            m = infer_seasonal_period_from_freq(self.freq)

        report.add(
            DiagnosticResult(
                step="seasonal_period",
                ok=True if m is not None else False,
                summary={"seasonal_period": m},
                warnings=[] if m is not None else ["No seasonal period available."],
            )
        )

        self._ensure_seasonal_variants(m)

        stationarity_res = self._stationarity_sweep(m=m, policy=vpol)
        report.add(stationarity_res)

        selection_res = self._select_stationary_variant(stationarity_res, policy=vpol)
        report.add(selection_res)

        selected = selection_res.summary.get("selected_stationary_variant")
        if selected is not None:
            report.add(self._acf_pacf(variant=selected, policy=vpol))
        else:
            report.add(DiagnosticResult(step="acf_pacf", ok=False, warnings=["No stationary variant selected."]))

        report.add(self._decomposition(base_variant=base_variant, m=m, robust=stl_robust))

        report.add(self._fourier_terms(m=m, K=fourier_K))

        if enable_exog:
            report.add(self._exog_correlation())
            report.add(self._multicollinearity())
            report.add(self._ccf(max_lag=ccf_max_lag))
        else:
            report.add(DiagnosticResult(step="exog", ok=True, notes=["Exogenous diagnostics disabled."]))

        return report

    def _choose_base_variant(self, policy: VariantSelectionPolicy) -> str:
        for name in policy.prefer_transform_order:
            if self.bundle.has(name):
                return name
        return "raw"

    def _detect_seasonality(self, base_variant: str, policy: SeasonalityPolicy) -> DiagnosticResult:
        if not self.bundle.has(base_variant):
            return DiagnosticResult(step="seasonality_assessment", ok=False, warnings=[f"Missing variant '{base_variant}'."])

        s = self.bundle.get(base_variant).dropna()

        if len(s) < int(policy.min_obs_for_periodogram):
            return DiagnosticResult(
                step="seasonality_assessment",
                ok=True,
                summary={
                    "base_variant": base_variant,
                    "status": "insufficient_data",
                    "selected_m": None,
                },
                notes=["Too few observations for reliable period detection."],
            )

        candidates = compute_fft_periodogram_candidates(s, top_k=policy.periodogram_top_k)
        cand_df = pd.DataFrame(candidates)

        selected_m = choose_period_from_candidates(
            cand_df,
            n_obs=len(s),
            max_period_frac=policy.max_period_frac,
            min_peak_ratio=policy.min_peak_ratio,
        )

        notes: List[str] = []
        status = "none"
        acf_at_m = None

        if selected_m is None:
            notes.append("No strong seasonal period detected from periodogram guardrails.")
        else:
            if policy.acf_gate:
                acf_vals = acf(s.values, nlags=min(int(selected_m), max(2, len(s) - 2)), fft=True)
                if len(acf_vals) > int(selected_m):
                    acf_at_m = float(acf_vals[int(selected_m)])
                else:
                    acf_at_m = None

                if acf_at_m is None or abs(acf_at_m) < float(policy.acf_min_peak):
                    notes.append(
                        f"Periodogram suggested m={selected_m}, but ACF at lag m was weak "
                        f"(acf_m={acf_at_m}); treating as no seasonality."
                    )
                    selected_m = None
                else:
                    status = "seasonal"
                    notes.append(f"Selected m={selected_m} (periodogram + ACF gate).")
            else:
                status = "seasonal"
                notes.append(f"Selected m={selected_m} (periodogram).")

        res = DiagnosticResult(
            step="seasonality_assessment",
            ok=True,
            summary={
                "base_variant": base_variant,
                "status": status,
                "selected_m": selected_m,
                "acf_at_m": acf_at_m,
            },
            notes=notes,
        )
        res.artifacts.append(DiagnosticArtifact(name="period_candidates", kind="table", payload=cand_df))
        return res

    def _ensure_seasonal_variants(self, m: Optional[int]) -> None:
        if m is None:
            return
        if self.transformer is None:
            return
        try:
            self.transformer.set_seasonal_period(int(m))
            self.transformer.add_seasonal_variants(self.bundle)
        except Exception:
            return

    def _overview(self) -> DiagnosticResult:
        y = self._get_series_for_variant("raw")
        s = y.dropna()
        return DiagnosticResult(
            step="overview",
            ok=True,
            summary={
                "n_obs_total": int(len(y)),
                "n_obs_non_missing": int(len(s)),
                "start": y.index.min(),
                "end": y.index.max(),
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
                "mean": float(s.mean()) if not s.empty else None,
                "std": float(s.std()) if not s.empty else None,
            },
        )

    def _missingness(self) -> DiagnosticResult:
        y = self._get_series_for_variant("raw")
        mask = y.isna()
        n_missing = int(mask.sum())
        longest = int(self._longest_true_run(mask.values))

        return DiagnosticResult(
            step="missingness",
            ok=True,
            summary={
                "n_missing": n_missing,
                "missing_frac": float(n_missing / len(y)) if len(y) else 0.0,
                "longest_missing_block": longest,
            },
        )

    def _longest_true_run(self, arr: np.ndarray) -> int:
        best = 0
        cur = 0
        for v in arr:
            if bool(v):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def _stationarity_sweep(self, m: Optional[int], policy: VariantSelectionPolicy) -> DiagnosticResult:
        candidates = self._candidate_variants(m=m)
        rows: List[Dict[str, Any]] = []

        for v in candidates:
            if not self.bundle.has(v):
                continue

            s = self.bundle.get(v).dropna()
            if len(s) < policy.min_n_for_tests:
                continue

            try:
                adf_p = float(adfuller(s.values, autolag="AIC")[1])
            except Exception:
                adf_p = np.nan

            try:
                kpss_p = float(kpss(s.values, regression="c", nlags="auto")[1])
            except Exception:
                kpss_p = np.nan

            d, D, m_used, base = self._parse_variant_signature(v)
            a1 = self._acf_lag1(s)

            stationary_adf = (not np.isnan(adf_p)) and (adf_p < policy.adf_alpha)
            stationary_kpss = (not np.isnan(kpss_p)) and (kpss_p > policy.kpss_alpha)

            rows.append(
                {
                    "variant": v,
                    "base": base,
                    "d": int(d),
                    "D": int(D),
                    "m": m_used,
                    "n": int(len(s)),
                    "adf_pvalue": adf_p,
                    "kpss_pvalue": kpss_p,
                    "acf_lag1": a1,
                    "stationary_adf": bool(stationary_adf),
                    "stationary_kpss": bool(stationary_kpss),
                }
            )

        if not rows:
            return DiagnosticResult(
                step="stationarity_sweep",
                ok=False,
                warnings=["No variants had enough observations for stationarity tests."],
            )

        df = pd.DataFrame(rows)
        df["stationary_both"] = df["stationary_adf"] & df["stationary_kpss"]
        df["total_diff"] = df["d"] + df["D"]

        res = DiagnosticResult(
            step="stationarity_sweep",
            ok=True,
            summary={
                "tested_variants": int(len(df)),
                "m": m,
                "adf_alpha": policy.adf_alpha,
                "kpss_alpha": policy.kpss_alpha,
            },
        )
        res.artifacts.append(DiagnosticArtifact(name="stationarity_table", kind="table", payload=df))
        return res

    def _select_stationary_variant(self, stationarity_res: DiagnosticResult, policy: VariantSelectionPolicy) -> DiagnosticResult:
        if not stationarity_res.ok or not stationarity_res.artifacts:
            return DiagnosticResult(
                step="variant_selection",
                ok=False,
                warnings=["Stationarity sweep did not produce a table; cannot select variant."],
            )

        df = stationarity_res.artifacts[0].payload
        if df is None or df.empty:
            return DiagnosticResult(step="variant_selection", ok=False, warnings=["Stationarity table is empty."])

        df2 = df.copy()
        df2["oversdiff_flag"] = df2["acf_lag1"] <= policy.oversdiff_neg_acf1_threshold

        df_pass = df2[df2["stationary_both"] == True].copy()
        pass_mode = "ADF+KPSS"

        if df_pass.empty:
            df_pass = df2[df2["stationary_adf"] == True].copy()
            pass_mode = "ADF_only"

        if df_pass.empty:
            return DiagnosticResult(step="variant_selection", ok=False, warnings=["No variants passed stationarity."])

        df_pass = df_pass.sort_values(
            by=["oversdiff_flag", "total_diff", "adf_pvalue"],
            ascending=[True, True, True],
        )

        selected = str(df_pass.iloc[0]["variant"])

        res = DiagnosticResult(
            step="variant_selection",
            ok=True,
            summary={
                "selected_stationary_variant": selected,
                "pass_mode": pass_mode,
            },
        )
        res.artifacts.append(DiagnosticArtifact(name="selection_ranked", kind="table", payload=df_pass))
        return res

    def _acf_pacf(self, variant: str, policy: VariantSelectionPolicy) -> DiagnosticResult:
        if not self.bundle.has(variant):
            return DiagnosticResult(step="acf_pacf", ok=False, warnings=[f"Missing variant '{variant}'."])

        s = self.bundle.get(variant).dropna()
        if len(s) < policy.min_n_for_tests:
            return DiagnosticResult(step="acf_pacf", ok=False, warnings=["Too few observations for ACF/PACF."])

        nlags = int(min(policy.max_lags, len(s) - 2))
        acf_vals = acf(s.values, nlags=nlags, fft=True)
        pacf_vals = pacf(s.values, nlags=nlags, method="ywm")

        df = pd.DataFrame(
            {
                "lag": np.arange(len(acf_vals)),
                "acf": acf_vals,
                "pacf": np.pad(pacf_vals, (0, len(acf_vals) - len(pacf_vals)), constant_values=np.nan),
            }
        )

        res = DiagnosticResult(step="acf_pacf", ok=True, summary={"variant": variant, "nlags": nlags})
        res.artifacts.append(DiagnosticArtifact(name="acf_pacf_values", kind="table", payload=df))
        return res

    def _decomposition(self, base_variant: str, m: Optional[int], robust: bool = True) -> DiagnosticResult:
        if not self.bundle.has(base_variant):
            return DiagnosticResult(step="stl", ok=False, warnings=[f"Missing variant '{base_variant}'."])

        s = self.bundle.get(base_variant).dropna()
        if len(s) < 20:
            return DiagnosticResult(step="stl", ok=False, warnings=["Too few observations for decomposition."])

        if m is not None and int(m) >= 2 and len(s) >= max(3 * int(m), 20):
            try:
                fit = STL(s, period=int(m), robust=bool(robust)).fit()
                df = pd.DataFrame({"observed": s, "trend": fit.trend, "seasonal": fit.seasonal, "resid": fit.resid})

                seasonal_strength = 1.0 - (np.var(df["resid"]) / np.var(df["resid"] + df["seasonal"]))
                trend_strength = 1.0 - (np.var(df["resid"]) / np.var(df["resid"] + df["trend"]))

                res = DiagnosticResult(
                    step="stl",
                    ok=True,
                    summary={
                        "base_variant": base_variant,
                        "method": "STL",
                        "period": int(m),
                        "seasonal_strength": float(seasonal_strength),
                        "trend_strength": float(trend_strength),
                    },
                    notes=[],
                )
                res.artifacts.append(DiagnosticArtifact(name="stl_components", kind="table", payload=df))
                return res
            except Exception:
                pass

        window = int(max(5, min(len(s) // 10, 31)))
        if window % 2 == 0:
            window += 1

        trend = s.rolling(window=window, center=True, min_periods=max(3, window // 3)).mean()
        resid = s - trend
        seasonal = pd.Series(0.0, index=s.index)

        df = pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal, "resid": resid})

        res = DiagnosticResult(
            step="stl",
            ok=True,
            summary={
                "base_variant": base_variant,
                "method": "trend_only",
                "period": None,
                "trend_window": int(window),
                "seasonal_strength": 0.0,
            },
            notes=[
                "No usable seasonal period detected; produced a trend-only decomposition (trend + residual; seasonal=0)."
            ],
        )
        res.artifacts.append(DiagnosticArtifact(name="stl_components", kind="table", payload=df))
        return res

    def _fourier_terms(self, m: Optional[int], K: int = 5) -> DiagnosticResult:
        if m is None or int(m) < 2:
            return DiagnosticResult(step="fourier_terms", ok=False, warnings=["No seasonal period available."])

        n = len(self.cleaned_df.index)
        t = np.arange(n, dtype=float)

        cols: Dict[str, np.ndarray] = {}
        for k in range(1, int(K) + 1):
            cols[f"sin_{k}_m{int(m)}"] = np.sin(2.0 * np.pi * k * t / float(m))
            cols[f"cos_{k}_m{int(m)}"] = np.cos(2.0 * np.pi * k * t / float(m))

        df = pd.DataFrame(cols, index=self.cleaned_df.index)
        res = DiagnosticResult(step="fourier_terms", ok=True, summary={"m": int(m), "K": int(K)})
        res.artifacts.append(DiagnosticArtifact(name="fourier_terms", kind="table", payload=df))
        return res

    def _candidate_variants(self, m: Optional[int]) -> List[str]:
        base = ["raw", "log", "log1p"]
        nonseasonal = ["diff1", "log_diff1", "log1p_diff1"]

        cands: List[str] = []
        for v in base + nonseasonal:
            cands.append(v)

        if m is not None:
            suffix = f"_seasdiff1_m{int(m)}"
            for v in base + nonseasonal:
                cands.append(v + suffix)

        out: List[str] = []
        seen = set()
        for v in cands:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _parse_variant_signature(self, name: str) -> Tuple[int, int, Optional[int], str]:
        if name.startswith("log1p"):
            base = "log1p"
        elif name.startswith("log"):
            base = "log"
        else:
            base = "raw"

        d = 1 if (name == "diff1" or "_diff1" in name or name.startswith("diff1_")) else 0

        D = 0
        m = None
        if "_seasdiff1_m" in name:
            D = 1
            try:
                m = int(name.split("_seasdiff1_m")[-1])
            except Exception:
                m = None

        return d, D, m, base

    def _acf_lag1(self, s: pd.Series) -> float:
        x = s.dropna().values
        if len(x) < 10:
            return np.nan
        vals = acf(x, nlags=1, fft=True)
        return float(vals[1]) if len(vals) > 1 else np.nan

    def _get_series_for_variant(self, name: str) -> pd.Series:
        if self.bundle.has(name):
            return self.bundle.get(name)
        if name == "raw":
            return self.cleaned_df[self.target_col]
        raise KeyError(f"Variant '{name}' not found.")

    def _exog_correlation(self) -> DiagnosticResult:
        if not self.exog_cols:
            return DiagnosticResult(step="exog_correlation", ok=True, notes=["No exogenous columns."])

        y = self._get_series_for_variant("raw")
        X = self.cleaned_df[self.exog_cols].copy()
        df = pd.DataFrame({"y": y}).join(X, how="inner").dropna()

        if df.empty:
            return DiagnosticResult(step="exog_correlation", ok=False, warnings=["No complete cases for y vs X correlation."])

        corr = df.corr(numeric_only=True)["y"].drop(labels=["y"], errors="ignore").sort_values(ascending=False)
        res = DiagnosticResult(step="exog_correlation", ok=True, summary={"n_complete_cases": int(len(df))})
        res.artifacts.append(DiagnosticArtifact(name="corr_yX", kind="series", payload=corr))
        return res

    def _multicollinearity(self) -> DiagnosticResult:
        import statsmodels.api as sm

        if not self.exog_cols:
            return DiagnosticResult(step="multicollinearity", ok=True, notes=["No exogenous columns."])

        X = self.cleaned_df[self.exog_cols].copy().dropna()
        if len(X) < 20:
            return DiagnosticResult(step="multicollinearity", ok=False, warnings=["Too few complete cases for multicollinearity."])

        Xn = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
        corr = Xn.corr()

        XtX = np.asarray(Xn.values.T @ Xn.values, dtype=float)
        try:
            eigvals = np.linalg.eigvalsh(XtX)
            eigvals = np.sort(eigvals)
            cond = float(np.sqrt(eigvals[-1] / max(eigvals[0], 1e-12)))
        except Exception:
            cond = np.nan

        vifs = []
        for col in Xn.columns:
            yj = Xn[col].values
            Xj = Xn.drop(columns=[col]).values
            Xj = sm.add_constant(Xj, has_constant="add")
            model = sm.OLS(yj, Xj).fit()
            r2 = float(model.rsquared)
            vif = float(1.0 / max(1.0 - r2, 1e-12))
            vifs.append({"feature": col, "vif": vif})

        df_vif = pd.DataFrame(vifs).sort_values("vif", ascending=False)

        res = DiagnosticResult(
            step="multicollinearity",
            ok=True,
            summary={"condition_number": cond, "n_complete_cases": int(len(Xn))},
        )
        res.artifacts.append(DiagnosticArtifact(name="corr_X", kind="table", payload=corr))
        res.artifacts.append(DiagnosticArtifact(name="vif", kind="table", payload=df_vif))
        return res

    def _ccf(self, max_lag: int = 30) -> DiagnosticResult:
        from statsmodels.tsa.stattools import ccf

        if not self.exog_cols:
            return DiagnosticResult(step="ccf", ok=True, notes=["No exogenous columns."])

        y = self._get_series_for_variant("diff1") if self.bundle.has("diff1") else self._get_series_for_variant("raw")
        y = y.dropna()

        if len(y) < 30:
            return DiagnosticResult(step="ccf", ok=False, warnings=["Too few observations for CCF."])

        rows = []
        for col in self.exog_cols:
            x = self.cleaned_df[col]
            df = pd.DataFrame({"y": y}).join(pd.DataFrame({"x": x}), how="inner").dropna()
            if len(df) < 30:
                continue

            yy = df["y"].values.astype(float)
            xx = df["x"].values.astype(float)

            yy = yy - yy.mean()
            xx = xx - xx.mean()

            vals = ccf(yy, xx)
            L = int(min(max_lag, len(vals) - 1))
            for k in range(0, L + 1):
                rows.append({"exog": col, "lag": -k, "ccf": float(vals[k])})

        if not rows:
            return DiagnosticResult(step="ccf", ok=False, warnings=["No exogenous series had enough overlap for CCF."])

        df_ccf = pd.DataFrame(rows)
        res = DiagnosticResult(step="ccf", ok=True, notes=["CCF shown for negative lags only (x leads y)."])
        res.artifacts.append(DiagnosticArtifact(name="ccf_values", kind="table", payload=df_ccf))
        return res
