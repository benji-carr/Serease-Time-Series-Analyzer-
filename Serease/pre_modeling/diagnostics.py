# serease/pre_modeling/diagnostics.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .containers import SeriesView, TransformedSeries


class DiagnosticsEngine:
    """
    Computes diagnostics on *views* (already transformed as needed).
    Covers:
      Step 1: Raw diagnostics
      Step 2: Variance assessment
      Step 3: Periodogram
      Step 4: Stationarity tests
      Step 5: ACF/PACF
      Step 6: STL
    """

    # -----------------------------
    # Helpers (pure computations)
    # -----------------------------
    def _missing_blocks(self, y: pd.Series) -> Dict[str, Any]:
        miss = y.isna()
        if miss.sum() == 0:
            return {"missing_count": 0, "missing_pct": 0.0, "blocks": []}

        blocks, in_block, start, prev_t = [], False, None, None
        for t, is_miss in miss.items():
            if is_miss and not in_block:
                in_block, start = True, t
            elif not is_miss and in_block:
                blocks.append({"start": str(start), "end": str(prev_t)})
                in_block = False
            prev_t = t

        if in_block and start is not None and prev_t is not None:
            blocks.append({"start": str(start), "end": str(prev_t)})

        return {
            "missing_count": int(miss.sum()),
            "missing_pct": float(miss.mean()),
            "blocks": blocks,
        }

    def _rolling_payload(self, y: pd.Series, window: int) -> Dict[str, Any]:
        s = y.astype(float)
        return {
            "window": int(window),
            "index": [str(x) for x in s.index],
            "rolling_mean": s.rolling(window, min_periods=max(2, window // 3)).mean().tolist(),
            "rolling_var": s.rolling(window, min_periods=max(2, window // 3)).var().tolist(),
        }

    def _window_mean_variance(self, y: pd.Series, n_windows: int = 12) -> Dict[str, Any]:
        s = y.dropna().astype(float)
        n = len(s)
        if n < 10:
            return {"n_windows": 0, "points": [], "slope_loglog": None, "r2_loglog": None}

        edges = np.linspace(0, n, num=n_windows + 1, dtype=int)
        points = []
        for i in range(n_windows):
            chunk = s.iloc[edges[i]:edges[i + 1]]
            if len(chunk) >= 3:
                m, v = chunk.mean(), chunk.var(ddof=1)
                if m > 0 and v > 0:
                    points.append({"window": i, "mean": float(m), "var": float(v)})

        if len(points) < 4:
            return {"n_windows": len(points), "points": points, "slope_loglog": None, "r2_loglog": None}

        x = np.log([p["mean"] for p in points])
        yv = np.log([p["var"] for p in points])
        b, a = np.polyfit(x, yv, 1)
        yhat = b * x + a
        r2 = 1 - np.sum((yv - yhat) ** 2) / np.sum((yv - yv.mean()) ** 2)

        return {
            "n_windows": len(points),
            "points": points,
            "slope_loglog": float(b),
            "r2_loglog": float(r2),
        }

    def _season_key(self, idx: pd.DatetimeIndex, freq_hint: Optional[str]) -> str:
        if freq_hint:
            f = freq_hint.upper()
            if "H" in f:
                return "hour"
            if "D" in f:
                return "dayofweek"
            if "W" in f:
                return "weekofyear"
            if "M" in f:
                return "month"
        return "month"

    def _cv_by_season(self, y: pd.Series, freq_hint: Optional[str]) -> Dict[str, Any]:
        s = y.dropna().astype(float)
        if not isinstance(s.index, pd.DatetimeIndex) or len(s) < 10:
            return {"group": None, "table": [], "cv_std": None}

        group = self._season_key(s.index, freq_hint)
        if group == "month":
            key = s.index.month
        elif group == "dayofweek":
            key = s.index.dayofweek
        elif group == "hour":
            key = s.index.hour
        else:
            key = s.index.month

        df = pd.DataFrame({"y": s.values, "g": key})
        agg = df.groupby("g")["y"].agg(["mean", "std", "count"])
        agg["cv"] = agg["std"] / agg["mean"]

        cvs = agg["cv"].dropna().values
        return {
            "group": group,
            "table": agg.reset_index().to_dict("records"),
            "cv_std": float(np.std(cvs)) if len(cvs) >= 2 else None,
        }

    def _ratio_payload(self, y: pd.Series, window: int) -> Dict[str, Any]:
        s = y.astype(float)
        ratio = (
            s.rolling(window, min_periods=max(2, window // 3)).std()
            / s.rolling(window, min_periods=max(2, window // 3)).mean()
        )
        return {
            "window": int(window),
            "index": [str(x) for x in s.index],
            "ratio_std_over_mean": ratio.tolist(),
        }

    def _distribution_stats(self, y: pd.Series) -> Dict[str, Any]:
        s = y.dropna().astype(float)
        if len(s) < 5:
            return {"skew": None, "kurtosis": None, "zero_pct": None}

        x = s.values
        mu, sd = x.mean(), x.std(ddof=1)
        skew = float(np.mean(((x - mu) / sd) ** 3)) if sd > 0 else 0.0
        kurt = float(np.mean(((x - mu) / sd) ** 4) - 3.0) if sd > 0 else 0.0

        return {
            "skew": skew,
            "kurtosis": kurt,
            "zero_pct": float(np.mean(x == 0.0)),
        }

    def _mad_outliers(self, y: pd.Series, thresh: float = 4.0) -> Dict[str, Any]:
        s = y.dropna().astype(float)
        if len(s) < 10:
            return {"method": "mad_z", "threshold": thresh, "indices": [], "count": 0}

        med = np.median(s.values)
        mad = np.median(np.abs(s.values - med)) + 1e-9
        z = 0.6745 * (s.values - med) / mad
        idxs = np.where(np.abs(z) > thresh)[0]

        return {
            "method": "mad_z",
            "threshold": thresh,
            "indices": [str(s.index[i]) for i in idxs],
            "count": int(len(idxs)),
        }

    # -----------------------------
    # STEP 1: RAW diagnostics
    # -----------------------------
    def raw_diagnostics(self, view: SeriesView, freq_hint: Optional[str] = None) -> Dict[str, Any]:
        y = view.y
        y_nonan = y.dropna()

        n = len(y_nonan)
        window = int(max(5, min(52, round(n * 0.10)))) if n else 5

        mean_var = self._window_mean_variance(y)
        cv_season = self._cv_by_season(y, freq_hint)

        variance_scales = (
            mean_var["slope_loglog"] is not None
            and mean_var["r2_loglog"] is not None
            and mean_var["slope_loglog"] > 0.7
            and mean_var["r2_loglog"] > 0.4
        )

        likely_mult = variance_scales and (cv_season["cv_std"] is not None and cv_season["cv_std"] < 0.25)

        return {
            "summary": {
                "n": int(len(y)),
                "n_nonmissing": n,
                "mean": float(y_nonan.mean()) if n else None,
                "std": float(y_nonan.std()) if n else None,
            },
            "missingness": self._missing_blocks(y),
            "rolling": self._rolling_payload(y, window),
            "mean_variance": mean_var,
            "cv_by_season": cv_season,
            "ratio": self._ratio_payload(y, window),
            "distribution": self._distribution_stats(y),
            "outliers": self._mad_outliers(y),
            "flags": {
                "variance_scales_with_level": variance_scales,
                "likely_multiplicative": likely_mult,
            },
        }

    # -----------------------------
    # STEP 2: Variance assessment
    # -----------------------------
    def _recommended_offset(self, y: pd.Series) -> float:
        s = y.dropna().astype(float)
        if len(s) == 0:
            return 0.0
        return float(-s.min() + 1e-6) if s.min() <= 0 else 0.0

    def _boxcox_mle_lambda(self, y_pos: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        try:
            from scipy import stats
            _, lam = stats.boxcox(y_pos)
            return float(lam), float(stats.boxcox_llf(lam, y_pos))
        except Exception:
            return None, None

    def variance_assessment(self, raw_view: SeriesView, freq_hint: Optional[str] = None) -> Dict[str, Any]:
        y = raw_view.y
        offset = self._recommended_offset(y)
        y_pos = y.dropna().values + offset

        lam, llf = (None, None)
        if len(y_pos) >= 20 and np.all(y_pos > 0):
            lam, llf = self._boxcox_mle_lambda(y_pos)

        mean_var = self._window_mean_variance(y)
        cv_season = self._cv_by_season(y, freq_hint)

        variance_scales = (
            mean_var["slope_loglog"] is not None
            and mean_var["r2_loglog"] is not None
            and mean_var["slope_loglog"] > 0.7
            and mean_var["r2_loglog"] > 0.4
        )

        needs = variance_scales or (cv_season["cv_std"] is not None and cv_season["cv_std"] < 0.25)

        return {
            "needs_stabilization": needs,
            "suggested_transform": "log" if needs else "none",
            "recommended_offset": offset,
            "boxcox_lambda_mle": lam,
            "boxcox_llf": llf,
        }

    # -----------------------------
    # STEP 3: Periodogram
    # -----------------------------
    def periodogram_candidates(self, view: SeriesView, top_k: int = 5) -> Dict[str, Any]:
        y = view.y.dropna().astype(float).values
        n = len(y)
        if n < 16:
            return {"candidates": [], "chosen_m": None, "notes": "Too short"}

        from scipy.signal import periodogram, find_peaks

        freqs, power = periodogram(y, fs=1.0)
        freqs, power = freqs[1:], power[1:]
        peaks, _ = find_peaks(power)

        candidates = {}
        for f, p in zip(freqs[peaks], power[peaks]):
            if f > 0:
                m = int(round(1 / f))
                if 2 <= m <= n // 2:
                    candidates[m] = max(candidates.get(m, 0), p)

        ranked = [{"m": m, "power": p} for m, p in sorted(candidates.items(), key=lambda x: x[1], reverse=True)]

        for c in ranked:
            m = c["m"]
            # Guard: correlation needs enough overlap points
            if (n - m) >= 3:
                c["acf_at_m"] = float(np.corrcoef(y[:-m], y[m:])[0, 1])
            else:
                c["acf_at_m"] = None

            c["verify_score"] = max(0.0, c["acf_at_m"] or 0.0)


        top = ranked[:top_k]
        chosen = max(top, key=lambda d: d.get("verify_score", 0.0)) if top else None

        return {
            "candidates": top,
            "chosen_m": int(chosen["m"]) if chosen else None,
        }

    # -----------------------------
    # STEP 4: Stationarity tests
    # -----------------------------
    def stationarity_tests(self, ts: TransformedSeries) -> Dict[str, Any]:
        y = ts.y.dropna().astype(float)
        n = len(y)
        if n < 12:
            return {"D": ts.plan.D, "d": ts.plan.d, "m": ts.plan.seasonal_period_m, "n": n}

        acf1 = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if y.std() > 0 else None

        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            adf_p = adfuller(y, autolag="AIC")[1]
            kpss_p = kpss(y, regression="c", nlags="auto")[1]
        except Exception:
            adf_p, kpss_p = None, None

        return {
            "D": ts.plan.D,
            "d": ts.plan.d,
            "m": ts.plan.seasonal_period_m,
            "n": n,
            "adf_pvalue": adf_p,
            "kpss_pvalue": kpss_p,
            "acf1": acf1,
        }

    # -----------------------------
    # STEP 5: ACF / PACF
    # -----------------------------
    def acf_pacf(self, view: SeriesView) -> Dict[str, Any]:
        y = view.y.dropna().astype(float)
        n = len(y)
        if n < 12:
            return {"n": n}

        from statsmodels.tsa.stattools import acf, pacf

        max_lag = min(60, n // 2)
        acf_vals, acf_ci = acf(y, nlags=max_lag, alpha=0.05)
        pacf_vals, pacf_ci = pacf(y, nlags=max_lag, alpha=0.05)

        return {
            "n": n,
            "nlags": int(max_lag),
            "acf": acf_vals.tolist(),
            "pacf": pacf_vals.tolist(),
            "acf_confint": acf_ci.tolist(),
            "pacf_confint": pacf_ci.tolist(),
        }

    # -----------------------------
    # STEP 6: STL
    # -----------------------------
    def stl(self, view: SeriesView, m: int | None) -> Dict[str, Any]:
        if m is None:
            return {"notes": "m not set"}

        from statsmodels.tsa.seasonal import STL

        y = view.y.dropna().astype(float)
        res = STL(y, period=m, robust=True).fit()

        return {
            "trend": res.trend.tolist(),
            "seasonal": res.seasonal.tolist(),
            "resid": res.resid.tolist(),
        }