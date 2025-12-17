from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover
    adfuller = None
    kpss = None


TransformName = Literal[
    "identity",
    "log",
    "log1p",
    "boxcox",
    "diff",
    "seas_diff",
]


@dataclass
class SeriesOperation:
    """
    A single, explicit operation applied to a series to produce a variant.

    Examples:
      - SeriesOperation("log", {"offset": 12.3})
      - SeriesOperation("diff", {"lag": 1, "order": 1})
      - SeriesOperation("seas_diff", {"m": 12, "order": 1})
    """
    name: TransformName
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeriesVariantMeta:
    """
    Metadata describing one series variant: how it was produced, basic stats,
    and (optional) stationarity test results.
    """
    name: str
    target_col: str
    freq: Optional[str]
    seasonal_period: Optional[int]
    operations: List[SeriesOperation] = field(default_factory=list)

    n_obs: int = 0
    n_missing: int = 0
    dropna_count: int = 0
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    adf_pvalue: Optional[float] = None
    kpss_pvalue: Optional[float] = None
    is_stationary: Optional[bool] = None

    lineage: Optional[str] = None
    recommended_for: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TransformBundle:
    """
    Container returned by TimeSeriesTransformer.fit().

    variants: mapping of variant name -> pd.Series
    metas:    mapping of variant name -> SeriesVariantMeta
    """
    variants: Dict[str, pd.Series] = field(default_factory=dict)
    metas: Dict[str, SeriesVariantMeta] = field(default_factory=dict)
    base_name: str = "raw"

    def get(self, name: str) -> pd.Series:
        if name not in self.variants:
            raise KeyError(f"Variant '{name}' not found. Available: {list(self.variants.keys())}")
        return self.variants[name]

    def meta(self, name: str) -> SeriesVariantMeta:
        if name not in self.metas:
            raise KeyError(f"Meta for variant '{name}' not found. Available: {list(self.metas.keys())}")
        return self.metas[name]

    def list_variants(self) -> List[str]:
        return list(self.variants.keys())

    def has(self, name: str) -> bool:
        return name in self.variants


class TimeSeriesTransformer:
    """
    Guided (non-combinatorial) variant creation:

      raw -> stationarity check
        -> (optional) variance stabilization -> check
        -> (optional) seasonal differencing -> check
        -> (optional) first differencing -> check

    Stops as soon as a stationary variant is found (when tests enabled).
    """

    def __init__(
        self,
        target_col: str,
        freq: Optional[str] = None,
        seasonal_period: Optional[int] = None,
        transforms: Sequence[Literal["identity", "log", "log1p"]] = ("identity", "log", "log1p"),
        difference_orders: Sequence[int] = (0, 1),
        seasonal_difference_orders: Sequence[int] = (0, 1),
        log_offset_strategy: Literal["auto", "fixed", "error"] = "auto",
        log_offset_value: Optional[float] = None,
        nan_policy: Literal["keep", "drop"] = "keep",
        enable_stationarity_tests: bool = False,
        min_obs_for_tests: int = 20,
        adf_alpha: float = 0.05,
        kpss_alpha: float = 0.05,
        max_variants: Optional[int] = 50,
    ) -> None:
        self.target_col = target_col
        self.freq = freq
        self.seasonal_period = seasonal_period

        self.transforms = tuple(transforms)
        self.difference_orders = tuple(int(x) for x in difference_orders)
        self.seasonal_difference_orders = tuple(int(x) for x in seasonal_difference_orders)

        self.log_offset_strategy = log_offset_strategy
        self.log_offset_value = log_offset_value

        self.nan_policy = nan_policy

        self.enable_stationarity_tests = enable_stationarity_tests
        self.min_obs_for_tests = int(min_obs_for_tests)
        self.adf_alpha = float(adf_alpha)
        self.kpss_alpha = float(kpss_alpha)

        self.max_variants = max_variants

        self._bundle: Optional[TransformBundle] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self, cleaned_df: pd.DataFrame) -> TransformBundle:
        """
        Build a TransformBundle using a sequential stationarity-first process.

        A) raw -> check
        B) variance stabilization (log/log1p) -> check after each
        C) seasonal differencing (if m set) -> check after each
        D) non-seasonal differencing -> check after each

        If tests are disabled/unavailable, we still create the variants but cannot stop early.
        """
        self._validate_input(cleaned_df)

        # NOTE: Keep this strict for now; coercion policy belongs upstream (cleaner).
        y_raw = cleaned_df[self.target_col].astype("float64")

        bundle = TransformBundle(base_name="raw")

        # --- register raw ---
        self._register_variant(
            bundle=bundle,
            name="raw",
            series=y_raw,
            ops=[SeriesOperation(name="identity", params={})],
            lineage=None,
        )

        chosen = "raw"
        ok = self._stationarity_checkpoint(bundle, chosen)
        if ok:
            bundle.meta(chosen).recommended_for.append("final_stationary")
            self._bundle = bundle
            return bundle

        # -----------------------------------------------------------------
        # Step 1: Variance stabilization (log / log1p) - best-effort
        # -----------------------------------------------------------------
        variance_candidates = [t for t in self.transforms if t != "identity"]

        for tname in variance_candidates:
            if self.max_variants is not None and len(bundle.variants) >= self.max_variants:
                bundle.meta(chosen).warnings.append("max_variants reached; stopping further variant creation.")
                break

            if tname == "log":
                try:
                    y_t, op = self._apply_log(y_raw)
                except Exception as e:
                    bundle.meta(chosen).notes.append(f"Skipped log: {e}")
                    continue

                vname = "log"
                ops = [SeriesOperation("identity", {}), op]

            elif tname == "log1p":
                try:
                    y_t, op = self._apply_log1p(y_raw)
                except Exception as e:
                    bundle.meta(chosen).notes.append(f"Skipped log1p: {e}")
                    continue

                vname = "log1p"
                ops = [SeriesOperation("identity", {}), op]

            else:
                # v1 ignores other transforms
                continue

            self._register_variant(
                bundle=bundle,
                name=vname,
                series=y_t,
                ops=ops,
                lineage=f"{chosen} -> {vname}",
            )

            # Optional: attach offset note for log variants (helps diagnostics later)
            if tname == "log":
                offset = float(op.params.get("offset", 0.0))
                if offset > 0:
                    bundle.meta(vname).notes.append(f"log offset applied: {offset}")

            ok = self._stationarity_checkpoint(bundle, vname)
            if ok:
                chosen = vname
                bundle.meta(chosen).recommended_for.append("final_stationary")
                self._bundle = bundle
                return bundle

        # If none achieved stationarity, pick the latest variance transform that exists
        if bundle.has("log1p"):
            chosen = "log1p"
        elif bundle.has("log"):
            chosen = "log"
        else:
            chosen = "raw"

        # -----------------------------------------------------------------
        # Step 2: Seasonal differencing (conditional on seasonal_period)
        # -----------------------------------------------------------------
        m = self.seasonal_period
        if m is not None and isinstance(m, int) and m >= 2:
            current_name = chosen
            current_series = bundle.get(current_name)
            current_ops = list(bundle.meta(current_name).operations)

            for sd in self.seasonal_difference_orders:
                sd = int(sd)
                if sd <= 0:
                    continue

                if self.max_variants is not None and len(bundle.variants) >= self.max_variants:
                    bundle.meta(current_name).warnings.append("max_variants reached; stopping seasonal differencing.")
                    break

                y_sd, sd_ops = self._apply_seasonal_diff(current_series, m=m, order=sd)
                next_name = self._append_seasdiff_name(current_name, sd=sd, m=m)

                self._register_variant(
                    bundle=bundle,
                    name=next_name,
                    series=y_sd,
                    ops=current_ops + sd_ops,
                    lineage=f"{current_name} -> {next_name}",
                )

                ok = self._stationarity_checkpoint(bundle, next_name)

                current_name = next_name
                current_series = y_sd
                current_ops = list(bundle.meta(current_name).operations)

                if ok:
                    chosen = current_name
                    bundle.meta(chosen).recommended_for.append("final_stationary")
                    self._bundle = bundle
                    return bundle
        else:
            bundle.meta(chosen).notes.append("seasonal_period not set; skipping seasonal differencing.")

        # -----------------------------------------------------------------
        # Step 3: Non-seasonal differencing (d)
        # -----------------------------------------------------------------
        current_name = chosen
        current_series = bundle.get(current_name)
        current_ops = list(bundle.meta(current_name).operations)

        for d in self.difference_orders:
            d = int(d)
            if d <= 0:
                continue

            if self.max_variants is not None and len(bundle.variants) >= self.max_variants:
                bundle.meta(current_name).warnings.append("max_variants reached; stopping differencing.")
                break

            y_d, d_ops = self._apply_diff(current_series, lag=1, order=d)
            next_name = self._append_diff_name(current_name, d=d)

            self._register_variant(
                bundle=bundle,
                name=next_name,
                series=y_d,
                ops=current_ops + d_ops,
                lineage=f"{current_name} -> {next_name}",
            )

            ok = self._stationarity_checkpoint(bundle, next_name)

            current_name = next_name
            current_series = y_d
            current_ops = list(bundle.meta(current_name).operations)

            if ok:
                chosen = current_name
                bundle.meta(chosen).recommended_for.append("final_stationary")
                self._bundle = bundle
                return bundle

        # If we got here, nothing passed stationarity (or tests unavailable/ambiguous).
        bundle.meta(current_name).warnings.append("No variant achieved stationarity under the configured path.")
        bundle.meta(current_name).recommended_for.append("best_effort_final")

        self._bundle = bundle
        return bundle

    def get_bundle(self) -> TransformBundle:
        """Return cached bundle from the most recent fit()."""
        if self._bundle is None:
            raise RuntimeError("No TransformBundle available. Call fit(cleaned_df) first.")
        return self._bundle

    def set_seasonal_period(self, seasonal_period: Optional[int]) -> None:
        """Set/clear seasonal period for seasonal differencing."""
        if seasonal_period is None:
            self.seasonal_period = None
            return
        m = int(seasonal_period)
        self.seasonal_period = m if m >= 2 else None

    # ---------------------------------------------------------------------
    # Validation & registration
    # ---------------------------------------------------------------------
    def _validate_input(self, cleaned_df: pd.DataFrame) -> None:
        """
        Validate cleaned_df meets transformer requirements.

        MVP checks:
          - target_col exists
          - index is DateTimeIndex
          - index sorted increasing
          - target is numeric or coercible to numeric
        """
        if self.target_col not in cleaned_df.columns:
            raise KeyError(
                f"target_col='{self.target_col}' not found. Available columns: {list(cleaned_df.columns)}"
            )

        if not isinstance(cleaned_df.index, pd.DatetimeIndex):
            raise TypeError("cleaned_df must have a DateTimeIndex.")

        if not cleaned_df.index.is_monotonic_increasing:
            raise ValueError("cleaned_df index must be sorted increasing.")

        y = cleaned_df[self.target_col]
        if not pd.api.types.is_numeric_dtype(y):
            y2 = pd.to_numeric(y, errors="coerce")
            if y2.notna().sum() == 0:
                raise TypeError(
                    f"target_col='{self.target_col}' is not numeric/coercible; all values became NaN after coercion."
                )

    def _register_variant(
        self,
        bundle: TransformBundle,
        name: str,
        series: pd.Series,
        ops: List[SeriesOperation],
        lineage: Optional[str],
    ) -> None:
        """
        Register a variant and its meta into the bundle.

        Responsibilities:
          - enforce unique variant name
          - enforce max_variants cap
          - apply nan_policy deterministically
          - compute basic meta stats
          - attach operations + lineage
        """
        if name in bundle.variants:
            raise ValueError(f"Variant '{name}' already exists in bundle.")

        if self.max_variants is not None and len(bundle.variants) >= self.max_variants:
            raise RuntimeError(f"max_variants={self.max_variants} reached; refusing to register '{name}'.")

        s_raw = series.copy()
        if not isinstance(s_raw.index, pd.DatetimeIndex):
            raise TypeError(f"Variant '{name}' must have a DateTimeIndex.")

        if self.nan_policy == "drop":
            s_store = s_raw.dropna()
            dropna_count = int(s_raw.isna().sum())
        else:
            s_store = s_raw
            dropna_count = 0

        meta = SeriesVariantMeta(
            name=name,
            target_col=self.target_col,
            freq=self.freq,
            seasonal_period=self.seasonal_period,
            operations=list(ops),
            lineage=lineage,
        )

        meta.n_obs = int(len(s_store))
        meta.n_missing = int(s_raw.isna().sum())
        meta.dropna_count = int(dropna_count)

        if len(s_store) > 0:
            meta.start = pd.Timestamp(s_store.index.min())
            meta.end = pd.Timestamp(s_store.index.max())
            nn = s_store.dropna()
            if len(nn) > 0:
                meta.min_value = float(nn.min())
                meta.max_value = float(nn.max())

        bundle.variants[name] = s_store
        bundle.metas[name] = meta

    # ---------------------------------------------------------------------
    # Stationarity testing (checkpoint + batch)
    # ---------------------------------------------------------------------
    def _stationarity_checkpoint(self, bundle: TransformBundle, name: str) -> bool:
        """
        Evaluate stationarity for a single variant and write results into its meta.

        Returns True only when both ADF and KPSS support stationarity:
          - ADF: p < adf_alpha
          - KPSS: p > kpss_alpha

        If tests are disabled/unavailable or results are ambiguous, returns False and sets
        meta.is_stationary to None.
        """
        meta = bundle.meta(name)

        if not self.enable_stationarity_tests:
            meta.notes.append("Stationarity tests disabled; cannot evaluate stationarity checkpoint.")
            meta.is_stationary = None
            return False

        if adfuller is None or kpss is None:
            meta.warnings.append("statsmodels unavailable; stationarity tests skipped.")
            meta.is_stationary = None
            return False

        s = bundle.get(name).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < self.min_obs_for_tests:
            meta.notes.append(
                f"Insufficient observations for stationarity tests "
                f"(n={len(s)} < min_obs_for_tests={self.min_obs_for_tests})."
            )
            meta.is_stationary = None
            return False

        adf_p: Optional[float] = None
        kpss_p: Optional[float] = None

        try:
            adf_p = self._adf_pvalue(s)
        except Exception as e:
            meta.warnings.append(f"ADF failed: {e}")

        try:
            kpss_p = self._kpss_pvalue(s)
        except Exception as e:
            meta.warnings.append(f"KPSS failed: {e}")

        meta.adf_pvalue = adf_p
        meta.kpss_pvalue = kpss_p

        if adf_p is None or kpss_p is None:
            meta.is_stationary = None
            return False

        adf_ok = adf_p < self.adf_alpha
        kpss_ok = kpss_p > self.kpss_alpha

        if adf_ok and kpss_ok:
            meta.is_stationary = True
            meta.notes.append("Stationarity supported: ADF and KPSS agree (stationary).")
            return True

        if adf_ok != kpss_ok:
            meta.is_stationary = None
            meta.notes.append("Stationarity ambiguous: ADF and KPSS disagree.")
            return False

        meta.is_stationary = False
        meta.notes.append("Non-stationary supported: ADF and KPSS agree (non-stationary).")
        return False

    def attach_stationarity_tests(self, bundle: Optional[TransformBundle] = None) -> None:
        """
        Batch-attach ADF/KPSS p-values (and derived is_stationary flag) to all variant metas.
        """
        if bundle is None:
            bundle = self.get_bundle()

        if not self.enable_stationarity_tests:
            for n in bundle.list_variants():
                m = bundle.meta(n)
                m.notes.append("Stationarity tests disabled; attach_stationarity_tests skipped.")
                m.is_stationary = None
            return

        if adfuller is None or kpss is None:
            for n in bundle.list_variants():
                m = bundle.meta(n)
                m.warnings.append("statsmodels unavailable; stationarity tests skipped.")
                m.is_stationary = None
            return

        for n in bundle.list_variants():
            meta = bundle.meta(n)
            s = bundle.get(n).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()

            if len(s) < self.min_obs_for_tests:
                meta.notes.append(
                    f"Insufficient observations for stationarity tests "
                    f"(n={len(s)} < min_obs_for_tests={self.min_obs_for_tests})."
                )
                meta.adf_pvalue = None
                meta.kpss_pvalue = None
                meta.is_stationary = None
                continue

            try:
                meta.adf_pvalue = self._adf_pvalue(s)
            except Exception as e:
                meta.adf_pvalue = None
                meta.warnings.append(f"ADF failed: {e}")

            try:
                meta.kpss_pvalue = self._kpss_pvalue(s)
            except Exception as e:
                meta.kpss_pvalue = None
                meta.warnings.append(f"KPSS failed: {e}")

            if meta.adf_pvalue is None or meta.kpss_pvalue is None:
                meta.is_stationary = None
                continue

            adf_ok = meta.adf_pvalue < self.adf_alpha
            kpss_ok = meta.kpss_pvalue > self.kpss_alpha

            if adf_ok and kpss_ok:
                meta.is_stationary = True
            elif adf_ok != kpss_ok:
                meta.is_stationary = None
                meta.notes.append("Stationarity ambiguous: ADF and KPSS disagree.")
            else:
                meta.is_stationary = False

    def _adf_pvalue(self, s: pd.Series) -> float:
        """Return ADF p-value. Caller may pass NaNs; we defensively clean."""
        if adfuller is None:
            raise RuntimeError("adfuller unavailable (statsmodels not installed/importable).")

        if not isinstance(s, pd.Series):
            raise TypeError("_adf_pvalue expects a pandas Series.")

        x = s.astype("float64").replace([np.inf, -np.inf], np.nan).dropna().values
        if x.size < 3:
            raise ValueError("ADF requires at least 3 finite observations.")

        res = adfuller(x, autolag="AIC")
        return float(res[1])

    def _kpss_pvalue(self, s: pd.Series) -> float:
        """Return KPSS p-value. Caller may pass NaNs; we defensively clean."""
        if kpss is None:
            raise RuntimeError("kpss unavailable (statsmodels not installed/importable).")

        if not isinstance(s, pd.Series):
            raise TypeError("_kpss_pvalue expects a pandas Series.")

        x = s.astype("float64").replace([np.inf, -np.inf], np.nan).dropna().values
        if x.size < 3:
            raise ValueError("KPSS requires at least 3 finite observations.")

        res = kpss(x, regression="c", nlags="auto")
        return float(res[1])

    # ---------------------------------------------------------------------
    # Transform primitives
    # ---------------------------------------------------------------------
    def _apply_log(self, y: pd.Series) -> Tuple[pd.Series, SeriesOperation]:
        """
        Apply log transform with offset handling.

        Ensures strict positivity on non-null values by optionally adding an offset.
        """
        if not isinstance(y, pd.Series):
            raise TypeError("_apply_log expects a pandas Series.")

        s = y.astype("float64")
        s_nonnull = s.dropna()

        if len(s_nonnull) == 0:
            # preserve NaNs/index; output is log(NaN)=NaN
            out = np.log(s)
            return out, SeriesOperation(name="log", params={"offset": 0.0})

        minv = float(s_nonnull.min())

        if minv > 0:
            offset = 0.0
        else:
            if self.log_offset_strategy == "error":
                raise ValueError(
                    "log transform requires strictly positive values. "
                    f"Found min={minv}. Consider log_offset_strategy='auto' or 'fixed'."
                )

            if self.log_offset_strategy == "fixed":
                if self.log_offset_value is None:
                    raise ValueError("log_offset_value must be provided when log_offset_strategy='fixed'.")
                offset = float(self.log_offset_value)
            elif self.log_offset_strategy == "auto":
                offset = float(self._compute_log_offset_auto(s_nonnull))
            else:
                raise ValueError(
                    f"Unknown log_offset_strategy='{self.log_offset_strategy}'. "
                    "Expected one of: 'auto', 'fixed', 'error'."
                )

            if float((s_nonnull + offset).min()) <= 0:
                raise ValueError(
                    "Computed/provided log offset does not make the series strictly positive. "
                    f"min(y+offset)={float((s_nonnull + offset).min())}."
                )

        out = np.log(s + offset)
        return out, SeriesOperation(name="log", params={"offset": offset})

    def _compute_log_offset_auto(self, y: pd.Series) -> float:
        """Compute offset so that min(y + offset) is slightly above 0."""
        minv = float(y.min())
        if minv > 0:
            return 0.0
        eps = 1e-6
        return abs(minv) + eps

    def _apply_log1p(self, y: pd.Series) -> Tuple[pd.Series, SeriesOperation]:
        """
        Apply log1p transform: log(1 + y). MVP rule: require y > -1 to avoid -inf.
        """
        if not isinstance(y, pd.Series):
            raise TypeError("_apply_log1p expects a pandas Series.")

        s = y.astype("float64")
        s_nonnull = s.dropna()

        if len(s_nonnull) == 0:
            return np.log1p(s), SeriesOperation(name="log1p", params={})

        minv = float(s_nonnull.min())
        if minv <= -1.0:
            raise ValueError(f"log1p requires all non-null values to be > -1. Found min={minv}.")

        out = np.log1p(s)
        return out, SeriesOperation(name="log1p", params={})

    def _apply_diff(self, y: pd.Series, lag: int, order: int = 1) -> Tuple[pd.Series, List[SeriesOperation]]:
        """Apply non-seasonal differencing repeatedly (preserves index)."""
        if not isinstance(y, pd.Series):
            raise TypeError("_apply_diff expects a pandas Series.")

        lag_i = int(lag)
        order_i = int(order)
        if lag_i <= 0:
            raise ValueError(f"lag must be >= 1, got {lag}.")
        if order_i <= 0:
            raise ValueError(f"order must be >= 1, got {order}.")

        s = y.astype("float64").copy()
        for _ in range(order_i):
            s = s.diff(lag_i)

        ops = [SeriesOperation(name="diff", params={"lag": lag_i, "order": order_i})]
        return s, ops

    def _apply_seasonal_diff(self, y: pd.Series, m: int, order: int = 1) -> Tuple[pd.Series, List[SeriesOperation]]:
        """Apply seasonal differencing repeatedly at seasonal period m (preserves index)."""
        if not isinstance(y, pd.Series):
            raise TypeError("_apply_seasonal_diff expects a pandas Series.")

        m_i = int(m)
        order_i = int(order)
        if m_i < 2:
            raise ValueError(f"seasonal period m must be >= 2, got {m}.")
        if order_i <= 0:
            raise ValueError(f"order must be >= 1, got {order}.")

        s = y.astype("float64").copy()
        for _ in range(order_i):
            s = s.diff(m_i)

        ops = [SeriesOperation(name="seas_diff", params={"m": m_i, "order": order_i})]
        return s, ops

    # ---------------------------------------------------------------------
    # Naming helpers & convenience
    # ---------------------------------------------------------------------
    def _append_seasdiff_name(self, base: str, sd: int, m: int) -> str:
        return f"{base}__sd{int(sd)}_m{int(m)}"

    def _append_diff_name(self, base: str, d: int) -> str:
        return f"{base}__d{int(d)}"

    def _make_name(self, base_name: str, d: int = 0) -> str:
        d_i = int(d)
        return base_name if d_i <= 0 else f"{base_name}__d{d_i}"

    def find(self, prefix: str) -> List[str]:
        """Find variant names that start with prefix from cached bundle."""
        bundle = self.get_bundle()
        return [k for k in bundle.variants.keys() if k.startswith(prefix)]

    # ---------------------------------------------------------------------
    # Inversion (v1)
    # ---------------------------------------------------------------------
    def inverse_transform(
        self,
        series_transformed: pd.Series,
        variant_name: str,
        original_y: pd.Series,
    ) -> pd.Series:
        """
        Invert a transformed series back to original scale using stored operations.

        Supported v1:
          - identity
          - log (with offset)
          - log1p
          - diff with lag=1, order=1 (no seasonal differencing)

        Raises NotImplementedError for seasonal differencing or higher-order diffs.
        """
        if not isinstance(series_transformed, pd.Series):
            raise TypeError("series_transformed must be a pandas Series.")
        if not isinstance(original_y, pd.Series):
            raise TypeError("original_y must be a pandas Series.")

        bundle = self.get_bundle()
        meta = bundle.meta(variant_name)
        ops = list(meta.operations)

        # block unsupported ops (v1)
        for op in ops:
            if op.name == "seas_diff":
                raise NotImplementedError("inverse_transform does not support seasonal differencing yet.")
            if op.name == "boxcox":
                raise NotImplementedError("inverse_transform does not support boxcox yet.")

        s = series_transformed.astype("float64").copy()

        def _invert_diff_lag1_order1(diff_series: pd.Series, anchor_series: pd.Series) -> pd.Series:
            ds = diff_series.astype("float64")
            mask = ds.notna()
            if mask.sum() == 0:
                return ds.copy()

            # first valid index in diff space
            first_idx = ds.index[np.flatnonzero(mask.values)[0]]

            oy = anchor_series.astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
            if len(oy) == 0:
                raise ValueError("original_y has no finite values; cannot anchor diff inversion.")

            oy_before = oy[oy.index < first_idx]
            anchor = float(oy_before.iloc[-1]) if len(oy_before) > 0 else float(oy.iloc[-1])

            out = pd.Series(index=ds.index, dtype="float64")
            out.loc[~mask] = np.nan

            cum = ds.loc[mask].cumsum()
            out.loc[mask] = anchor + cum.values
            return out

        # invert in reverse operation order
        for op in reversed(ops):
            if op.name == "identity":
                continue

            if op.name == "log":
                offset = float(op.params.get("offset", 0.0))
                s = np.exp(s) - offset
                continue

            if op.name == "log1p":
                s = np.expm1(s)
                continue

            if op.name == "diff":
                lag = int(op.params.get("lag", 1))
                order = int(op.params.get("order", 1))
                if lag != 1 or order != 1:
                    raise NotImplementedError("inverse_transform supports only diff(lag=1, order=1) in v1.")
                s = _invert_diff_lag1_order1(s, original_y)
                continue

            raise NotImplementedError(f"inverse_transform does not support operation '{op.name}' yet.")

        return s