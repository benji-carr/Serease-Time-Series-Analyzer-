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
    name: TransformName
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeriesVariantMeta:
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
    Builds transformed/differenced variants of a target series y for diagnostics and modeling.

    This class assumes the input is already cleaned:
    - DateTimeIndex
    - sorted index
    - missing timestamps reindexed (potential NaNs present)
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

    def fit(self, cleaned_df: pd.DataFrame) -> TransformBundle:
        self._validate_input(cleaned_df)

        y = cleaned_df[self.target_col].copy()
        bundle = TransformBundle(base_name="raw")

        base_variants = self._generate_base_transform_variants(y)

        for base_name, base_series, base_ops in base_variants:
            for d in self.difference_orders:
                if d == 0:
                    name = base_name
                    ops = list(base_ops)
                    s = base_series
                    self._register_variant(bundle, name, s, ops, lineage=None)
                else:
                    s_d, ops_d = self._apply_diff(base_series, lag=1, order=d)
                    name = self._make_name(base_name, d=d)
                    ops = list(base_ops) + ops_d
                    self._register_variant(bundle, name, s_d, ops, lineage=base_name)

        if self.seasonal_period is not None:
            self._add_seasonal_variants_inplace(bundle)

        if self.enable_stationarity_tests:
            self.attach_stationarity_tests(bundle)

        self._bundle = bundle
        return bundle

    def get_bundle(self) -> TransformBundle:
        if self._bundle is None:
            raise ValueError("No bundle available. Call fit(cleaned_df) first.")
        return self._bundle

    def set_seasonal_period(self, seasonal_period: Optional[int]) -> None:
        if seasonal_period is None:
            self.seasonal_period = None
            return
        m = int(seasonal_period)
        if m < 2:
            raise ValueError("seasonal_period must be >= 2.")
        self.seasonal_period = m

    def add_seasonal_variants(self, bundle: Optional[TransformBundle] = None) -> TransformBundle:
        if self.seasonal_period is None:
            raise ValueError("seasonal_period is not set. Call set_seasonal_period(m) first.")
        if bundle is None:
            bundle = self.get_bundle()
        self._add_seasonal_variants_inplace(bundle)
        if self.enable_stationarity_tests:
            self.attach_stationarity_tests(bundle)
        return bundle

    def attach_stationarity_tests(self, bundle: Optional[TransformBundle] = None) -> None:
        if bundle is None:
            bundle = self.get_bundle()

        if adfuller is None or kpss is None:
            for name in bundle.list_variants():
                bundle.meta(name).warnings.append(
                    "statsmodels not available; cannot compute ADF/KPSS."
                )
            return

        for name, s in bundle.variants.items():
            meta = bundle.metas[name]
            s2 = s.dropna()

            if len(s2) < self.min_obs_for_tests:
                meta.notes.append("Too few observations for stationarity tests.")
                continue

            try:
                meta.adf_pvalue = float(self._adf_pvalue(s2))
            except Exception:
                meta.warnings.append("ADF test failed.")
                meta.adf_pvalue = None

            try:
                meta.kpss_pvalue = float(self._kpss_pvalue(s2))
            except Exception:
                meta.warnings.append("KPSS test failed.")
                meta.kpss_pvalue = None

            if meta.adf_pvalue is not None and meta.kpss_pvalue is not None:
                meta.is_stationary = (meta.adf_pvalue < self.adf_alpha) and (meta.kpss_pvalue > self.kpss_alpha)

    def inverse_transform(
        self,
        series_transformed: pd.Series,
        variant_name: str,
        original_y: pd.Series,
    ) -> pd.Series:
        """
        Invert a transformed series back to the original scale using metadata.

        Supports v1 inversion for:
        - identity
        - log (with stored offset)
        - log1p
        - diff (lag=1, order=1) using last observed original_y

        For more complex pipelines (seasonal differencing, higher-order diffs),
        this raises NotImplementedError for now.
        """
        bundle = self.get_bundle()
        meta = bundle.meta(variant_name)

        ops = meta.operations
        if not ops:
            raise ValueError(f"Variant '{variant_name}' has no operations metadata.")

        s = series_transformed.copy()

        diff_ops = [op for op in ops if op.name in ("diff", "seas_diff")]
        if any(op.name == "seas_diff" for op in diff_ops):
            raise NotImplementedError("Inverse for seasonal differencing not implemented in v1.")
        if len([op for op in diff_ops if op.name == "diff"]) > 1:
            raise NotImplementedError("Inverse for multiple diff steps not implemented in v1.")

        transform_ops = [op for op in ops if op.name in ("identity", "log", "log1p", "boxcox")]
        if len(transform_ops) == 0:
            raise ValueError("No base transform operation found.")
        if len(transform_ops) > 1:
            raise NotImplementedError("Inverse for multiple base transforms not implemented in v1.")

        base_op = transform_ops[0]
        diff_op = next((op for op in ops if op.name == "diff"), None)

        if diff_op is not None:
            lag = int(diff_op.params.get("lag", 1))
            order = int(diff_op.params.get("order", 1))
            if lag != 1 or order != 1:
                raise NotImplementedError("Inverse for diff lag!=1 or order!=1 not implemented in v1.")
            if original_y.dropna().empty:
                raise ValueError("original_y must contain at least one non-missing value for inverse differencing.")
            last_val = float(original_y.dropna().iloc[-1])
            s = s.fillna(0.0).cumsum() + last_val

        if base_op.name == "identity":
            return s
        if base_op.name == "log":
            offset = float(base_op.params.get("offset", 0.0))
            return np.exp(s) - offset
        if base_op.name == "log1p":
            return np.expm1(s)

        raise NotImplementedError(f"Inverse transform not implemented for base transform '{base_op.name}'.")

    def _validate_input(self, cleaned_df: pd.DataFrame) -> None:
        if not isinstance(cleaned_df, pd.DataFrame):
            raise ValueError("cleaned_df must be a pandas DataFrame.")
        if not isinstance(cleaned_df.index, pd.DatetimeIndex):
            raise ValueError("TimeSeriesTransformer expects cleaned_df with a DateTimeIndex.")
        if self.target_col not in cleaned_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in cleaned_df.")
        if not pd.api.types.is_numeric_dtype(cleaned_df[self.target_col]):
            raise ValueError(f"Target column '{self.target_col}' must be numeric for transformations.")
        if cleaned_df.index.has_duplicates:
            raise ValueError("cleaned_df index has duplicates; expected cleaned index from TimeSeriesCleaner.")
        if not cleaned_df.index.is_monotonic_increasing:
            raise ValueError("cleaned_df index must be sorted ascending.")

    def _generate_base_transform_variants(
        self, y: pd.Series
    ) -> List[Tuple[str, pd.Series, List[SeriesOperation]]]:
        out: List[Tuple[str, pd.Series, List[SeriesOperation]]] = []

        for tname in self.transforms:
            if tname == "identity":
                out.append(("raw", y, [SeriesOperation("identity", {})]))
            elif tname == "log":
                y_log, op = self._apply_log(y)
                out.append(("log", y_log, [op]))
            elif tname == "log1p":
                y_log1p, op = self._apply_log1p(y)
                out.append(("log1p", y_log1p, [op]))
            else:
                raise ValueError(f"Unsupported base transform '{tname}'.")

        return out

    def _apply_log(self, y: pd.Series) -> Tuple[pd.Series, SeriesOperation]:
        if self.log_offset_strategy == "fixed":
            if self.log_offset_value is None:
                raise ValueError("log_offset_value must be set when log_offset_strategy='fixed'.")
            offset = float(self.log_offset_value)
        elif self.log_offset_strategy == "error":
            if (y.dropna() <= 0).any():
                raise ValueError("log transform requires y > 0 when log_offset_strategy='error'.")
            offset = 0.0
        else:
            offset = float(self._compute_log_offset_auto(y))

        s = np.log(y + offset)
        return s, SeriesOperation("log", {"offset": offset})

    def _compute_log_offset_auto(self, y: pd.Series) -> float:
        y2 = y.dropna()
        if y2.empty:
            return 0.0
        min_val = float(y2.min())
        if min_val > 0:
            return 0.0
        return abs(min_val) + 1e-6

    def _apply_log1p(self, y: pd.Series) -> Tuple[pd.Series, SeriesOperation]:
        y2 = y.dropna()
        if not y2.empty and float(y2.min()) < -1:
            raise ValueError("log1p requires y >= -1 for all non-missing values.")
        return np.log1p(y), SeriesOperation("log1p", {})

    def _apply_diff(self, y: pd.Series, lag: int, order: int = 1) -> Tuple[pd.Series, List[SeriesOperation]]:
        s = y.copy()
        for _ in range(int(order)):
            s = s.diff(int(lag))
        return s, [SeriesOperation("diff", {"lag": int(lag), "order": int(order)})]

    def _apply_seasonal_diff(self, y: pd.Series, m: int, order: int = 1) -> Tuple[pd.Series, List[SeriesOperation]]:
        s = y.copy()
        for _ in range(int(order)):
            s = s.diff(int(m))
        return s, [SeriesOperation("seas_diff", {"lag": int(m), "order": int(order)})]

    def _make_name(self, base_name: str, d: int = 0) -> str:
        if d == 0:
            return base_name
        if base_name == "raw":
            return f"diff{d}"
        return f"{base_name}_diff{d}"

    def _register_variant(
        self,
        bundle: TransformBundle,
        name: str,
        series: pd.Series,
        ops: List[SeriesOperation],
        lineage: Optional[str],
    ) -> None:
        if self.max_variants is not None and len(bundle.variants) >= int(self.max_variants):
            return

        if name in bundle.variants:
            return

        s = series.copy()
        if self.nan_policy == "drop":
            s = s.dropna()

        y_non_na = s.dropna()
        min_v = float(y_non_na.min()) if not y_non_na.empty else None
        max_v = float(y_non_na.max()) if not y_non_na.empty else None

        meta = SeriesVariantMeta(
            name=name,
            target_col=self.target_col,
            freq=self.freq,
            seasonal_period=self.seasonal_period,
            operations=list(ops),
            n_obs=int(len(s)),
            n_missing=int(s.isna().sum()),
            dropna_count=int(series.isna().sum() - s.isna().sum()),
            start=s.index.min() if len(s) else None,
            end=s.index.max() if len(s) else None,
            min_value=min_v,
            max_value=max_v,
            lineage=lineage,
        )

        bundle.variants[name] = s
        bundle.metas[name] = meta

    def _add_seasonal_variants_inplace(self, bundle: TransformBundle) -> None:
        if self.seasonal_period is None:
            return
        m = int(self.seasonal_period)
        if m < 2:
            return

        existing = list(bundle.variants.items())

        for base_name, s in existing:
            base_meta = bundle.meta(base_name)

            if any(op.name == "seas_diff" for op in base_meta.operations):
                continue

            for D in self.seasonal_difference_orders:
                if D == 0:
                    continue

                sD, opsD = self._apply_seasonal_diff(s, m=m, order=D)
                new_name = f"{base_name}_seasdiff{D}_m{m}"

                new_ops = list(base_meta.operations) + opsD
                self._register_variant(bundle, new_name, sD, new_ops, lineage=base_name)

    def _adf_pvalue(self, s: pd.Series) -> float:
        res = adfuller(s.values, autolag="AIC")
        return float(res[1])

    def _kpss_pvalue(self, s: pd.Series) -> float:
        res = kpss(s.values, regression="c", nlags="auto")
        return float(res[1])
