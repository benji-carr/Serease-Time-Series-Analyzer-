from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from Serease.schema import SchemaMetadata


@dataclass
class TimeSeriesMeta:
    """
    Metadata describing a cleaned, modeling-ready time series.

    This dataclass is produced by ``TimeSeriesCleaner`` after date parsing,
    sorting, duplicate handling, and frequency inference have been applied.

    Parameters
    ----------
    date_col : str
        Name of the column used as the time index.
    target_col : str
        Name of the primary target (y) column.
    exog_cols : list of str
        Names of the columns treated as exogenous features.
    start : pandas.Timestamp
        First timestamp in the cleaned series.
    end : pandas.Timestamp
        Last timestamp in the cleaned series.
    freq : str or None
        Inferred frequency string (e.g. ``'D'``, ``'MS'``), or ``None`` if
        frequency could not be reliably inferred.
    n_obs : int
        Number of observations in the cleaned time series.
    has_missing_dates : bool
        Whether gaps exist in the date index relative to the inferred
        frequency.
    n_missing_dates : int
        Count of missing timestamps between ``start`` and ``end`` at the
        inferred frequency.
    has_duplicates : bool
        Whether duplicate timestamps were detected (and resolved) during
        cleaning.
    notes : list of str
        Human-readable notes and warnings generated during cleaning
        (e.g., irregular frequency, unresolved parsing issues).
    """
    date_col: str
    target_col: str
    exog_cols: List[str]

    start: pd.Timestamp
    end: pd.Timestamp
    freq: Optional[str]

    n_obs: int
    has_missing_dates: bool
    n_missing_dates: int
    has_duplicates: bool

    notes: List[str] = field(default_factory=list)


DuplicatesPolicy = Literal["sum", "mean", "first", "last"]
MissingDatesPolicy = Literal["keep", "reindex"]  # "keep" = do not fill, just flag


class TimeSeriesCleaner:
    """
    Clean and standardize a raw time-series dataset based on detected schema.

    Responsibilities
    ----------------
    - Parse the date column into a proper ``DateTimeIndex``.
    - Sort observations chronologically.
    - Detect and resolve duplicate timestamps.
    - Infer the series frequency when possible.
    - Detect missing timestamps relative to the inferred frequency.
    - Produce a cleaned DataFrame plus a ``TimeSeriesMeta`` summary.

    This class does not perform value imputation, transformations
    (log/Box-Cox), or differencing. Those are handled by later modules
    in the pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ingested dataset.
    schema : SchemaMetadata
        Schema information produced by ``SchemaDetector`` (date, target,
        exogenous columns, and basic diagnostics).
    freq : str, optional
        User-specified frequency (e.g., ``'D'``, ``'H'``, ``'MS'``). If provided,
        this overrides automatic frequency inference.
    duplicates : {'sum', 'mean', 'first', 'last}, default 'sum'
        Strategy to resolve duplicate timestamps:
        - 'sum'   : aggregate duplicates by summing numeric values
        - 'mean'  : aggregate duplicates by mean of numeric values
        - 'first' : keep the first occurrence
        - 'last'  : keep the last occurrence
    missing : {'keep', 'reindex'}, default 'keep'
        Strategy for handling missing dates relative to the inferred
        frequency:
        - 'keep'    : do not reindex; only flag missing dates in metadata
        - 'reindex' : reindex to a complete date range and introduce NaNs
    date_format : str, optional
        Explicit strftime-compatible format string for parsing the date
        column (e.g. ``'%Y-%m-%d %H:%M:%S'``). When provided, this format
        is passed to ``pd.to_datetime`` to avoid slow per-element parsing
        and to ensure consistent interpretation.

    Notes
    -----
    The cleaned DataFrame returned by ``clean()`` always has a
    ``DateTimeIndex`` and is sorted in ascending order by time. The
    index is not necessarily regular unless a frequency can be inferred.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: SchemaMetadata,
        freq: Optional[str] = None,
        duplicates: DuplicatesPolicy = "sum",
        missing: MissingDatesPolicy = "keep",
        date_format: Optional[str] = None,
    ) -> None:
        self.df = df.copy()
        self.schema = schema
        self.freq_override = freq
        self.duplicates_policy: DuplicatesPolicy = duplicates
        self.missing_policy: MissingDatesPolicy = missing
        self.date_format = date_format  # new: used in _ensure_datetime_index

        self.notes: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clean(self) -> Tuple[pd.DataFrame, TimeSeriesMeta]:
        """
        Run the complete time-series cleaning step.

        This method orchestrates:
        - date parsing and index setting
        - chronological sorting
        - duplicate timestamp detection and resolution
        - frequency inference (unless overridden)
        - missing date detection (and optional reindexing)

        Returns
        -------
        cleaned_df : pandas.DataFrame
            Cleaned dataset with a ``DateTimeIndex``, sorted in ascending
            time order.
        ts_meta : TimeSeriesMeta
            Metadata describing the cleaned series (start, end, freq,
            n_obs, missing dates, duplicates, and notes).
        """
        # 1. Ensure DateTimeIndex and sort
        df_ts = self._ensure_datetime_index()

        # 2. Resolve duplicate timestamps if present
        df_ts, has_duplicates = self._handle_duplicates(df_ts)

        # 3. Infer or use provided frequency
        freq, is_regular = self._infer_frequency(df_ts.index)

        # 4. Detect missing dates (and optionally reindex)
        df_ts, has_missing, n_missing = self._handle_missing_dates(df_ts, freq)

        # 5. Build TimeSeriesMeta
        ts_meta = self._build_meta(
            df_ts=df_ts,
            freq=freq,
            has_missing=has_missing,
            n_missing=n_missing,
            has_duplicates=has_duplicates,
        )

        return df_ts, ts_meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_datetime_index(self) -> pd.DataFrame:
        """
        Convert the detected date column to a ``DateTimeIndex`` using the
        optional user-specified ``date_format`` if provided, then sort.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by datetime and sorted in ascending order.

        Raises
        ------
        ValueError
            If no date column is available or if parsing fails completely.
        """
        date_col = self.schema.date_col
        if date_col is None:
            raise ValueError("No date column available in schema for TimeSeriesCleaner.")

        if date_col not in self.df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

        series = self.df[date_col]
        if pd.api.types.is_numeric_dtype(series):
            series_for_parse = series.astype(str)
        else:
            series_for_parse = series

        # Parse to datetime (using explicit date_format if provided)
        try:
            dt = pd.to_datetime(
                series_for_parse,
                errors="coerce",
                format=self.date_format if self.date_format else None,
            )
        except Exception as exc:
            # Explicit format was provided but parsing failed hard
            if self.date_format:
                raise ValueError(
                    f"Failed to parse date column '{date_col}' using "
                    f"date_format='{self.date_format}'. Original error: {exc}"
                )
            else:
                raise

        # Evaluate parsing success
        if dt.isna().all():
            if self.date_format:
                raise ValueError(
                    f"Failed to parse any values in date column '{date_col}' "
                    f"using date_format='{self.date_format}'."
                )
            else:
                raise ValueError(
                    f"Failed to parse any values in date column '{date_col}' as datetime."
                )

        # Warn (via notes) about partial failures
        if dt.isna().any():
            n_bad = dt.isna().sum()
            if self.date_format:
                self.notes.append(
                    f"Date column '{date_col}' has {n_bad} values that do not match "
                    f"date_format='{self.date_format}'; dropping those rows."
                )
            else:
                self.notes.append(
                    f"Date column '{date_col}' contains {n_bad} unparseable values; "
                    "these rows will be dropped during cleaning."
                )

        # Drop invalid rows and finalize
        mask_valid = ~dt.isna()
        df_ts = self.df.loc[mask_valid].copy()
        df_ts[date_col] = dt[mask_valid]

        # Set index and sort chronologically
        df_ts = df_ts.set_index(date_col)
        df_ts = df_ts.sort_index()

        return df_ts

    def _handle_duplicates(self, df_ts: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """
        Detect and resolve duplicate timestamps according to the configured policy.

        Parameters
        ----------
        df_ts : pandas.DataFrame
            DataFrame with a ``DateTimeIndex``.

        Returns
        -------
        cleaned : pandas.DataFrame
            DataFrame with duplicate timestamps resolved.
        has_duplicates : bool
            True if duplicates were detected (and handled), False otherwise.
        """
        idx = df_ts.index
        has_dupes = idx.duplicated().any()
        if not has_dupes:
            return df_ts, False

        self.notes.append("Duplicate timestamps detected in index; resolving via "
                          f"policy='{self.duplicates_policy}'.")

        if self.duplicates_policy in {"sum", "mean"}:
            agg = "sum" if self.duplicates_policy == "sum" else "mean"
            numeric_cols = df_ts.select_dtypes(include="number").columns.tolist()
            # Keep non-numeric columns from the first occurrence
            non_numeric_cols = [c for c in df_ts.columns if c not in numeric_cols]

            agg_dict = {col: agg for col in numeric_cols}
            for col in non_numeric_cols:
                agg_dict[col] = "first"

            cleaned = df_ts.groupby(level=0).agg(agg_dict)
        elif self.duplicates_policy == "first":
            cleaned = df_ts[~idx.duplicated(keep="first")]
        elif self.duplicates_policy == "last":
            cleaned = df_ts[~idx.duplicated(keep="last")]
        else:
            raise ValueError(f"Unsupported duplicates policy: {self.duplicates_policy}")

        return cleaned, True

    def _infer_frequency(
            self,
            index: pd.DatetimeIndex,
    ) -> Tuple[Optional[str], bool]:
        """
        Infer the frequency of a ``DateTimeIndex``.

        Strategy
        --------
        1. Use ``freq_override`` if provided.
        2. Try ``pd.infer_freq``.
        3. If that fails, fall back to the most common time delta between
           consecutive timestamps. If this dominant delta explains the vast
           majority of gaps (>= 95%), treat it as the effective frequency,
           but mark the series as irregular.
        """
        # 1. User override wins
        if self.freq_override is not None:
            return self.freq_override, True

        # 2. Ask pandas first
        try:
            freq = pd.infer_freq(index)
        except ValueError:
            freq = None

        if freq is not None:
            # Basic regularity check: reindex and compare
            full_index = pd.date_range(start=index[0], end=index[-1], freq=freq)
            is_regular = len(full_index) == len(index) and index.equals(full_index)

            if not is_regular:
                self.notes.append(
                    f"Series appears irregular relative to inferred frequency '{freq}'."
                )
            return freq, is_regular

        # 3. Fallback: dominant time delta heuristic
        if len(index) < 3:
            self.notes.append("Too few points to infer a robust frequency.")
            return None, False

        diffs = index.to_series().diff().dropna()
        counts = diffs.value_counts()
        dominant_delta = counts.index[0]
        dominant_frac = counts.iloc[0] / counts.sum()

        if dominant_frac < 0.95:
            # No clearly dominant cadence
            self.notes.append(
                "Could not reliably infer a regular frequency; "
                "no dominant time delta between observations."
            )
            return None, False

        # Try to convert Timedelta -> frequency string
        try:
            freq = pd.tseries.frequencies.to_offset(dominant_delta).freqstr
        except Exception:
            self.notes.append(
                "Identified a dominant time delta "
                f"({dominant_delta}) but could not map it to a frequency string."
            )
            return None, False

        self.notes.append(
            "Pandas could not infer a frequency, but a dominant cadence of "
            f"{dominant_delta} (~{dominant_frac:.1%} of gaps) was detected "
            f"and mapped to freq='{freq}'. Series may still be irregular."
        )

        return freq, False  # False = not perfectly regular

    def _handle_missing_dates(
        self,
        df_ts: pd.DataFrame,
        freq: Optional[str],
    ) -> Tuple[pd.DataFrame, bool, int]:
        """
        Detect missing timestamps and reindex the series to a complete grid.

        Parameters
        ----------
        df_ts : pandas.DataFrame
            Time-indexed DataFrame after duplicate handling.
        freq : str or None
            Inferred or user-provided frequency.

        Returns
        -------
        cleaned : pandas.DataFrame
            Reindexed DataFrame, with a complete date range from the first
            to the last timestamp at the specified frequency. Any dates that
            did not exist in the original data will appear with NaN values.
        has_missing : bool
            Whether missing dates were detected in the original index.
        n_missing : int
            Number of missing timestamps between start and end in the
            original series.
        """
        index = df_ts.index

        if freq is None or len(index) == 0:
            # Without a frequency, we cannot define "missing dates".
            return df_ts, False, 0

        # Build the full expected index at the given frequency
        full_index = pd.date_range(start=index[0], end=index[-1], freq=freq)

        # Compare original index to the full grid
        missing = full_index.difference(index)
        n_missing = len(missing)
        has_missing = n_missing > 0

        if has_missing:
            self.notes.append(
                f"Detected {n_missing} missing timestamps between {index[0]} and {index[-1]} "
                f"at frequency '{freq}'. The series has been reindexed to a full {freq} grid; "
                "NaN values mark originally missing observations."
            )
        else:
            self.notes.append(
                f"No missing timestamps detected between {index[0]} and {index[-1]} "
                f"at frequency '{freq}'. Index has been reindexed to a full grid."
            )

        # Always reindex to the full grid; do not fill values here.
        df_ts = df_ts.reindex(full_index)

        return df_ts, has_missing, n_missing

    def _build_meta(
        self,
        df_ts: pd.DataFrame,
        freq: Optional[str],
        has_missing: bool,
        n_missing: int,
        has_duplicates: bool,
    ) -> TimeSeriesMeta:
        """
        Construct a TimeSeriesMeta object from the cleaned dataset and notes.

        Parameters
        ----------
        df_ts : pandas.DataFrame
            Cleaned time series (after reindexing to a full date grid).
        freq : str or None
            Inferred or user-provided frequency.
        has_missing : bool
            Whether missing dates were present in the original series.
        n_missing : int
            Number of missing timestamps in the original series.
        has_duplicates : bool
            Whether duplicates were detected and resolved.

        Returns
        -------
        TimeSeriesMeta
            Metadata summary of the cleaned series.
        """
        if df_ts.empty:
            # This should be extremely rare; upstream checks should prevent it.
            raise ValueError("Cleaned time series is empty; cannot build TimeSeriesMeta.")

        idx = df_ts.index

        start = idx[0]
        end = idx[-1]
        n_obs = len(df_ts)

        meta = TimeSeriesMeta(
            date_col=self.schema.date_col or "",
            target_col=self.schema.target_col or "",
            exog_cols=self.schema.exog_cols,
            start=start,
            end=end,
            freq=freq,
            n_obs=n_obs,
            has_missing_dates=has_missing,
            n_missing_dates=n_missing,
            has_duplicates=has_duplicates,
            notes=self.notes.copy(),
        )
        return meta
