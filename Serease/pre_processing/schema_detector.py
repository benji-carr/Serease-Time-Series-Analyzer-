from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Serease.pre_processing import DataIngestor, IngestionMetadata

@dataclass
class SchemaMetadata:
    date_col: Optional[str]
    target_col: Optional[str]
    exog_cols: List[str]
    all_numeric: List[str]
    all_categorical: List[str]
    notes: List[str] = field(default_factory=list)


class SchemaDetector:
    """
    Lightweight schema detection for Serease.

    Inputs:
    - df: raw ingested DataFrame
    - ingestion_meta: IngestionMetadata from DataIngestor (optional)
    - user_date_col, user_target_col, user_exog_cols: optional overrides
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ingestion_meta: Optional[object] = None,
        user_date_col: Optional[str] = None,
        user_target_col: Optional[str] = None,
        user_exog_cols: Optional[Sequence[str]] = None,
        date_format: Optional[str] = None,
    ) -> None:
        self.df = df
        self.ingestion_meta = ingestion_meta

        self.user_date_col = user_date_col
        self.user_target_col = user_target_col
        self.user_exog_cols = list(user_exog_cols) if user_exog_cols else None
        self.date_format = date_format

        # Will be filled by detect()
        self.date_col: Optional[str] = None
        self.target_col: Optional[str] = None
        self.exog_cols: List[str] = []
        self.all_numeric: List[str] = []
        self.all_categorical: List[str] = []
        self.notes: List[str] = []

    # ======================================================================
    # Public API
    # ======================================================================
    def detect(self) -> SchemaMetadata:
        """
        Run all schema-detection steps and return a ``SchemaMetadata`` summary.

        This method orchestrates:
        - numeric vs. categorical splitting
        - date column detection
        - target column detection
        - exogenous feature selection
        - schema validation and note collection

        Returns
        -------
        SchemaMetadata
            A dataclass describing the detected time-series schema.
        """
        # 1. Basic numeric vs categorical split
        numeric_cols, categorical_cols = self._split_numeric_categorical()
        self.all_numeric = numeric_cols
        self.all_categorical = categorical_cols

        # 2. Detect date column
        date_col = self._detect_date_column()
        self.date_col = date_col

        # 3. Detect target column (exclude date col if present)
        target_col = self._detect_target_column(
            exclude=[date_col] if date_col else None
        )
        self.target_col = target_col

        # 4. Detect exogenous columns (exclude date + target)
        exclude = [c for c in [date_col, target_col] if c]
        exog_cols = self._detect_exog_columns(exclude=exclude)
        self.exog_cols = exog_cols

        # 5. Validate schema & collect notes
        notes = self._validate_columns(date_col, target_col, exog_cols)
        self.notes = notes

        return SchemaMetadata(
            date_col=date_col,
            target_col=target_col,
            exog_cols=exog_cols,
            all_numeric=numeric_cols,
            all_categorical=categorical_cols,
            notes=notes,
        )

    # ======================================================================
    # Internal helpers
    # ======================================================================
    def _split_numeric_categorical(self) -> Tuple[List[str], List[str]]:
        """
        Separate columns into numeric and non-numeric groups.

        Returns
        -------
        numeric_cols : list of str
            Columns with a numeric dtype.
        non_numeric_cols : list of str
            Columns with any non-numeric dtype (including datetime, strings,
            objects, categoricals, etc.).
        """
        df = self.df

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        return numeric_cols, non_numeric_cols

    def _detect_date_column(self) -> Optional[str]:
        """
        Infer the dataset's date column.

        Detection priority
        ------------------
        1. Use ``user_date_col`` if provided and parseable.
        2. Prefer native pandas datetime64 columns (with name-based scoring
           that favors plain 'date' over 'date_start' / 'date_end', etc.).
        3. Among non-datetime columns, look at:
           - column name (date/time/timestamp/period tokens)
           - sample values (head(10+), parsed with pandas as datetimes)
           and pick the best-scoring candidate.
        4. If all heuristics fail, return ``None`` and append a warning note.

        Returns
        -------
        str or None
            The inferred date column name, or ``None`` if no suitable column
            can be found.
        """
        df = self.df
        date_format = getattr(self, "date_format", None)

        # ------------------------------------------------------------------
        # 0. If the dataframe index is already datetime-like, allow using it
        # ------------------------------------------------------------------
        index_is_datetime = isinstance(df.index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(df.index)

        # If user did NOT explicitly provide a date column, prefer the index.
        # This avoids "date_col=None" when the dataset is already properly indexed.
        if index_is_datetime and not getattr(self, "user_date_col", None):
            self.notes.append("Using DateTimeIndex as date column (date_col='__index__').")
            return "__index__"

        # ------------------------------------------------------------------
        # Helper: name-based score for "date-ness"
        #   - strong preference for exact 'date'
        #   - then 'date_*' / '*_date'
        #   - then generic 'date', 'time', 'timestamp', 'period' in name
        #   - small penalty for 'start' / 'end' variants
        # ------------------------------------------------------------------
        preferred_tokens = ["date", "time", "timestamp", "period"]

        def _name_score(col_name: str) -> int:
            name = col_name.lower()
            score = 0

            # Exact 'date' gets a huge boost
            if name == "date":
                score += 1000

            for tok in preferred_tokens:
                if name == tok:
                    score += 800
                elif name.endswith("_" + tok) or name.startswith(tok + "_"):
                    score += 400
                elif tok in name:
                    score += 150

            # Slight penalty for 'start'/'end' to favor plain 'date'
            if ("start" in name or "end" in name) and "date" in name and name != "date":
                score -= 100

            return score

        # ------------------------------------------------------------------
        # Helper: content-based score — how date-like are the values?
        #   Use up to max_sample rows from head(), try to parse with
        #   pd.to_datetime (with date_format if provided).
        # ------------------------------------------------------------------
        def _content_score(series: pd.Series, max_sample: int = 50) -> float:
            s = series.head(max_sample).dropna()
            if s.empty:
                return 0.0

            # Convert to string to be safe
            s = s.astype(str)

            try:
                parsed = pd.to_datetime(
                    s,
                    errors="coerce",
                    format=date_format if date_format else None,
                )
            except Exception:
                return 0.0

            # Fraction of sample that parses to a datetime
            frac = (~parsed.isna()).mean()
            return float(frac)

        # ------------------------------------------------------------------
        # 1. User override
        # ------------------------------------------------------------------
        if getattr(self, "user_date_col", None):
            col = self.user_date_col
            if col in df.columns:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    series_for_parse = series.astype(str)
                else:
                    series_for_parse = series
                try:
                    _ = pd.to_datetime(
                        series_for_parse,
                        errors="raise",
                        format=date_format if date_format else None,
                    )
                    return col
                except Exception:
                    if date_format:
                        self.notes.append(
                            f"user_date_col='{col}' could not be parsed using "
                            f"date_format='{date_format}'."
                        )
                    else:
                        self.notes.append(
                            f"user_date_col='{col}' is not parseable as datetime; ignoring."
                        )

        # ------------------------------------------------------------------
        # 1b. DatetimeIndex override (strong signal)
        # ------------------------------------------------------------------
        # If ingestion already indicates a DatetimeIndex, prefer using it as the date axis.
        # We return a sentinel string so downstream schema logic can treat "date_col"
        # as coming from the index rather than a column.
        ingestion_meta = getattr(self, "ingestion_meta", None)
        index_is_datetime = False

        if ingestion_meta is not None and getattr(ingestion_meta, "index_is_datetime", False):
            index_is_datetime = True
        elif isinstance(df.index, pd.DatetimeIndex):
            index_is_datetime = True

        if index_is_datetime:
            self.notes.append(
                "Using DatetimeIndex as date axis (ingestion_meta.index_is_datetime=True)."
                if ingestion_meta is not None and getattr(ingestion_meta, "index_is_datetime", False)
                else "Using DatetimeIndex as date axis (df.index is DatetimeIndex)."
            )
            return "__index__"

        # ------------------------------------------------------------------
        # 2. Native datetime64 columns (already parsed)
        #    If multiple, choose by name_score (favor 'date' over 'date_start').
        # ------------------------------------------------------------------
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

        if len(datetime_cols) == 1:
            return datetime_cols[0]

        elif len(datetime_cols) > 1:
            scored = [(col, _name_score(col)) for col in datetime_cols]
            scored.sort(key=lambda x: x[1], reverse=True)
            # Even if all scores are 0, pick the first; they're all datetime anyway
            return scored[0][0]

        # ------------------------------------------------------------------
        # 3. Non-datetime columns: use name + sample content
        # ------------------------------------------------------------------
        best_col: Optional[str] = None
        best_score: float = -1.0

        for col in df.columns:
            # Skip columns that are already numeric datetime types (handled above)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            # Compute name-based score
            name_score = _name_score(col)

            # Compute content-based score on a sample of at least 10 rows (if available)
            content_frac = _content_score(df[col])
            # content_frac is in [0, 1]; treat >0 as some evidence of date-like values

            # If neither the name nor the content look remotely like dates, skip
            if name_score == 0 and content_frac < 0.3:
                continue

            # Combine scores: weight name and content
            # (adjust weights as needed; this strongly prefers 'date' but
            #  still requires content to look reasonably date-like)
            combined = name_score * 1.0 + content_frac * 500.0

            if combined > best_score:
                best_score = combined
                best_col = col

        if best_col is not None and best_score > 0:
            # Optional: log what we did
            self.notes.append(
                f"Inferred date column '{best_col}' via name/content heuristics "
                f"(combined_score={best_score:.1f})."
            )
            return best_col

        # ------------------------------------------------------------------
        # 4. Give up
        # ------------------------------------------------------------------
        msg = "No clear date/time column detected."
        if date_format:
            msg += f" Tried parsing with date_format='{date_format}'."
        self.notes.append(msg)

        return None

    def _detect_target_column(
        self,
        exclude: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        """
        Heuristically infer the target (y) column.

        Detection priority:
        1. Use ``user_target_col`` if provided and numeric.
        2. Consider remaining numeric columns (excluding ``exclude``).
        3. Remove ID-like columns (monotonic counters, *_id, etc.).
        4. Prefer columns with target-like names (sales, price, demand, etc.).
        5. Fall back to the column with highest ``variance × completeness``.

        Parameters
        ----------
        exclude : sequence of str, optional
            Column names to remove from consideration (usually date + known
            exogenous columns).

        Returns
        -------
        str or None
            The best target column candidate, or ``None`` if none is suitable.
        """
        df = self.df
        exclude_set = set(exclude or [])

        # 1. User override (if valid)
        if getattr(self, "user_target_col", None):
            col = self.user_target_col
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                return col

        # 2. Start from numeric columns (minus excluded)
        numeric_cols = (
            df.select_dtypes(include=["number"])
              .columns
              .difference(list(exclude_set))
              .tolist()
        )
        if not numeric_cols:
            return None

        # 3. Helper: detect ID-like / index-like numeric columns
        def _looks_like_id(col_name: str, series: pd.Series) -> bool:
            name = col_name.lower()

            # Name-based hints
            if name in {"id", "index"}:
                return True
            if name.endswith("_id") or name.startswith("id_"):
                return True
            if "id" in name and any(tok in name for tok in ["user", "row", "record", "cust", "client"]):
                return True

            # Pattern-based hints: monotonic integer sequence with nearly-constant step
            if pd.api.types.is_integer_dtype(series):
                s = series.dropna()
                if s.empty:
                    return False

                # If it's basically unique + monotonic, it's probably an index
                if s.is_monotonic_increasing and s.nunique() >= 0.9 * len(s):
                    diffs = s.diff().dropna()
                    if not diffs.empty:
                        most_common = diffs.mode().iloc[0]
                        if (diffs == most_common).mean() >= 0.9:
                            return True

            return False

        # 4. Filter out obvious ID-like numeric columns
        candidate_cols: List[str] = []
        for col in numeric_cols:
            s = df[col]
            if not _looks_like_id(col, s):
                candidate_cols.append(col)

        # If we filtered everything out, just fall back to raw numeric_cols
        if not candidate_cols:
            candidate_cols = numeric_cols

        # 5. Name-based heuristics (target-like names)
        name_priority = [
            "target",
            "y",
            "value",
            "values",
            "series",
            "sales",
            "revenue",
            "amount",
            "qty",
            "quantity",
            "demand",
            "load",
            "power",
            "traffic",
            "volume",
            "price",
            "count",
        ]

        def _name_score(col_name: str) -> int:
            """Higher score = more likely to be the target based on the name."""
            name = col_name.lower()
            score = 0
            for rank, token in enumerate(name_priority):
                if name == token:
                    # Exact match is highest weight
                    score += 1000 - rank
                elif name.endswith("_" + token) or name.startswith(token + "_"):
                    score += 500 - rank
                elif token in name:
                    score += 100 - rank
            return score

        scored = [(col, _name_score(col)) for col in candidate_cols]
        scored.sort(key=lambda x: x[1], reverse=True)

        if scored and scored[0][1] > 0:
            # At least one column had a positive name score
            return scored[0][0]

        # 6. Statistical heuristic: variance * completeness
        best_col = None
        best_score = -np.inf

        for col in candidate_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            non_null = s.notna().sum()
            if non_null == 0:
                continue

            completeness = non_null / len(s)
            var = float(s.var()) if non_null > 1 else 0.0
            score = completeness * var

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_col = col

        return best_col

    def _detect_exog_columns(
        self,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Select exogenous (X) feature columns.

        Detection rules:
        - If ``user_exog_cols`` is provided, intersect with existing columns
          and return those.
        - Otherwise use numeric columns excluding the date and target.
        - Drop constant columns (≤ 1 unique value).

        Parameters
        ----------
        exclude : sequence of str, optional
            Columns that must not be treated as exogenous features.

        Returns
        -------
        list of str
            Selected exogenous feature columns.
        """
        df = self.df
        exclude_set = set(exclude or [])

        # 1. User override
        if self.user_exog_cols:
            cols = [c for c in self.user_exog_cols if c in df.columns and c not in exclude_set]
            return cols

        # 2. Numeric exogs (minus date/target/etc.)
        exog_cols = (
            df.select_dtypes(include=["number"])
              .columns
              .difference(list(exclude_set))
              .tolist()
        )

        # Optional: drop "too-constant" columns (no signal)
        clean_exog_cols: List[str] = []
        for col in exog_cols:
            s = df[col]
            if s.nunique(dropna=True) <= 1:
                continue
            clean_exog_cols.append(col)

        return clean_exog_cols

    def _validate_columns(
        self,
        date_col: Optional[str],
        target_col: Optional[str],
        exog_cols: List[str],
    ) -> List[str]:
        """
        Validate detected schema components and accumulate human-readable notes.

        Validates:
        - presence of date and target columns
        - parseability and uniqueness of the date column
        - sortedness of the date index
        - numeric type of the target column
        - missingness levels for exogenous columns

        Returns
        -------
        list of str
            Warning and validation notes produced during schema checking.
        """
        notes: List[str] = []

        if date_col is None:
            notes.append("No date column detected. You may need to specify user_date_col.")
        else:
            # Check if date is unique / sorted (typical TS assumption)
            s = pd.to_datetime(self.df[date_col], errors="coerce")
            if s.isna().any():
                notes.append(f"Date column '{date_col}' has unparseable values.")
            if s.duplicated().any():
                notes.append(f"Date column '{date_col}' has duplicate timestamps.")
            if not s.is_monotonic_increasing:
                notes.append(f"Date column '{date_col}' is not sorted; pipeline will need to sort.")

        if target_col is None:
            notes.append("No numeric target column detected. You may need to specify user_target_col.")
        else:
            if not pd.api.types.is_numeric_dtype(self.df[target_col]):
                notes.append(f"Target column '{target_col}' is not numeric; SARIMAX may fail.")

        if not exog_cols:
            notes.append("No exogenous columns detected. Model will be univariate.")
        else:
            # Mild sanity check: very high missingness?
            for col in exog_cols:
                missing_frac = self.df[col].isna().mean()
                if missing_frac > 0.4:
                    notes.append(
                        f"Exogenous column '{col}' has {missing_frac:.1%} missing values."
                    )

        return notes
