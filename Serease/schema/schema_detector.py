from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Serease.ingestion import DataIngestor, IngestionMetadata

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
    ) -> None:
        self.df = df
        self.ingestion_meta = ingestion_meta

        self.user_date_col = user_date_col
        self.user_target_col = user_target_col
        self.user_exog_cols = list(user_exog_cols) if user_exog_cols else None

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

        Detection priority:
        1. Use ``user_date_col`` if provided and parseable.
        2. Prefer native pandas datetime64 columns.
        3. Attempt to parse columns with date-like names.
        4. If all heuristics fail, return ``None`` and append a warning note.

        Returns
        -------
        str or None
            The inferred date column name, or ``None`` if no suitable column
            can be found.
        """
        df = self.df

        # 1. User override
        if getattr(self, "user_date_col", None):
            col = self.user_date_col
            if col in df.columns:
                try:
                    _ = pd.to_datetime(df[col], errors="raise")
                    return col
                except Exception:
                    self.notes.append(
                        f"user_date_col='{col}' is not parseable as datetime; ignoring."
                    )

        # 2. Existing datetime columns
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
        if len(datetime_cols) == 1:
            return datetime_cols[0]
        elif len(datetime_cols) > 1:
            # Prefer names that look like dates
            preferred_tokens = ["date", "time", "timestamp", "period"]
            scored = []
            for col in datetime_cols:
                name = col.lower()
                score = sum(tok in name for tok in preferred_tokens)
                scored.append((col, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > 0:
                return scored[0][0]
            # Otherwise, just take the first datetime column
            return datetime_cols[0]

        # 3. Try to parse any column with a date-like name
        candidate_cols = [
            c for c in df.columns
            if any(tok in c.lower() for tok in ["date", "time", "timestamp", "period"])
        ]

        for col in candidate_cols:
            try:
                _ = pd.to_datetime(df[col], errors="raise")
                return col
            except Exception:
                continue

        # 4. Give up: no reliable date column
        self.notes.append("No clear date/time column detected.")
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
