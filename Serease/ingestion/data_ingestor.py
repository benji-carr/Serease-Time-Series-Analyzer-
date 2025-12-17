from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from pandas.errors import ParserError
import io
import chardet


@dataclass
class IngestionMetadata:
    """
    Metadata describing the result of a data ingestion operation.

    This dataclass is produced by ``DataIngestor`` and captures basic
    structural information about the loaded dataset along with ingestion-
    related settings such as inferred file type, encoding, and delimiter.
    Other components of the Serease pipeline use this metadata to perform
    validation, schema detection, and diagnostics.

    Parameters
    ----------
    n_rows : int
        Number of rows in the loaded dataset.
    n_cols : int
        Number of columns in the dataset.
    column_names : list of str
        List of column names in the order they appear in the DataFrame.
    dtypes : dict of {str: str}
        Mapping of column names to their stringified pandas dtypes.
    file_type : str
        The detected or declared file type (e.g., ``'csv'`` or ``'excel'``).
    encoding : str or None
        Inferred file encoding, or ``None`` if not applicable.
    delimiter : str or None
        Inferred delimiter for CSV files (e.g., ``','`` or ``'\t'``);
        ``None`` for non-CSV files.
    warnings : list of str
        List of human-readable ingestion warnings, such as:
        - low row count
        - duplicate column names
        - absence of numeric or datetime columns
        - encoding detection fallback

    Notes
    -----
    This dataclass does not store the dataset itself—only structural
    metadata useful for downstream schema detection and model preparation.
    All fields are simple Python types to ensure JSON-serializability if
    needed for logging or API responses.
    """
    n_rows: int
    n_cols: int
    column_names: List[str]
    dtypes: Dict[str, str]
    file_type: str
    encoding: Optional[str]
    delimiter: Optional[str]
    warnings: List[str]
    index_is_datetime: bool


class DataIngestor:
    """
    Class responsible for reading user-uploaded CSV/Excel data
    and producing a raw DataFrame with basic metadata.

    Responsibilities:
    - Detect file type if not provided
    - Detect encoding (basic heuristic initially)
    - Detect delimiter for CSV (use simple rules or csv.Sniffer)
    - Load data into a DataFrame
    - Run minimal validation checks and record warnings

    NOT responsible for:
    - Schema detection (date/target/exog) → handled by SchemaDetector
    - Cleaning/resampling → handled by TimeSeriesCleaner
    - Transformations or diagnostics → later modules
    """

    def __init__(
            self,
            source: str | Path | object,
            file_type: Optional[str] = None,
            encoding: Optional[str] = None,
            delimiter: Optional[str] = None,
            max_rows: Optional[int] = None,
    ) -> None:
        if isinstance(source, (str, Path)):
            self.source = Path(source)
        else:
            self.source = source

        self.file_type = file_type
        self.encoding = encoding
        self.delimiter = delimiter
        self.max_rows = max_rows

        self.df: Optional[pd.DataFrame] = None
        self.warnings: list[str] = []

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        """
        if self.file_type is None:
            self.file_type = self._detect_file_type()

        if self.file_type == 'csv':
            df = self._load_csv()
        elif self.file_type == 'excel':
            df = self._load_excel()
        else:
            raise ValueError(f"Unsupported file type in load(): {self.file_type}")

        self.df = df

        self._basic_validate()

        return df

    def preview(self, n: int = 5) -> pd.DataFrame:
        """
        """
        if self.df is None:
            self.load()
        return self.df.head(n)

    def get_metadata(self) -> IngestionMetadata:
        """
        """
        if self.df is None:
            self.load()

        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

        return IngestionMetadata(
            n_rows=len(self.df),
            n_cols=self.df.shape[1],
            column_names=list(self.df.columns),
            dtypes=dtypes,
            file_type=self.file_type or 'unknown',
            encoding=self.encoding,
            delimiter=getattr(self, 'delimiter', None),
            warnings=self.warnings.copy(),
            index_is_datetime=index_is_datetime,
        )

    def validate(self) -> List[str]:
        """
        Run basic validation checks on the loaded DataFrame and return
        a list of warnings.

        This is a public wrapper around _basic_validate().
        """
        if self.df is None:
            self.load()

        self._basic_validate()
        return self.warnings.copy()
    # ----------------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------------

    def _detect_file_type(self):
        """
        Use the file extenstion to determine whether the file is an excel or a csv. Raise an error if the file type is unsupported.
        """
        # checking file via filepath suffix or by filename if the filepath isn't available
        if isinstance(self.source, (str, Path)):
            suffix = Path(self.source).suffix.lower()
        elif hasattr(self.source, "filename"):
            suffix = Path(self.source.filename).suffix.lower()
        else:
            raise ValueError("Cannot detect file type from this source")
        # once suffix is found we check it against
        if suffix == ".csv":
            return "csv"
        elif suffix in {".xlsx", ".xls"}:
            return "excel"
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

    import io
    from pathlib import Path
    from typing import Optional

    import chardet  # if you don't already import this
    import pandas as pd

    # inside class DataIngestor ...

    def _detect_encoding(self) -> Optional[str]:
        """
        Detect the file encoding for CSV sources.

        Strategy
        --------
        - If user set `self.encoding`, honor it.
        - For path-like sources, use chardet on a sample of bytes.
        - Treat 'ascii' as 'utf-8' (ascii is a subset and most Kaggle files
          are actually utf-8).
        - If detection fails, default to 'utf-8'.
        """
        if self.encoding is not None:
            return self.encoding

        # Only attempt detection for path-like sources; for file-like we let
        # pandas decide (or user can pass encoding).
        if isinstance(self.source, (str, Path)):
            path = Path(self.source)

            try:
                with open(path, "rb") as f:
                    raw = f.read(2_000_000)  # up to ~2MB sample
            except OSError:
                # If reading fails, just fall back to utf-8
                self.encoding = "utf-8"
                self.warnings.append(
                    f"Could not read bytes for encoding detection; defaulting to 'utf-8'."
                )
                return self.encoding

            result = chardet.detect(raw)
            enc = result.get("encoding")
            conf = float(result.get("confidence") or 0.0)

            if not enc:
                # No clear guess → default
                self.encoding = "utf-8"
                self.warnings.append(
                    f"Encoding detection inconclusive (confidence={conf:.2f}); "
                    "defaulting to 'utf-8'."
                )
                return self.encoding

            enc = enc.lower()

            # IMPORTANT: chardet often returns 'ascii' for utf-8 files that
            # happen to be mostly 7-bit; treat that as utf-8.
            if enc in ("ascii", "us-ascii"):
                self.encoding = "utf-8"
                self.warnings.append(
                    "Encoding detection returned 'ascii'; using 'utf-8' instead."
                )
                return self.encoding

            # Otherwise accept chardet's guess
            self.encoding = enc
            self.warnings.append(
                f"Detected encoding '{self.encoding}' (confidence={conf:.2f})."
            )
            return self.encoding

        # Non-path-like sources → let pandas infer by default
        return None

    def _detect_delimiter(self):
        """
        Assume the delimiter is a comma, if not fallback on some other options/infer from the file contents.
        """
        if getattr(self, "delimiter", None) is not None:
            return self.delimiter

        default = ","

        if not isinstance(self.source, (str, Path)):
            return default

        try:
            import csv
            with open(self.source, "r", newline="") as f:
                sample = f.read(2048)
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            return default


    def _load_csv(self):
        """
        Load a CSV file into a DataFrame.

        Strategy
        --------
        1. Detect encoding and delimiter.
        2. First attempt: pandas.read_csv with the detected encoding/delimiter.
        3. If we hit a UnicodeDecodeError or ParserError, log a warning and
           retry with safer defaults / alternate encodings.
        """
        encoding = self._detect_encoding()
        delimiter = self._detect_delimiter()

        # ----------------------------------------------------------
        # Helper: attempt to read with given encoding + delimiter
        # ----------------------------------------------------------
        def _read_core(src, enc, delim):
            return pd.read_csv(
                src,
                encoding=enc,
                delimiter=delim,
                nrows=self.max_rows,
            )

        # ----------------------------------------------------------
        # Fallback logic for path-like sources
        # ----------------------------------------------------------
        def _read_from_path():
            # First attempt: our detected encoding + delimiter
            try:
                return _read_core(self.source, encoding, delimiter)
            except UnicodeDecodeError as exc:
                self.warnings.append(
                    f"UnicodeDecodeError with encoding={repr(encoding)}: {exc}. "
                    "Retrying with fallback encodings."
                )
            except ParserError as exc:
                self.warnings.append(
                    f"ParserError with delimiter={repr(delimiter)}: {exc}. "
                    "Retrying with pandas defaults (engine='python', sep inference)."
                )
                # For ParserError we immediately jump to python engine:
                return pd.read_csv(
                    self.source,
                    encoding=encoding,
                    nrows=self.max_rows,
                    engine="python",
                    sep=None,
                    on_bad_lines="warn",
                )

            # If we got here, it was a UnicodeDecodeError – try fallback encodings
            fallback_encodings = ["utf-8", "latin-1"]
            for enc in fallback_encodings:
                if enc == encoding:
                    continue
                try:
                    df = pd.read_csv(
                        self.source,
                        encoding=enc,
                        delimiter=delimiter,
                        nrows=self.max_rows,
                    )
                    self.warnings.append(
                        f"Successfully re-read CSV using fallback encoding '{enc}'."
                    )
                    # Update self.encoding so metadata reports the actual one used
                    self.encoding = enc
                    return df
                except UnicodeDecodeError:
                    continue

            # As a last resort, let pandas' python engine try to figure it out
            self.warnings.append(
                "All fallback encodings failed; using engine='python', "
                "sep=None, on_bad_lines='warn'."
            )
            return pd.read_csv(
                self.source,
                encoding=encoding or "utf-8",
                nrows=self.max_rows,
                engine="python",
                sep=None,
                on_bad_lines="warn",
            )

        # ----------------------------------------------------------
        # Fallback logic for file-like sources
        # ----------------------------------------------------------
        def _read_from_file_obj(file_obj):
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)

            try:
                return _read_core(file_obj, encoding, delimiter)
            except UnicodeDecodeError as exc:
                self.warnings.append(
                    f"UnicodeDecodeError with encoding={repr(encoding)} on file-like "
                    f"source: {exc}. Retrying with fallback encodings."
                )
            except ParserError as exc:
                self.warnings.append(
                    f"ParserError with delimiter={repr(delimiter)} on file-like source: "
                    f"{exc}. Retrying with pandas defaults (engine='python')."
                )
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                return pd.read_csv(
                    file_obj,
                    encoding=encoding,
                    nrows=self.max_rows,
                    engine="python",
                    sep=None,
                    on_bad_lines="warn",
                )

            # UnicodeDecodeError path: try fallbacks
            fallback_encodings = ["utf-8", "latin-1"]
            for enc in fallback_encodings:
                if enc == encoding:
                    continue
                try:
                    if hasattr(file_obj, "seek"):
                        file_obj.seek(0)
                    df = pd.read_csv(
                        file_obj,
                        encoding=enc,
                        delimiter=delimiter,
                        nrows=self.max_rows,
                    )
                    self.warnings.append(
                        f"Successfully re-read CSV using fallback encoding '{enc}' "
                        "for file-like source."
                    )
                    self.encoding = enc
                    return df
                except UnicodeDecodeError:
                    continue

            # Last resort for file-like
            self.warnings.append(
                "All fallback encodings failed on file-like source; using "
                "engine='python', sep=None, on_bad_lines='warn'."
            )
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            return pd.read_csv(
                file_obj,
                encoding=encoding or "utf-8",
                nrows=self.max_rows,
                engine="python",
                sep=None,
                on_bad_lines="warn",
            )

        # ----------------------------------------------------------
        # Dispatch based on source type
        # ----------------------------------------------------------
        if isinstance(self.source, (str, Path)):
            return _read_from_path()

        elif hasattr(self.source, "read"):
            return _read_from_file_obj(self.source)

        else:
            raise ValueError("Unsupported source type for CSV loading.")

    def _load_excel(self):
        """
        Use pandas read_excel, set a minimum or maximum, then return the loaded dataframe
        """

        if isinstance(self.source, (str, Path)):
            df = pd.read_excel(
                self.source,
                nrows=self.max_rows
            )
            return df
        elif hasattr(self.source, 'read'):
            file_obj = self.source

            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)

            df = pd.read_excel(
                file_obj,
                nrows=self.max_rows
            )
            return df

        else:
            raise ValueError("Unsupported source type for Excel loading.")

    def _basic_validate(self) -> None:
        """
        Run minimal sanity checks on self.df and populate self.warnings.

        - Raise if df is None or completely empty (catastrophic).
        - Warn if very few rows (< 10).
        - Warn if there are duplicate column names.
        - Warn if there are no numeric or datetime-like columns at all.
        """
        if self.df is None:
            raise RuntimeError("No DataFrame loaded. Call load() before _basic_validate().")

        # Catastrophic: empty df
        if self.df.empty:
            raise ValueError("Loaded DataFrame is empty; cannot proceed with modeling.")

        # Very few rows
        if len(self.df) < 10:
            self.warnings.append("Very few rows detected (< 10); models may be unreliable.")

        # Duplicate column names
        if self.df.columns.duplicated().any():
            self.warnings.append("Duplicate column names detected in DataFrame.")

        # Check whether there is at least one numeric or datetime-like column
        has_numeric = any(pd.api.types.is_numeric_dtype(dtype) for dtype in self.df.dtypes)
        has_datetime = any(pd.api.types.is_datetime64_any_dtype(dtype) for dtype in self.df.dtypes)

        if not has_numeric and not has_datetime:
            self.warnings.append(
                "No numeric or datetime-like columns detected; "
                "this may not be suitable for time series modeling."
            )
        # DatetimeIndex check (important for time-series pipelines)
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.warnings.append(
                "DataFrame index is a DatetimeIndex; downstream schema detection "
                "may treat the index as the primary date axis."
            )
