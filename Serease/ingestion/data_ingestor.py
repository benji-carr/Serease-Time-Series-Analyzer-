from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Union, BinaryIO

import io
import chardet
import pandas as pd
from pandas.errors import ParserError


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
SourceT = Union[str, Path, bytes, bytearray, BinaryIO]


@dataclass
class IngestionMetadata:
    """
    Metadata describing the result of a data ingestion operation.

    This dataclass is produced by ``DataIngestor`` and captures basic
    structural information about the loaded dataset along with ingestion-
    related settings such as inferred file type, encoding, and delimiter.
    Other components of the Serease pipeline use this metadata to perform
    validation, schema detection, and diagnostics.
    """
    n_rows: int
    n_cols: int
    column_names: List[str]
    dtypes: Dict[str, str]
    file_type: str
    encoding: Optional[str]
    delimiter: Optional[str]
    warnings: List[str]
    index_is_datetime: bool = False


class DataIngestor:
    """
    Class responsible for reading user-uploaded CSV/Excel data
    and producing a raw DataFrame with basic metadata.

    Supports:
    - Path sources (csv/excel)
    - File-like sources (streams) via .read() (e.g., FastAPI UploadFile.file)
    - Raw bytes/bytearray

    Important:
    - For stream sources, pass filename=... (or file_type=...) so file type
      detection works reliably.
    """

    def __init__(
        self,
        source: SourceT,
        file_type: Optional[str] = None,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_rows: Optional[int] = None,
        filename: Optional[str] = None,  # NEW: used for stream sources
    ) -> None:
        self.source: Union[Path, bytes, bytearray, BinaryIO]
        if isinstance(source, (str, Path)):
            self.source = Path(source)
        else:
            self.source = source

        self.filename = filename  # NEW
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
        if self.file_type is None:
            self.file_type = self._detect_file_type()

        if self.file_type == "csv":
            df = self._load_csv()
        elif self.file_type == "excel":
            df = self._load_excel()
        else:
            raise ValueError(f"Unsupported file type in load(): {self.file_type}")

        self.df = df
        self._basic_validate()
        return df

    def preview(self, n: int = 5) -> pd.DataFrame:
        if self.df is None:
            self.load()
        return self.df.head(n)

    def get_metadata(self) -> IngestionMetadata:
        if self.df is None:
            self.load()

        idx = self.df.index
        index_is_datetime = isinstance(idx, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(idx)

        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

        return IngestionMetadata(
            n_rows=len(self.df),
            n_cols=self.df.shape[1],
            column_names=list(self.df.columns),
            dtypes=dtypes,
            file_type=self.file_type or "unknown",
            encoding=self.encoding,
            delimiter=self.delimiter,
            warnings=self.warnings.copy(),
            index_is_datetime=index_is_datetime,
        )

    def validate(self) -> List[str]:
        if self.df is None:
            self.load()
        self._basic_validate()
        return self.warnings.copy()

    # ----------------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------------

    def _rewind(self, obj) -> None:
        if hasattr(obj, "seek"):
            try:
                obj.seek(0)
            except Exception:
                pass

    def _read_sample_bytes(self, n: int = 2_000_000) -> bytes:
        """
        Read a sample of bytes from the source without consuming it permanently
        (rewinds file-like sources where possible).
        """
        # Path-like
        if isinstance(self.source, Path):
            with open(self.source, "rb") as f:
                return f.read(n)

        # Raw bytes
        if isinstance(self.source, (bytes, bytearray)):
            return bytes(self.source[:n])

        # File-like
        if hasattr(self.source, "read"):
            self._rewind(self.source)
            chunk = self.source.read(n)
            self._rewind(self.source)
            if isinstance(chunk, (bytes, bytearray)):
                return bytes(chunk)
            # Extremely defensive fallback: treat as text
            return str(chunk).encode("utf-8", errors="ignore")

        raise ValueError("Unsupported source type for sampling bytes.")

    def _detect_file_type(self) -> str:
        """
        Use extension to infer file type.
        - Path sources: use Path suffix
        - Stream sources: use filename hint (self.filename) OR .filename attr if present
        """
        suffix: Optional[str] = None

        if isinstance(self.source, Path):
            suffix = self.source.suffix.lower()
        elif self.filename:
            suffix = Path(self.filename).suffix.lower()
        elif hasattr(self.source, "filename"):
            # supports some wrappers that carry filename
            suffix = Path(getattr(self.source, "filename")).suffix.lower()

        if not suffix:
            raise ValueError(
                "Cannot detect file type from this source. "
                "Provide file_type=... or filename=... for stream sources."
            )

        if suffix == ".csv":
            return "csv"
        if suffix in {".xlsx", ".xls"}:
            return "excel"
        raise ValueError(f"Unsupported file extension: {suffix}")

    def _detect_encoding(self) -> Optional[str]:
        """
        Detect encoding from a sample of bytes for BOTH path and stream sources.
        - Honors user-provided encoding
        - Treats ascii as utf-8
        - Defaults to utf-8 if inconclusive
        """
        if self.encoding is not None:
            return self.encoding

        try:
            raw = self._read_sample_bytes()
        except Exception:
            self.encoding = "utf-8"
            self.warnings.append("Could not sample bytes for encoding detection; defaulting to 'utf-8'.")
            return self.encoding

        result = chardet.detect(raw)
        enc = (result.get("encoding") or "").lower()
        conf = float(result.get("confidence") or 0.0)

        if not enc:
            self.encoding = "utf-8"
            self.warnings.append(
                f"Encoding detection inconclusive (confidence={conf:.2f}); defaulting to 'utf-8'."
            )
            return self.encoding

        if enc in ("ascii", "us-ascii"):
            self.encoding = "utf-8"
            self.warnings.append("Encoding detection returned 'ascii'; using 'utf-8' instead.")
            return self.encoding

        self.encoding = enc
        self.warnings.append(f"Detected encoding '{self.encoding}' (confidence={conf:.2f}).")
        return self.encoding

    def _detect_delimiter(self) -> str:
        """
        Detect delimiter for CSV sources using csv.Sniffer on a small text sample.
        Works for both paths and streams.
        """
        if self.delimiter is not None:
            return self.delimiter

        default = ","
        try:
            raw = self._read_sample_bytes(n=4096)
            enc = self.encoding or "utf-8"
            text = raw.decode(enc, errors="replace")

            import csv
            dialect = csv.Sniffer().sniff(text)
            self.delimiter = dialect.delimiter
            return self.delimiter
        except Exception:
            self.delimiter = default
            return self.delimiter

    def _load_csv(self) -> pd.DataFrame:
        """
        Load CSV from path OR stream/bytes without writing temp files.

        Strategy
        --------
        1. Detect encoding and delimiter.
        2. Attempt read with detected settings.
        3. If UnicodeDecodeError or ParserError -> retry with fallbacks.
        """
        encoding = self._detect_encoding() or "utf-8"
        delimiter = self._detect_delimiter()

        def _read_core(src, enc, delim):
            return pd.read_csv(
                src,
                encoding=enc,
                delimiter=delim,
                nrows=self.max_rows,
            )

        # -------------------------
        # Path-like
        # -------------------------
        if isinstance(self.source, Path):
            try:
                return _read_core(self.source, encoding, delimiter)
            except UnicodeDecodeError as exc:
                self.warnings.append(
                    f"UnicodeDecodeError with encoding={repr(encoding)}: {exc}. Retrying with fallback encodings."
                )
            except ParserError as exc:
                self.warnings.append(
                    f"ParserError with delimiter={repr(delimiter)}: {exc}. "
                    "Retrying with pandas defaults (engine='python', sep inference)."
                )
                return pd.read_csv(
                    self.source,
                    encoding=encoding,
                    nrows=self.max_rows,
                    engine="python",
                    sep=None,
                    on_bad_lines="warn",
                )

            # UnicodeDecodeError fallback encodings
            for enc in ["utf-8", "latin-1"]:
                if enc == encoding:
                    continue
                try:
                    df = pd.read_csv(
                        self.source,
                        encoding=enc,
                        delimiter=delimiter,
                        nrows=self.max_rows,
                    )
                    self.warnings.append(f"Successfully re-read CSV using fallback encoding '{enc}'.")
                    self.encoding = enc
                    return df
                except UnicodeDecodeError:
                    continue

            self.warnings.append(
                "All fallback encodings failed; using engine='python', sep=None, on_bad_lines='warn'."
            )
            return pd.read_csv(
                self.source,
                encoding=encoding,
                nrows=self.max_rows,
                engine="python",
                sep=None,
                on_bad_lines="warn",
            )

        # -------------------------
        # Bytes / bytearray
        # -------------------------
        if isinstance(self.source, (bytes, bytearray)):
            raw = bytes(self.source)
            bio = io.BytesIO(raw)
            text = io.TextIOWrapper(bio, encoding=encoding, errors="replace", newline="")
            try:
                return pd.read_csv(text, delimiter=delimiter, nrows=self.max_rows)
            except ParserError as exc:
                self.warnings.append(
                    f"ParserError on bytes source with delimiter={repr(delimiter)}: {exc}. "
                    "Retrying with python engine (sep inference)."
                )
                bio2 = io.BytesIO(raw)
                text2 = io.TextIOWrapper(bio2, encoding=encoding, errors="replace", newline="")
                return pd.read_csv(
                    text2,
                    nrows=self.max_rows,
                    engine="python",
                    sep=None,
                    on_bad_lines="warn",
                )

        # -------------------------
        # File-like stream
        # -------------------------
        if hasattr(self.source, "read"):
            # Read the stream once into memory (MVP-safe; add file-size limits at API boundary)
            self._rewind(self.source)
            raw = self.source.read()
            if not isinstance(raw, (bytes, bytearray)):
                raw = str(raw).encode("utf-8", errors="ignore")
            raw = bytes(raw)

            bio = io.BytesIO(raw)
            text = io.TextIOWrapper(bio, encoding=encoding, errors="replace", newline="")

            try:
                return pd.read_csv(text, delimiter=delimiter, nrows=self.max_rows)
            except UnicodeDecodeError as exc:
                self.warnings.append(
                    f"UnicodeDecodeError on stream with encoding={repr(encoding)}: {exc}. Retrying with fallback encodings."
                )
            except ParserError as exc:
                self.warnings.append(
                    f"ParserError on stream with delimiter={repr(delimiter)}: {exc}. "
                    "Retrying with python engine (sep inference)."
                )
                bio2 = io.BytesIO(raw)
                text2 = io.TextIOWrapper(bio2, encoding=encoding, errors="replace", newline="")
                return pd.read_csv(
                    text2,
                    nrows=self.max_rows,
                    engine="python",
                    sep=None,
                    on_bad_lines="warn",
                )

            # UnicodeDecodeError fallbacks for stream bytes
            for enc in ["utf-8", "latin-1"]:
                if enc == encoding:
                    continue
                try:
                    bio3 = io.BytesIO(raw)
                    text3 = io.TextIOWrapper(bio3, encoding=enc, errors="replace", newline="")
                    df = pd.read_csv(text3, delimiter=delimiter, nrows=self.max_rows)
                    self.warnings.append(f"Successfully re-read stream CSV using fallback encoding '{enc}'.")
                    self.encoding = enc
                    return df
                except UnicodeDecodeError:
                    continue

            self.warnings.append(
                "All fallback encodings failed on stream; using engine='python', sep=None, on_bad_lines='warn'."
            )
            bio4 = io.BytesIO(raw)
            text4 = io.TextIOWrapper(bio4, encoding=encoding, errors="replace", newline="")
            return pd.read_csv(
                text4,
                nrows=self.max_rows,
                engine="python",
                sep=None,
                on_bad_lines="warn",
            )

        raise ValueError("Unsupported source type for CSV loading.")

    def _load_excel(self) -> pd.DataFrame:
        """
        Load Excel from path OR stream/bytes without writing temp files.
        """
        if isinstance(self.source, Path):
            return pd.read_excel(self.source, nrows=self.max_rows)

        if isinstance(self.source, (bytes, bytearray)):
            return pd.read_excel(io.BytesIO(bytes(self.source)), nrows=self.max_rows)

        if hasattr(self.source, "read"):
            self._rewind(self.source)
            raw = self.source.read()
            if not isinstance(raw, (bytes, bytearray)):
                raw = str(raw).encode("utf-8", errors="ignore")
            return pd.read_excel(io.BytesIO(bytes(raw)), nrows=self.max_rows)

        raise ValueError("Unsupported source type for Excel loading.")

    def _basic_validate(self) -> None:
        """
        Run minimal sanity checks on self.df and populate self.warnings.
        """
        if self.df is None:
            raise RuntimeError("No DataFrame loaded. Call load() before _basic_validate().")

        if self.df.empty:
            raise ValueError("Loaded DataFrame is empty; cannot proceed with modeling.")

        if len(self.df) < 10:
            self.warnings.append("Very few rows detected (< 10); models may be unreliable.")

        if self.df.columns.duplicated().any():
            self.warnings.append("Duplicate column names detected in DataFrame.")

        has_numeric = any(pd.api.types.is_numeric_dtype(dtype) for dtype in self.df.dtypes)
        has_datetime = any(pd.api.types.is_datetime64_any_dtype(dtype) for dtype in self.df.dtypes)

        if not has_numeric and not has_datetime:
            self.warnings.append(
                "No numeric or datetime-like columns detected; this may not be suitable for time series modeling."
            )

        if isinstance(self.df.index, pd.DatetimeIndex):
            self.warnings.append(
                "DataFrame index is a DatetimeIndex; downstream schema detection may treat the index as the primary date axis."
            )
