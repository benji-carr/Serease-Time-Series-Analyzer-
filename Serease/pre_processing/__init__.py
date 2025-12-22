"""
Serease pre-processing layer.

Contains ingestion, schema detection, and time-series cleaning primitives.
"""

from .data_ingestor import DataIngestor
from .schema_detector import SchemaDetector
from .time_series_cleaner import TimeSeriesCleaner

__all__ = ["DataIngestor", "SchemaDetector", "TimeSeriesCleaner"]
