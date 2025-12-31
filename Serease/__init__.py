# Serease/__init__.py

# -------------------------
# Pre-processing layer
# -------------------------
from .pre_processing.data_ingestor import DataIngestor, IngestionMetadata
from .pre_processing.schema_detector import SchemaDetector, SchemaMetadata
from .pre_processing.time_series_cleaner import TimeSeriesCleaner

# -------------------------
# Pre-modeling layer
# -------------------------
from .pre_modeling.orchestrator import PreModelOrchestrator
from .pre_modeling.containers import (
    SeriesBundle,
    PreModelState,
    TransformPlan,
)

# Optional: expose stationarity selection utilities
from .pre_modeling.selection import (
    StationarityPolicy,
    select_stationary_candidate,
)

__all__ = [
    # pre_processing
    "DataIngestor",
    "IngestionMetadata",
    "SchemaDetector",
    "SchemaMetadata",
    "TimeSeriesCleaner",
    # pre_modeling
    "PreModelOrchestrator",
    "SeriesBundle",
    "PreModelState",
    "TransformPlan",
    # selection
    "StationarityPolicy",
    "select_stationary_candidate",
]
