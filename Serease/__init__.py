# Serease/__init__.py

# Expose ingestion layer
from .ingestion import DataIngestor, IngestionMetadata

# Expose schema layer
from .schema import SchemaDetector, SchemaMetadata

__all__ = [
    "DataIngestor",
    "IngestionMetadata",
    "SchemaDetector",
    "SchemaMetadata",
]