# SchemaDetector

## Purpose

`SchemaDetector` is the second stage of the Serease pipeline.  
Given a raw `pandas.DataFrame` from `DataIngestor`, it:

- Identifies the **date column**
- Identifies the **target (y) column**
- Chooses **exogenous feature columns (X)**
- Splits all columns into **numeric** vs **categorical**
- Emits **notes/warnings** about potential problems

It returns a `SchemaMetadata` dataclass with this information.

---

## Public API

### SchemaMetadata

```python
@dataclass
class SchemaMetadata:
    date_col: Optional[str]
    target_col: Optional[str]
    exog_cols: List[str]
    all_numeric: List[str]
    all_categorical: List[str]
    notes: List[str]

sd = SchemaDetector(
    df,
    ingestion_meta=ingestion_meta,        # optional
    user_date_col=None,                   # optional override
    user_target_col=None,                 # optional override
    user_exog_cols=None,                  # optional override
)

meta = sd.detect()


