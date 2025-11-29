import pandas as pd
from Serease import SchemaDetector


def test_schema_detector_basic():
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=5, freq="D"),
        "sales": [10, 20, 15, 30, 25],
        "temp": [70, 72, 68, 75, 73],
    })

    sd = SchemaDetector(df)
    meta = sd.detect()

    assert meta.date_col == "date"
    assert meta.target_col == "sales"
    assert "temp" in meta.exog_cols


def test_respects_user_target_override():
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=5, freq="D"),
        "weird_name": [1, 2, 3, 4, 5],
        "other": [10, 11, 12, 13, 14],
    })

    sd = SchemaDetector(df, user_target_col="weird_name")
    meta = sd.detect()

    assert meta.target_col == "weird_name"
    assert "other" in meta.exog_cols


def test_ignores_id_like_column():
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=5, freq="D"),
        "row_id": [1, 2, 3, 4, 5],        # should be treated as ID
        "demand": [100, 120, 110, 130, 125],
    })

    sd = SchemaDetector(df)
    meta = sd.detect()

    assert meta.date_col == "date"
    assert meta.target_col == "demand"
    assert "row_id" not in meta.exog_cols  # ID shouldn't be used


def test_no_date_column_produces_note():
    df = pd.DataFrame({
        "sales": [10, 20, 15, 30, 25],
        "temp": [70, 72, 68, 75, 73],
    })

    sd = SchemaDetector(df)
    meta = sd.detect()

    assert meta.date_col is None
    assert any("No date column detected" in n for n in meta.notes)
