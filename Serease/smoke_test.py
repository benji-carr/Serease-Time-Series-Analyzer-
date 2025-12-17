# Serease/dev/smoke_test_mvp.py
import argparse
from pathlib import Path

from Serease.ingestion import DataIngestor
from Serease.schema import SchemaDetector
from Serease.ingestion.time_series_cleaner import TimeSeriesCleaner
from Serease.preprocessing.transformer import TimeSeriesTransformer
from Serease.diagnostics.diagnostics_engine import DiagnosticsEngine, DiagnosticsConfig
from Serease.reporting.diagnostics_reporter import DiagnosticsReporter, ReporterConfig


def run(csv_path: str, target_col: str, seasonal_period: int = 7, dataset_name: str = "smoke_test"):
    ingestor = DataIngestor(csv_path)
    df_raw = ingestor.load()
    ing_meta = ingestor.get_metadata()

    schema_detector = SchemaDetector(df=df_raw, ingestion_meta=ing_meta, user_target_col=target_col)
    schema_meta = schema_detector.detect()

    cleaner = TimeSeriesCleaner(df=df_raw, schema=schema_meta)
    cleaned_df, ts_meta = cleaner.clean()

    transformer = TimeSeriesTransformer(
        target_col=schema_meta.target_col,
        freq=ts_meta.freq,
        seasonal_period=seasonal_period
    )
    bundle = transformer.fit(cleaned_df)

    engine = DiagnosticsEngine(DiagnosticsConfig(dataset_name=dataset_name))
    report = engine.run(
        cleaned_df=cleaned_df,
        schema_meta=schema_meta,
        ts_meta=ts_meta,
        transform_bundle=bundle,
        df_raw=df_raw,
        ingestion_meta=ing_meta,
    )

    reporter = DiagnosticsReporter(
        report=report,
        transform_bundle=bundle,
        cleaned_df=cleaned_df,
        schema_meta=schema_meta,
        ts_meta=ts_meta,
        config=ReporterConfig(),
    )

    # You want a render method for web/CLI usage:
    html = reporter.render_html()  # implement if not present
    out_path = Path("artifacts") / f"{dataset_name}_diagnostics.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"[OK] Wrote report: {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--m", type=int, default=7)
    ap.add_argument("--name", default="smoke_test")
    args = ap.parse_args()

    run(args.csv, args.target, args.m, args.name)
