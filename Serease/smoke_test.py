# Serease/dev/smoke_test_mvp.py
import argparse
from pathlib import Path

from Serease.ingestion import DataIngestor
from Serease.schema import SchemaDetector
from Serease.ingestion.time_series_cleaner import TimeSeriesCleaner
from Serease.preprocessing.transformer import TimeSeriesTransformer
from Serease.diagnostics.diagnostics_engine import DiagnosticsEngine, DiagnosticsConfig
from Serease.reporting.diagnostics_reporter import DiagnosticsReporter, ReporterConfig


def _make_ingestor(csv_path: str, mode: str) -> DataIngestor:
    p = Path(csv_path)

    if mode == "path":
        # Option 1: force CSV even if the saved upload path has no .csv extension
        return DataIngestor(str(p), file_type="csv")

    if mode == "stream":
        f = open(p, "rb")
        return DataIngestor(f, filename=p.name, file_type="csv")

    if mode == "bytes":
        raw = p.read_bytes()
        return DataIngestor(raw, filename=p.name, file_type="csv")

    raise ValueError(f"Unknown mode: {mode}")


def run(
    csv_path: str,
    target_col: str,
    seasonal_period: int = 7,
    dataset_name: str = "smoke_test",
    mode: str = "path",
):
    ingestor = _make_ingestor(csv_path, mode)

    try:
        df_raw = ingestor.load()
        ing_meta = ingestor.get_metadata()
    finally:
        # If we created an open stream, close it
        src = getattr(ingestor, "source", None)
        if mode == "stream" and hasattr(src, "close"):
            try:
                src.close()
            except Exception:
                pass

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

    # Web/CLI-friendly output
    html = reporter.render_html()  # ensure this exists

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

    # NEW: choose how ingestion happens
    ap.add_argument(
        "--mode",
        choices=["path", "stream", "bytes"],
        default="path",
        help="Ingestion mode. Use 'stream' to mimic web upload streams without temp files."
    )

    args = ap.parse_args()
    run(args.csv, args.target, args.m, args.name, mode=args.mode)
