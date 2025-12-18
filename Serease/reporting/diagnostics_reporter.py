from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import html
import pathlib

import pandas as pd

from Serease.diagnostics.report_types import DiagnosticsReport, StepResult
from .plot_utils import (
    fig_to_base64,
    periodogram_plot,
    stationarity_scatter_plot,
    acf_plot_from_payload,
    pacf_plot_from_payload,
    stl_components_plot,
)


@dataclass
class ReporterConfig:
    """
    Controls which optional sections/tables are rendered.

    MVP defaults keep the report readable and avoid giant tables.
    """
    show_period_table: bool = False
    show_acf_pacf_table: bool = False
    show_stationarity_tables: bool = False

    max_table_rows: int = 50


from __future__ import annotations

import html
import pathlib
from typing import Any, Optional

import pandas as pd

from Serease.diagnostics.report_types import DiagnosticsReport, StepResult
from Serease.reporting.plot_utils import (
    fig_to_base64,
    acf_plot_from_payload,
    pacf_plot_from_payload,
    stl_components_plot,
    stationarity_scatter_plot,
    periodogram_plot,
)

# ReporterConfig should already exist in this module in your project.
# If it's defined elsewhere, adjust the import accordingly.
# from Serease.reporting.diagnostics_reporter import ReporterConfig  # (avoid circular)
# Ensure ReporterConfig is available in this file.


class DiagnosticsReporter:
    """
    Renders a DiagnosticsReport into notebook-viewable HTML (and exportable HTML file).

    Contract:
      - Reporter never computes diagnostics; it only renders StepResults and Artifacts.
      - Reporter must tolerate missing steps/artifacts (render placeholders, not crashes).
    """

    def __init__(
        self,
        report: DiagnosticsReport,
        transform_bundle: Optional[Any] = None,
        cleaned_df: Optional[pd.DataFrame] = None,
        schema_meta: Optional[Any] = None,
        ts_meta: Optional[Any] = None,
        config: Optional["ReporterConfig"] = None,
    ) -> None:
        self.report = report
        self.transform_bundle = transform_bundle
        self.cleaned_df = cleaned_df
        self.schema_meta = schema_meta
        self.ts_meta = ts_meta
        self.config = config or ReporterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render_html(self) -> str:
        """
        Web/CLI-friendly renderer.

        This is the method your smoke test expects.
        We keep `to_html_string()` as the underlying implementation so you don't
        break existing notebook usage.
        """
        return self.to_html_string()

    def show_in_notebook(self) -> None:
        """
        Display the report HTML in a Jupyter notebook cell.

        This method requires IPython.display, but it is only imported inside
        the function to keep the module import-safe in non-notebook environments.
        """
        from IPython.display import HTML as IPyHTML, display  # type: ignore

        display(IPyHTML(self.to_html_string()))

    def to_html(self, path: str) -> str:
        """
        Write the HTML report to disk and return the resolved path.
        """
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        html_text = self.to_html_string()
        p.write_text(html_text, encoding="utf-8")
        return str(p.resolve())

    def to_html_string(self) -> str:
        """
        Return the full HTML document as a string.

        Contract:
          - Always renders all stable step sections in a fixed order.
          - Must not crash if steps or artifacts are missing.
        """
        parts: list[str] = []
        parts.append(self._render_header())

        for step_name in [
            "overview",
            "missingness",
            "seasonality_assessment",
            "variant_selection",
            "stationarity_sweep",
            "acf_pacf",
            "stl",
            "exog",
        ]:
            parts.append(self._render_step(step_name))

        return self._wrap_html("\n".join(parts))

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_header(self) -> str:
        """
        Render the report title and lightweight metadata.
        """
        title = html.escape(f"Serease Diagnostics Report — {self.report.dataset_name}")

        meta_items: list[str] = []
        for k, v in (self.report.meta or {}).items():
            meta_items.append(
                f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
            )
        meta_html = (
            "<ul>" + "\n".join(meta_items) + "</ul>"
            if meta_items
            else "<p><i>(no meta)</i></p>"
        )

        return f"<h1>{title}</h1>\n{meta_html}\n<hr/>"

    def _render_step(self, step_name: str) -> str:
        """
        Render one step section.
        """
        step = self.report.get(step_name)
        if step is None:
            return f"<h2>{html.escape(step_name)}</h2><p><i>(step not present)</i></p><hr/>"

        parts: list[str] = [f"<h2>{html.escape(step.step_name)}</h2>"]
        parts.append(self._render_summary(step))
        parts.append(self._render_notes(step))
        parts.append(self._render_warnings(step))
        parts.append(self._render_artifacts(step))
        parts.append("<hr/>")
        return "\n".join([p for p in parts if p])

    def _render_summary(self, step: StepResult) -> str:
        if not step.summary:
            return "<p><i>(no summary)</i></p>"

        items = []
        for k, v in step.summary.items():
            items.append(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>")
        return "<ul>" + "\n".join(items) + "</ul>"

    def _render_notes(self, step: StepResult) -> str:
        if not step.notes:
            return ""
        items = "\n".join(f"<li>{html.escape(str(n))}</li>" for n in step.notes)
        return f"<details open><summary><b>Notes</b></summary><ul>{items}</ul></details>"

    def _render_warnings(self, step: StepResult) -> str:
        if not step.warnings:
            return ""
        items = "\n".join(f"<li>{html.escape(str(w))}</li>" for w in step.warnings)
        return f"<details open><summary><b>Warnings</b></summary><ul>{items}</ul></details>"

    def _render_artifacts(self, step: StepResult) -> str:
        if not step.artifacts:
            return "<p><i>(no artifacts)</i></p>"

        out: list[str] = []
        for a in step.artifacts:
            out.append(f"<h3>Artifact: {html.escape(a.name)}</h3>")
            out.append(self._render_artifact_payload(step.step_name, a.name, a.payload))
        return "\n".join(out)

    def _render_artifact_payload(self, step_name: str, artifact_name: str, payload: Any) -> str:
        """
        Render known artifacts with stable visualizations.
        """
        # missingness
        if step_name == "missingness" and artifact_name == "missing_blocks":
            return self._render_table(payload, title="Missing blocks")

        # seasonality assessment
        if step_name == "seasonality_assessment" and artifact_name == "period_candidates":
            fig = periodogram_plot(payload or [])
            fig_html = self._img(fig)
            tbl_html = self._render_table(payload, title="Period candidates") if self.config.show_period_table else ""
            return fig_html + tbl_html

        # stationarity sweep
        if step_name == "stationarity_sweep" and artifact_name == "stationarity_table":
            rows = payload or []
            fig = stationarity_scatter_plot(rows)
            fig_html = self._img(fig)

            tbl_html = self._render_table(rows, title="Stationarity sweep") if self.config.show_stationarity_tables else ""

            hint = (
                "<p><i>Interpretation: prefer variants with low ADF p-values and high KPSS p-values. "
                "Disagreement is labeled ambiguous.</i></p>"
            )
            return hint + fig_html + tbl_html

        # ACF / PACF
        if step_name == "acf_pacf" and artifact_name == "acf_pacf_payload":
            p = payload or {}
            fig_acf = acf_plot_from_payload(p, title="ACF (stem)")
            fig_pacf = pacf_plot_from_payload(p, title="PACF (stem)")
            imgs = self._img(fig_acf) + self._img(fig_pacf)

            tbl_html = self._render_table([p], title="ACF/PACF payload") if self.config.show_acf_pacf_table else ""
            return imgs + tbl_html

        # STL
        if step_name == "stl" and artifact_name == "stl_components":
            fig = stl_components_plot(payload or {})
            return self._img(fig)

        # Variant selection
        if step_name == "variant_selection" and artifact_name == "selection_ranked":
            return self._render_table(payload, title="Variant ranking")

        # Fallback
        return f"<p><i>(unrendered payload type: {html.escape(type(payload).__name__)})</i></p>"

    def _render_table(self, rows: Any, title: str = "Table") -> str:
        if not isinstance(rows, list) or len(rows) == 0:
            return f"<p><i>({html.escape(title)} not available)</i></p>"

        rows = rows[: self.config.max_table_rows]

        cols: list[str] = []
        for r in rows:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in cols:
                        cols.append(str(k))

        if not cols:
            return f"<p><i>({html.escape(title)} has no columns)</i></p>"

        header = "".join(f"<th>{html.escape(c)}</th>" for c in cols)

        body_rows: list[str] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            tds = "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols)
            body_rows.append(f"<tr>{tds}</tr>")

        return (
            f"<h4>{html.escape(title)}</h4>"
            f"<div style='overflow-x:auto; max-width: 100%;'>"
            f"<table class='sr-table'>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            f"</table></div>"
        )

    def _img(self, fig) -> str:
        b64 = fig_to_base64(fig)
        return f"<img src='data:image/png;base64,{b64}' style='max-width:100%; height:auto;'/>"

    def _wrap_html(self, body: str) -> str:
        css = """
        body { font-family: Arial, sans-serif; line-height: 1.35; padding: 12px; }
        h1 { margin: 0 0 8px 0; }
        h2 { margin: 18px 0 6px 0; }
        h3 { margin: 12px 0 6px 0; }
        h4 { margin: 8px 0 6px 0; }
        hr { margin: 18px 0; }

        details { margin: 8px 0; }

        .sr-table { border-collapse: collapse; width: 100%; }
        .sr-table th, .sr-table td { border: 1px solid #ccc; padding: 6px 8px; font-size: 12px; }
        .sr-table th { background: #eee; white-space: nowrap; }
        .sr-table td { vertical-align: top; }
        """
        return (
            "<!DOCTYPE html>"
            "<html><head><meta charset='utf-8'>"
            f"<style>{css}</style>"
            "</head><body>"
            f"{body}"
            "</body></html>"
        )
