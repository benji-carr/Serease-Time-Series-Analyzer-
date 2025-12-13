from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from IPython.display import HTML, display

from .plot_utils import (
    PlotTheme,
    apply_theme,
    fig_to_base64,
    lineplot_series,
    multi_lineplot,
    histogram_kde,
    periodogram_plot,
    stationarity_scatter_plot,
    acf_plot_from_payload,
    pacf_plot_from_payload,
    stl_components_plot,
)


@dataclass
class ReporterConfig:
    theme: PlotTheme = field(default_factory=PlotTheme)
    dpi: int = 150
    max_table_rows: int = 40
    show_period_table: bool = False
    show_acf_pacf_table: bool = False
    show_stationarity_tables: bool = False


class DiagnosticsReporter:
    def __init__(
        self,
        diagnostics_report: Any,
        transform_bundle: Any,
        cleaned_df: Optional[pd.DataFrame] = None,
        schema_meta: Optional[Any] = None,
        ts_meta: Optional[Any] = None,
        config: Optional[ReporterConfig] = None,
        title: str = "Serease Diagnostics Report",
    ) -> None:
        self.report = diagnostics_report
        self.bundle = transform_bundle
        self.cleaned_df = cleaned_df
        self.schema = schema_meta
        self.ts_meta = ts_meta
        self.config = config or ReporterConfig()
        self.title = title

        apply_theme(self.config.theme)

    def to_html(self, path: str) -> str:
        html = self._build_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def show_in_notebook(self) -> None:
        html = self._build_html()
        display(HTML(html))

    def _build_html(self) -> str:
        sections: List[str] = []
        sections.append(self._html_header())
        sections.append(self._section_overview())
        sections.append(self._section_seasonality())
        sections.append(self._section_stationarity())
        sections.append(self._section_acf_pacf())
        sections.append(self._section_stl())
        sections.append(self._html_footer())
        return "\n".join(sections)

    def _get_step(self, step: str) -> Optional[Any]:
        if self.report is None:
            return None
        if hasattr(self.report, "get"):
            try:
                return self.report.get(step)
            except Exception:
                return None
        if hasattr(self.report, "results") and isinstance(self.report.results, dict):
            return self.report.results.get(step)
        return None

    def _artifact_payload(self, step: str, artifact_name: str) -> Optional[Any]:
        r = self._get_step(step)
        if r is None:
            return None
        arts = getattr(r, "artifacts", None)
        if not arts:
            return None
        for a in arts:
            if getattr(a, "name", None) == artifact_name:
                return getattr(a, "payload", None)
        return None

    def _summary(self, step: str) -> Dict[str, Any]:
        r = self._get_step(step)
        if r is None:
            return {}
        return getattr(r, "summary", {}) or {}

    def _table_html(self, obj: Any, max_rows: Optional[int] = None) -> str:
        if obj is None:
            return "<div class='small'>No table available.</div>"

        mr = self.config.max_table_rows if max_rows is None else int(max_rows)

        if isinstance(obj, pd.Series):
            df = obj.to_frame("value")
        elif isinstance(obj, pd.DataFrame):
            df = obj
        else:
            return "<div class='small'>Unsupported table payload.</div>"

        if len(df) > mr:
            df_show = df.head(mr).copy()
            extra = f"<div class='small'>Showing first {mr} rows of {len(df)}.</div>"
        else:
            df_show = df
            extra = ""

        return extra + df_show.to_html(index=True, escape=False)

    def _img_html(self, fig_b64: str, caption: str = "") -> str:
        cap = f"<div class='small'>{caption}</div>" if caption else ""
        return f"<div class='card'><img src='data:image/png;base64,{fig_b64}'/>{cap}</div>"

    def _meta_line(self) -> str:
        bits = []
        if self.schema is not None:
            bits.append(f"<span class='pill'>target: {getattr(self.schema, 'target_col', None)}</span>")
            bits.append(f"<span class='pill'>date: {getattr(self.schema, 'date_col', None)}</span>")
        if self.ts_meta is not None:
            bits.append(f"<span class='pill'>freq: {getattr(self.ts_meta, 'freq', None)}</span>")
            bits.append(f"<span class='pill'>n_obs: {getattr(self.ts_meta, 'n_obs', None)}</span>")
        return " ".join(bits)

    def _html_header(self) -> str:
        return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{self.title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; }}
  h1 {{ margin-bottom: 6px; }}
  h2 {{ margin-top: 28px; border-bottom: 1px solid #e5e5e5; padding-bottom: 6px; }}
  .meta {{ color: #444; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
  .card {{ border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px; background: #fff; }}
  .small {{ font-size: 13px; color: #444; }}
  .pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; background: #f3f4f6; font-size: 12px; margin-right: 6px; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #e6e6e6; padding: 8px; text-align: left; font-size: 13px; }}
  th {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<div class="meta">{self._meta_line()}</div>
"""

    def _html_footer(self) -> str:
        return """
</body>
</html>
"""

    def _section_overview(self) -> str:
        ov = self._summary("overview")
        y = None
        if self.cleaned_df is not None and self.schema is not None:
            tc = getattr(self.schema, "target_col", None)
            if tc in self.cleaned_df.columns:
                y = self.cleaned_df[tc]

        parts: List[str] = []
        parts.append("<h2>Overview</h2>")
        parts.append("<div class='grid'>")

        if y is not None:
            fig = lineplot_series(y, title="Target series (raw)", xlabel="time", ylabel="value")
            b64 = fig_to_base64(fig, dpi=self.config.dpi)
            parts.append(self._img_html(b64, "Raw target series"))
        else:
            parts.append("<div class='card'><div class='small'>No target series available.</div></div>")

        parts.append("<div class='card'>")
        parts.append("<div><b>Summary</b></div>")
        parts.append(self._table_html(pd.Series(ov)))
        parts.append("</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_seasonality(self) -> str:
        cand = self._artifact_payload("seasonality_assessment", "period_candidates")
        parts: List[str] = []
        parts.append("<h2>Seasonality</h2>")
        parts.append("<div class='grid'>")

        fig = periodogram_plot(cand, title="Periodogram candidates")
        b64 = fig_to_base64(fig, dpi=self.config.dpi)
        parts.append(self._img_html(b64, "Seasonality candidates"))

        if self.config.show_period_table and isinstance(cand, (pd.DataFrame, pd.Series)):
            parts.append("<div class='card'>")
            parts.append("<div><b>Period candidates table</b></div>")
            parts.append(self._table_html(cand))
            parts.append("</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_stationarity(self) -> str:
        st = self._artifact_payload("stationarity_sweep", "stationarity_table")
        parts: List[str] = []
        parts.append("<h2>Stationarity</h2>")
        parts.append("<div class='grid'>")

        fig = stationarity_scatter_plot(st, title="Stationarity diagnostics")
        b64 = fig_to_base64(fig, dpi=self.config.dpi)
        parts.append(self._img_html(b64, "Stationarity summary plot"))

        if self.config.show_stationarity_tables and isinstance(st, (pd.DataFrame, pd.Series)):
            parts.append("<div class='card'>")
            parts.append("<div><b>Stationarity sweep table</b></div>")
            parts.append(self._table_html(st))
            parts.append("</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_acf_pacf(self) -> str:
        payload = self._artifact_payload("acf_pacf", "acf_pacf_payload")
        parts: List[str] = []
        parts.append("<h2>ACF and PACF</h2>")
        parts.append("<div class='grid'>")

        fig1 = acf_plot_from_payload(payload, title="ACF")
        b64_1 = fig_to_base64(fig1, dpi=self.config.dpi)
        parts.append(self._img_html(b64_1, "ACF plot"))

        fig2 = pacf_plot_from_payload(payload, title="PACF")
        b64_2 = fig_to_base64(fig2, dpi=self.config.dpi)
        parts.append(self._img_html(b64_2, "PACF plot"))

        parts.append("</div>")
        return "\n".join(parts)

    def _section_stl(self) -> str:
        stl = self._artifact_payload("stl", "stl_components")
        parts: List[str] = []
        parts.append("<h2>STL decomposition</h2>")
        parts.append("<div class='grid'>")

        fig = stl_components_plot(stl, title="STL components")
        b64 = fig_to_base64(fig, dpi=self.config.dpi)
        parts.append(self._img_html(b64, "STL decomposition"))

        parts.append("</div>")
        return "\n".join(parts)
