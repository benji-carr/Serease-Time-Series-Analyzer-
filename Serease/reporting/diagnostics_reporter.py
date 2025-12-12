from __future__ import annotations

from dataclasses import dataclass
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
    acf_pacf_plot,
    stl_components_plot,
)


@dataclass
class ReporterConfig:
    theme: PlotTheme = PlotTheme()
    dpi: int = 150
    max_table_rows: int = 40


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
        sections.append(self._section_variants_story())
        sections.append(self._section_stationarity())
        sections.append(self._section_acf_pacf())
        sections.append(self._section_stl())
        sections.append(self._section_recommendations())
        sections.append(self._html_footer())
        return "\n".join(sections)

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
  .note {{ color: #333; }}
  .warn {{ color: #9b1c1c; }}
  .kv {{ display: grid; grid-template-columns: 220px 1fr; gap: 6px 12px; }}
  .kv div {{ padding: 2px 0; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #e6e6e6; padding: 8px; text-align: left; font-size: 13px; }}
  th {{ background: #fafafa; }}
  .small {{ font-size: 13px; color: #444; }}
  .pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; background: #f3f4f6; font-size: 12px; margin-right: 6px; }}
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

    def _meta_line(self) -> str:
        bits = []
        if self.schema is not None:
            bits.append(f"<span class='pill'>target: {getattr(self.schema, 'target_col', None)}</span>")
            bits.append(f"<span class='pill'>date: {getattr(self.schema, 'date_col', None)}</span>")
        if self.ts_meta is not None:
            bits.append(f"<span class='pill'>freq: {getattr(self.ts_meta, 'freq', None)}</span>")
            bits.append(f"<span class='pill'>n_obs: {getattr(self.ts_meta, 'n_obs', None)}</span>")
        return " ".join(bits)

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

    def _warnings(self, step: str) -> List[str]:
        r = self._get_step(step)
        if r is None:
            return []
        return list(getattr(r, "warnings", []) or [])

    def _notes(self, step: str) -> List[str]:
        r = self._get_step(step)
        if r is None:
            return []
        return list(getattr(r, "notes", []) or [])

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

    def _section_overview(self) -> str:
        ov = self._summary("overview")
        miss = self._summary("missingness")

        y = self._bundle_series_safe("raw")
        fig_raw = lineplot_series(y, title="Target series (raw)", xlabel="time", ylabel="value")
        raw_b64 = fig_to_base64(fig_raw, dpi=self.config.dpi)

        parts = []
        parts.append("<h2>Overview</h2>")
        parts.append("<div class='grid'>")
        parts.append(self._img_html(raw_b64, "Raw target series"))

        parts.append("<div class='card'>")
        parts.append("<div class='kv'>")
        for k in ["n_obs_total", "n_obs_non_missing", "start", "end", "min", "max", "mean", "std"]:
            if k in ov:
                parts.append(f"<div><b>{k}</b></div><div>{ov.get(k)}</div>")
        for k in ["n_missing", "missing_frac", "n_missing_blocks", "longest_missing_block"]:
            if k in miss:
                parts.append(f"<div><b>{k}</b></div><div>{miss.get(k)}</div>")
        parts.append("</div>")
        parts.append("</div>")
        parts.append("</div>")

        mb = self._artifact_payload("missingness", "missing_blocks")
        if mb is not None:
            parts.append("<div class='card'>")
            parts.append("<div><b>Missing blocks</b></div>")
            parts.append(self._table_html(mb))
            parts.append("</div>")

        return "\n".join(parts)

    def _section_seasonality(self) -> str:
        base = self._summary("base_variant").get("base_variant", None)
        m = self._summary("seasonal_period").get("seasonal_period", None)
        cand = self._artifact_payload("period_detection", "period_candidates")

        parts = []
        parts.append("<h2>Seasonality</h2>")
        parts.append("<div class='grid'>")

        info = []
        info.append("<div class='card'>")
        info.append("<div class='kv'>")
        info.append(f"<div><b>base_variant</b></div><div>{base}</div>")
        info.append(f"<div><b>seasonal_period</b></div><div>{m}</div>")
        for w in self._warnings("seasonal_period"):
            info.append(f"<div><b>warning</b></div><div class='warn'>{w}</div>")
        info.append("</div>")
        info.append("</div>")
        parts.append("\n".join(info))

        if isinstance(cand, pd.DataFrame) and not cand.empty:
            fig = periodogram_plot(cand, title="Periodogram candidates (top peaks)")
            b64 = fig_to_base64(fig, dpi=self.config.dpi)
            parts.append(self._img_html(b64, "Periodogram candidate periods"))
            parts.append("<div class='card'>")
            parts.append("<div><b>Period candidates table</b></div>")
            parts.append(self._table_html(cand))
            parts.append("</div>")
        else:
            parts.append("<div class='card'><div class='small'>No period candidates available.</div></div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_variants_story(self) -> str:
        selected = self._summary("variant_selection").get("selected_stationary_variant", None)

        variants_to_show = self._choose_story_variants(selected)
        series_map = {name: self._bundle_series_safe(name) for name in variants_to_show}

        fig = multi_lineplot(series_map, title="Variant comparison (raw / transforms / differencing)")
        b64 = fig_to_base64(fig, dpi=self.config.dpi)

        parts = []
        parts.append("<h2>Transformations and differencing</h2>")
        parts.append("<div class='grid'>")
        parts.append(self._img_html(b64, "Comparing key variants on the same time axis"))

        if selected is not None:
            s_sel = self._bundle_series_safe(selected)
            fig2 = histogram_kde(s_sel, title=f"Distribution of selected variant: {selected}", xlabel="value")
            b64_2 = fig_to_base64(fig2, dpi=self.config.dpi)
            parts.append(self._img_html(b64_2, "Histogram + KDE of selected stationary candidate"))

        parts.append("<div class='card'>")
        parts.append("<div><b>Selected stationary variant</b></div>")
        parts.append(f"<div class='note'>{selected}</div>")
        parts.append("</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_stationarity(self) -> str:
        st = self._artifact_payload("stationarity_sweep", "stationarity_table")
        sel_table = self._artifact_payload("variant_selection", "selection_ranked")
        sel_summary = self._summary("variant_selection")

        parts = []
        parts.append("<h2>Stationarity</h2>")
        parts.append("<div class='grid'>")

        parts.append("<div class='card'>")
        parts.append("<div><b>Selection summary</b></div>")
        parts.append("<div class='kv'>")
        for k, v in sel_summary.items():
            parts.append(f"<div><b>{k}</b></div><div>{v}</div>")
        parts.append("</div>")
        parts.append("</div>")

        if isinstance(st, pd.DataFrame) and not st.empty:
            cols = [c for c in ["adf_pvalue", "kpss_pvalue", "acf_lag1", "total_diff"] if c in st.columns]
            view = st.set_index("variant")[cols].copy() if "variant" in st.columns else st[cols].copy()
            fig = None
            try:
                fig = heatmap_table(view.astype(float), title="Stationarity diagnostics heatmap")
            except Exception:
                fig = None
            if fig is not None:
                b64 = fig_to_base64(fig, dpi=self.config.dpi)
                parts.append(self._img_html(b64, "Heatmap over candidate variants"))
            parts.append("<div class='card'>")
            parts.append("<div><b>Stationarity sweep table</b></div>")
            parts.append(self._table_html(st))
            parts.append("</div>")
        else:
            parts.append("<div class='card'><div class='small'>No stationarity sweep table available.</div></div>")

        if isinstance(sel_table, pd.DataFrame) and not sel_table.empty:
            parts.append("<div class='card'>")
            parts.append("<div><b>Ranked candidates</b></div>")
            parts.append(self._table_html(sel_table))
            parts.append("</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_acf_pacf(self) -> str:
        ap = self._artifact_payload("acf_pacf", "acf_pacf_values")
        variant = self._summary("acf_pacf").get("variant", None)

        parts = []
        parts.append("<h2>ACF and PACF</h2>")
        parts.append("<div class='grid'>")

        if isinstance(ap, pd.DataFrame) and not ap.empty:
            fig = acf_pacf_plot(ap, title=f"ACF / PACF (variant: {variant})")
            b64 = fig_to_base64(fig, dpi=self.config.dpi)
            parts.append(self._img_html(b64, "ACF/PACF values"))
            parts.append("<div class='card'>")
            parts.append("<div><b>ACF/PACF table</b></div>")
            parts.append(self._table_html(ap))
            parts.append("</div>")
        else:
            parts.append("<div class='card'><div class='small'>No ACF/PACF data available.</div></div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_stl(self) -> str:
        stl = self._artifact_payload("stl", "stl_components")
        stl_sum = self._summary("stl")

        parts = []
        parts.append("<h2>STL decomposition</h2>")
        parts.append("<div class='grid'>")

        if isinstance(stl, pd.DataFrame) and not stl.empty:
            fig = stl_components_plot(stl, title="STL components")
            b64 = fig_to_base64(fig, dpi=self.config.dpi)
            parts.append(self._img_html(b64, "Observed, trend, seasonal, residual"))

            parts.append("<div class='card'>")
            parts.append("<div><b>STL summary</b></div>")
            parts.append("<div class='kv'>")
            for k, v in stl_sum.items():
                parts.append(f"<div><b>{k}</b></div><div>{v}</div>")
            parts.append("</div>")
            parts.append("</div>")
        else:
            parts.append("<div class='card'><div class='small'>No STL components available.</div></div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _section_recommendations(self) -> str:
        recs = self._auto_recommendations()

        parts = []
        parts.append("<h2>Notes and recommendations</h2>")
        parts.append("<div class='card'>")
        if not recs:
            parts.append("<div class='small'>No recommendations generated.</div>")
        else:
            parts.append("<ul>")
            for r in recs:
                parts.append(f"<li class='note'>{r}</li>")
            parts.append("</ul>")
        parts.append("</div>")
        return "\n".join(parts)

    def _bundle_series_safe(self, name: str) -> pd.Series:
        if self.bundle is not None and hasattr(self.bundle, "has") and self.bundle.has(name):
            return self.bundle.get(name)
        if name == "raw" and self.cleaned_df is not None and self.schema is not None:
            tc = getattr(self.schema, "target_col", None)
            if tc is not None and tc in self.cleaned_df.columns:
                return self.cleaned_df[tc]
        if self.cleaned_df is not None and self.schema is not None:
            tc = getattr(self.schema, "target_col", None)
            if tc is not None and tc in self.cleaned_df.columns:
                return self.cleaned_df[tc]
        raise KeyError(f"Cannot find series '{name}' in bundle or cleaned_df.")

    def _choose_story_variants(self, selected: Optional[str]) -> List[str]:
        candidates = ["raw", "log", "log1p", "diff1", "log_diff1", "log1p_diff1"]
        out = [v for v in candidates if self.bundle is not None and hasattr(self.bundle, "has") and self.bundle.has(v)]
        if selected is not None and selected not in out:
            if self.bundle is not None and hasattr(self.bundle, "has") and self.bundle.has(selected):
                out.append(selected)
        if "raw" not in out:
            out.insert(0, "raw")
        return out

    def _auto_recommendations(self) -> List[str]:
        recs: List[str] = []

        m = self._summary("seasonal_period").get("seasonal_period", None)
        if m is None:
            recs.append("No seasonal period was selected; STL was skipped and seasonal structure may be under-modeled.")
        else:
            recs.append(f"Seasonal period selected: m={m}. Use this as the seasonal period for STL and seasonal ARIMA terms.")

        sel = self._summary("variant_selection")
        selected = sel.get("selected_stationary_variant", None)
        if selected is None:
            recs.append("No stationary variant was selected. Consider providing a seasonal period or increasing observation count.")
            return recs

        recs.append(f"Selected stationary variant: {selected}.")

        st = self._artifact_payload("stationarity_sweep", "stationarity_table")
        if isinstance(st, pd.DataFrame) and "variant" in st.columns:
            row = st[st["variant"] == selected]
            if not row.empty:
                r0 = row.iloc[0].to_dict()
                adf_p = r0.get("adf_pvalue", None)
                kpss_p = r0.get("kpss_pvalue", None)
                a1 = r0.get("acf_lag1", None)
                recs.append(f"Stationarity diagnostics (selected): ADF p={adf_p}, KPSS p={kpss_p}, ACF lag1={a1}.")
                if a1 is not None and pd.notna(a1) and float(a1) <= -0.6:
                    recs.append("Potential over-differencing: ACF lag1 is strongly negative; consider a less differenced variant.")

        recs.extend(self._notes("overview"))
        recs.extend(self._notes("missingness"))
        recs.extend(self._notes("seasonal_period"))
        recs.extend(self._notes("variant_selection"))

        for w in self._warnings("stationarity_sweep"):
            recs.append(f"Stationarity sweep warning: {w}")

        return recs
