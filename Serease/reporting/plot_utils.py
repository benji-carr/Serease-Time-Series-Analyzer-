from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PlotTheme:
    font_scale: float = 1.0
    style: str = "default"


def apply_theme(theme: PlotTheme) -> None:
    try:
        plt.style.use(theme.style)
    except Exception:
        plt.style.use("default")


def fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    import io
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def lineplot_series(s: pd.Series, title: str = "", xlabel: str = "", ylabel: str = "") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(s.index, s.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def multi_lineplot(series_map: Mapping[str, pd.Series], title: str = "") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, s in series_map.items():
        ax.plot(s.index, s.values, label=name)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def histogram_kde(s: pd.Series, title: str = "", xlabel: str = "") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = s.dropna().values
    ax.hist(x, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def periodogram_plot(candidates: Any, title: str = "Periodogram") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.text(0.01, 0.5, "Periodogram plot not implemented in skeleton.", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def stationarity_scatter_plot(stationarity_table: Any, title: str = "Stationarity") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.text(0.01, 0.5, "Stationarity plot not implemented in skeleton.", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def acf_plot_from_payload(payload: Any, title: str = "ACF") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.text(0.01, 0.5, "ACF plot not implemented in skeleton.", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def pacf_plot_from_payload(payload: Any, title: str = "PACF") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.text(0.01, 0.5, "PACF plot not implemented in skeleton.", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def stl_components_plot(stl_df: Any, title: str = "STL decomposition") -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.text(0.01, 0.5, "STL plot not implemented in skeleton.", transform=ax.transAxes)
    fig.tight_layout()
    return fig
