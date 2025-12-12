from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PlotTheme:
    style: str = "whitegrid"
    context: str = "notebook"
    palette: str = "deep"
    font_scale: float = 1.0


def apply_theme(theme: Optional[PlotTheme] = None) -> None:
    t = theme or PlotTheme()
    sns.set_theme(style=t.style, context=t.context, palette=t.palette, font_scale=t.font_scale)


def fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def lineplot_series(
    s: pd.Series,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    df = s.rename("value").to_frame()
    df = df.reset_index().rename(columns={df.index.name or "index": "time"})
    sns.lineplot(data=df, x="time", y="value", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def multi_lineplot(
    series_map: Dict[str, pd.Series],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    frames = []
    for name, s in series_map.items():
        if s is None:
            continue
        tmp = s.rename("value").to_frame()
        tmp["variant"] = name
        tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "time"})
        frames.append(tmp)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        sns.lineplot(data=df, x="time", y="value", hue="variant", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def histogram_kde(
    s: pd.Series,
    title: str,
    xlabel: str = "",
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    x = s.dropna().values
    if len(x) == 0:
        ax.set_title(title)
        return fig
    sns.histplot(x, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return fig


def heatmap_table(
    df: pd.DataFrame,
    title: str,
    figsize: Tuple[int, int] = (12, 6),
    fmt: str = ".3g",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.set_title(title)
        return fig
    sns.heatmap(df, annot=False, cmap="viridis", ax=ax)
    ax.set_title(title)
    return fig


def periodogram_plot(
    candidates_df: pd.DataFrame,
    title: str = "Periodogram candidates",
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if candidates_df is None or candidates_df.empty:
        ax.set_title(title)
        return fig

    df = candidates_df.copy()
    if "period_steps" not in df.columns and "period_int" in df.columns:
        df["period_steps"] = df["period_int"].astype(float)

    df = df[df["period_steps"].notna()].copy()
    if df.empty:
        ax.set_title(title)
        return fig

    df = df.sort_values("power", ascending=False).head(12)
    sns.barplot(data=df, x="period_steps", y="power", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("candidate period (steps)")
    ax.set_ylabel("power")
    return fig


def acf_pacf_plot(
    acf_pacf_df: pd.DataFrame,
    title: str = "ACF / PACF",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if acf_pacf_df is None or acf_pacf_df.empty:
        ax.set_title(title)
        return fig

    df = acf_pacf_df.copy()
    df_long = df.melt(id_vars=["lag"], value_vars=["acf", "pacf"], var_name="type", value_name="value")
    sns.lineplot(data=df_long, x="lag", y="value", hue="type", ax=ax)
    ax.axhline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("lag")
    ax.set_ylabel("value")
    return fig


def stl_components_plot(
    stl_df: pd.DataFrame,
    title: str = "STL components",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    cols = ["observed", "trend", "seasonal", "resid"]
    for ax, c in zip(axes, cols):
        if c in stl_df.columns:
            sns.lineplot(x=stl_df.index, y=stl_df[c].values, ax=ax)
            ax.set_ylabel(c)
        else:
            ax.set_ylabel(c)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time")
    return fig
