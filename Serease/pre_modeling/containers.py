# serease/pre_modeling/containers.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any, Dict, List
import pandas as pd


# ---------------------------
# Input container
# ---------------------------
@dataclass(frozen=True)
class SeriesBundle:
    """The ground-truth inputs to pre-model."""
    y_raw: pd.Series
    exog_raw: Optional[pd.DataFrame] = None
    freq: Optional[str] = None
    name: str = "y"

    def __post_init__(self):
        # Ensure the series has a stable, human-readable name
        if self.y_raw.name is None or self.y_raw.name != self.name:
            object.__setattr__(self, "y_raw", self.y_raw.rename(self.name))


# ---------------------------
# Transform provenance
# ---------------------------
@dataclass
class TransformStepRecord:
    """
    Records one transformation applied to a series.
    Used for provenance, reporting, and invertibility tracking.
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    invertible: bool = True
    notes: str = ""


# ---------------------------
# Transform plan (decision container)
# ---------------------------
@dataclass
class TransformPlan:
    """
    Declarative description of how the series should be transformed.
    No logic — just decisions.
    """
    # variance stabilization
    variance_transform: Literal["none", "log", "boxcox"] = "none"
    boxcox_lambda: Optional[float] = None
    boxcox_offset: float = 0.0  # used if zeros/negatives exist

    # period detection prep
    detrend_for_periodogram: Literal["none", "linear", "diff1"] = "linear"

    # seasonality
    seasonal_period_m: Optional[int] = None

    # stationarity diffs
    d: int = 0
    D: int = 0

    # guardrails
    max_d: int = 2
    max_D: int = 1


# ---------------------------
# Views + transformed outputs
# ---------------------------
@dataclass
class SeriesView:
    """
    A transformed *view* of the series created to satisfy a diagnostic's needs.
    Views are temporary and diagnostic-specific.
    """
    view_id: str
    y: pd.Series
    exog: Optional[pd.DataFrame]
    applied_steps: List[TransformStepRecord] = field(default_factory=list)


@dataclass
class TransformedSeries:
    """
    A selected series intended for modeling (stationary).
    This is the output of the pre-model layer.
    """
    y: pd.Series
    exog: Optional[pd.DataFrame]
    plan: TransformPlan
    applied_steps: List[TransformStepRecord] = field(default_factory=list)
    inverse_fn: Optional[Callable[[pd.Series], pd.Series]] = None


# ---------------------------
# Diagnostics artifacts
# (thin containers — expand to typed dataclasses later)
# ---------------------------
@dataclass
class DiagnosticsPacket:
    raw: Dict[str, Any] = field(default_factory=dict)
    variance: Dict[str, Any] = field(default_factory=dict)

    period_candidates: Dict[str, Any] = field(default_factory=dict)
    stationarity_table: List[Dict[str, Any]] = field(default_factory=list)
    stationarity_selection: Dict[str, Any] = field(default_factory=dict)

    acf_pacf_payload: Dict[str, Any] = field(default_factory=dict)
    stl_components: Dict[str, Any] = field(default_factory=dict)

    notes: List[str] = field(default_factory=list)


# ---------------------------
# Global pre-model state
# ---------------------------
@dataclass
class PreModelState:
    """
    Mutable state object passed through the pre-model orchestration.
    Owns:
      - inputs
      - decisions
      - diagnostics
      - cached views
      - candidate series
      - final selected stationary series
    """
    bundle: SeriesBundle
    plan: TransformPlan = field(default_factory=TransformPlan)
    diagnostics: DiagnosticsPacket = field(default_factory=DiagnosticsPacket)

    # caches
    views: Dict[str, SeriesView] = field(default_factory=dict)
    candidates: List[TransformedSeries] = field(default_factory=list)

    # final selected stationary series
    final: Optional[TransformedSeries] = None
