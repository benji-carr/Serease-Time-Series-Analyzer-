# serease/pre_modeling/specs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ViewType = Literal[
    "raw",
    "var_stable",
    "for_periodogram",
    "stationary_candidate",
    "final_stationary",
    "stl_interpretation",
]


@dataclass(frozen=True)
class DiagnosticSpec:
    """
    Declares what view is required for a diagnostic to make sense.

    Today: ViewBuilder primarily uses view_type routing.
    Future: the needs_* flags can be used to dynamically assemble views.
    """
    name: str
    view_type: ViewType

    # Requirements
    needs_variance_stable: bool = False
    needs_detrended: bool = False
    needs_stationary: bool = False

    # Optional prerequisites / preferences
    needs_seasonal_period: bool = False
    prefers_minimal_differencing: bool = False

    # Where to store outputs in DiagnosticsPacket
    artifact_key: str = ""
