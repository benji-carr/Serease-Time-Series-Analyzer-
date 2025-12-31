# serease/pre_modeling/steps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .specs import DiagnosticSpec
from .containers import PreModelState


@dataclass(frozen=True)
class Step:
    spec: DiagnosticSpec
    run: Callable[[PreModelState], None]


def build_step_registry() -> List[Step]:
    """
    Registry defines the orchestration order.
    Steps can update state.plan and write artifacts into state.diagnostics.

    Note: In the current architecture, the orchestrator performs the actual execution
    and these Step.run callables remain placeholders.
    """
    return [
        Step(
            spec=DiagnosticSpec(
                name="raw_diagnostics",
                view_type="raw",
                artifact_key="raw",
            ),
            run=lambda state: None,
        ),
        Step(
            spec=DiagnosticSpec(
                name="variance_assessment",
                view_type="raw",
                artifact_key="variance",
            ),
            run=lambda state: None,
        ),
        Step(
            spec=DiagnosticSpec(
                name="periodogram",
                view_type="for_periodogram",
                needs_variance_stable=True,
                needs_detrended=True,
                artifact_key="period_candidates",
            ),
            run=lambda state: None,
        ),
        # Step 4 is orchestrated explicitly (candidate sweep + ADF/KPSS selection),
        # but we keep a registry entry so the pipeline contract is complete.
        Step(
            spec=DiagnosticSpec(
                name="stationarity_sweep",
                view_type="stationary_candidate",
                needs_stationary=True,
                needs_seasonal_period=True,
                artifact_key="stationarity_table",
            ),
            run=lambda state: None,
        ),
        Step(
            spec=DiagnosticSpec(
                name="acf_pacf_final",
                view_type="final_stationary",
                needs_stationary=True,
                artifact_key="acf_pacf_payload",
            ),
            run=lambda state: None,
        ),
        Step(
            spec=DiagnosticSpec(
                name="stl_interpretation",
                view_type="stl_interpretation",
                needs_variance_stable=True,
                needs_seasonal_period=True,
                artifact_key="stl_components",
            ),
            run=lambda state: None,
        ),
    ]