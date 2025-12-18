from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    """
    A named payload produced by a diagnostics step.

    Contract:
      - `name` must be stable because the reporter keys on it.
      - `payload` can be any serializable object or a rich python object
        (e.g., pandas DataFrame/Series), depending on how the reporter renders it.
    """
    name: str
    payload: Any = None


@dataclass
class StepResult:
    """
    Output of a single diagnostics step.

    Fields
    ------
    step_name:
        Stable step identifier (e.g., "acf_pacf", "stl").
    summary:
        Small dictionary of headline values that the reporter can render as key/value.
    notes:
        Non-fatal, explanatory messages for the user ("why this was skipped", "assumptions").
    warnings:
        Non-fatal but important issues ("insufficient obs", "statsmodels missing", etc.).
    artifacts:
        List of Artifact objects keyed by stable artifact names.
    """
    step_name: str
    summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)

    def get_artifact(self, name: str) -> Optional[Artifact]:
        for a in self.artifacts:
            if a.name == name:
                return a
        return None

    def artifact_payload(self, name: str, default: Any = None) -> Any:
        a = self.get_artifact(name)
        return default if a is None else a.payload

    def add_artifact(self, name: str, payload: Any) -> None:
        for i, a in enumerate(self.artifacts):
            if a.name == name:
                self.artifacts[i] = Artifact(name=name, payload=payload)
                return
        self.artifacts.append(Artifact(name=name, payload=payload))


@dataclass
class DiagnosticsReport:
    """
    Container for all step outputs produced by DiagnosticsEngine.

    This is the stable boundary between diagnostics computation and reporting.
    """
    dataset_name: str = "dataset"
    steps: Dict[str, StepResult] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, step: StepResult) -> None:
        self.steps[step.step_name] = step

    def get(self, step_name: str) -> Optional[StepResult]:
        return self.steps.get(step_name)

    def ensure_step(self, step_name: str) -> StepResult:
        if step_name not in self.steps:
            self.steps[step_name] = StepResult(step_name=step_name)
        return self.steps[step_name]

    def list_steps(self, order: Optional[List[str]] = None) -> List[str]:
        keys = list(self.steps.keys())
        if not order:
            return keys

        out: List[str] = []
        seen = set()

        for k in order:
            if k in self.steps:
                out.append(k)
                seen.add(k)

        for k in keys:
            if k not in seen:
                out.append(k)

        return out
