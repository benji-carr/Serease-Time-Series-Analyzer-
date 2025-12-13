from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    name: str
    payload: Any


@dataclass
class StepResult:
    step: str
    summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)


@dataclass
class DiagnosticsReport:
    results: Dict[str, StepResult] = field(default_factory=dict)

    def add(self, step_result: StepResult) -> None:
        self.results[step_result.step] = step_result

    def get(self, step: str) -> Optional[StepResult]:
        return self.results.get(step)

    def steps(self) -> List[str]:
        return list(self.results.keys())
