from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class DiagnosticArtifact:
    name: str
    kind: str
    payload: Any


@dataclass
class DiagnosticResult:
    step: str
    ok: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[DiagnosticArtifact] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DiagnosticsReport:
    target_col: str
    date_col: str
    freq: Optional[str]
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp

    results: Dict[str, DiagnosticResult] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add(self, result: DiagnosticResult) -> None:
        self.results[result.step] = result

    def get(self, step: str) -> DiagnosticResult:
        if step not in self.results:
            raise KeyError(f"Diagnostic step '{step}' not found.")
        return self.results[step]
