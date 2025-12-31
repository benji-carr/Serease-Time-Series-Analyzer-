# serease/pre_modeling/views.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from typing import List

from .containers import PreModelState, SeriesView, TransformedSeries
from .specs import DiagnosticSpec
from .transforms import TransformEngine


class ViewBuilder:
    """Builds and caches SeriesViews required by diagnostics."""

    def __init__(self, transform_engine: TransformEngine):
        self.xf = transform_engine

    def _stable_hash(self, obj) -> str:
        """
        Hash any object deterministically (best effort).
        - Dataclasses: use asdict
        - dict-like: json dumps sorted
        - fallback: str(obj)
        """
        if is_dataclass(obj):
            payload = asdict(obj)
        elif isinstance(obj, dict):
            payload = obj
        else:
            payload = {"repr": str(obj)}

        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

    def _plan_hash(self, state: PreModelState) -> str:
        return self._stable_hash(state.plan)

    def _spec_hash(self, spec: DiagnosticSpec) -> str:
        # If DiagnosticSpec is a dataclass (recommended), hash all its fields.
        # Otherwise, fall back to hashing its view_type only.
        try:
            return self._stable_hash(spec)
        except Exception:
            return self._stable_hash({"view_type": getattr(spec, "view_type", "unknown")})

    def get_view(self, state: PreModelState, spec: DiagnosticSpec) -> SeriesView:
        """
        Build or retrieve a cached SeriesView.
        Cache key must include:
          - view_type
          - plan hash
          - spec hash (important for candidate-specific specs / future requirements)
        """
        plan_h = self._plan_hash(state)
        spec_h = self._spec_hash(spec)
        key = f"{spec.view_type}:{plan_h}:{spec_h}"

        if key in state.views:
            return state.views[key]

        b = state.bundle
        p = state.plan

        if spec.view_type == "raw":
            view = self.xf.build_raw_view(b)

        elif spec.view_type == "var_stable":
            view = self.xf.build_var_stable_view(b, p)

        elif spec.view_type == "for_periodogram":
            view = self.xf.build_for_periodogram_view(b, p)

        elif spec.view_type == "stl_interpretation":
            view = self.xf.build_stl_interpretation_view(b, p)

        elif spec.view_type in ("final_stationary", "stationary_candidate"):
            # Stationary outputs come from TransformEngine as TransformedSeries
            ts = self.xf.build_stationary_view(b, p)
            view = SeriesView(
                view_id=spec.view_type,
                y=ts.y,
                exog=ts.exog,
                applied_steps=ts.applied_steps,
            )

        else:
            raise ValueError(f"Unknown view_type: {spec.view_type}")

        state.views[key] = view
        return view

    def build_stationarity_candidates(self, state: PreModelState) -> List[TransformedSeries]:
        """
        Create a small, ordered grid of candidate (D,d) combos.
        IMPORTANT: This ordering enforces your preference:
          (0,0) -> (1,0) -> (0,1) -> (1,1) -> (0,2) -> (1,2)
        TransformEngine enforces application order: seasonal diff (D) before regular diff (d).
        """
        b = state.bundle
        base = state.plan
        m = base.seasonal_period_m

        candidates: List[TransformedSeries] = []

        combos = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
        ]

        for D, d in combos:
            if D > base.max_D or d > base.max_d:
                continue
            if D > 0 and (m is None or m <= 1):
                continue

            plan_i = replace(base, D=D, d=d)

            try:
                ts = self.xf.build_stationary_view(b, plan_i)
                candidates.append(ts)
            except Exception as e:
                # Keep orchestration alive, but don't fail silently.
                if hasattr(state, "diagnostics") and hasattr(state.diagnostics, "notes"):
                    state.diagnostics.notes.append(
                        f"Stationarity candidate failed for (D={D}, d={d}, m={m}): {type(e).__name__}: {e}"
                    )
                continue

        return candidates