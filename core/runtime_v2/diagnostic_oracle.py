"""Hard isolation boundary for privileged diagnostic interventions.

This module is intentionally not imported by :mod:`core.runtime_v2`.  A caller
must explicitly opt into a diagnostic profile before it can even construct an
intervention.  Final-method configs therefore cannot accidentally leak LIBERO
state into policy evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DIAGNOSTIC_PROFILE_SUFFIX = "diagnostic_oracle"


def require_diagnostic_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    profile = str(cfg.get("research_profile") or "")
    runtime = cfg.get("runtime_v2")
    runtime = runtime if isinstance(runtime, dict) else {}
    oracle = runtime.get("diagnostic_oracle")
    oracle = oracle if isinstance(oracle, dict) else {}
    if not bool(oracle.get("enabled", False)):
        raise RuntimeError("privileged intervention requested while oracle is disabled")
    if not profile.endswith(DIAGNOSTIC_PROFILE_SUFFIX):
        raise RuntimeError(
            "privileged intervention requires a research_profile ending in "
            f"{DIAGNOSTIC_PROFILE_SUFFIX!r}"
        )
    return oracle


@dataclass(frozen=True)
class SlipInjection:
    """Metadata contract for an environment-owned transport perturbation."""

    frame_id: int
    lateral_m: float
    downward_m: float
    policy_observation_exposed: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "transport_slip",
            "frame_id": int(self.frame_id),
            "lateral_m": float(self.lateral_m),
            "downward_m": float(self.downward_m),
            "privileged_state_to_policy": False,
            "policy_observation_exposed": bool(self.policy_observation_exposed),
        }


def validate_slip_injection(spec: SlipInjection) -> SlipInjection:
    """Bound perturbations so diagnostics cannot become scripted placement."""
    if spec.frame_id < 0:
        raise ValueError("slip frame_id must be non-negative")
    if not 0.0 < abs(float(spec.lateral_m)) <= 0.04:
        raise ValueError("lateral slip must be non-zero and at most 4 cm")
    if not 0.0 <= float(spec.downward_m) <= 0.04:
        raise ValueError("downward slip must be between 0 and 4 cm")
    if spec.policy_observation_exposed:
        raise ValueError("oracle intervention metadata may not enter policy observation")
    return spec
