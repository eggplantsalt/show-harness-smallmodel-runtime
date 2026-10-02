from __future__ import annotations

import pytest

from core.runtime_v2.diagnostic_oracle import (
    SlipInjection,
    require_diagnostic_profile,
    validate_slip_injection,
)


def test_oracle_requires_explicit_isolated_profile() -> None:
    with pytest.raises(RuntimeError):
        require_diagnostic_profile(
            {
                "research_profile": "clean_qwen3vl_verified_capability_runtime_v2",
                "runtime_v2": {"diagnostic_oracle": {"enabled": True}},
            }
        )
    enabled = {
        "research_profile": "vcr_slip_diagnostic_oracle",
        "runtime_v2": {"diagnostic_oracle": {"enabled": True}},
    }
    assert require_diagnostic_profile(enabled)["enabled"] is True


def test_slip_contract_never_exposes_privileged_state() -> None:
    spec = validate_slip_injection(
        SlipInjection(frame_id=20, lateral_m=0.02, downward_m=0.01)
    )
    assert spec.metadata()["privileged_state_to_policy"] is False
    with pytest.raises(ValueError):
        validate_slip_injection(
            SlipInjection(
                frame_id=20,
                lateral_m=0.02,
                downward_m=0.01,
                policy_observation_exposed=True,
            )
        )
