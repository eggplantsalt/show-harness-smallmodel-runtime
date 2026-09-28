from core.capabilities.verified_runtime import VerifiedEmbodiedRuntime


def evidence(
    dx,
    dy,
    *,
    height=0.20,
    horizontal="MV_RIGHT",
    vertical="MV_FWD",
):
    return {
        "frame_id": 1,
        "stage": "APPROACH",
        "target": "test object",
        "camera": "agentview",
        "visible": True,
        "confidence": 0.9,
        "source": "cpu_template_tracker",
        "geometry": {
            "target_minus_eef_px": [dx, dy],
            "alignment_ready": max(abs(dx), abs(dy)) <= 10,
            "eef_height_m": height,
            "calibrated_correction_candidates": {
                "horizontal": horizontal,
                "vertical": vertical,
            },
        },
    }


def test_uses_prior_then_learns_real_effect():
    rt = VerifiedEmbodiedRuntime(
        enabled=True,
        mode="active",
        alignment_px=10,
        final_height_max_m=0.14,
    )

    d1 = rt.observe(
        stage="APPROACH",
        evidence=evidence(30, 5),
        previous_action=None,
    )
    assert d1["action_token"] == "MV_RIGHT"

    # MV_RIGHT actually improved 30 -> 20.
    d2 = rt.observe(
        stage="APPROACH",
        evidence=evidence(20, 5),
        previous_action="MV_RIGHT",
    )
    assert d2["transition"]["status"] == "IMPROVING"
    assert "MV_RIGHT" in d2["action_effects"]


def test_wrong_direction_is_reversed():
    rt = VerifiedEmbodiedRuntime(
        enabled=True,
        mode="active",
        alignment_px=10,
        final_height_max_m=0.14,
    )

    first = rt.observe(
        stage="APPROACH",
        evidence=evidence(30, 5),
        previous_action=None,
    )
    assert first["action_token"] == "MV_RIGHT"

    # Contrary to the prior, MV_RIGHT made error worse.
    second = rt.observe(
        stage="APPROACH",
        evidence=evidence(40, 5),
        previous_action="MV_RIGHT",
    )
    assert second["transition"]["status"] == "WRONG_DIRECTION"
    assert second["action_token"] == "MV_LEFT"


def test_high_pose_does_not_confuse_parallax_with_depth():
    rt = VerifiedEmbodiedRuntime(
        enabled=True,
        mode="active",
        alignment_px=10,
        final_height_max_m=0.14,
    )

    decision = rt.observe(
        stage="APPROACH",
        evidence=evidence(5, 50, height=0.22),
        previous_action=None,
    )
    assert decision["takeover"] is False
    assert decision["status"] == "WAITING_FOR_HEIGHT_TRANSITION"


def test_depth_is_externalized_only_at_final_height():
    rt = VerifiedEmbodiedRuntime(
        enabled=True,
        mode="active",
        alignment_px=10,
        final_height_max_m=0.14,
    )

    decision = rt.observe(
        stage="APPROACH",
        evidence=evidence(5, 50, height=0.13),
        previous_action=None,
    )
    assert decision["takeover"] is True
    assert decision["action_token"] == "MV_FWD"
