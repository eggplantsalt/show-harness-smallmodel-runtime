from plugins.recovery.plugin import RecoveryDecision, RecoveryPlugin


def test_transport_visual_held_rejects_width_only_lost_grasp():
    plugin = RecoveryPlugin(enabled=True)
    mechanical = RecoveryDecision(
        event="lost_grasp",
        reason="closed gripper became empty during TRANSPORT",
        release=True,
        rollback_index=0,
    )
    accepted, telemetry = plugin.arbitrate_transport_holding(
        mechanical,
        stage="TRANSPORT",
        visual_holding_state="HELD",
    )
    assert accepted is None
    assert telemetry["accepted"] is False
    assert telemetry["visual_holding_state"] == "HELD"


def test_transport_visual_lost_keeps_mechanical_recovery():
    plugin = RecoveryPlugin(enabled=True)
    mechanical = RecoveryDecision(
        event="lost_grasp",
        reason="closed gripper became empty during TRANSPORT",
        release=True,
        rollback_index=0,
    )
    accepted, telemetry = plugin.arbitrate_transport_holding(
        mechanical,
        stage="TRANSPORT",
        visual_holding_state="LOST",
    )
    assert accepted is mechanical
    assert telemetry is None


def test_transport_suspected_loss_waits_for_qwen_visual_confirmation():
    plugin = RecoveryPlugin(enabled=True)
    mechanical = RecoveryDecision(
        event="lost_grasp",
        reason="closed gripper width is near zero",
        release=True,
        rollback_index=0,
    )
    accepted, telemetry = plugin.arbitrate_transport_holding(
        mechanical,
        stage="TRANSPORT",
        visual_holding_state="SUSPECTED_LOST",
    )
    assert accepted is None
    assert telemetry["visual_holding_state"] == "SUSPECTED_LOST"
    assert telemetry["reason"] == (
        "width_only_loss_waiting_for_temporal_visual_confirmation"
    )


def test_grasp_empty_close_is_never_suppressed_by_transport_arbiter():
    plugin = RecoveryPlugin(enabled=True)
    mechanical = RecoveryDecision(
        event="empty_grasp",
        reason="gripper closed empty",
        release=True,
        rollback_index=1,
    )
    accepted, telemetry = plugin.arbitrate_transport_holding(
        mechanical,
        stage="GRASP",
        visual_holding_state="HELD",
    )
    assert accepted is mechanical
    assert telemetry is None
