from __future__ import annotations

from pathlib import Path

import pytest

from core.runtime_v3.calibration import compute_contract_metrics, run_calibration_trial
from core.runtime_v3.observer import RobotObservation


ROOT = Path(__file__).resolve().parents[2]


def test_calibration_metric_computes_projection_ratio_off_axis_and_norm():
    metrics = compute_contract_metrics(
        [0.0, 0.0, 0.0],
        [0.0, 0.003, 0.004],
        [0.0, 0.0, 1.0],
        0.005,
    )
    assert metrics["projection_m"] == pytest.approx(0.004)
    assert metrics["realization_ratio"] == pytest.approx(0.8)
    assert metrics["off_axis_vector_m"] == pytest.approx([0.0, 0.003, 0.0])
    assert metrics["off_axis_magnitude_m"] == pytest.approx(0.003)
    assert metrics["observed_norm_m"] == pytest.approx(0.005)
    assert metrics["projection_mm"] == pytest.approx(4.0)
    assert metrics["off_axis_magnitude_mm"] == pytest.approx(3.0)


def test_calibration_projection_is_signed_against_commanded_axis():
    metrics = compute_contract_metrics(
        [0.0, 0.0, 0.3], [0.0, 0.0, 0.298], [0.0, 0.0, 1.0], 0.005
    )
    assert metrics["projection_m"] == pytest.approx(-0.002)
    assert metrics["realization_ratio"] == pytest.approx(-0.4)


def test_calibration_off_axis_error_removes_the_target_projection():
    metrics = compute_contract_metrics(
        [0.0, 0.0, 0.0], [0.003, 0.004, 0.0], [1.0, 0.0, 0.0], 0.005
    )
    assert metrics["off_axis_vector_m"] == pytest.approx([0.0, 0.004, 0.0])
    assert metrics["off_axis_magnitude_m"] == pytest.approx(0.004)


def test_calibration_direction_cosine_uses_projection_over_total_motion():
    metrics = compute_contract_metrics(
        [0.0, 0.0, 0.0], [0.003, 0.0, 0.004], [0.0, 0.0, 1.0], 0.005
    )
    assert metrics["direction_cosine"] == pytest.approx(0.8)
    assert metrics["observed_norm_m"] == pytest.approx(0.005)


def test_calibration_requires_a_unit_command_direction():
    with pytest.raises(ValueError, match="unit length"):
        compute_contract_metrics([0, 0, 0], [0, 0, 0.1], [0, 0, 2], 0.005)


def test_calibration_trial_resets_before_each_single_executor_action():
    class Environment:
        def __init__(self):
            self.position = [0.0, 0.0, 0.3]
            self.reset_count = 0
            self.step_count = 0

        def reset(self):
            self.reset_count += 1
            self.step_count = 0
            self.position = [0.0, 0.0, 0.3]

        def step(self, _command):
            self.step_count += 1
            self.position[2] += 0.001
            return {}, 0.0, False, {}

    class Observer:
        def __init__(self):
            self.frame = 0

        def observe(self, environment):
            self.frame += 1
            return RobotObservation(
                observation_id=f"frame-{self.frame}",
                frame_id=self.frame,
                proprioception={
                    "end_effector_state": {"position_xyz": list(environment.position)},
                },
                evidence={"stage": "CALIBRATION"},
                evidence_refs=(f"frame-ref-{self.frame}",),
                fresh=True,
            )

    class Controller:
        def __init__(self):
            self.calls = []

        def action_for_atomic(self, token, *, step_m=None):
            self.calls.append((token, step_m))
            return (token, step_m)

    environment = Environment()
    observer = Observer()
    first = run_calibration_trial(
        environment,
        observer,
        Controller(),
        task_id="mock:0",
        token="MV_UP",
        direction_unit=[0.0, 0.0, 1.0],
        commanded_step_m=0.005,
    )
    first_before = first["state_before_eef_xyz_m"]
    second = run_calibration_trial(
        environment,
        observer,
        Controller(),
        task_id="mock:0",
        token="MV_UP",
        direction_unit=[0.0, 0.0, 1.0],
        commanded_step_m=0.005,
    )

    assert environment.reset_count == 2
    assert first["actions"] == second["actions"] == 1
    assert first["selection"] == second["selection"] == "CALIBRATION_MV_UP"
    assert first["approved_action"] == second["approved_action"] == "CALIBRATION_MV_UP"
    assert first["backend_execution"] and second["backend_execution"]
    assert environment.step_count == 1
    assert first_before == second["state_before_eef_xyz_m"] == [0.0, 0.0, 0.3]
    assert first["contract_metrics"]["projection_m"] == pytest.approx(0.001)
    assert second["contract_metrics"]["projection_m"] == pytest.approx(0.001)


def test_calibration_trial_skips_a_target_outside_the_configured_z_workspace():
    class Environment:
        def __init__(self):
            self.position = [0.0, 0.0, 0.599]
            self.steps = 0

        def reset(self):
            self.position = [0.0, 0.0, 0.599]

        def step(self, _command):
            self.steps += 1
            return {}, 0.0, False, {}

    class Observer:
        def observe(self, environment):
            return RobotObservation(
                observation_id="frame-1",
                frame_id=1,
                proprioception={
                    "end_effector_state": {"position_xyz": list(environment.position)},
                },
                evidence_refs=("frame-ref-1",),
                fresh=True,
            )

    class Controller:
        def action_for_atomic(self, _token, *, step_m=None):
            return ("command", step_m)

    environment = Environment()
    outcome = run_calibration_trial(
        environment,
        Observer(),
        Controller(),
        task_id="mock:0",
        token="MV_UP",
        direction_unit=[0.0, 0.0, 1.0],
        commanded_step_m=0.005,
        workspace_z_bounds_m=(0.02, 0.60),
    )
    assert outcome["skip_reason"] == "SKIPPED_WORKSPACE_BOUNDARY"
    assert outcome["actions"] == 0
    assert environment.steps == 0


def test_calibration_runner_routes_action_through_runtime_executor():
    module_source = (ROOT / "core/runtime_v3/calibration.py").read_text()
    runner_source = (ROOT / "core/runtime_v3/runner.py").read_text()
    script_source = (ROOT / "scripts/runtime_v3_calibrate_primitives.py").read_text()
    assert "Executor(backend, arbiter)" in module_source
    assert "LiberoPrimitiveBackend(environment, controller, arbiter)" in module_source
    assert "self.executor.execute(action)" in runner_source
    assert "env.step(" not in script_source
    assert "environment.step(" not in script_source
    assert "run_calibration_trial(" in script_source
