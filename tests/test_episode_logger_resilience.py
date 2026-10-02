from __future__ import annotations

import json
import shutil

from core.record.episode_logger import EpisodeLogger


def test_logger_recovers_directory_removed_before_first_frame(tmp_path) -> None:
    logger = EpisodeLogger(tmp_path, task_id=0, variant="test")
    logger.write_metadata({"research_profile": "vcr-v2", "api_key": "secret"})
    run_dir = logger.run_dir
    shutil.rmtree(run_dir)

    logger.write_plan({"subgoals": []})
    logger.write_planner_diagnostics({"route": "ok"})
    logger.close(success=False, fps=2.0)

    assert json.loads((run_dir / "metadata.json").read_text())["api_key"] == "[REDACTED]"
    assert json.loads((run_dir / "subgoals.json").read_text()) == {"subgoals": []}
    assert (run_dir / "steps.json").is_file()
