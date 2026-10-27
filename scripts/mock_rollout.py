#!/usr/bin/env python3
"""Exercise the robot rollout boundary without hardware or model weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_rl_real.rollout import RobotFeedback, run_episode


class MockRobot:
    def __init__(self) -> None:
        self.position = 0.0
        self.stopped = False

    def reset(self) -> None:
        self.position = 0.0

    def observe(self) -> dict:
        return {"observation/state": np.array([self.position], dtype=np.float32)}

    def validate_action(self, action: np.ndarray) -> None:
        if action.shape != (1,) or not 0.0 <= action[0] <= 0.1:
            raise ValueError("mock action outside [0, 0.1]")

    def step(self, action: np.ndarray) -> RobotFeedback:
        self.position += float(action[0])
        done = self.position >= 0.3
        return RobotFeedback(reward=1.0 if done else 0.0, done=done)

    def stop(self) -> None:
        self.stopped = True


class MockPolicy:
    def infer(self, observation: dict) -> np.ndarray:
        assert observation["prompt"] == "move to target"
        return np.full((5, 1), 0.05, dtype=np.float32)


def main() -> None:
    robot = MockRobot()
    steps: list[int] = []
    result = run_episode(
        robot, MockPolicy(), "move to target", max_action_steps=10,
        execute_horizon=3, on_step=lambda step, *_: steps.append(step),
    )
    assert robot.stopped and result.done and result.action_steps == len(steps)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
