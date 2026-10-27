"""A small robot boundary for OpenPI action-chunk inference.

The robot adapter owns action units, coordinate frames, hard limits, timing,
reset, and emergency stop. This module never sends a command to hardware by
itself. RLinf's real-world environment remains the online learner interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class RobotFeedback:
    reward: float = 0.0
    done: bool = False
    info: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeResult:
    action_steps: int
    policy_calls: int
    reward_sum: float
    done: bool
    elapsed_seconds: float


class RobotAdapter(Protocol):
    def reset(self) -> None: ...
    def observe(self) -> Mapping[str, Any]: ...
    def validate_action(self, action: np.ndarray) -> None: ...
    def step(self, action: np.ndarray) -> RobotFeedback: ...
    def stop(self) -> None: ...


class PolicyAdapter(Protocol):
    def infer(self, observation: Mapping[str, Any]) -> np.ndarray: ...


class RemoteOpenPIPolicy:
    """Use the official lightweight WebSocket client on a robot computer."""

    def __init__(self, host: str, port: int = 8000) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=host, port=port)

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        response = self._client.infer(dict(observation))
        return np.asarray(response["actions"], dtype=np.float32)


def run_episode(
    robot: RobotAdapter,
    policy: PolicyAdapter,
    prompt: str,
    *,
    max_action_steps: int,
    execute_horizon: int,
    on_step: Callable[[int, Mapping[str, Any], np.ndarray, RobotFeedback], None] | None = None,
) -> EpisodeResult:
    """Execute action chunks with replanning and a recorder hook.

    `on_step` receives raw observations and actions so callers can persist them
    in LeRobot or another task-specific format. `robot.stop()` is always called,
    including if inference, validation, logging, or execution fails.
    """

    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_action_steps < 1 or execute_horizon < 1:
        raise ValueError("max_action_steps and execute_horizon must be positive")
    start = monotonic()
    steps = 0
    calls = 0
    reward_sum = 0.0
    done = False
    try:
        robot.reset()
        while steps < max_action_steps and not done:
            observation = dict(robot.observe())
            observation["prompt"] = prompt
            actions = np.asarray(policy.infer(observation), dtype=np.float32)
            calls += 1
            if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
                raise ValueError(f"policy must return [horizon, action_dim], got {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("policy returned NaN or infinite actions")
            for action in actions[: min(execute_horizon, max_action_steps - steps)]:
                robot.validate_action(action)
                feedback = robot.step(action)
                reward_sum += float(feedback.reward)
                if on_step:
                    on_step(steps, observation, action.copy(), feedback)
                steps += 1
                if feedback.done:
                    done = True
                    break
        return EpisodeResult(steps, calls, reward_sum, done, monotonic() - start)
    finally:
        robot.stop()
