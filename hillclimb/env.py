"""Gymnasium environment wrapping the real Hill Climb Racing 2 game."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hillclimb.capture import ScreenCapture
from hillclimb.config import cfg
from hillclimb.controller import ADBController, Action
from hillclimb.navigator import Navigator
from hillclimb.vision import GameState, VisionAnalyzer, VisionState


class HillClimbEnv(gym.Env):
    """Gymnasium-compatible environment for Hill Climb Racing 2.

    Observation: 8-dim float32 vector
        [fuel, rpm, boost, tilt_norm, terrain_slope_norm,
         airborne, speed_estimate, distance_norm]

    Action: Discrete(3)
        0 = nothing, 1 = gas, 2 = brake

    Reward:
        +0.1 * distance_delta_m  (actual metres gained)
        -10.0 if crashed (DRIVER_DOWN)
        -0.1 * fuel_consumed
        +0.5 if moving with fuel remaining
        -0.3 if extreme tilt (>45 deg)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode

        self._capture = ScreenCapture()
        self._vision = VisionAnalyzer()
        self._controller = ADBController()
        self._navigator = Navigator(
            self._controller, self._capture, self._vision,
        )

        self._prev_state: VisionState | None = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        ok = self._navigator.restart_game(timeout=20.0)
        if not ok:
            time.sleep(2.0)
            self._navigator.restart_game(timeout=20.0)

        time.sleep(0.5)

        frame = self._capture.grab()
        state = self._vision.analyze(frame)
        self._prev_state = state
        self._step_count = 0

        return state.to_array(), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        self._controller.execute(Action(action))
        time.sleep(cfg.loop_interval_sec)

        frame = self._capture.grab()
        state = self._vision.analyze(frame)

        reward = self._compute_reward(self._prev_state, state, action)

        terminated = state.game_state in (
            GameState.DRIVER_DOWN,
            GameState.TOUCH_TO_CONTINUE,
            GameState.RESULTS,
        )
        truncated = False

        if state.fuel <= 0.01 and state.speed_estimate < 0.05:
            terminated = True

        self._prev_state = state

        info: dict = {
            "game_state": state.game_state.name,
            "fuel": state.fuel,
            "distance_m": state.distance_m,
            "step": self._step_count,
        }

        # Capture results data if available
        if state.game_state == GameState.RESULTS:
            info["results_coins"] = state.results_coins
            info["results_distance_m"] = state.results_distance_m

        if self.render_mode == "human":
            import cv2
            debug = self._vision.draw_debug(frame, state)
            cv2.imshow("hillclimb-env", debug)
            cv2.waitKey(1)

        return state.to_array(), reward, terminated, truncated, info

    def close(self) -> None:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reward(
        prev: VisionState | None,
        curr: VisionState,
        action: int,
    ) -> float:
        reward = 0.0

        # Crash penalty
        if curr.game_state == GameState.DRIVER_DOWN:
            return -10.0

        if curr.game_state != GameState.RACING:
            return 0.0

        # Primary: distance gained (via OCR)
        if prev is not None and prev.distance_m > 0 and curr.distance_m > prev.distance_m:
            delta = curr.distance_m - prev.distance_m
            reward += 0.1 * delta
        else:
            # Fallback: speed as proxy when OCR not available
            reward += 1.0 * curr.speed_estimate

        # Fuel efficiency penalty
        if prev is not None:
            fuel_used = max(0.0, prev.fuel - curr.fuel)
            reward -= 0.1 * fuel_used

        # Bonus for moving with fuel
        if curr.fuel > 0.0 and curr.speed_estimate > 0.1:
            reward += 0.5

        # Stability: penalise extreme tilt
        if abs(curr.tilt) > 45:
            reward -= 0.3

        return reward
