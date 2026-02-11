"""Gymnasium environment wrapping the real Hill Climb Racing 2 game."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hillclimb.capture import ScreenCapture
from hillclimb.config import cfg
from hillclimb.controller import ADBConnectionError, ADBController, Action
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
        +0.1 * distance_delta_m  (actual metres gained via OCR)
        -10.0 if crashed (DRIVER_DOWN)
        +0.5 if moving with fuel remaining
        tilt-based balance bonus/penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        adb_serial: str = "localhost:5555",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode

        self._serial = adb_serial
        self._capture = ScreenCapture(adb_serial=adb_serial)
        self._vision = VisionAnalyzer()
        self._controller = ADBController(
            adb_serial=adb_serial,
            gas_x=cfg.gas_button.x,
            gas_y=cfg.gas_button.y,
            brake_x=cfg.brake_button.x,
            brake_y=cfg.brake_button.y,
            action_hold_ms=cfg.action_hold_ms,
        )
        self._navigator = Navigator(
            self._controller, self._capture, self._vision,
        )

        self._prev_state: VisionState | None = None
        self._step_count = 0
        self._max_distance_m = 0.0
        self._zero_speed_count = 0

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

        try:
            ok = self._navigator.restart_game(timeout=20.0)
            if not ok:
                time.sleep(2.0)
                self._navigator.restart_game(timeout=20.0)
        except ADBConnectionError:
            print(f"[ENV {self._serial}] ADB connection lost during reset — waiting 10s...")
            time.sleep(10.0)
            try:
                self._navigator.restart_game(timeout=20.0)
            except Exception:
                raise ADBConnectionError(
                    f"ADB connection lost on {self._serial} and could not recover."
                )

        time.sleep(0.5)

        frame = self._capture.capture()
        state = self._vision.analyze(frame)
        self._prev_state = state
        self._step_count = 0
        self._max_distance_m = 0.0
        self._zero_speed_count = 0

        return state.to_array(), {}

    def step(
        self, action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        action_enum = Action(int(action))

        # Execute action
        try:
            self._controller.execute(action_enum, duration_ms=cfg.action_hold_ms)
        except ADBConnectionError:
            print(f"[ENV {self._serial}] ADB connection lost during step — ending episode")
            obs = self._prev_state.to_array() if self._prev_state else np.zeros(8, dtype=np.float32)
            return obs, -10.0, True, False, {
                "game_state": "ADB_DISCONNECTED",
                "max_distance_m": self._max_distance_m,
                "step": self._step_count,
            }

        frame = self._capture.capture()
        state = self._vision.analyze(frame)

        # Mid-race popup: CAPTCHA and DOUBLE_COINS can interrupt racing
        if state.game_state in (GameState.CAPTCHA, GameState.DOUBLE_COINS_POPUP):
            print(f"  [ENV] Mid-race popup: {state.game_state.name} — navigating back...")
            ok = self._navigator.ensure_racing(timeout=30.0)
            if ok:
                frame = self._capture.capture()
                state = self._vision.analyze(frame)
            else:
                print(f"  [ENV] Could not recover from {state.game_state.name} — ending episode")
                obs = self._prev_state.to_array() if self._prev_state else np.zeros(8, dtype=np.float32)
                return obs, 0.0, True, False, {
                    "game_state": state.game_state.name,
                    "max_distance_m": self._max_distance_m,
                    "step": self._step_count,
                }

        reward = self._compute_reward(self._prev_state, state)

        terminated = state.game_state not in (
            GameState.RACING,
            GameState.CAPTCHA,
            GameState.DOUBLE_COINS_POPUP,
        )
        truncated = False

        if state.fuel <= 0.01 and state.speed_estimate < 0.05:
            terminated = True

        # Stuck detection: speed≈0 for 5+ steps → end episode
        if state.game_state == GameState.RACING and state.speed_estimate < 0.02:
            self._zero_speed_count += 1
            if self._zero_speed_count >= 5:
                print(f"  [ENV] Car stuck (speed=0 for {self._zero_speed_count} steps) — ending episode")
                terminated = True
        else:
            self._zero_speed_count = 0

        # Track max distance (filter OCR jumps > 100m)
        if state.game_state == GameState.RACING and state.distance_m > self._max_distance_m:
            jump = state.distance_m - self._max_distance_m
            if jump < 100 or self._max_distance_m == 0:
                self._max_distance_m = state.distance_m

        self._prev_state = state

        info: dict = {
            "game_state": state.game_state.name,
            "fuel": state.fuel,
            "distance_m": state.distance_m,
            "max_distance_m": self._max_distance_m,
            "step": self._step_count,
        }

        if self.render_mode == "human":
            import cv2
            debug = self._vision.draw_debug(frame, state)
            cv2.imshow(f"hillclimb-env-{self._serial}", debug)
            cv2.waitKey(1)

        return state.to_array(), reward, terminated, truncated, info

    def close(self) -> None:
        self._capture.close()
        self._controller.close()
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
    ) -> float:
        reward = 0.0

        # Crash penalty
        if curr.game_state == GameState.DRIVER_DOWN:
            return -10.0

        if curr.game_state != GameState.RACING:
            return 0.0

        # === Primary: distance gained (via OCR) ===
        if prev is not None and prev.distance_m > 0 and curr.distance_m > prev.distance_m:
            delta = curr.distance_m - prev.distance_m
            reward += 0.1 * delta
        else:
            # Fallback: speed as proxy when OCR not available
            reward += 1.0 * curr.speed_estimate

        # === Bonus for forward movement ===
        if curr.speed_estimate > 0.1:
            reward += 0.5

        # === Tilt-based balance rewards (key for HCR2) ===
        relative_tilt = abs(curr.tilt - curr.terrain_slope)
        if relative_tilt < 15:
            reward += 0.3    # stable — parallel to ground
        elif relative_tilt > 30:
            reward -= 0.5    # danger zone — about to flip
        if relative_tilt > 60:
            reward -= 1.0    # critical — almost crashed

        return reward
