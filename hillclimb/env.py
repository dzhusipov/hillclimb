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

# Длительности удержания (мс) — модель выбирает одну из них
HOLD_DURATIONS = [200, 500, 1000, 2000, 3000]

# Действия: 0=gas, 1=brake (без NOTHING — газ нужен почти всегда)
ACTION_MAP = [Action.GAS, Action.BRAKE]


class HillClimbEnv(gym.Env):
    """Gymnasium-compatible environment for Hill Climb Racing 2.

    Observation: 8-dim float32 vector
        [fuel, rpm, boost, tilt_norm, terrain_slope_norm,
         airborne, speed_estimate, distance_norm]

    Action: MultiDiscrete([2, 5])
        dim 0: тип действия — 0=gas, 1=brake
        dim 1: длительность удержания — индекс в HOLD_DURATIONS
               [200ms, 500ms, 1000ms, 2000ms, 3000ms]

    Reward:
        +0.1 * distance_delta_m  (actual metres gained)
        -10.0 if crashed (DRIVER_DOWN)
        +0.5 if moving with fuel remaining
        -0.3 if extreme tilt (>45 deg)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32,
        )
        # MultiDiscrete: [тип_действия (2: gas/brake), длительность (5)]
        self.action_space = spaces.MultiDiscrete([2, len(HOLD_DURATIONS)])
        self.render_mode = render_mode

        self._capture = ScreenCapture()
        self._vision = VisionAnalyzer()
        self._controller = ADBController()
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
            print("[ENV] ADB connection lost during reset — waiting 10s for recovery...")
            time.sleep(10.0)
            try:
                self._controller._verify_connection()
                self._navigator.restart_game(timeout=20.0)
            except Exception:
                raise ADBConnectionError(
                    "ADB connection lost and could not recover. "
                    "Check scrcpy and USB connection."
                )

        time.sleep(0.5)

        frame = self._capture.grab()
        state = self._vision.analyze(frame)
        self._prev_state = state
        self._step_count = 0
        self._max_distance_m = 0.0
        self._zero_speed_count = 0

        return state.to_array(), {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # Декодируем MultiDiscrete: [тип (0=gas,1=brake), длительность]
        action_type = int(action[0])
        hold_ms = HOLD_DURATIONS[int(action[1])]
        actual_action = ACTION_MAP[action_type]

        # Удерживаем газ/тормоз на выбранную длительность
        try:
            self._controller.execute(actual_action, duration_ms=hold_ms)
        except ADBConnectionError:
            print("[ENV] ADB connection lost during step — ending episode")
            obs = self._prev_state.to_array() if self._prev_state else np.zeros(8, dtype=np.float32)
            return obs, -10.0, True, False, {
                "game_state": "ADB_DISCONNECTED",
                "max_distance_m": self._max_distance_m,
                "step": self._step_count,
            }

        frame = self._capture.grab()
        state = self._vision.analyze(frame)

        # Mid-race popup: only CAPTCHA and DOUBLE_COINS can interrupt racing
        # without ending the episode. Everything else = race ended → terminate.
        if state.game_state in (GameState.CAPTCHA, GameState.DOUBLE_COINS_POPUP):
            print(f"  [ENV] Mid-race popup: {state.game_state.name} — navigating back...")
            ok = self._navigator.ensure_racing(timeout=30.0)
            if ok:
                frame = self._capture.grab()
                state = self._vision.analyze(frame)
            else:
                print(f"  [ENV] Could not recover from {state.game_state.name} — ending episode")
                obs = self._prev_state.to_array() if self._prev_state else np.zeros(8, dtype=np.float32)
                return obs, 0.0, True, False, {
                    "game_state": state.game_state.name,
                    "max_distance_m": self._max_distance_m,
                    "step": self._step_count,
                }

        reward = self._compute_reward(self._prev_state, state, actual_action)

        terminated = state.game_state not in (
            GameState.RACING,
            GameState.CAPTCHA,
            GameState.DOUBLE_COINS_POPUP,
        )
        truncated = False

        if state.fuel <= 0.01 and state.speed_estimate < 0.05:
            terminated = True

        # Stuck detection: if speed≈0 for 5+ steps, car is stuck → end episode
        if state.game_state == GameState.RACING and state.speed_estimate < 0.02:
            self._zero_speed_count += 1
            if self._zero_speed_count >= 5:
                print(f"  [ENV] Car stuck (speed=0 for {self._zero_speed_count} steps) — ending episode")
                terminated = True
        else:
            self._zero_speed_count = 0

        # Трекаем максимальную дистанцию из RACING
        # Фильтр: скачок > 100м за один шаг = OCR ошибка, игнорируем
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
            "hold_ms": hold_ms,
        }

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
        action: Action,
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
        # Relative tilt = car tilt minus terrain slope
        # If terrain is 30° and car is 30°, relative tilt ≈ 0 (stable)
        # If terrain is 0° and car is 30°, relative tilt = 30° (danger)
        relative_tilt = abs(curr.tilt - curr.terrain_slope)
        if relative_tilt < 15:
            reward += 0.3    # stable — parallel to ground
        elif relative_tilt > 30:
            reward -= 0.5    # danger zone — about to flip
        if relative_tilt > 60:
            reward -= 1.0    # critical — almost crashed

        return reward
