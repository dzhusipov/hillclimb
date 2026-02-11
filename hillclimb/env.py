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

# Discrete(7): 5 газов + 2 тормоза → рандомная политика 71% газ
# Газ — длинные нажатия для прироста, тормоз — короткие тапы для баланса
ACTIONS = [
    (Action.GAS, 500),     # 0: gas 500ms
    (Action.GAS, 1000),    # 1: gas 1s
    (Action.GAS, 2000),    # 2: gas 2s
    (Action.GAS, 3000),    # 3: gas 3s
    (Action.GAS, 5000),    # 4: gas 5s
    (Action.BRAKE, 300),   # 5: brake 300ms (баланс)
    (Action.BRAKE, 800),   # 6: brake 800ms (сильная коррекция)
]


class HillClimbEnv(gym.Env):
    """Gymnasium-compatible environment for Hill Climb Racing 2.

    Observation: 8-dim float32 vector
        [fuel, rpm, boost, tilt_norm, terrain_slope_norm,
         airborne, speed_estimate, distance_norm]

    Action: Discrete(7)
        0-4: gas с разной длительностью [500,1000,2000,3000,5000]мс
        5-6: brake для баланса [300,800]мс
        Соотношение 5:2 газ:тормоз

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
        self.action_space = spaces.Discrete(len(ACTIONS))
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
        self, action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # Декодируем Discrete(7): индекс в ACTIONS
        action_idx = int(action)
        actual_action, hold_ms = ACTIONS[action_idx]

        # === Rule-based балансировка: override при опасном крене ===
        # Крен назад (задирается перед) → тормоз (опускает перед)
        # Крен вперёд (клюёт нос) → газ (опускает зад)
        if self._prev_state is not None and self._prev_state.game_state == GameState.RACING:
            rel_tilt = self._prev_state.tilt - self._prev_state.terrain_slope
            if rel_tilt > 30:   # сильный крен назад
                actual_action = Action.BRAKE
                hold_ms = 300
            elif rel_tilt < -30:  # сильный крен вперёд
                actual_action = Action.GAS
                hold_ms = 300

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
