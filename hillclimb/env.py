"""Gymnasium environment wrapping the real Hill Climb Racing 2 game."""

from __future__ import annotations

import subprocess
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hillclimb.capture import ScreenCapture, create_capture
from hillclimb.config import cfg
from hillclimb.controller import ADBConnectionError, ADBController, Action
from hillclimb.navigator import Navigator
from hillclimb.vision import GameState, VisionAnalyzer, VisionState, extract_game_field


class HillClimbEnv(gym.Env):
    """Gymnasium-compatible environment for Hill Climb Racing 2.

    Observation: Dict
        "image": (84, 84, 1) uint8 — grayscale game field crop
        "vector": (6,) float32 — [fuel, distance_m, speed, fuel_delta,
                                   distance_delta, time_since_progress]

    Action: Discrete(3)
        0 = nothing, 1 = gas, 2 = brake

    Reward:
        +distance_delta * 1.0 (main signal)
        +speed * 0.1 (speed bonus)
        -5.0 if crashed
        +0.5 if fuel picked up
        -0.05 * (0.2 - fuel) / 0.2 when fuel < 0.2
        -0.1 if stalled (no progress while RACING)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        adb_serial: str = "localhost:5555",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        size = cfg.game_field_size
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(size, size, 1), dtype=np.uint8),
            "vector": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, -1.0, -100.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 10000.0, 1.0, 1.0, 100.0, 300.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode

        self._serial = adb_serial
        self._capture = create_capture(
            adb_serial=adb_serial,
            backend=cfg.capture_backend,
            max_fps=cfg.scrcpy_max_fps,
            max_size=cfg.scrcpy_max_size,
            bitrate=cfg.scrcpy_bitrate,
            server_jar=cfg.scrcpy_server_jar,
            dashboard_url=cfg.dashboard_url,
        )
        self._vision = VisionAnalyzer()
        self._controller = ADBController(
            adb_serial=adb_serial,
            gas_x=cfg.gas_button.x,
            gas_y=cfg.gas_button.y,
            brake_x=cfg.brake_button.x,
            brake_y=cfg.brake_button.y,
            action_hold_ms=cfg.action_hold_ms,
        )
        # Container name for watchdog restart
        port = int(adb_serial.split(":")[-1])
        self._container_name = f"hcr2-{port - 5555}"
        self._env_index = port - 5555

        self._navigator = Navigator(
            self._controller, self._capture, self._vision,
            env_index=self._env_index,
        )

        self._prev_state: VisionState | None = None
        self._step_count = 0
        self._max_distance_m = 0.0
        # Tracking for vector obs
        self._prev_distance_m = 0.0
        self._prev_fuel = 1.0
        self._last_progress_time = 0.0
        self._episode_start_time = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        try:
            ok = self._navigator.restart_game(timeout=20.0)
            if not ok:
                time.sleep(2.0)
                self._navigator.restart_game(timeout=20.0)
        except (ADBConnectionError, RuntimeError) as e:
            print(f"[ENV {self._serial}] Error during reset: {e} — restarting container...")
            self._restart_container()
            try:
                self._navigator.restart_game(timeout=30.0)
            except Exception:
                print(f"[ENV {self._serial}] Still failing after container restart")

        time.sleep(0.5)

        try:
            frame = self._capture.capture()
        except (RuntimeError, Exception):
            self._prev_state = None
            self._step_count = 0
            self._max_distance_m = 0.0
            self._prev_distance_m = 0.0
            self._prev_fuel = 1.0
            self._episode_start_time = time.time()
            self._last_progress_time = time.time()
            return self._zero_obs(), {}

        state = self._vision.analyze(frame)
        self._prev_state = state
        self._step_count = 0
        self._max_distance_m = 0.0
        self._prev_distance_m = state.distance_m
        self._prev_fuel = state.fuel
        self._episode_start_time = time.time()
        self._last_progress_time = time.time()

        return self._build_obs(frame, state), {}

    def step(
        self, action: int | np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        self._step_count += 1

        action_enum = Action(int(action))

        # Fire action in background — overlaps with capture below
        try:
            self._controller.execute_async(action_enum, duration_ms=cfg.action_hold_ms)
        except (ADBConnectionError, RuntimeError):
            print(f"[ENV {self._serial}] ADB error during step — ending episode")
            obs = self._prev_obs if hasattr(self, '_prev_obs') else self._zero_obs()
            return obs, -5.0, True, False, {
                "game_state": "ADB_DISCONNECTED",
                "max_distance_m": self._max_distance_m,
                "step": self._step_count,
            }

        # Capture while action is executing (action_hold=200ms < capture≈300ms,
        # so action fully completes before capture data returns)
        try:
            frame = self._capture.capture()
        except (RuntimeError, Exception) as e:
            print(f"[ENV {self._serial}] Capture failed: {e} — ending episode")
            obs = self._prev_obs if hasattr(self, '_prev_obs') else self._zero_obs()
            return obs, -5.0, True, False, {
                "game_state": "CAPTURE_FAILED",
                "max_distance_m": self._max_distance_m,
                "step": self._step_count,
            }

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
                obs = self._prev_obs if hasattr(self, '_prev_obs') else self._zero_obs()
                return obs, 0.0, True, False, {
                    "game_state": state.game_state.name,
                    "max_distance_m": self._max_distance_m,
                    "step": self._step_count,
                }

        reward = self._compute_reward(self._prev_state, state)
        obs = self._build_obs(frame, state)

        # Episode termination — only explicit crash/end states
        terminated = state.game_state in (
            GameState.DRIVER_DOWN,
            GameState.TOUCH_TO_CONTINUE,
            GameState.RESULTS,
        )
        truncated = False

        # UNKNOWN — try to navigate back, don't kill episode
        if state.game_state == GameState.UNKNOWN:
            self._navigator.ensure_racing(timeout=5.0)

        # Track max distance (filter OCR jumps > 100m)
        if state.game_state == GameState.RACING and state.distance_m > self._max_distance_m:
            jump = state.distance_m - self._max_distance_m
            if jump < 100 or self._max_distance_m == 0:
                self._max_distance_m = state.distance_m

        self._prev_state = state
        self._prev_obs = obs

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

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, frame: np.ndarray, state: VisionState) -> dict:
        """Build Dict observation from frame and vision state."""
        image = extract_game_field(frame)

        # Compute deltas
        distance_delta = state.distance_m - self._prev_distance_m
        # Filter OCR glitches
        if abs(distance_delta) > 100:
            distance_delta = 0.0
        fuel_delta = state.fuel - self._prev_fuel

        # Track progress time
        if distance_delta > 0.5:
            self._last_progress_time = time.time()
        time_since_progress = time.time() - self._last_progress_time

        # Speed: use RPM as proxy (more reliable than optical flow)
        speed = state.rpm

        vector = np.array([
            state.fuel,
            state.distance_m,
            speed,
            fuel_delta,
            distance_delta,
            time_since_progress,
        ], dtype=np.float32)

        # Update tracking
        self._prev_distance_m = state.distance_m
        self._prev_fuel = state.fuel

        return {"image": image, "vector": vector}

    def _zero_obs(self) -> dict:
        """Return zeroed observation for error cases."""
        size = cfg.game_field_size
        return {
            "image": np.zeros((size, size, 1), dtype=np.uint8),
            "vector": np.zeros(6, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Container restart
    # ------------------------------------------------------------------

    def _restart_container(self) -> None:
        """Restart the Docker container and reconnect ADB."""
        print(f"[ENV {self._serial}] Restarting container {self._container_name}...")
        try:
            subprocess.run(
                ["docker", "restart", self._container_name],
                timeout=60, capture_output=True,
            )
        except Exception as e:
            print(f"[ENV {self._serial}] docker restart failed: {e}")
            return
        print(f"[ENV {self._serial}] Container restarted, waiting for boot...")
        time.sleep(15)
        # Reconnect
        self._capture = create_capture(
            adb_serial=self._serial,
            backend=cfg.capture_backend,
            max_fps=cfg.scrcpy_max_fps,
            max_size=cfg.scrcpy_max_size,
            bitrate=cfg.scrcpy_bitrate,
            server_jar=cfg.scrcpy_server_jar,
            dashboard_url=cfg.dashboard_url,
        )
        self._controller = ADBController(
            adb_serial=self._serial,
            gas_x=cfg.gas_button.x,
            gas_y=cfg.gas_button.y,
            brake_x=cfg.brake_button.x,
            brake_y=cfg.brake_button.y,
            action_hold_ms=cfg.action_hold_ms,
        )
        self._navigator = Navigator(
            self._controller, self._capture, self._vision,
            env_index=self._env_index,
        )
        self._controller.shell("am start -n com.fingersoft.hcr2/.AppActivity")
        time.sleep(8)
        print(f"[ENV {self._serial}] Recovery complete")

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
        if curr.game_state in (GameState.DRIVER_DOWN, GameState.TOUCH_TO_CONTINUE):
            return -5.0

        if curr.game_state != GameState.RACING:
            return 0.0

        # === Primary: distance gained (via OCR) ===
        if prev is not None and prev.distance_m > 0 and curr.distance_m > prev.distance_m:
            delta = curr.distance_m - prev.distance_m
            if delta < 100:  # filter OCR glitches
                reward += 1.0 * delta

        # === Speed bonus ===
        reward += 0.1 * curr.rpm

        # === Fuel pickup bonus ===
        if prev is not None and curr.fuel > prev.fuel + 0.02:
            reward += 0.5

        # === Low fuel penalty ===
        if curr.fuel < 0.2:
            reward -= 0.05 * (0.2 - curr.fuel) / 0.2

        # === Stall penalty (no distance gain while racing) ===
        if prev is not None and prev.game_state == GameState.RACING:
            delta = curr.distance_m - prev.distance_m
            if abs(delta) < 0.5:
                reward -= 0.1

        return reward
