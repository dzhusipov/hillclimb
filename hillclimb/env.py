"""Gymnasium environment wrapping the real Hill Climb Racing 2 game."""

from __future__ import annotations

import subprocess
import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hillclimb.capture import ScreenCapture, create_capture
from hillclimb.config import cfg
from hillclimb.controller import ADBConnectionError, ADBController, Action
from hillclimb.memory_reader import MemoryReader
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

    Reward v2:
        +1.0/step alive bonus (main dense signal)
        +0.5 * distance_delta (OCR, clamped 5m)
        +0.05 * rpm (small speed bonus)
        +2.0 fuel pickup
        crash: -2.0 + dist*0.05 (proportional)
        fuel-out: -1.0 + dist*0.05
        results: +dist*0.05

    Action repeat: each step repeats the action 3 times (frame skip).
    Grace period: first 15 steps cannot terminate (vision false positives).
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
                low=np.array([0, 0, 0, -1, -100, 0, -50, -1, -1, -1, -1], dtype=np.float32),
                high=np.array([1, 10000, 1, 1, 100, 300, 50, 1, 1, 1, 1], dtype=np.float32),
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
            stale_timeout=5.0,  # reduce screencap fallbacks (less emulator CPU spikes)
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
            save_nav_frames=True,
        )

        self._mem_reader: MemoryReader | None = None
        self._mem_scan_thread: threading.Thread | None = None
        self._last_car_state = None  # CarState from MemoryReader

        self._prev_state: VisionState | None = None
        self._step_count = 0
        self._max_distance_m = 0.0
        # Tracking for vector obs
        self._prev_distance_m = 0.0
        self._prev_fuel = 1.0
        self._last_progress_time = 0.0
        self._episode_start_time = 0.0
        # Grace period: ignore terminal states for first N steps (vision false positives)
        self._grace_period = 15
        # Action repeat (frame skip): repeat each action N times
        self._action_repeat = cfg.action_repeat

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

        # Stop previous memory reader (if any) before starting new episode
        if self._mem_reader is not None:
            self._mem_reader.stop()
            self._mem_reader = None
        self._last_car_state = None

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

        # Poll until we are in RACING (up to 5 attempts, ~2.5s)
        frame = None
        state = None
        for attempt in range(5):
            time.sleep(0.5)
            try:
                frame = self._capture.capture()
                state = self._vision.analyze(frame)
                if state.game_state == GameState.RACING:
                    break
            except (RuntimeError, Exception):
                frame = None
                state = None

        if frame is None or state is None:
            self._prev_state = None
            self._step_count = 0
            self._max_distance_m = 0.0
            self._prev_distance_m = 0.0
            self._prev_fuel = 1.0
            self._episode_start_time = time.time()
            self._last_progress_time = time.time()
            return self._zero_obs(), {}

        self._prev_state = state
        self._step_count = 0
        self._max_distance_m = 0.0
        # Reset to 0 — don't carry stale OCR values from previous episode
        self._prev_distance_m = 0.0
        self._prev_fuel = state.fuel
        self._episode_start_time = time.time()
        self._last_progress_time = time.time()

        # Create MemoryReader — scan will be triggered from _single_step
        # at grace_period when car is confirmed moving.
        self._mem_reader = MemoryReader(container=self._container_name)

        return self._build_obs(frame, state), {}

    def step(
        self, action: int | np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute action with action repeat (frame skip).

        Repeats the same action `_action_repeat` times, accumulating reward.
        Returns the observation from the last sub-step.
        """
        total_reward = 0.0
        obs = self._prev_obs if hasattr(self, '_prev_obs') else self._zero_obs()
        terminated = False
        truncated = False
        info: dict = {}

        for repeat_i in range(self._action_repeat):
            obs_i, reward_i, terminated_i, truncated_i, info_i = self._single_step(action)
            total_reward += reward_i
            obs = obs_i
            info = info_i
            terminated = terminated_i
            truncated = truncated_i
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

    def _single_step(
        self, action: int | np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute one atomic step (action + capture + reward)."""
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

        # MemoryReader: launch background scan early (consensus doesn't need movement)
        if (self._mem_reader is not None
                and not self._mem_reader.is_active
                and self._step_count == 3
                and state.game_state == GameState.RACING):
            reader = self._mem_reader

            def _bg_scan():
                if self._mem_reader is not reader:
                    return
                if reader.scan(timeout=15):
                    print(f"[ENV {self._serial}] MemoryReader attached!", flush=True)
                else:
                    print(f"[ENV {self._serial}] MemoryReader scan failed — OCR fallback", flush=True)
                    if self._mem_reader is reader:
                        self._mem_reader = None

            self._mem_scan_thread = threading.Thread(target=_bg_scan, daemon=True)
            self._mem_scan_thread.start()

        # MemoryReader: override OCR distance with memory-based distance
        if self._mem_reader is not None and self._mem_reader.is_active:
            car_state = self._mem_reader.read()
            if car_state.valid:
                state.distance_m = car_state.distance
                self._last_car_state = car_state

        # Mid-race interruption: any non-RACING state that isn't a terminal
        # state means we left the race (OFFLINE popup, CAPTCHA, menu, etc.)
        _interrupt_states = (
            GameState.CAPTCHA,
            GameState.DOUBLE_COINS_POPUP,
            GameState.MAIN_MENU,
            GameState.VEHICLE_SELECT,
        )
        if state.game_state in _interrupt_states:
            print(f"  [ENV] Mid-race interrupt: {state.game_state.name} — navigating back...")
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

        # Episode termination — only explicit crash/end states
        is_terminal = state.game_state in (
            GameState.DRIVER_DOWN,
            GameState.TOUCH_TO_CONTINUE,
            GameState.RESULTS,
        )

        # Grace period: suppress termination AND crash penalty during first N steps
        # (vision false positives on first frames after race start)
        in_grace = self._step_count < self._grace_period
        if is_terminal and in_grace:
            # Treat false terminal as RACING: give alive bonus instead of crash penalty
            reward = 1.0
            terminated = False
        else:
            reward = self._compute_reward(self._prev_state, state, self._max_distance_m)
            terminated = is_terminal

        obs = self._build_obs(frame, state)

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

        # Extended physics from MemoryReader (or neutral defaults)
        if self._last_car_state and self._last_car_state.valid:
            cs = self._last_car_state
            vel_raw = cs.vel_raw
            sin_rot, cos_rot = cs.sin_rot, cs.cos_rot
            sin_tilt, cos_tilt = cs.sin_tilt, cs.cos_tilt
        else:
            vel_raw = 0.0
            sin_rot, cos_rot = 0.0, 1.0
            sin_tilt, cos_tilt = 0.0, 1.0

        vector = np.array([
            state.fuel,
            state.distance_m,
            speed,
            fuel_delta,
            distance_delta,
            time_since_progress,
            vel_raw,
            sin_rot,
            cos_rot,
            sin_tilt,
            cos_tilt,
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
            "vector": np.zeros(11, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Container restart
    # ------------------------------------------------------------------

    def _restart_container(self) -> None:
        """Restart the Docker container and reconnect ADB."""
        if self._mem_reader is not None:
            self._mem_reader.stop()
            self._mem_reader = None
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
            stale_timeout=5.0,
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
            save_nav_frames=True,
        )
        self._controller.shell("am start -n com.fingersoft.hcr2/.AppActivity")
        time.sleep(8)
        print(f"[ENV {self._serial}] Recovery complete")

    def close(self) -> None:
        if self._mem_reader is not None:
            self._mem_reader.stop()
            self._mem_reader = None
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
        max_distance_m: float = 0.0,
    ) -> float:
        """Reward v2: alive bonus is the primary dense signal.

        Components:
            +1.0/step alive (RACING)
            +0.5 * distance_delta (OCR, clamped 5m)
            +0.05 * rpm (small speed bonus)
            +2.0 fuel pickup
            DRIVER_DOWN: -2.0 + dist*0.05 (proportional — far crash less bad)
            TOUCH_TO_CONTINUE: -1.0 + dist*0.05 (fuel-out, not crash)
            RESULTS: +dist*0.05 (completed episode bonus)
        """
        dist = max_distance_m

        # --- Terminal states ---
        if curr.game_state == GameState.DRIVER_DOWN:
            return -2.0 + dist * 0.05

        if curr.game_state == GameState.TOUCH_TO_CONTINUE:
            return -1.0 + dist * 0.05

        if curr.game_state == GameState.RESULTS:
            return dist * 0.05

        if curr.game_state != GameState.RACING:
            return 0.0

        # --- RACING: dense reward ---
        reward = 1.0  # alive bonus

        # Distance delta (OCR)
        if prev is not None and prev.distance_m > 0 and curr.distance_m > prev.distance_m:
            delta = curr.distance_m - prev.distance_m
            delta = min(delta, 5.0)
            reward += 0.5 * delta

        # Speed bonus (RPM proxy)
        reward += 0.05 * curr.rpm

        # Fuel pickup bonus
        if prev is not None and curr.fuel > prev.fuel + 0.02:
            reward += 2.0

        return reward
