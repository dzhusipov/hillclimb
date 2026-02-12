"""RL agent wrapper: loads a trained PPO model and provides a decide() interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hillclimb.config import cfg
from hillclimb.controller import Action
from hillclimb.vision import VisionState, extract_game_field


class RLAgent:
    """Wraps a Stable-Baselines3 PPO model for use in the game loop."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        from stable_baselines3 import PPO

        if model_path is None:
            model_path = Path(cfg.model_dir) / "ppo_hillclimb"

        model_path = Path(model_path)

        # Try to load with .zip extension
        if model_path.suffix != ".zip":
            zip_path = model_path.with_suffix(".zip")
            if zip_path.exists():
                model_path = zip_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run `python -m hillclimb.train` first."
            )

        self._model = PPO.load(str(model_path))
        print(f"Loaded RL model from {model_path}")

        # Tracking for vector obs deltas
        self._prev_distance_m = 0.0
        self._prev_fuel = 1.0

    def decide(self, state: VisionState, frame: np.ndarray | None = None) -> Action:
        """Choose action given vision state and raw frame.

        Args:
            state: VisionState from vision analyzer.
            frame: Raw BGR frame (required for CNN observation).
        """
        if frame is None:
            # Fallback: no frame → gas
            return Action.GAS

        image = extract_game_field(frame)

        # Compute deltas
        distance_delta = state.distance_m - self._prev_distance_m
        if abs(distance_delta) > 100:
            distance_delta = 0.0
        fuel_delta = state.fuel - self._prev_fuel
        speed = state.rpm

        vector = np.array([
            state.fuel,
            state.distance_m,
            speed,
            fuel_delta,
            distance_delta,
            0.0,  # time_since_progress not tracked in game_loop
        ], dtype=np.float32)

        self._prev_distance_m = state.distance_m
        self._prev_fuel = state.fuel

        obs = {"image": image, "vector": vector}
        action, _ = self._model.predict(obs, deterministic=True)
        return Action(int(action))
