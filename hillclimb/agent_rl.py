"""RL agent wrapper: loads a trained PPO model and provides a decide() interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hillclimb.config import cfg
from hillclimb.controller import Action
from hillclimb.vision import VisionState


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

    def decide(self, state: VisionState) -> Action:
        obs = state.to_array()
        action, _ = self._model.predict(obs, deterministic=True)
        return Action(int(action))
