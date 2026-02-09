"""Data logger: write state + action to CSV, save debug frames."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

from hillclimb.config import cfg
from hillclimb.controller import Action
from hillclimb.vision import VisionState


class Logger:
    """Asynchronously logs game state and actions."""

    def __init__(self, session_name: str | None = None) -> None:
        ts = session_name or time.strftime("%Y%m%d_%H%M%S")
        self._dir = Path(cfg.log_dir) / ts
        self._dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self._dir / "log.csv"
        self._frame_count = 0
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "timestamp", "game_state", "fuel", "rpm", "boost",
            "tilt", "terrain_slope", "airborne", "speed_estimate", "action",
        ])

    def log(
        self,
        state: VisionState,
        action: Action | int,
        frame: np.ndarray | None = None,
    ) -> None:
        """Log one step. Optionally save a frame PNG (async)."""
        self._frame_count += 1

        self._writer.writerow([
            time.time(),
            state.game_state.name,
            f"{state.fuel:.4f}",
            f"{state.rpm:.4f}",
            f"{state.boost:.4f}",
            f"{state.tilt:.2f}",
            f"{state.terrain_slope:.2f}",
            int(state.airborne),
            f"{state.speed_estimate:.4f}",
            int(action),
        ])

        # Flush periodically
        if self._frame_count % 10 == 0:
            self._csv_file.flush()

        # Save frame every N steps
        if frame is not None and self._frame_count % cfg.log_frame_every_n == 0:
            path = self._dir / f"frame_{self._frame_count:06d}.png"
            Thread(target=cv2.imwrite, args=(str(path), frame), daemon=True).start()

    def close(self) -> None:
        self._csv_file.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
