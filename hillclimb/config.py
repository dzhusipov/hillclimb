"""Configuration: button coordinates, ROI regions, thresholds.

All coordinates are in the scrcpy window pixel space.
Run `python -m hillclimb.calibrate` to recalibrate for your device.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Rect:
    """Rectangle defined by top-left corner (x, y) and size (w, h)."""
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 20


@dataclass
class Point:
    x: int = 0
    y: int = 0


# ---------------------------------------------------------------------------
# Default configuration (Galaxy S-series ~1080x2400 in landscape via scrcpy)
# Adjust via calibrate.py or by editing config.json
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # -- scrcpy window capture ------------------------------------------------
    scrcpy_window_title: str = "scrcpy"

    # -- ADB ------------------------------------------------------------------
    adb_device: str = ""  # empty = first connected device

    # -- Button positions (in Android screen coords, landscape) ---------------
    gas_button: Point = field(default_factory=lambda: Point(x=2200, y=900))
    brake_button: Point = field(default_factory=lambda: Point(x=200, y=900))

    # -- Menu buttons ---------------------------------------------------------
    play_button: Point = field(default_factory=lambda: Point(x=1200, y=800))
    retry_button: Point = field(default_factory=lambda: Point(x=1200, y=700))

    # -- Gauge ROIs (in scrcpy window pixel coords) ---------------------------
    fuel_gauge_roi: Rect = field(default_factory=lambda: Rect(x=50, y=10, w=200, h=20))
    # RPM and boost may not always be visible; these are optional
    rpm_gauge_roi: Rect = field(default_factory=lambda: Rect(x=300, y=10, w=150, h=20))
    boost_gauge_roi: Rect = field(default_factory=lambda: Rect(x=500, y=10, w=150, h=20))

    # -- Vehicle ROI (where the car typically sits) ---------------------------
    vehicle_roi: Rect = field(default_factory=lambda: Rect(x=200, y=300, w=400, h=200))

    # -- Terrain ROI (ground line area) ---------------------------------------
    terrain_roi: Rect = field(default_factory=lambda: Rect(x=0, y=400, w=800, h=200))

    # -- HSV ranges for gauge bar color detection ----------------------------
    # Default: greenish fuel bar — tune after calibration
    fuel_hsv_lower: list[int] = field(default_factory=lambda: [30, 80, 80])
    fuel_hsv_upper: list[int] = field(default_factory=lambda: [90, 255, 255])

    # -- Thresholds -----------------------------------------------------------
    crash_template_threshold: float = 0.7
    menu_template_threshold: float = 0.7
    tilt_smoothing_window: int = 5
    gauge_smoothing_window: int = 5

    # -- Game loop timing ----------------------------------------------------
    loop_interval_sec: float = 0.1  # target 10 decisions/sec
    action_hold_ms: int = 100       # ADB swipe hold duration

    # -- Logging --------------------------------------------------------------
    log_dir: str = "logs"
    log_frame_every_n: int = 30     # save PNG every N frames

    # -- RL training ----------------------------------------------------------
    model_dir: str = "models"
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    total_timesteps: int = 200_000

    # -- Template paths -------------------------------------------------------
    template_dir: str = str(Path(__file__).resolve().parent.parent / "templates")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str | None = None) -> None:
        path = Path(path or CONFIG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = Path(path or CONFIG_PATH)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        for key, value in data.items():
            if not hasattr(cfg, key):
                continue
            field_val = getattr(cfg, key)
            if isinstance(field_val, Point):
                setattr(cfg, key, Point(**value))
            elif isinstance(field_val, Rect):
                setattr(cfg, key, Rect(**value))
            else:
                setattr(cfg, key, value)
        return cfg


# Singleton — import and use everywhere
cfg = Config.load()
