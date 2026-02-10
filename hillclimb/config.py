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


@dataclass
class CircleROI:
    """Circular region defined by center (cx, cy) and radius."""
    cx: int = 0
    cy: int = 0
    radius: int = 50


# ---------------------------------------------------------------------------
# Default configuration (Redmi Note 8 Pro 2340x1080 landscape via scrcpy)
# Adjust via calibrate.py or by editing config.json
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # -- scrcpy window capture ------------------------------------------------
    scrcpy_window_title: str = "scrcpy"

    # -- ADB ------------------------------------------------------------------
    adb_path: str = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
    adb_device: str = ""  # empty = first connected device

    # -- Capture method: "adb" (native res, ~200ms) or "mss" (scrcpy window, ~5ms)
    capture_method: str = "adb"

    # -- Gas / Brake (in Android screen coords, landscape) --------------------
    gas_button: Point = field(default_factory=lambda: Point(x=2200, y=900))
    brake_button: Point = field(default_factory=lambda: Point(x=200, y=900))

    # -- Navigation buttons (2340x1080 landscape) ----------------------------
    race_button: Point = field(default_factory=lambda: Point(x=1170, y=900))
    start_button: Point = field(default_factory=lambda: Point(x=2088, y=976))
    back_button: Point = field(default_factory=lambda: Point(x=200, y=950))
    skip_button: Point = field(default_factory=lambda: Point(x=870, y=875))
    close_popup_button: Point = field(default_factory=lambda: Point(x=1170, y=600))
    retry_button: Point = field(default_factory=lambda: Point(x=180, y=1000))
    next_button: Point = field(default_factory=lambda: Point(x=2150, y=1000))
    center_screen: Point = field(default_factory=lambda: Point(x=1170, y=540))

    # -- Legacy aliases (kept for controller.py compatibility) -----------------
    play_button: Point = field(default_factory=lambda: Point(x=1170, y=900))

    # -- Dial gauge ROIs (circular, калиброваны для Redmi Note 8 Pro 2340x1080)
    rpm_dial_roi: CircleROI = field(default_factory=lambda: CircleROI(cx=920, cy=955, radius=70))
    fuel_dial_roi: CircleROI = field(default_factory=lambda: CircleROI(cx=1170, cy=975, radius=70))
    boost_dial_roi: CircleROI = field(default_factory=lambda: CircleROI(cx=1420, cy=955, radius=70))

    # -- Needle colour (red, wraps around H=0/180 in OpenCV HSV) --------------
    needle_hsv_lower1: list[int] = field(default_factory=lambda: [0, 100, 100])
    needle_hsv_upper1: list[int] = field(default_factory=lambda: [10, 255, 255])
    needle_hsv_lower2: list[int] = field(default_factory=lambda: [170, 100, 100])
    needle_hsv_upper2: list[int] = field(default_factory=lambda: [180, 255, 255])

    # -- Needle angle calibration (degrees, atan2 convention: 0=right, CCW+) --
    # Стрелки идут по часовой от ~10 часов (150°) до ~4 часов (-30°)
    needle_min_angle: float = 150.0    # позиция стрелки при 0% (10 часов)
    needle_max_angle: float = -30.0    # позиция стрелки при 100% (4 часа)

    # -- Gauge ROIs (horizontal bars — kept for fuel, legacy) -----------------
    fuel_gauge_roi: Rect = field(default_factory=lambda: Rect(x=50, y=10, w=200, h=20))
    rpm_gauge_roi: Rect = field(default_factory=lambda: Rect(x=300, y=10, w=150, h=20))
    boost_gauge_roi: Rect = field(default_factory=lambda: Rect(x=500, y=10, w=150, h=20))

    # -- OCR ROIs (calibrated for Redmi Note 8 Pro results screen) -----------
    distance_text_roi: Rect = field(default_factory=lambda: Rect(x=1130, y=840, w=160, h=45))
    coins_text_roi: Rect = field(default_factory=lambda: Rect(x=780, y=710, w=140, h=80))
    results_coins_roi: Rect = field(default_factory=lambda: Rect(x=780, y=710, w=140, h=80))
    results_distance_roi: Rect = field(default_factory=lambda: Rect(x=1200, y=710, w=320, h=80))

    # -- Vehicle ROI (where the car typically sits) ---------------------------
    vehicle_roi: Rect = field(default_factory=lambda: Rect(x=200, y=300, w=400, h=200))

    # -- Terrain ROI (ground line area) ---------------------------------------
    terrain_roi: Rect = field(default_factory=lambda: Rect(x=0, y=400, w=800, h=200))

    # -- HSV ranges for gauge bar colour detection ----------------------------
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

    # -- OCR ------------------------------------------------------------------
    tesseract_cmd: str = "/opt/homebrew/bin/tesseract"
    ocr_backend: str = "tesseract"  # or "easyocr"

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
            elif isinstance(field_val, CircleROI):
                setattr(cfg, key, CircleROI(**value))
            else:
                setattr(cfg, key, value)
        return cfg


# Singleton — import and use everywhere
cfg = Config.load()
