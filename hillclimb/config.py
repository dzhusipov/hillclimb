"""Configuration: button coordinates, ROI regions, thresholds.

Default coordinates are for ReDroid 800x480 landscape.
Run `python -m hillclimb.calibrate` to recalibrate for your device.
"""

from __future__ import annotations

import json
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
# Default configuration for ReDroid 800x480 landscape
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # -- ADB ------------------------------------------------------------------
    adb_serial: str = "localhost:5555"

    # -- Capture --------------------------------------------------------------
    capture_backend: str = "png"  # "png" or "raw"

    # -- Emulators (for parallel training) ------------------------------------
    num_emulators: int = 2
    adb_port_base: int = 5555

    # -- Gas / Brake (800x480 landscape) --------------------------------------
    gas_button: Point = field(default_factory=lambda: Point(x=700, y=400))
    brake_button: Point = field(default_factory=lambda: Point(x=100, y=400))

    # -- Navigation buttons (800x480 landscape, approximate) ------------------
    race_button: Point = field(default_factory=lambda: Point(x=400, y=400))
    start_button: Point = field(default_factory=lambda: Point(x=700, y=430))
    back_button: Point = field(default_factory=lambda: Point(x=60, y=430))
    skip_button: Point = field(default_factory=lambda: Point(x=300, y=380))
    close_popup_button: Point = field(default_factory=lambda: Point(x=400, y=240))
    retry_button: Point = field(default_factory=lambda: Point(x=60, y=440))
    next_button: Point = field(default_factory=lambda: Point(x=740, y=440))
    center_screen: Point = field(default_factory=lambda: Point(x=400, y=240))

    # -- Dial gauge ROIs (approximate for 800x480, needs calibration) ---------
    rpm_dial_roi: CircleROI = field(default_factory=lambda: CircleROI(cx=315, cy=420, radius=30))
    fuel_dial_roi: CircleROI = field(default_factory=lambda: CircleROI(cx=400, cy=430, radius=30))
    boost_dial_roi: CircleROI = field(default_factory=lambda: CircleROI(cx=485, cy=420, radius=30))

    # -- Needle colour (red, wraps around H=0/180 in OpenCV HSV) --------------
    needle_hsv_lower1: list[int] = field(default_factory=lambda: [0, 100, 100])
    needle_hsv_upper1: list[int] = field(default_factory=lambda: [10, 255, 255])
    needle_hsv_lower2: list[int] = field(default_factory=lambda: [170, 100, 100])
    needle_hsv_upper2: list[int] = field(default_factory=lambda: [180, 255, 255])

    # -- Needle angle calibration (atan2 convention: 0=right, CCW+) -----------
    needle_min_angle: float = 150.0    # 0% position (~10 o'clock)
    needle_max_angle: float = -30.0    # 100% position (~4 o'clock)

    # -- Gauge ROIs (horizontal bars, legacy) ---------------------------------
    fuel_gauge_roi: Rect = field(default_factory=lambda: Rect(x=15, y=3, w=70, h=8))

    # -- OCR ROIs (800x480 landscape, approximate) ----------------------------
    distance_text_roi: Rect = field(default_factory=lambda: Rect(x=380, y=355, w=60, h=20))

    # -- Results screen OCR ROIs ----------------------------------------------
    results_coins_roi: Rect = field(default_factory=lambda: Rect(x=260, y=300, w=60, h=30))
    results_distance_roi: Rect = field(default_factory=lambda: Rect(x=410, y=300, w=120, h=30))

    # Kept for backward compat
    coins_text_roi: Rect = field(default_factory=lambda: Rect(x=260, y=300, w=60, h=30))

    # -- Vehicle ROI (where the car sits, for tilt detection) -----------------
    vehicle_roi: Rect = field(default_factory=lambda: Rect(x=70, y=130, w=140, h=80))

    # -- Terrain ROI (ground area) --------------------------------------------
    terrain_roi: Rect = field(default_factory=lambda: Rect(x=0, y=170, w=280, h=80))

    # -- HSV ranges for gauge bar colour detection ----------------------------
    fuel_hsv_lower: list[int] = field(default_factory=lambda: [30, 80, 80])
    fuel_hsv_upper: list[int] = field(default_factory=lambda: [90, 255, 255])

    # -- Thresholds -----------------------------------------------------------
    crash_template_threshold: float = 0.7
    menu_template_threshold: float = 0.7
    tilt_smoothing_window: int = 5
    gauge_smoothing_window: int = 5

    # -- OCR ------------------------------------------------------------------
    ocr_backend: str = "template"  # "template" (default) or "tesseract"
    ocr_confidence_threshold: float = 0.85
    tesseract_cmd: str = "tesseract"

    # -- Game loop timing ----------------------------------------------------
    loop_interval_sec: float = 0.1  # target 10 decisions/sec
    action_hold_ms: int = 100       # ADB swipe hold duration

    # -- Logging --------------------------------------------------------------
    log_dir: str = "logs"
    log_frame_every_n: int = 30     # save PNG every N frames

    # -- RL training ----------------------------------------------------------
    model_dir: str = "models"
    learning_rate: float = 2.5e-4
    batch_size: int = 256
    n_steps: int = 128
    total_timesteps: int = 10_000_000

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
            elif isinstance(field_val, CircleROI):
                setattr(cfg, key, CircleROI(**value))
            else:
                setattr(cfg, key, value)
        return cfg

    def emulator_serial(self, index: int) -> str:
        """Return ADB serial for emulator at given index."""
        return f"localhost:{self.adb_port_base + index}"


# Singleton — import and use everywhere
cfg = Config.load()
