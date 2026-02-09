"""Computer Vision module: game state classifier, gauge reader, tilt & terrain."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from hillclimb.config import Rect, cfg


# ---------------------------------------------------------------------------
# Game state enum
# ---------------------------------------------------------------------------

class GameState(Enum):
    UNKNOWN = auto()
    MENU = auto()
    RACING = auto()
    CRASHED = auto()
    RESULTS = auto()


# ---------------------------------------------------------------------------
# State vector produced by the vision pipeline
# ---------------------------------------------------------------------------

@dataclass
class VisionState:
    game_state: GameState = GameState.UNKNOWN
    fuel: float = 1.0       # 0..1
    rpm: float = 0.0        # 0..1
    boost: float = 0.0      # 0..1
    tilt: float = 0.0       # degrees, positive = nose up
    terrain_slope: float = 0.0  # degrees
    airborne: bool = False
    speed_estimate: float = 0.0  # 0..1 rough estimate

    def to_array(self) -> np.ndarray:
        """Return normalised 7-element float32 vector for the RL agent."""
        return np.array([
            self.fuel,
            self.rpm,
            self.boost,
            (self.tilt + 90.0) / 180.0,        # normalise -90..90 → 0..1
            (self.terrain_slope + 90.0) / 180.0,
            float(self.airborne),
            self.speed_estimate,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Vision analyser
# ---------------------------------------------------------------------------

class VisionAnalyzer:
    """Processes a single BGR frame and returns a VisionState."""

    def __init__(self) -> None:
        # Smoothing buffers
        self._fuel_buf: deque[float] = deque(maxlen=cfg.gauge_smoothing_window)
        self._rpm_buf: deque[float] = deque(maxlen=cfg.gauge_smoothing_window)
        self._boost_buf: deque[float] = deque(maxlen=cfg.gauge_smoothing_window)
        self._tilt_buf: deque[float] = deque(maxlen=cfg.tilt_smoothing_window)

        # Optional templates (loaded lazily)
        self._templates: dict[str, np.ndarray] = {}
        self._prev_frame_gray: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> VisionState:
        """Run the full vision pipeline on a BGR frame."""
        state = VisionState()
        state.game_state = self._classify_state(frame)

        if state.game_state == GameState.RACING:
            state.fuel = self._read_gauge(frame, cfg.fuel_gauge_roi,
                                          cfg.fuel_hsv_lower, cfg.fuel_hsv_upper,
                                          self._fuel_buf)
            state.rpm = self._read_gauge(frame, cfg.rpm_gauge_roi,
                                         [0, 80, 80], [10, 255, 255],
                                         self._rpm_buf)
            state.boost = self._read_gauge(frame, cfg.boost_gauge_roi,
                                           [100, 80, 80], [130, 255, 255],
                                           self._boost_buf)
            state.tilt = self._detect_tilt(frame)
            state.terrain_slope = self._detect_terrain_slope(frame)
            state.airborne = self._detect_airborne(frame)
            state.speed_estimate = self._estimate_speed(frame)

        return state

    # ------------------------------------------------------------------
    # Game State Classifier
    # ------------------------------------------------------------------

    def _classify_state(self, frame: np.ndarray) -> GameState:
        """Classify the current game state using colour heuristics + templates."""
        h, w = frame.shape[:2]

        # Convert a small strip at top + bottom to HSV for fast analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- CRASHED detection: look for reddish overlay or dark vignette ---
        # When crashed, the screen darkens and often has a red/orange tint
        center_region = hsv[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
        dark_mask = center_region[:, :, 2] < 60  # low value = dark
        dark_ratio = np.mean(dark_mask)
        if dark_ratio > 0.5:
            return GameState.CRASHED

        # --- RESULTS detection: bright popup with text in centre ---
        bright_mask = center_region[:, :, 2] > 200
        bright_ratio = np.mean(bright_mask)
        low_sat = center_region[:, :, 1] < 50
        low_sat_ratio = np.mean(low_sat)
        if bright_ratio > 0.4 and low_sat_ratio > 0.4:
            return GameState.RESULTS

        # --- MENU detection: large buttons / high saturation UI elements ---
        # Menus typically have vivid coloured buttons
        bottom_half = hsv[h // 2 :, :]
        vivid = (bottom_half[:, :, 1] > 150) & (bottom_half[:, :, 2] > 150)
        vivid_ratio = np.mean(vivid)
        if vivid_ratio > 0.15:
            # Could be menu — also check if the top portion is mostly sky/bg
            top_region = hsv[: h // 4, :]
            top_low_sat = top_region[:, :, 1] < 80
            if np.mean(top_low_sat) > 0.5:
                return GameState.MENU

        # --- RACING: if we have fuel gauge bar visible ---
        fuel_roi = self._crop_roi(frame, cfg.fuel_gauge_roi)
        if fuel_roi is not None:
            fuel_hsv = cv2.cvtColor(fuel_roi, cv2.COLOR_BGR2HSV)
            lower = np.array(cfg.fuel_hsv_lower, dtype=np.uint8)
            upper = np.array(cfg.fuel_hsv_upper, dtype=np.uint8)
            mask = cv2.inRange(fuel_hsv, lower, upper)
            if np.mean(mask) > 5:  # at least some coloured pixels
                return GameState.RACING

        # Template matching fallback
        tmpl_result = self._match_template(frame, "menu")
        if tmpl_result is not None and tmpl_result > cfg.menu_template_threshold:
            return GameState.MENU

        tmpl_result = self._match_template(frame, "crash")
        if tmpl_result is not None and tmpl_result > cfg.crash_template_threshold:
            return GameState.CRASHED

        # Default: assume racing if nothing else matches
        return GameState.RACING

    # ------------------------------------------------------------------
    # Gauge Reader
    # ------------------------------------------------------------------

    def _read_gauge(
        self,
        frame: np.ndarray,
        roi: Rect,
        hsv_lower: list[int],
        hsv_upper: list[int],
        buf: deque[float],
    ) -> float:
        """Read a horizontal gauge bar as a 0..1 fill fraction."""
        crop = self._crop_roi(frame, roi)
        if crop is None or crop.size == 0:
            return float(np.mean(buf)) if buf else 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower = np.array(hsv_lower, dtype=np.uint8)
        upper = np.array(hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Project along X-axis: find rightmost column with colour
        col_sums = np.sum(mask, axis=0)
        total_cols = mask.shape[1]

        if total_cols == 0:
            return 0.0

        # Find rightmost column with significant fill
        threshold = mask.shape[0] * 128  # at least ~50% of row height
        filled_cols = np.where(col_sums > threshold)[0]
        if len(filled_cols) == 0:
            fill = 0.0
        else:
            fill = (filled_cols[-1] + 1) / total_cols

        fill = float(np.clip(fill, 0.0, 1.0))
        buf.append(fill)
        return float(np.mean(buf))

    # ------------------------------------------------------------------
    # Vehicle Tilt Detection
    # ------------------------------------------------------------------

    def _detect_tilt(self, frame: np.ndarray) -> float:
        """Estimate vehicle tilt angle in degrees (positive = nose up)."""
        crop = self._crop_roi(frame, cfg.vehicle_roi)
        if crop is None or crop.size == 0:
            return self._smooth_tilt(0.0)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours — use the largest one as the vehicle body
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._smooth_tilt(0.0)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return self._smooth_tilt(0.0)

        rect = cv2.minAreaRect(largest)
        angle = rect[2]  # -90..0 from OpenCV

        # Normalise: OpenCV minAreaRect angle is in [-90, 0)
        # We want: positive = nose up, negative = nose down
        w, h = rect[1]
        if w < h:
            angle = angle + 90

        angle = float(np.clip(angle, -90, 90))
        return self._smooth_tilt(angle)

    def _smooth_tilt(self, angle: float) -> float:
        self._tilt_buf.append(angle)
        return float(np.mean(self._tilt_buf))

    # ------------------------------------------------------------------
    # Terrain Slope Detection
    # ------------------------------------------------------------------

    def _detect_terrain_slope(self, frame: np.ndarray) -> float:
        """Estimate terrain slope under the vehicle via edge/line detection."""
        crop = self._crop_roi(frame, cfg.terrain_roi)
        if crop is None or crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=40, minLineLength=30, maxLineGap=10,
        )
        if lines is None or len(lines) == 0:
            return 0.0

        # Average angle of detected line segments
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:
                continue
            angle_deg = float(np.degrees(np.arctan2(-dy, dx)))  # screen Y is inverted
            angles.append(angle_deg)

        if not angles:
            return 0.0

        return float(np.median(angles))

    # ------------------------------------------------------------------
    # Airborne Detection
    # ------------------------------------------------------------------

    def _detect_airborne(self, frame: np.ndarray) -> bool:
        """Heuristic: if terrain under vehicle ROI is mostly sky/empty, we are airborne."""
        crop = self._crop_roi(frame, cfg.terrain_roi)
        if crop is None or crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Sky is typically blue-ish and bright or very light
        sky_mask = (
            (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 140) &
            (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 100)
        )
        sky_ratio = np.mean(sky_mask)
        return bool(sky_ratio > 0.5)

    # ------------------------------------------------------------------
    # Speed Estimation (optical flow)
    # ------------------------------------------------------------------

    def _estimate_speed(self, frame: np.ndarray) -> float:
        """Rough speed estimate via optical flow magnitude (0..1)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 90))

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray_small
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_frame_gray, gray_small,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        self._prev_frame_gray = gray_small

        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        avg_mag = float(np.mean(mag))
        # Normalise: typical max magnitude ~20 pixels at 160px width
        return float(np.clip(avg_mag / 20.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Template Matching
    # ------------------------------------------------------------------

    def _match_template(self, frame: np.ndarray, name: str) -> float | None:
        """Match a named template against the frame. Returns max confidence or None."""
        tmpl = self._load_template(name)
        if tmpl is None:
            return None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

        # Resize template if frame is much larger/smaller
        if gray_tmpl.shape[0] > gray_frame.shape[0]:
            return None

        result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    def _load_template(self, name: str) -> np.ndarray | None:
        if name in self._templates:
            return self._templates[name]
        path = Path(cfg.template_dir) / f"{name}.png"
        if not path.exists():
            self._templates[name] = None  # type: ignore[assignment]
            return None
        tmpl = cv2.imread(str(path))
        self._templates[name] = tmpl
        return tmpl

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_roi(frame: np.ndarray, roi: Rect) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1 = max(0, roi.x)
        y1 = max(0, roi.y)
        x2 = min(w, roi.x + roi.w)
        y2 = min(h, roi.y + roi.h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Debug Overlay
    # ------------------------------------------------------------------

    def draw_debug(self, frame: np.ndarray, state: VisionState) -> np.ndarray:
        """Draw debug overlay on a copy of the frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # State text
        cv2.putText(out, f"State: {state.game_state.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out, f"Fuel: {state.fuel:.0%}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Tilt: {state.tilt:.1f} deg", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Terrain: {state.terrain_slope:.1f} deg", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Airborne: {state.airborne}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Speed: {state.speed_estimate:.0%}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw ROI rectangles
        for label, roi in [
            ("fuel", cfg.fuel_gauge_roi),
            ("vehicle", cfg.vehicle_roi),
            ("terrain", cfg.terrain_roi),
        ]:
            cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h),
                          (255, 0, 0), 1)

        return out
