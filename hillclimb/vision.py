"""Computer Vision module: game state classifier, dial gauge reader, OCR, tilt & terrain.

Designed for HCR2 on Redmi Note 8 Pro (2340x1080 landscape via scrcpy).
- Circular dial gauges with red needles (RPM, Boost)
- Distance text via Tesseract OCR
- 8 distinct game screens classified by button/colour heuristics
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from hillclimb.config import CircleROI, Rect, cfg


# ---------------------------------------------------------------------------
# Game state enum (8 states covering full HCR2 flow)
# ---------------------------------------------------------------------------

class GameState(Enum):
    UNKNOWN = auto()
    MAIN_MENU = auto()
    VEHICLE_SELECT = auto()
    DOUBLE_COINS_POPUP = auto()
    RACING = auto()
    DRIVER_DOWN = auto()
    TOUCH_TO_CONTINUE = auto()
    RESULTS = auto()

    # Legacy aliases
    MENU = MAIN_MENU
    CRASHED = DRIVER_DOWN


# ---------------------------------------------------------------------------
# State vector produced by the vision pipeline
# ---------------------------------------------------------------------------

@dataclass
class VisionState:
    game_state: GameState = GameState.UNKNOWN
    fuel: float = 1.0           # 0..1
    rpm: float = 0.0            # 0..1
    boost: float = 0.0          # 0..1
    tilt: float = 0.0           # degrees, positive = nose up
    terrain_slope: float = 0.0  # degrees
    airborne: bool = False
    speed_estimate: float = 0.0  # 0..1 rough estimate
    distance_m: float = 0.0     # metres read via OCR
    coins: int = 0              # in-game coins (top-left)
    results_coins: int = 0      # coins on results screen
    results_distance_m: float = 0.0  # distance on results screen

    def to_array(self) -> np.ndarray:
        """Return normalised 8-element float32 vector for the RL agent."""
        return np.array([
            self.fuel,
            self.rpm,
            self.boost,
            (self.tilt + 90.0) / 180.0,           # normalise -90..90 -> 0..1
            (self.terrain_slope + 90.0) / 180.0,
            float(self.airborne),
            self.speed_estimate,
            min(self.distance_m / 1000.0, 1.0),    # normalise 0..1000m -> 0..1
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
        self._templates: dict[str, np.ndarray | None] = {}
        self._prev_frame_gray: np.ndarray | None = None

        # OCR engine (lazy init)
        self._ocr_ready = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> VisionState:
        """Run the full vision pipeline on a BGR frame."""
        state = VisionState()
        state.game_state = self._classify_state(frame)

        if state.game_state == GameState.RACING:
            state.rpm = self._read_dial_gauge(
                frame, cfg.rpm_dial_roi, self._rpm_buf)
            state.boost = self._read_dial_gauge(
                frame, cfg.boost_dial_roi, self._boost_buf)
            # Fuel: keep horizontal bar reader (fuel bar is still horizontal in HCR2)
            state.fuel = self._read_gauge(
                frame, cfg.fuel_gauge_roi,
                cfg.fuel_hsv_lower, cfg.fuel_hsv_upper,
                self._fuel_buf)
            state.tilt = self._detect_tilt(frame)
            state.terrain_slope = self._detect_terrain_slope(frame)
            state.airborne = self._detect_airborne(frame)
            state.speed_estimate = self._estimate_speed(frame)
            state.distance_m = self._read_distance_ocr(frame)

        elif state.game_state == GameState.RESULTS:
            results = self._read_results_screen(frame)
            state.results_coins = results[0]
            state.results_distance_m = results[1]

        return state

    # ------------------------------------------------------------------
    # Game State Classifier (priority-based for HCR2)
    # ------------------------------------------------------------------

    def _classify_state(self, frame: np.ndarray) -> GameState:
        """Classify the current game state using colour heuristics + button detection."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. DRIVER_DOWN: orange star burst in upper-center area
        #    Screen has dark overlay + orange/yellow "DRIVER DOWN" text
        upper_center = hsv[h // 6 : h // 3, w // 3 : 2 * w // 3]
        orange_mask = cv2.inRange(
            upper_center,
            np.array([10, 100, 150], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
        orange_ratio = np.mean(orange_mask > 0)
        # Also check for dark overlay
        center_v = hsv[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4, 2]
        dark_ratio = np.mean(center_v < 60)
        if orange_ratio > 0.05 and dark_ratio > 0.3:
            return GameState.DRIVER_DOWN

        # 2. TOUCH_TO_CONTINUE: dark overlay with text at bottom
        #    Similar dark overlay but no orange burst; text prompt at bottom
        bottom_strip = hsv[int(h * 0.8):, w // 4 : 3 * w // 4]
        bottom_white = np.mean(bottom_strip[:, :, 2] > 180)
        if dark_ratio > 0.4 and bottom_white > 0.05 and orange_ratio < 0.03:
            return GameState.TOUCH_TO_CONTINUE

        # 3. RESULTS: green buttons at bottom (RETRY left, NEXT right)
        #    Look for green button pixels at bottom corners
        bottom_left = hsv[int(h * 0.85):, : w // 3]
        bottom_right = hsv[int(h * 0.85):, 2 * w // 3 :]
        green_lower = np.array([35, 80, 80], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)
        green_bl = np.mean(cv2.inRange(bottom_left, green_lower, green_upper) > 0)
        green_br = np.mean(cv2.inRange(bottom_right, green_lower, green_upper) > 0)
        if green_bl > 0.08 and green_br > 0.08:
            return GameState.RESULTS

        # 4. DOUBLE_COINS_POPUP: SKIP button in center area
        #    Look for vivid button in center-bottom region
        center_bottom = hsv[h // 2 : int(h * 0.8), w // 3 : 2 * w // 3]
        vivid_center = (
            (center_bottom[:, :, 1] > 100) & (center_bottom[:, :, 2] > 120)
        )
        vivid_center_ratio = np.mean(vivid_center)
        # Also check for popup background (bright, low-sat top half)
        top_half = hsv[: h // 2, w // 4 : 3 * w // 4]
        bright_low_sat = (top_half[:, :, 2] > 180) & (top_half[:, :, 1] < 60)
        popup_ratio = np.mean(bright_low_sat)
        if vivid_center_ratio > 0.05 and popup_ratio > 0.15:
            return GameState.DOUBLE_COINS_POPUP

        # 5. VEHICLE_SELECT: START button bottom-right + BACK button bottom-left
        #    START is typically a large green/yellow button
        vivid_br = (
            (bottom_right[:, :, 1] > 100) & (bottom_right[:, :, 2] > 120)
        )
        vivid_bl = (
            (bottom_left[:, :, 1] > 60) & (bottom_left[:, :, 2] > 100)
        )
        if np.mean(vivid_br) > 0.10 and np.mean(vivid_bl) > 0.03:
            # Distinguish from RESULTS: RESULTS has two green buttons,
            # VEHICLE_SELECT has one green + one grey/arrow
            if green_bl < 0.05:  # left button is NOT green = not RESULTS
                return GameState.VEHICLE_SELECT

        # 6. MAIN_MENU: RACE button in centre-bottom, vivid colours
        bottom_center = hsv[int(h * 0.7) :, w // 3 : 2 * w // 3]
        vivid_bc = (
            (bottom_center[:, :, 1] > 120) & (bottom_center[:, :, 2] > 120)
        )
        if np.mean(vivid_bc) > 0.10:
            return GameState.MAIN_MENU

        # 7. RACING: dial gauges visible at bottom-center
        #    Check if we can see the dial region (not black/empty)
        dial_region = self._crop_circle_roi(frame, cfg.rpm_dial_roi)
        if dial_region is not None:
            dial_hsv = cv2.cvtColor(dial_region, cv2.COLOR_BGR2HSV)
            # Dials have moderate brightness, not all-black
            dial_brightness = np.mean(dial_hsv[:, :, 2])
            if dial_brightness > 30:
                return GameState.RACING

        # Fallback: horizontal fuel gauge check (legacy)
        fuel_roi = self._crop_roi(frame, cfg.fuel_gauge_roi)
        if fuel_roi is not None:
            fuel_hsv = cv2.cvtColor(fuel_roi, cv2.COLOR_BGR2HSV)
            lower = np.array(cfg.fuel_hsv_lower, dtype=np.uint8)
            upper = np.array(cfg.fuel_hsv_upper, dtype=np.uint8)
            mask = cv2.inRange(fuel_hsv, lower, upper)
            if np.mean(mask) > 5:
                return GameState.RACING

        # Template matching fallback
        tmpl_result = self._match_template(frame, "menu")
        if tmpl_result is not None and tmpl_result > cfg.menu_template_threshold:
            return GameState.MAIN_MENU

        tmpl_result = self._match_template(frame, "crash")
        if tmpl_result is not None and tmpl_result > cfg.crash_template_threshold:
            return GameState.DRIVER_DOWN

        return GameState.UNKNOWN

    # ------------------------------------------------------------------
    # Dial Gauge Reader (circular dials with red needle)
    # ------------------------------------------------------------------

    def _read_dial_gauge(
        self,
        frame: np.ndarray,
        dial_roi: CircleROI,
        buf: deque[float],
    ) -> float:
        """Read a circular dial gauge via red needle angle detection.

        1. Crop square around dial center
        2. HSV mask for red needle (two ranges for H wrapping)
        3. Morphological cleanup
        4. Filter pixels far from center (needle tip, not hub)
        5. atan2 from center to mean of far red pixels -> angle
        6. Map angle to 0..1 via calibrated min/max
        """
        crop = self._crop_circle_roi(frame, dial_roi)
        if crop is None or crop.size == 0:
            return float(np.mean(buf)) if buf else 0.0

        h_crop, w_crop = crop.shape[:2]
        cx_local = w_crop // 2
        cy_local = h_crop // 2

        # HSV mask for red needle (wraps around H=0/180)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower1 = np.array(cfg.needle_hsv_lower1, dtype=np.uint8)
        upper1 = np.array(cfg.needle_hsv_upper1, dtype=np.uint8)
        lower2 = np.array(cfg.needle_hsv_lower2, dtype=np.uint8)
        upper2 = np.array(cfg.needle_hsv_upper2, dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 | mask2

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find red pixels
        ys, xs = np.where(mask > 0)
        if len(xs) < 5:
            return float(np.mean(buf)) if buf else 0.0

        # Filter: keep pixels far from center (needle tip, not hub cap)
        distances = np.sqrt((xs - cx_local) ** 2 + (ys - cy_local) ** 2)
        min_dist = dial_roi.radius * 0.3  # ignore center 30%
        far_mask = distances > min_dist
        xs_far = xs[far_mask]
        ys_far = ys[far_mask]

        if len(xs_far) < 3:
            return float(np.mean(buf)) if buf else 0.0

        # Compute angle from center to mean of far needle pixels
        mean_x = float(np.mean(xs_far)) - cx_local
        mean_y = float(np.mean(ys_far)) - cy_local
        # atan2 with screen coords (y-down): negate y for standard math angle
        angle_rad = math.atan2(-mean_y, mean_x)
        angle_deg = math.degrees(angle_rad)

        # Map angle to 0..1 via calibrated min/max
        min_a = cfg.needle_min_angle
        max_a = cfg.needle_max_angle
        if abs(max_a - min_a) < 1.0:
            fill = 0.5
        else:
            fill = (angle_deg - min_a) / (max_a - min_a)

        fill = float(np.clip(fill, 0.0, 1.0))
        buf.append(fill)
        return float(np.mean(buf))

    # ------------------------------------------------------------------
    # Horizontal Gauge Reader (legacy, used for fuel bar)
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

        col_sums = np.sum(mask, axis=0)
        total_cols = mask.shape[1]

        if total_cols == 0:
            return 0.0

        threshold = mask.shape[0] * 128
        filled_cols = np.where(col_sums > threshold)[0]
        if len(filled_cols) == 0:
            fill = 0.0
        else:
            fill = (filled_cols[-1] + 1) / total_cols

        fill = float(np.clip(fill, 0.0, 1.0))
        buf.append(fill)
        return float(np.mean(buf))

    # ------------------------------------------------------------------
    # OCR: Distance text ("103m")
    # ------------------------------------------------------------------

    def _read_distance_ocr(self, frame: np.ndarray) -> float:
        """Read distance text above gauges via OCR. Returns metres as float."""
        crop = self._crop_roi(frame, cfg.distance_text_roi)
        if crop is None or crop.size == 0:
            return 0.0

        return self._ocr_distance_from_crop(crop)

    def _ocr_distance_from_crop(self, crop: np.ndarray) -> float:
        """Extract distance value from a cropped image region."""
        # Preprocess: grayscale -> threshold (white text on dark bg) -> scale up
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        text = self._run_ocr(scaled)
        return self._parse_distance(text)

    @staticmethod
    def _parse_distance(text: str) -> float:
        """Parse OCR text like '103m' or '1,234m' into float metres."""
        text = text.strip().replace(",", "").replace(" ", "")
        match = re.search(r"(\d+)", text)
        if match:
            return float(match.group(1))
        return 0.0

    # ------------------------------------------------------------------
    # OCR: Results screen
    # ------------------------------------------------------------------

    def _read_results_screen(
        self, frame: np.ndarray,
    ) -> tuple[int, float]:
        """Read coins and distance from the results screen.

        Returns:
            (coins, distance_m)
        """
        coins = 0
        distance_m = 0.0

        # Results coins
        crop_coins = self._crop_roi(frame, cfg.results_coins_roi)
        if crop_coins is not None and crop_coins.size > 0:
            gray = cv2.cvtColor(crop_coins, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            text = self._run_ocr(scaled)
            text_clean = text.strip().replace(",", "").replace(" ", "")
            digits = re.search(r"(\d+)", text_clean)
            if digits:
                coins = int(digits.group(1))

        # Results distance
        crop_dist = self._crop_roi(frame, cfg.results_distance_roi)
        if crop_dist is not None and crop_dist.size > 0:
            distance_m = self._ocr_distance_from_crop(crop_dist)

        return coins, distance_m

    # ------------------------------------------------------------------
    # OCR engine
    # ------------------------------------------------------------------

    def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on a preprocessed grayscale/binary image."""
        if cfg.ocr_backend == "tesseract":
            return self._run_tesseract(image)
        return ""

    @staticmethod
    def _run_tesseract(image: np.ndarray) -> str:
        """Run Tesseract OCR. Returns raw text string."""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd
            text = pytesseract.image_to_string(
                image,
                config="--psm 7 -c tessedit_char_whitelist=0123456789m,.",
            )
            return text.strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Vehicle Tilt Detection (unchanged from original)
    # ------------------------------------------------------------------

    def _detect_tilt(self, frame: np.ndarray) -> float:
        """Estimate vehicle tilt angle in degrees (positive = nose up)."""
        crop = self._crop_roi(frame, cfg.vehicle_roi)
        if crop is None or crop.size == 0:
            return self._smooth_tilt(0.0)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._smooth_tilt(0.0)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return self._smooth_tilt(0.0)

        rect = cv2.minAreaRect(largest)
        angle = rect[2]

        w, h_rect = rect[1]
        if w < h_rect:
            angle = angle + 90

        angle = float(np.clip(angle, -90, 90))
        return self._smooth_tilt(angle)

    def _smooth_tilt(self, angle: float) -> float:
        self._tilt_buf.append(angle)
        return float(np.mean(self._tilt_buf))

    # ------------------------------------------------------------------
    # Terrain Slope Detection (unchanged from original)
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

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:
                continue
            angle_deg = float(np.degrees(np.arctan2(-dy, dx)))
            angles.append(angle_deg)

        if not angles:
            return 0.0

        return float(np.median(angles))

    # ------------------------------------------------------------------
    # Airborne Detection (unchanged from original)
    # ------------------------------------------------------------------

    def _detect_airborne(self, frame: np.ndarray) -> bool:
        """Heuristic: if terrain under vehicle ROI is mostly sky/empty, we are airborne."""
        crop = self._crop_roi(frame, cfg.terrain_roi)
        if crop is None or crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sky_mask = (
            (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 140) &
            (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 100)
        )
        sky_ratio = np.mean(sky_mask)
        return bool(sky_ratio > 0.5)

    # ------------------------------------------------------------------
    # Speed Estimation (unchanged from original)
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

        if gray_tmpl.shape[0] > gray_frame.shape[0]:
            return None

        result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    def _load_template(self, name: str) -> np.ndarray | None:
        if name in self._templates:
            return self._templates[name]
        path = Path(cfg.template_dir) / f"{name}.png"
        if not path.exists():
            self._templates[name] = None
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

    @staticmethod
    def _crop_circle_roi(frame: np.ndarray, roi: CircleROI) -> np.ndarray | None:
        """Crop a square region around a circular ROI."""
        h, w = frame.shape[:2]
        x1 = max(0, roi.cx - roi.radius)
        y1 = max(0, roi.cy - roi.radius)
        x2 = min(w, roi.cx + roi.radius)
        y2 = min(h, roi.cy + roi.radius)
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
        cv2.putText(out, f"RPM: {state.rpm:.0%}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Boost: {state.boost:.0%}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Tilt: {state.tilt:.1f} deg", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Terrain: {state.terrain_slope:.1f} deg", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Airborne: {state.airborne}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Speed: {state.speed_estimate:.0%}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"Dist: {state.distance_m:.0f}m", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw ROI rectangles
        for label, roi in [
            ("fuel", cfg.fuel_gauge_roi),
            ("distance", cfg.distance_text_roi),
            ("vehicle", cfg.vehicle_roi),
            ("terrain", cfg.terrain_roi),
        ]:
            cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h),
                          (255, 0, 0), 1)

        # Draw dial ROI circles
        for label, droi in [
            ("RPM", cfg.rpm_dial_roi),
            ("Boost", cfg.boost_dial_roi),
        ]:
            cv2.circle(out, (droi.cx, droi.cy), droi.radius, (0, 255, 255), 1)
            cv2.putText(out, label,
                        (droi.cx - droi.radius, droi.cy - droi.radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return out
