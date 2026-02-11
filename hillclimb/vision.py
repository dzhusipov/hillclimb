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
    CAPTCHA = auto()

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

def _classify_digit_heuristic(holes: int, fill: float) -> int:
    """Fallback: classify a digit by contour hole count and fill ratio.

    Returns digit 0-9, or -1 if unrecognized.
    """
    if holes >= 2:
        return 8
    if holes == 1:
        if fill >= 0.70:
            return 0
        if fill < 0.55:
            return 4
        return 6  # 6 and 9 are similar
    # 0 holes
    if fill < 0.52:
        return 1
    if fill < 0.56:
        return 7
    return 5  # 2, 3, 5 are hard to distinguish


# Results digit templates (loaded lazily)
_results_templates: dict[int, np.ndarray] | None = None


def _load_results_templates() -> dict[int, np.ndarray]:
    """Load digit templates extracted from RESULTS screen."""
    global _results_templates
    if _results_templates is not None:
        return _results_templates
    _results_templates = {}
    tdir = Path("templates/digits_results")
    if not tdir.exists():
        return _results_templates
    for digit in range(10):
        path = tdir / f"{digit}.png"
        if path.exists():
            tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                _results_templates[digit] = tmpl
    return _results_templates


def _classify_digit(
    digit_img: np.ndarray, holes: int, fill: float,
) -> int:
    """Classify a digit: template matching first, heuristic fallback.

    Args:
        digit_img: binary (white on black) cropped digit image.
        holes: number of contour holes.
        fill: pixel fill ratio within bounding box.
    """
    templates = _load_results_templates()
    if templates:
        best_digit = -1
        best_score = 0.0
        for d, tmpl in templates.items():
            # Resize digit to template size for matching
            resized = cv2.resize(digit_img, (tmpl.shape[1], tmpl.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            # Normalised correlation
            score = cv2.matchTemplate(
                resized, tmpl, cv2.TM_CCOEFF_NORMED,
            )
            s = float(score[0, 0]) if score.size == 1 else float(score.max())
            if s > best_score:
                best_score = s
                best_digit = d
        if best_score > 0.5:
            return best_digit

    return _classify_digit_heuristic(holes, fill)


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
        self._digit_templates: dict[int, np.ndarray] | None = None
        self._prev_frame_gray: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> VisionState:
        """Run the full vision pipeline on a BGR frame."""
        state = VisionState()
        state.game_state = self._classify_state(frame)

        if state.game_state == GameState.RACING:
            state.rpm = self._read_dial_gauge(
                frame, cfg.rpm_dial_roi, self._rpm_buf,
                cfg.rpm_needle_min_angle, cfg.rpm_needle_max_angle)
            state.boost = self._read_dial_gauge(
                frame, cfg.boost_dial_roi, self._boost_buf,
                cfg.boost_needle_min_angle, cfg.boost_needle_max_angle)
            # Fuel — тоже циферблат в HCR2 (центральный)
            state.fuel = self._read_dial_gauge(
                frame, cfg.fuel_dial_roi, self._fuel_buf,
                cfg.fuel_needle_min_angle, cfg.fuel_needle_max_angle)
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

        # 0. CAPTCHA ("ARE YOU A ROBOT?"): dark overlay covering entire screen.
        #    Real CAPTCHA has very low overall brightness (~35).
        #    Main menu with OFFLINE badge has dark top bar but bright center (~95).
        #    Dark racing maps have visible dials with red needles.
        overall_v = np.mean(hsv[:, :, 2])
        if overall_v < 75:
            top_8 = hsv[:int(h * 0.08), :]
            t8_dark = np.mean(top_8[:, :, 2] < 80)
            if t8_dark > 0.7:
                left_e = hsv[:, :int(w * 0.08)]
                right_e = hsv[:, int(w * 0.92):]
                edges_dark = (np.mean(left_e[:, :, 2] < 80) +
                              np.mean(right_e[:, :, 2] < 80)) / 2
                if edges_dark > 0.6:
                    # Guard: if RPM dial with red needle visible → dark map, not CAPTCHA
                    _dial = self._crop_circle_roi(frame, cfg.rpm_dial_roi)
                    _is_racing = False
                    if _dial is not None:
                        _dh = cv2.cvtColor(_dial, cv2.COLOR_BGR2HSV)
                        _db = np.mean(_dh[:, :, 2])
                        _nm = (
                            cv2.inRange(_dh,
                                        np.array(cfg.needle_hsv_lower1, dtype=np.uint8),
                                        np.array(cfg.needle_hsv_upper1, dtype=np.uint8))
                            | cv2.inRange(_dh,
                                          np.array(cfg.needle_hsv_lower2, dtype=np.uint8),
                                          np.array(cfg.needle_hsv_upper2, dtype=np.uint8))
                        )
                        if _db > 30 and np.mean(_nm > 0) > 0.025:
                            _is_racing = True
                    if not _is_racing:
                        return GameState.CAPTCHA

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

        # 1b. OUT OF FUEL / TOUCH TO CONTINUE: red text in upper half
        bottom_strip = hsv[int(h * 0.8):, w // 4 : 3 * w // 4]
        bottom_white = np.mean(bottom_strip[:, :, 2] > 180)
        upper_half = hsv[:h // 2, w // 4 : 3 * w // 4]
        red_mask = (
            cv2.inRange(upper_half,
                        np.array([0, 100, 100], dtype=np.uint8),
                        np.array([10, 255, 255], dtype=np.uint8))
            | cv2.inRange(upper_half,
                          np.array([160, 100, 100], dtype=np.uint8),
                          np.array([180, 255, 255], dtype=np.uint8))
        )
        red_upper = np.mean(red_mask > 0)
        # Green RESPAWN button in center = OUT OF FUEL (treat as DRIVER_DOWN)
        green_lower = np.array([35, 80, 80], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)
        if red_upper > 0.08:
            center_zone = hsv[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
            green_center = np.mean(
                cv2.inRange(center_zone, green_lower, green_upper) > 0)
            if green_center > 0.15:
                return GameState.DRIVER_DOWN  # BACK skips respawn → results
            if bottom_white > 0.04:
                return GameState.TOUCH_TO_CONTINUE

        # 2. TOUCH_TO_CONTINUE: dark overlay with text at bottom
        #    Similar dark overlay but no orange burst; text prompt at bottom
        if dark_ratio > 0.4 and bottom_white > 0.05 and orange_ratio < 0.03:
            return GameState.TOUCH_TO_CONTINUE

        # 3. RACING: циферблаты с красной стрелкой — самый надёжный сигнал
        #    Проверяем ДО RESULTS, потому что зелёная трава Countryside
        #    ложно срабатывает как зелёные кнопки RETRY/NEXT
        dial_region = self._crop_circle_roi(frame, cfg.rpm_dial_roi)
        has_racing_dial = False
        if dial_region is not None:
            dial_hsv = cv2.cvtColor(dial_region, cv2.COLOR_BGR2HSV)
            dial_brightness = np.mean(dial_hsv[:, :, 2])
            needle_mask = (
                cv2.inRange(dial_hsv,
                            np.array(cfg.needle_hsv_lower1, dtype=np.uint8),
                            np.array(cfg.needle_hsv_upper1, dtype=np.uint8))
                | cv2.inRange(dial_hsv,
                              np.array(cfg.needle_hsv_lower2, dtype=np.uint8),
                              np.array(cfg.needle_hsv_upper2, dtype=np.uint8))
            )
            has_needle = np.mean(needle_mask > 0) > 0.025
            if dial_brightness > 30 and has_needle:
                has_racing_dial = True
                # Check for "second chance" screen: big white hand cursor
                # in center of screen = car suspended, NOT actually racing
                hand_zone = hsv[int(h * 0.40):int(h * 0.75),
                                int(w * 0.35):int(w * 0.65)]
                white_hand = (
                    (hand_zone[:, :, 1] < 50) &   # low saturation
                    (hand_zone[:, :, 2] > 200)     # high brightness
                )
                if np.mean(white_hand) > 0.06:
                    return GameState.DRIVER_DOWN
                return GameState.RACING

        # 4. RESULTS: зелёные кнопки внизу (RETRY слева, NEXT справа)
        #    Иногда NEXT не видна (анимация) — достаточно RETRY + серая панель
        #    Exclude splash screen: too vivid/colorful center
        center_quarter = hsv[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        vivid_center = np.mean(
            (center_quarter[:, :, 1] > 80) & (center_quarter[:, :, 2] > 80))
        bottom_left = hsv[int(h * 0.85):, : w // 3]
        bottom_right = hsv[int(h * 0.85):, 2 * w // 3 :]
        green_bl = np.mean(cv2.inRange(bottom_left, green_lower, green_upper) > 0)
        green_br = np.mean(cv2.inRange(bottom_right, green_lower, green_upper) > 0)
        if green_bl > 0.08 and green_br > 0.08 and vivid_center < 0.25:
            return GameState.RESULTS
        # Fallback: only RETRY visible + gray results panel in center
        if green_bl > 0.08:
            panel = hsv[int(h * 0.3):int(h * 0.7), w // 4 : 3 * w // 4]
            panel_gray = (
                (panel[:, :, 1] < 40) &  # low saturation
                (panel[:, :, 2] > 140) & (panel[:, :, 2] < 210)  # mid brightness
            )
            if np.mean(panel_gray) > 0.12:
                return GameState.RESULTS

        # 5. DOUBLE_COINS_POPUP: жёлтые монеты/2x сверху + синяя SKIP внизу
        #    Жёлтый в верхней-центральной зоне (монеты, "2x")
        top_center = hsv[int(h * 0.15) : int(h * 0.55),
                         int(w * 0.25) : int(w * 0.75)]
        yellow_lower = np.array([18, 100, 150], dtype=np.uint8)
        yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
        yellow_top = np.mean(cv2.inRange(top_center, yellow_lower, yellow_upper) > 0)
        #    Синяя SKIP кнопка внизу-по-центру
        skip_zone = hsv[int(h * 0.70) : int(h * 0.90),
                        int(w * 0.30) : int(w * 0.55)]
        blue_lower = np.array([100, 60, 80], dtype=np.uint8)
        blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        blue_skip = np.mean(cv2.inRange(skip_zone, blue_lower, blue_upper) > 0)
        if yellow_top > 0.10 and blue_skip > 0.05:
            return GameState.DOUBLE_COINS_POPUP

        # 6. VEHICLE_SELECT: зелёная кнопка START в правом нижнем углу
        green_start = np.mean(cv2.inRange(
            bottom_right, green_lower, green_upper) > 0)
        if green_start > 0.05 and green_bl < 0.05:
            return GameState.VEHICLE_SELECT

        # 7. MAIN_MENU: яркий центр-низ + НЕТ зелёной START кнопки справа
        bottom_center = hsv[int(h * 0.7) :, w // 3 : 2 * w // 3]
        vivid_bc = (
            (bottom_center[:, :, 1] > 120) & (bottom_center[:, :, 2] > 120)
        )
        if np.mean(vivid_bc) > 0.10 and green_start < 0.03:
            return GameState.MAIN_MENU

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
        min_angle: float,
        max_angle: float,
    ) -> float:
        """Read a circular dial gauge via contour-based needle detection.

        1. Crop square around dial center + circular mask (0.85r)
        2. HSV mask for red needle (two ranges for H wrapping)
        3. Morphological cleanup
        4. findContours → select most elongated contour (aspect ratio)
        5. Tip angle: farthest point from center → atan2 → degrees
        6. Map angle → 0..1 via per-dial min_angle/max_angle calibration
        """
        crop = self._crop_circle_roi(frame, dial_roi)
        if crop is None or crop.size == 0:
            return float(np.mean(buf)) if buf else 0.0

        h_crop, w_crop = crop.shape[:2]
        cx_local = w_crop // 2
        cy_local = h_crop // 2
        radius = min(cx_local, cy_local)

        # Circular mask to exclude UI elements outside the dial face
        circle_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
        cv2.circle(circle_mask, (cx_local, cy_local), int(radius * 0.85), 255, -1)

        # HSV mask for red needle (wraps around H=0/180)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower1 = np.array(cfg.needle_hsv_lower1, dtype=np.uint8)
        upper1 = np.array(cfg.needle_hsv_upper1, dtype=np.uint8)
        lower2 = np.array(cfg.needle_hsv_lower2, dtype=np.uint8)
        upper2 = np.array(cfg.needle_hsv_upper2, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        # Apply circular mask
        mask = cv2.bitwise_and(mask, circle_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours and select the most elongated one (needle)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return float(np.mean(buf)) if buf else 0.0

        best_contour = None
        best_aspect = 0.0
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if min(rw, rh) < 1:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect > best_aspect:
                best_aspect = aspect
                best_contour = cnt

        if best_contour is None or best_aspect < 2.0:
            return float(np.mean(buf)) if buf else 0.0

        # Tip angle: find the point on the contour farthest from dial center
        pts = best_contour.reshape(-1, 2)
        dists = np.sqrt((pts[:, 0] - cx_local) ** 2 + (pts[:, 1] - cy_local) ** 2)
        tip_idx = np.argmax(dists)
        tip_x = float(pts[tip_idx, 0]) - cx_local
        tip_y = float(pts[tip_idx, 1]) - cy_local
        # atan2 with screen coords (y-down): negate y for standard math angle
        angle_deg = math.degrees(math.atan2(-tip_y, tip_x))

        # Map angle → 0..1 using per-dial calibration (CW sweep from min to max)
        total_sweep = (min_angle - max_angle) % 360
        if total_sweep < 1.0:
            fill = 0.5
        else:
            cw_from_min = (min_angle - angle_deg) % 360
            fill = cw_from_min / total_sweep

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
    # OCR: Distance text ("103m") via template matching
    # ------------------------------------------------------------------

    def _load_digit_templates(self) -> None:
        """Load digit templates 0-9 from templates/digits/ directory."""
        if self._digit_templates is not None:
            return
        self._digit_templates = {}
        digits_dir = Path(cfg.template_dir) / "digits"
        if not digits_dir.exists():
            return
        for digit in range(10):
            path = digits_dir / f"{digit}.png"
            if path.exists():
                tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    self._digit_templates[digit] = tmpl

    def _read_distance_ocr(self, frame: np.ndarray) -> float:
        """Read distance text via template matching digits."""
        crop = self._crop_roi(frame, cfg.distance_text_roi)
        if crop is None or crop.size == 0:
            return 0.0
        return self._ocr_number_from_crop(crop)

    def _ocr_number_from_crop(self, crop: np.ndarray) -> float:
        """Extract a number from a cropped region using template matching.

        Primary: HSV yellow filter (for HCR2 yellow distance text).
        Fallback: grayscale thresholding at multiple levels.
        Falls back to Tesseract if digit templates are not available.
        """
        self._load_digit_templates()

        # Fall back to Tesseract if no templates
        if not self._digit_templates:
            return self._ocr_number_tesseract(crop)

        best_value = 0.0
        best_ndigits = 0
        best_confidence = 0.0

        # Method 1: HSV yellow filter (best for in-race distance text)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (18, 150, 180), (35, 255, 255))
        value, ndigits, confidence = self._match_digits(yellow_mask)
        if ndigits > best_ndigits or (
            ndigits == best_ndigits and confidence > best_confidence
        ):
            best_value = value
            best_ndigits = ndigits
            best_confidence = confidence

        # Method 2: grayscale thresholding (fallback for other text)
        if best_ndigits == 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            for thresh_val in (150, 170, 190):
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                value, ndigits, confidence = self._match_digits(thresh)
                if ndigits > best_ndigits or (
                    ndigits == best_ndigits and confidence > best_confidence
                ):
                    best_value = value
                    best_ndigits = ndigits
                    best_confidence = confidence

        return best_value

    def _match_digits(
        self, binary_image: np.ndarray,
    ) -> tuple[float, int, float]:
        """Match digit templates against a binary image.

        Returns (value, num_digits, avg_confidence).
        """
        threshold = cfg.ocr_confidence_threshold
        matches: list[tuple[int, int, float]] = []  # (x, digit, confidence)

        for digit, tmpl in self._digit_templates.items():
            # Use template at native size (extracted from same resolution)
            if (tmpl.shape[0] > binary_image.shape[0] or
                    tmpl.shape[1] > binary_image.shape[1]):
                continue

            result = cv2.matchTemplate(
                binary_image, tmpl, cv2.TM_CCOEFF_NORMED,
            )
            locations = np.where(result >= threshold)
            for y, x in zip(*locations):
                conf = float(result[y, x])
                matches.append((int(x), digit, conf))

        if not matches:
            return 0.0, 0, 0.0

        # Non-maximum suppression: keep best match per x-region
        matches.sort(key=lambda m: m[0])
        filtered: list[tuple[int, int, float]] = []
        for x, digit, conf in matches:
            if filtered and abs(x - filtered[-1][0]) < 5:
                # Same region — keep higher confidence
                if conf > filtered[-1][2]:
                    filtered[-1] = (x, digit, conf)
            else:
                filtered.append((x, digit, conf))

        # Build number from left-to-right digits
        value = 0
        for _, digit, _ in filtered:
            value = value * 10 + digit

        avg_conf = sum(c for _, _, c in filtered) / len(filtered)
        return float(value), len(filtered), avg_conf

    @staticmethod
    def _ocr_number_tesseract(crop: np.ndarray) -> float:
        """Fallback: Tesseract OCR if available."""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
            inv = cv2.bitwise_not(thresh)
            padded = cv2.copyMakeBorder(
                inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255,
            )
            scaled = cv2.resize(
                padded, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR,
            )
            text = pytesseract.image_to_string(
                scaled,
                config="--psm 7 -c tessedit_char_whitelist=0123456789m,.",
            ).strip()
            text = text.replace(",", "").replace(" ", "")
            match = re.search(r"(\d+)", text)
            return float(match.group(1)) if match else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # OCR: Results screen
    # ------------------------------------------------------------------

    def _read_results_screen(
        self, frame: np.ndarray,
    ) -> tuple[int, float]:
        """Read coins and distance from the results screen.

        Uses contour-based OCR: white bold text on gray panel.
        Returns:
            (coins, distance_m)
        """
        coins = 0
        distance_m = 0.0

        crop_coins = self._crop_roi(frame, cfg.results_coins_roi)
        if crop_coins is not None and crop_coins.size > 0:
            coins = int(self._ocr_white_text(crop_coins))

        crop_dist = self._crop_roi(frame, cfg.results_distance_roi)
        if crop_dist is not None and crop_dist.size > 0:
            distance_m = self._ocr_white_text(crop_dist)

        return coins, distance_m

    @staticmethod
    def _ocr_white_text(crop: np.ndarray) -> float:
        """OCR for white bold text on gray background (RESULTS screen).

        Segments digits via contour detection, classifies by hole count
        and aspect ratio.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours or hierarchy is None:
            return 0.0

        # Collect top-level contours (digits + 'm')
        digits_info: list[tuple[int, int, int, int, float, int]] = []
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # skip children (holes)
                continue
            x, y, w, ht = cv2.boundingRect(contours[i])
            if ht < 8 or w < 3:
                continue
            # Count holes
            holes = 0
            child = h[2]
            while child != -1:
                holes += 1
                child = hierarchy[0][child][0]
            area = cv2.contourArea(contours[i])
            fill = area / (w * ht) if w * ht > 0 else 0.0
            digits_info.append((x, w, ht, holes, fill, i))

        if not digits_info:
            return 0.0

        digits_info.sort(key=lambda d: d[0])

        # Classify each contour into a digit
        value = 0
        for x, w, ht, holes, fill, idx in digits_info:
            aspect = w / ht
            # Skip 'm' (wide, aspect > 1.2)
            if aspect > 1.2:
                continue
            # Extract binary digit image for template matching
            bx, by, bw, bh = cv2.boundingRect(contours[idx])
            digit_img = thresh[by:by + bh, bx:bx + bw]
            digit = _classify_digit(digit_img, holes, fill)
            if digit >= 0:
                value = value * 10 + digit

        return float(value)

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
            ("Fuel", cfg.fuel_dial_roi),
            ("Boost", cfg.boost_dial_roi),
        ]:
            cv2.circle(out, (droi.cx, droi.cy), droi.radius, (0, 255, 255), 1)
            cv2.putText(out, label,
                        (droi.cx - droi.radius, droi.cy - droi.radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return out
