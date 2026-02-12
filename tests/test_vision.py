"""Tests for the vision module.

These tests use synthetic frames to verify CV logic without needing
a real device. Place real screenshots in tests/screenshots/ for
integration testing.
"""

from __future__ import annotations

import math
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hillclimb.config import CircleROI, Rect, cfg
from hillclimb.vision import GameState, VisionAnalyzer, VisionState


@pytest.fixture
def analyzer() -> VisionAnalyzer:
    return VisionAnalyzer()


# ---------------------------------------------------------------------------
# VisionState
# ---------------------------------------------------------------------------

class TestVisionState:
    def test_to_array_shape(self):
        state = VisionState()
        arr = state.to_array()
        assert arr.shape == (8,)
        assert arr.dtype == np.float32

    def test_to_array_defaults(self):
        state = VisionState(fuel=0.5, tilt=0.0)
        arr = state.to_array()
        assert arr[0] == pytest.approx(0.5)  # fuel
        # tilt=0 -> (0+90)/180 = 0.5
        assert arr[3] == pytest.approx(0.5)

    def test_to_array_bounds(self):
        state = VisionState(fuel=1.0, tilt=90.0, terrain_slope=-90.0)
        arr = state.to_array()
        assert 0.0 <= arr[0] <= 1.0
        assert 0.0 <= arr[3] <= 1.0
        assert 0.0 <= arr[4] <= 1.0

    def test_to_array_distance(self):
        state = VisionState(distance_m=500.0)
        arr = state.to_array()
        assert arr[7] == pytest.approx(0.5)  # 500/1000

    def test_to_array_distance_clamped(self):
        state = VisionState(distance_m=2000.0)
        arr = state.to_array()
        assert arr[7] == pytest.approx(1.0)  # clamped at 1.0


# ---------------------------------------------------------------------------
# Horizontal Gauge Reader (legacy, used for fuel)
# ---------------------------------------------------------------------------

class TestGaugeReader:
    def _make_gauge_frame(self, fill: float, roi: Rect) -> np.ndarray:
        """Create a synthetic frame with a green bar at the given fill level."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        filled_width = int(roi.w * fill)
        # Green in BGR
        frame[roi.y : roi.y + roi.h, roi.x : roi.x + filled_width] = [0, 200, 0]
        return frame

    def test_full_gauge(self, analyzer: VisionAnalyzer):
        frame = self._make_gauge_frame(1.0, cfg.fuel_gauge_roi)
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_gauge(
            frame, cfg.fuel_gauge_roi,
            cfg.fuel_hsv_lower, cfg.fuel_hsv_upper, buf,
        )
        assert val > 0.8

    def test_empty_gauge(self, analyzer: VisionAnalyzer):
        frame = self._make_gauge_frame(0.0, cfg.fuel_gauge_roi)
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_gauge(
            frame, cfg.fuel_gauge_roi,
            cfg.fuel_hsv_lower, cfg.fuel_hsv_upper, buf,
        )
        assert val < 0.1

    def test_half_gauge(self, analyzer: VisionAnalyzer):
        frame = self._make_gauge_frame(0.5, cfg.fuel_gauge_roi)
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_gauge(
            frame, cfg.fuel_gauge_roi,
            cfg.fuel_hsv_lower, cfg.fuel_hsv_upper, buf,
        )
        assert 0.3 < val < 0.7


# ---------------------------------------------------------------------------
# Dial Gauge Reader (circular dials with red needle)
# ---------------------------------------------------------------------------

class TestDialGaugeReader:
    def _make_dial_frame(
        self,
        angle_deg: float,
        dial_roi: CircleROI,
        frame_h: int = 1080,
        frame_w: int = 2340,
    ) -> np.ndarray:
        """Create a synthetic frame with a red line at the given angle."""
        import cv2

        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        # Draw a grey circle (dial background)
        cv2.circle(frame, (dial_roi.cx, dial_roi.cy), dial_roi.radius, (100, 100, 100), -1)
        # Draw a red needle line from center to edge at the given angle
        rad = math.radians(angle_deg)
        ex = int(dial_roi.cx + dial_roi.radius * 0.9 * math.cos(rad))
        ey = int(dial_roi.cy - dial_roi.radius * 0.9 * math.sin(rad))
        # Red in BGR = (0, 0, 255) -> HSV H~0, S=255, V=255
        cv2.line(frame, (dial_roi.cx, dial_roi.cy), (ex, ey), (0, 0, 255), 3)
        return frame

    def test_needle_at_min_angle(self, analyzer: VisionAnalyzer):
        """Needle at min angle should read ~0.0."""
        dial = cfg.rpm_dial_roi
        frame = self._make_dial_frame(cfg.rpm_needle_min_angle, dial)
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_dial_gauge(
            frame, dial, buf,
            cfg.rpm_needle_min_angle, cfg.rpm_needle_max_angle)
        assert val < 0.3  # close to 0

    def test_needle_at_max_angle(self, analyzer: VisionAnalyzer):
        """Needle at max angle should read ~1.0."""
        dial = cfg.rpm_dial_roi
        frame = self._make_dial_frame(cfg.rpm_needle_max_angle, dial)
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_dial_gauge(
            frame, dial, buf,
            cfg.rpm_needle_min_angle, cfg.rpm_needle_max_angle)
        assert val > 0.7  # close to 1

    def test_needle_at_mid_angle(self, analyzer: VisionAnalyzer):
        """Needle at midpoint angle should read ~0.5."""
        dial = cfg.rpm_dial_roi
        # CW angular midpoint (not arithmetic mean, which doesn't work for wrapped angles)
        total_sweep = (cfg.rpm_needle_min_angle - cfg.rpm_needle_max_angle) % 360
        mid_angle = cfg.rpm_needle_min_angle - total_sweep / 2
        frame = self._make_dial_frame(mid_angle, dial)
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_dial_gauge(
            frame, dial, buf,
            cfg.rpm_needle_min_angle, cfg.rpm_needle_max_angle)
        assert 0.2 < val < 0.8  # roughly mid-range

    def test_no_needle_returns_zero(self, analyzer: VisionAnalyzer):
        """Empty dial (no red pixels) should return 0."""
        frame = np.zeros((1080, 2340, 3), dtype=np.uint8)
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_dial_gauge(
            frame, cfg.rpm_dial_roi, buf,
            cfg.rpm_needle_min_angle, cfg.rpm_needle_max_angle)
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Game State Classifier
# ---------------------------------------------------------------------------

class TestGameStateClassifier:
    def test_dark_with_orange_is_driver_down(self, analyzer: VisionAnalyzer):
        """Dark frame with orange burst in upper-center -> DRIVER_DOWN."""
        frame = np.full((600, 800, 3), 30, dtype=np.uint8)
        # Edges NOT dark (game HUD visible) — prevents CAPTCHA false positive
        frame[:, :64] = [100, 100, 100]
        frame[:, 736:] = [100, 100, 100]
        # Add orange pixels in upper-center region (H~15 in HSV)
        import cv2
        upper_center = frame[100:200, 260:540]
        # Orange in BGR: ~(0, 165, 255)
        upper_center[:] = [0, 130, 230]
        state = analyzer._classify_state(frame)
        assert state == GameState.DRIVER_DOWN

    def test_bright_frame_with_green_buttons_is_results(self, analyzer: VisionAnalyzer):
        """Frame with green buttons at both bottom corners -> RESULTS."""
        frame = np.full((600, 800, 3), 150, dtype=np.uint8)
        # Green buttons at bottom-left and bottom-right
        # Green in BGR: (0, 200, 0)
        frame[510:600, 0:260] = [0, 180, 0]    # bottom-left
        frame[510:600, 540:800] = [0, 180, 0]  # bottom-right
        state = analyzer._classify_state(frame)
        assert state == GameState.RESULTS

    def test_generic_frame_with_dials_is_racing(self, analyzer: VisionAnalyzer):
        """Frame with dial + red needle -> RACING."""
        import cv2
        frame = np.full((480, 800, 3), 120, dtype=np.uint8)
        roi = cfg.rpm_dial_roi
        # Draw dial background
        cv2.circle(frame, (roi.cx, roi.cy), roi.radius, (100, 100, 100), -1)
        # Draw thick red needle (must exceed 2.5% of dial crop for detection)
        total_sweep = (cfg.rpm_needle_min_angle - cfg.rpm_needle_max_angle) % 360
        mid_angle = cfg.rpm_needle_min_angle - total_sweep / 2
        rad = math.radians(mid_angle)
        ex = int(roi.cx + roi.radius * 0.8 * math.cos(rad))
        ey = int(roi.cy - roi.radius * 0.8 * math.sin(rad))
        cv2.line(frame, (roi.cx, roi.cy), (ex, ey), (0, 0, 255), 8)
        state = analyzer._classify_state(frame)
        assert state == GameState.RACING


# ---------------------------------------------------------------------------
# Crop ROI
# ---------------------------------------------------------------------------

class TestCropROI:
    def test_valid_roi(self, analyzer: VisionAnalyzer):
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        crop = analyzer._crop_roi(frame, Rect(10, 10, 100, 50))
        assert crop is not None
        assert crop.shape == (50, 100, 3)

    def test_out_of_bounds_roi_clipped(self, analyzer: VisionAnalyzer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = analyzer._crop_roi(frame, Rect(80, 80, 50, 50))
        assert crop is not None
        assert crop.shape == (20, 20, 3)

    def test_fully_outside_returns_none(self, analyzer: VisionAnalyzer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = analyzer._crop_roi(frame, Rect(200, 200, 50, 50))
        assert crop is None


class TestCropCircleROI:
    def test_valid_circle_roi(self, analyzer: VisionAnalyzer):
        frame = np.zeros((1080, 2340, 3), dtype=np.uint8)
        roi = CircleROI(cx=100, cy=100, radius=50)
        crop = analyzer._crop_circle_roi(frame, roi)
        assert crop is not None
        assert crop.shape == (100, 100, 3)

    def test_circle_roi_clipped(self, analyzer: VisionAnalyzer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = CircleROI(cx=90, cy=90, radius=50)
        crop = analyzer._crop_circle_roi(frame, roi)
        assert crop is not None
        # Should be clipped to fit within frame

    def test_circle_roi_fully_outside(self, analyzer: VisionAnalyzer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = CircleROI(cx=500, cy=500, radius=50)
        crop = analyzer._crop_circle_roi(frame, roi)
        assert crop is None


# ---------------------------------------------------------------------------
# Speed Estimation
# ---------------------------------------------------------------------------

class TestSpeedEstimation:
    def test_static_frame_zero_speed(self, analyzer: VisionAnalyzer):
        frame = np.full((600, 800, 3), 128, dtype=np.uint8)
        analyzer._estimate_speed(frame)
        speed = analyzer._estimate_speed(frame)
        assert speed < 0.1

    def test_shifted_frame_nonzero_speed(self, analyzer: VisionAnalyzer):
        frame1 = np.zeros((600, 800, 3), dtype=np.uint8)
        frame1[:, 100:200] = 255

        frame2 = np.zeros((600, 800, 3), dtype=np.uint8)
        frame2[:, 130:230] = 255  # shifted right by 30px

        analyzer._estimate_speed(frame1)
        speed = analyzer._estimate_speed(frame2)
        assert speed > 0.0


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_analyze_returns_vision_state(self, analyzer: VisionAnalyzer):
        frame = np.full((1080, 2340, 3), 50, dtype=np.uint8)
        # Make dial region visible to trigger RACING
        import cv2
        roi = cfg.rpm_dial_roi
        cv2.circle(frame, (roi.cx, roi.cy), roi.radius, (100, 100, 100), -1)
        state = analyzer.analyze(frame)
        assert isinstance(state, VisionState)
        assert isinstance(state.game_state, GameState)

    def test_debug_overlay(self, analyzer: VisionAnalyzer):
        frame = np.full((1080, 2340, 3), 100, dtype=np.uint8)
        state = VisionState()
        out = analyzer.draw_debug(frame, state)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)


# ---------------------------------------------------------------------------
# Navigator (state transitions)
# ---------------------------------------------------------------------------

class TestNavigator:
    def test_ensure_racing_from_main_menu(self):
        """Navigator should tap RACE button when in MAIN_MENU."""
        from hillclimb.navigator import Navigator

        mock_ctrl = MagicMock()
        mock_cap = MagicMock()
        mock_vision = MagicMock()

        state_menu = MagicMock()
        state_menu.game_state = GameState.MAIN_MENU
        state_racing = MagicMock()
        state_racing.game_state = GameState.RACING

        mock_cap.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_vision.analyze.side_effect = [state_menu, state_racing]

        nav = Navigator(mock_ctrl, mock_cap, mock_vision)
        result = nav.ensure_racing(timeout=10.0)

        assert result is True
        # RACE кнопка тапается по координатам из cfg
        mock_ctrl.tap.assert_any_call(cfg.race_button.x, cfg.race_button.y)

    def test_ensure_racing_from_results(self):
        """Navigator should capture results and tap retry when in RESULTS."""
        from hillclimb.navigator import Navigator

        mock_ctrl = MagicMock()
        mock_cap = MagicMock()
        mock_vision = MagicMock()

        state_results = VisionState(
            game_state=GameState.RESULTS,
            results_coins=500,
            results_distance_m=123.0,
        )
        state_racing = MagicMock()
        state_racing.game_state = GameState.RACING

        mock_cap.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_vision.analyze.side_effect = [state_results, state_racing]

        nav = Navigator(mock_ctrl, mock_cap, mock_vision)
        result = nav.ensure_racing(timeout=10.0)

        assert result is True
        assert nav.last_results is not None
        assert nav.last_results.results_coins == 500
        mock_ctrl.tap.assert_called_once_with(cfg.retry_button.x, cfg.retry_button.y)

    def test_ensure_racing_from_double_coins(self):
        """Navigator should dismiss popups when in DOUBLE_COINS_POPUP."""
        from hillclimb.navigator import Navigator

        mock_ctrl = MagicMock()
        mock_cap = MagicMock()
        mock_vision = MagicMock()

        state_popup = MagicMock()
        state_popup.game_state = GameState.DOUBLE_COINS_POPUP
        state_racing = MagicMock()
        state_racing.game_state = GameState.RACING

        mock_cap.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_vision.analyze.side_effect = [state_popup, state_racing]

        nav = Navigator(mock_ctrl, mock_cap, mock_vision)
        result = nav.ensure_racing(timeout=10.0)

        assert result is True
        # _dismiss_popups тапает SKIP в обеих позициях (COINS + TOKENS)
        mock_ctrl.tap.assert_any_call(cfg.skip_button.x, cfg.skip_button.y)
        mock_ctrl.tap.assert_any_call(990, 830)
