"""Tests for the vision module.

These tests use synthetic frames to verify CV logic without needing
a real device. Place real screenshots in tests/screenshots/ for
integration testing.
"""

from __future__ import annotations

import numpy as np
import pytest

from hillclimb.config import Rect, cfg
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
        assert arr.shape == (7,)
        assert arr.dtype == np.float32

    def test_to_array_defaults(self):
        state = VisionState(fuel=0.5, tilt=0.0)
        arr = state.to_array()
        assert arr[0] == pytest.approx(0.5)  # fuel
        # tilt=0 → (0+90)/180 = 0.5
        assert arr[3] == pytest.approx(0.5)

    def test_to_array_bounds(self):
        state = VisionState(fuel=1.0, tilt=90.0, terrain_slope=-90.0)
        arr = state.to_array()
        assert 0.0 <= arr[0] <= 1.0
        assert 0.0 <= arr[3] <= 1.0
        assert 0.0 <= arr[4] <= 1.0


# ---------------------------------------------------------------------------
# Gauge Reader
# ---------------------------------------------------------------------------

class TestGaugeReader:
    def _make_gauge_frame(self, fill: float, roi: Rect) -> np.ndarray:
        """Create a synthetic frame with a green bar at the given fill level."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        # Fill the ROI region with a green bar up to `fill` fraction
        filled_width = int(roi.w * fill)
        # Green in BGR
        frame[roi.y : roi.y + roi.h, roi.x : roi.x + filled_width] = [0, 200, 0]
        return frame

    def test_full_gauge(self, analyzer: VisionAnalyzer):
        frame = self._make_gauge_frame(1.0, cfg.fuel_gauge_roi)
        from collections import deque
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_gauge(
            frame, cfg.fuel_gauge_roi,
            cfg.fuel_hsv_lower, cfg.fuel_hsv_upper, buf,
        )
        assert val > 0.8

    def test_empty_gauge(self, analyzer: VisionAnalyzer):
        frame = self._make_gauge_frame(0.0, cfg.fuel_gauge_roi)
        from collections import deque
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_gauge(
            frame, cfg.fuel_gauge_roi,
            cfg.fuel_hsv_lower, cfg.fuel_hsv_upper, buf,
        )
        assert val < 0.1

    def test_half_gauge(self, analyzer: VisionAnalyzer):
        frame = self._make_gauge_frame(0.5, cfg.fuel_gauge_roi)
        from collections import deque
        buf: deque[float] = deque(maxlen=5)
        val = analyzer._read_gauge(
            frame, cfg.fuel_gauge_roi,
            cfg.fuel_hsv_lower, cfg.fuel_hsv_upper, buf,
        )
        assert 0.3 < val < 0.7


# ---------------------------------------------------------------------------
# Game State Classifier
# ---------------------------------------------------------------------------

class TestGameStateClassifier:
    def test_dark_frame_is_crashed(self, analyzer: VisionAnalyzer):
        """A mostly-dark frame should be classified as CRASHED."""
        frame = np.full((600, 800, 3), 20, dtype=np.uint8)
        state = analyzer._classify_state(frame)
        assert state == GameState.CRASHED

    def test_bright_frame_is_results(self, analyzer: VisionAnalyzer):
        """A bright, low-saturation centre should be classified as RESULTS."""
        frame = np.full((600, 800, 3), 220, dtype=np.uint8)  # bright grey
        state = analyzer._classify_state(frame)
        assert state == GameState.RESULTS

    def test_normal_frame_defaults_to_racing(self, analyzer: VisionAnalyzer):
        """A generic frame without clear indicators should default to RACING."""
        frame = np.full((600, 800, 3), 100, dtype=np.uint8)
        # Add some colour variation to avoid triggering crash/results
        frame[0:150, :] = [180, 200, 220]   # top area — sky-ish
        frame[400:, :] = [50, 100, 50]       # bottom — ground
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


# ---------------------------------------------------------------------------
# Speed Estimation
# ---------------------------------------------------------------------------

class TestSpeedEstimation:
    def test_static_frame_zero_speed(self, analyzer: VisionAnalyzer):
        frame = np.full((600, 800, 3), 128, dtype=np.uint8)
        # First call initialises, second should detect no movement
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
        frame = np.full((600, 800, 3), 100, dtype=np.uint8)
        frame[0:150, :] = [180, 200, 220]
        frame[400:, :] = [50, 100, 50]
        state = analyzer.analyze(frame)
        assert isinstance(state, VisionState)
        assert isinstance(state.game_state, GameState)

    def test_debug_overlay(self, analyzer: VisionAnalyzer):
        frame = np.full((600, 800, 3), 100, dtype=np.uint8)
        state = VisionState()
        out = analyzer.draw_debug(frame, state)
        assert out.shape == frame.shape
        # Should have drawn something (not identical to input)
        assert not np.array_equal(out, frame)
