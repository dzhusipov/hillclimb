"""Calibration utility: interactive ROI selection for gauges, buttons, and dials.

Usage:
    python -m hillclimb.calibrate

Shows the captured screen and lets you click to define ROI regions.
Press keys to switch modes, click/drag to set regions.

Modes:
  VIEW         — just view the frame
  FUEL         — drag to set fuel gauge ROI (horizontal bar)
  RPM_DIAL     — click center, then click edge for radius (circle ROI)
  BOOST_DIAL   — click center, then click edge for radius (circle ROI)
  DISTANCE_TEXT — drag to set distance OCR ROI
  VEHICLE      — drag to set vehicle ROI
  TERRAIN      — drag to set terrain ROI
  GAS_BTN      — click to set gas button position
  BRAKE_BTN    — click to set brake button position
  RACE_BTN     — click to set RACE button position
  START_BTN    — click to set START button position
  SKIP_BTN     — click to set SKIP button position
  RETRY_BTN    — click to set RETRY button position

Keys:
  TAB    — switch mode
  s      — save config
  r      — refresh frame
  d      — toggle debug overlay
  t      — toggle live gauge test overlay
  n      — record needle min angle (current needle = 0%)
  m      — record needle max angle (current needle = 100%)
  q      — quit
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from hillclimb.capture import ScreenCapture
from hillclimb.config import CircleROI, Point, Rect, cfg
from hillclimb.vision import VisionAnalyzer


class Calibrator:
    # Modes
    MODE_VIEW = "view"
    MODE_FUEL = "fuel"
    MODE_RPM_DIAL = "rpm_dial"
    MODE_BOOST_DIAL = "boost_dial"
    MODE_DISTANCE_TEXT = "distance_text"
    MODE_VEHICLE = "vehicle"
    MODE_TERRAIN = "terrain"
    MODE_GAS = "gas_button"
    MODE_BRAKE = "brake_button"
    MODE_RACE = "race_button"
    MODE_START = "start_button"
    MODE_SKIP = "skip_button"
    MODE_RETRY = "retry_button"

    MODES = [
        MODE_VIEW, MODE_FUEL,
        MODE_RPM_DIAL, MODE_BOOST_DIAL, MODE_DISTANCE_TEXT,
        MODE_VEHICLE, MODE_TERRAIN,
        MODE_GAS, MODE_BRAKE, MODE_RACE, MODE_START, MODE_SKIP, MODE_RETRY,
    ]

    # Modes that use single-click for button position
    BUTTON_MODES = {
        MODE_GAS, MODE_BRAKE, MODE_RACE, MODE_START, MODE_SKIP, MODE_RETRY,
    }

    # Modes that use drag for rectangular ROI
    RECT_MODES = {MODE_FUEL, MODE_DISTANCE_TEXT, MODE_VEHICLE, MODE_TERRAIN}

    # Modes that use two clicks for circle ROI (center + edge)
    CIRCLE_MODES = {MODE_RPM_DIAL, MODE_BOOST_DIAL}

    def __init__(self) -> None:
        self._capture = ScreenCapture()
        self._vision = VisionAnalyzer()
        self._mode_idx = 0
        self._drag_start: tuple[int, int] | None = None
        self._circle_center: tuple[int, int] | None = None
        self._frame: np.ndarray | None = None
        self._show_debug = False
        self._show_gauge_test = False
        self._last_needle_angle: float | None = None

    @property
    def mode(self) -> str:
        return self.MODES[self._mode_idx]

    def run(self) -> None:
        """Main calibration loop."""
        cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("calibrate", self._on_mouse)

        print("=== Hill Climb Racing 2 Calibration ===")
        print("Keys:")
        print("  TAB  — switch mode")
        print("  s    — save config")
        print("  r    — refresh frame")
        print("  d    — toggle debug overlay")
        print("  t    — toggle live gauge test")
        print("  n    — record needle min angle (0%)")
        print("  m    — record needle max angle (100%)")
        print("  q    — quit")
        print(f"Current mode: {self.mode}")

        while True:
            frame = self._capture.grab()
            self._frame = frame.copy()

            if self._show_debug:
                state = self._vision.analyze(frame)
                display = self._vision.draw_debug(frame, state)
            else:
                display = frame.copy()

            # Live gauge test overlay
            if self._show_gauge_test:
                display = self._draw_gauge_test(display, frame)

            # Show current mode
            cv2.putText(display, f"Mode: {self.mode}", (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw current ROI for the active mode
            roi = self._get_current_roi()
            if roi is not None:
                cv2.rectangle(display, (roi.x, roi.y),
                              (roi.x + roi.w, roi.y + roi.h), (0, 0, 255), 2)

            circle = self._get_current_circle()
            if circle is not None:
                cv2.circle(display, (circle.cx, circle.cy), circle.radius,
                           (0, 0, 255), 2)

            # Show pending circle center (first click, waiting for edge click)
            if self._circle_center is not None and self.mode in self.CIRCLE_MODES:
                cv2.circle(display, self._circle_center, 5, (0, 255, 255), -1)
                cv2.putText(display, "Click edge for radius",
                            (self._circle_center[0] + 10, self._circle_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("calibrate", display)
            key = cv2.waitKey(100) & 0xFF

            if key == ord("q"):
                break
            elif key == 9:  # TAB
                self._mode_idx = (self._mode_idx + 1) % len(self.MODES)
                self._circle_center = None
                print(f"Mode: {self.mode}")
            elif key == ord("s"):
                cfg.save()
                print("Config saved!")
            elif key == ord("d"):
                self._show_debug = not self._show_debug
            elif key == ord("t"):
                self._show_gauge_test = not self._show_gauge_test
            elif key == ord("r"):
                self._capture.reset_window()
            elif key == ord("n"):
                if self._last_needle_angle is not None:
                    cfg.rpm_needle_min_angle = self._last_needle_angle
                    cfg.boost_needle_min_angle = self._last_needle_angle
                    print(f"RPM/Boost needle min angle set to {self._last_needle_angle:.1f} deg")
            elif key == ord("m"):
                if self._last_needle_angle is not None:
                    cfg.rpm_needle_max_angle = self._last_needle_angle
                    cfg.boost_needle_max_angle = self._last_needle_angle
                    print(f"RPM/Boost needle max angle set to {self._last_needle_angle:.1f} deg")

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.mode == self.MODE_VIEW:
            return

        # Button modes: single click
        if self.mode in self.BUTTON_MODES:
            if event == cv2.EVENT_LBUTTONDOWN:
                point = Point(x=x, y=y)
                attr = {
                    self.MODE_GAS: "gas_button",
                    self.MODE_BRAKE: "brake_button",
                    self.MODE_RACE: "race_button",
                    self.MODE_START: "start_button",
                    self.MODE_SKIP: "skip_button",
                    self.MODE_RETRY: "retry_button",
                }[self.mode]
                setattr(cfg, attr, point)
                print(f"{attr} set to ({x}, {y})")
            return

        # Circle modes: first click = center, second click = edge
        if self.mode in self.CIRCLE_MODES:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._circle_center is None:
                    self._circle_center = (x, y)
                    print(f"Circle center set to ({x}, {y}). Click edge for radius.")
                else:
                    cx, cy = self._circle_center
                    radius = int(math.hypot(x - cx, y - cy))
                    circle = CircleROI(cx=cx, cy=cy, radius=max(radius, 5))
                    if self.mode == self.MODE_RPM_DIAL:
                        cfg.rpm_dial_roi = circle
                        print(f"RPM dial ROI: center=({cx},{cy}) radius={radius}")
                    elif self.mode == self.MODE_BOOST_DIAL:
                        cfg.boost_dial_roi = circle
                        print(f"Boost dial ROI: center=({cx},{cy}) radius={radius}")
                    self._circle_center = None
            return

        # Rect modes: drag selection
        if self.mode in self.RECT_MODES:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drag_start = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
                sx, sy = self._drag_start
                roi = Rect(
                    x=min(sx, x),
                    y=min(sy, y),
                    w=abs(x - sx),
                    h=abs(y - sy),
                )
                self._set_current_roi(roi)
                print(f"{self.mode} ROI set to ({roi.x}, {roi.y}, {roi.w}, {roi.h})")
                self._drag_start = None

    # ------------------------------------------------------------------
    # Live gauge test overlay
    # ------------------------------------------------------------------

    def _draw_gauge_test(self, display: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Draw needle mask and computed values for RPM/Boost dials."""
        for label, droi in [("RPM", cfg.rpm_dial_roi), ("Boost", cfg.boost_dial_roi)]:
            crop = self._vision._crop_circle_roi(frame, droi)
            if crop is None:
                continue

            h_crop, w_crop = crop.shape[:2]
            cx_local, cy_local = w_crop // 2, h_crop // 2

            # HSV red needle mask
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            l1 = np.array(cfg.needle_hsv_lower1, dtype=np.uint8)
            u1 = np.array(cfg.needle_hsv_upper1, dtype=np.uint8)
            l2 = np.array(cfg.needle_hsv_lower2, dtype=np.uint8)
            u2 = np.array(cfg.needle_hsv_upper2, dtype=np.uint8)
            mask = cv2.inRange(hsv, l1, u1) | cv2.inRange(hsv, l2, u2)

            # Find needle angle
            ys, xs = np.where(mask > 0)
            angle_deg = None
            if len(xs) > 5:
                dists = np.sqrt((xs - cx_local) ** 2 + (ys - cy_local) ** 2)
                far = dists > droi.radius * 0.3
                if np.sum(far) > 3:
                    mean_x = float(np.mean(xs[far])) - cx_local
                    mean_y = float(np.mean(ys[far])) - cy_local
                    angle_deg = math.degrees(math.atan2(-mean_y, mean_x))
                    self._last_needle_angle = angle_deg

            # Draw needle mask as red overlay on display
            x1 = max(0, droi.cx - droi.radius)
            y1 = max(0, droi.cy - droi.radius)
            x2 = x1 + mask.shape[1]
            y2 = y1 + mask.shape[0]
            if y2 <= display.shape[0] and x2 <= display.shape[1]:
                overlay_region = display[y1:y2, x1:x2]
                overlay_region[mask > 0] = [0, 0, 255]

            # Draw angle line
            if angle_deg is not None:
                rad = math.radians(angle_deg)
                line_len = droi.radius
                ex = int(droi.cx + line_len * math.cos(rad))
                ey = int(droi.cy - line_len * math.sin(rad))
                cv2.line(display, (droi.cx, droi.cy), (ex, ey), (0, 255, 0), 2)
                cv2.putText(display, f"{label}: {angle_deg:.1f}deg",
                            (droi.cx - droi.radius, droi.cy + droi.radius + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return display

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------

    def _get_current_roi(self) -> Rect | None:
        mapping = {
            self.MODE_FUEL: cfg.fuel_gauge_roi,
            self.MODE_DISTANCE_TEXT: cfg.distance_text_roi,
            self.MODE_VEHICLE: cfg.vehicle_roi,
            self.MODE_TERRAIN: cfg.terrain_roi,
        }
        return mapping.get(self.mode)

    def _get_current_circle(self) -> CircleROI | None:
        mapping = {
            self.MODE_RPM_DIAL: cfg.rpm_dial_roi,
            self.MODE_BOOST_DIAL: cfg.boost_dial_roi,
        }
        return mapping.get(self.mode)

    def _set_current_roi(self, roi: Rect) -> None:
        if self.mode == self.MODE_FUEL:
            cfg.fuel_gauge_roi = roi
        elif self.mode == self.MODE_DISTANCE_TEXT:
            cfg.distance_text_roi = roi
        elif self.mode == self.MODE_VEHICLE:
            cfg.vehicle_roi = roi
        elif self.mode == self.MODE_TERRAIN:
            cfg.terrain_roi = roi


def main() -> None:
    cal = Calibrator()
    cal.run()


if __name__ == "__main__":
    main()
