"""Calibration utility: interactive ROI selection for gauges and buttons.

Usage:
    python -m hillclimb.calibrate

Shows the captured screen and lets you click to define ROI regions.
Press keys to switch modes, click/drag to set regions.
"""

from __future__ import annotations

import cv2
import numpy as np

from hillclimb.capture import ScreenCapture
from hillclimb.config import Rect, Point, cfg
from hillclimb.vision import VisionAnalyzer


# ---------------------------------------------------------------------------
# Interactive calibration
# ---------------------------------------------------------------------------

class Calibrator:
    MODE_VIEW = "view"
    MODE_FUEL = "fuel"
    MODE_VEHICLE = "vehicle"
    MODE_TERRAIN = "terrain"
    MODE_GAS = "gas_button"
    MODE_BRAKE = "brake_button"

    MODES = [MODE_VIEW, MODE_FUEL, MODE_VEHICLE, MODE_TERRAIN, MODE_GAS, MODE_BRAKE]

    def __init__(self) -> None:
        self._capture = ScreenCapture()
        self._vision = VisionAnalyzer()
        self._mode_idx = 0
        self._drag_start: tuple[int, int] | None = None
        self._frame: np.ndarray | None = None

    @property
    def mode(self) -> str:
        return self.MODES[self._mode_idx]

    def run(self) -> None:
        """Main calibration loop."""
        cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("calibrate", self._on_mouse)

        print("=== Hill Climb Racing Calibration ===")
        print("Keys:")
        print("  TAB  — switch mode")
        print("  s    — save config")
        print("  r    — refresh frame")
        print("  d    — show debug overlay")
        print("  q    — quit")
        print(f"Current mode: {self.mode}")

        show_debug = False

        while True:
            frame = self._capture.grab()
            self._frame = frame.copy()

            if show_debug:
                state = self._vision.analyze(frame)
                display = self._vision.draw_debug(frame, state)
            else:
                display = frame.copy()

            # Show current mode
            cv2.putText(display, f"Mode: {self.mode}", (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw current ROI for the active mode
            roi = self._get_current_roi()
            if roi is not None:
                cv2.rectangle(display, (roi.x, roi.y),
                              (roi.x + roi.w, roi.y + roi.h), (0, 0, 255), 2)

            cv2.imshow("calibrate", display)
            key = cv2.waitKey(100) & 0xFF

            if key == ord("q"):
                break
            elif key == 9:  # TAB
                self._mode_idx = (self._mode_idx + 1) % len(self.MODES)
                print(f"Mode: {self.mode}")
            elif key == ord("s"):
                cfg.save()
                print(f"Config saved to {cfg.save.__func__}")
                print("Saved!")
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("r"):
                self._capture.reset_window()

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.mode == self.MODE_VIEW:
            return

        if self.mode in (self.MODE_GAS, self.MODE_BRAKE):
            # Single click to set button position
            if event == cv2.EVENT_LBUTTONDOWN:
                point = Point(x=x, y=y)
                if self.mode == self.MODE_GAS:
                    cfg.gas_button = point
                    print(f"Gas button set to ({x}, {y})")
                else:
                    cfg.brake_button = point
                    print(f"Brake button set to ({x}, {y})")
            return

        # ROI drag selection
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
    # ROI helpers
    # ------------------------------------------------------------------

    def _get_current_roi(self) -> Rect | None:
        mapping = {
            self.MODE_FUEL: cfg.fuel_gauge_roi,
            self.MODE_VEHICLE: cfg.vehicle_roi,
            self.MODE_TERRAIN: cfg.terrain_roi,
        }
        return mapping.get(self.mode)

    def _set_current_roi(self, roi: Rect) -> None:
        if self.mode == self.MODE_FUEL:
            cfg.fuel_gauge_roi = roi
        elif self.mode == self.MODE_VEHICLE:
            cfg.vehicle_roi = roi
        elif self.mode == self.MODE_TERRAIN:
            cfg.terrain_roi = roi


def main() -> None:
    cal = Calibrator()
    cal.run()


if __name__ == "__main__":
    main()
