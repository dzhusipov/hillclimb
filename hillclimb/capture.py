"""Screen capture: grab scrcpy window via mss or ADB screencap fallback."""

from __future__ import annotations

import subprocess
import time

import cv2
import numpy as np

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

from hillclimb.config import cfg


class ScreenCapture:
    """Captures the scrcpy window on macOS."""

    def __init__(self) -> None:
        self._sct: mss.mss | None = None
        self._monitor: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grab(self) -> np.ndarray:
        """Return a BGR numpy frame.

        Uses cfg.capture_method: "adb" for native resolution (2340x1080),
        "mss" for scrcpy window capture (fast but scaled).
        """
        if cfg.capture_method == "adb":
            return self._grab_adb()
        frame = self._grab_mss()
        if frame is not None:
            return frame
        return self._grab_adb()

    # ------------------------------------------------------------------
    # mss-based capture
    # ------------------------------------------------------------------

    def _find_scrcpy_window(self) -> dict | None:
        """Use macOS Accessibility / Quartz to find the scrcpy window bounds."""
        try:
            # Use screencapture with window ID — but first find window via osascript
            script = (
                'tell application "System Events" to tell process "scrcpy" to '
                "get {position, size} of window 1"
            )
            out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            # output like: "100, 200, 800, 600"
            parts = [int(p.strip()) for p in out.split(",")]
            if len(parts) == 4:
                x, y, w, h = parts
                return {"left": x, "top": y, "width": w, "height": h}
        except Exception:
            pass
        return None

    def _grab_mss(self) -> np.ndarray | None:
        if not HAS_MSS:
            return None
        try:
            if self._sct is None:
                self._sct = mss.mss()
            if self._monitor is None:
                self._monitor = self._find_scrcpy_window()
            if self._monitor is None:
                return None
            shot = self._sct.grab(self._monitor)
            frame = np.array(shot)  # BGRA
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception:
            # Window may have moved / closed; reset monitor cache
            self._monitor = None
            return None

    # ------------------------------------------------------------------
    # ADB screencap fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _grab_adb() -> np.ndarray:
        """Capture via `adb exec-out screencap -p`. Slow but reliable."""
        device_args = ["-s", cfg.adb_device] if cfg.adb_device else []
        cmd = [cfg.adb_path] + device_args + ["exec-out", "screencap", "-p"]
        raw = subprocess.check_output(cmd)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("ADB screencap returned invalid image data")
        return frame

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset_window(self) -> None:
        """Force re-detection of scrcpy window on next grab."""
        self._monitor = None


def main() -> None:
    """Quick smoke test: capture and display one frame."""
    cap = ScreenCapture()
    frame = cap.grab()
    print(f"Captured frame: {frame.shape}")
    cv2.imshow("capture", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
