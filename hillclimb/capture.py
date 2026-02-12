"""Screen capture from Android via ADB (adbutils).

Each ScreenCapture instance maintains a persistent connection to a specific
Android device (ReDroid emulator or physical phone) identified by ADB serial.
"""

from __future__ import annotations

import struct
import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Capture frames from a single Android device via ADB."""

    def __init__(
        self,
        adb_serial: str = "localhost:5555",
        backend: str = "png",
    ) -> None:
        """
        Args:
            adb_serial: ADB device serial (e.g. "localhost:5555" for ReDroid).
            backend: "png" (smaller transfer, default) or "raw" (no compression).
        """
        self._serial = adb_serial
        self._backend = backend
        self._device = None
        self._reconnect_counter = 0
        self._connect()

    def _connect(self) -> None:
        """Establish persistent ADB connection."""
        import adbutils
        client = adbutils.AdbClient()
        try:
            client.connect(self._serial, timeout=5)
        except Exception:
            pass  # already connected or direct USB
        self._device = client.device(self._serial)
        logger.info("ADB connected: %s", self._serial)

    def capture(self) -> np.ndarray:
        """Capture a single frame. Returns BGR numpy array (H, W, 3).

        Retries up to 3 times on failure. Reconnects every 1000 captures
        to prevent ADB memory leaks.
        """
        self._reconnect_counter += 1
        if self._reconnect_counter >= 1000:
            self._reconnect_counter = 0
            self._connect()

        for attempt in range(3):
            try:
                if self._backend == "raw":
                    return self._capture_raw()
                return self._capture_png()
            except Exception as e:
                logger.warning(
                    "Capture failed (attempt %d/3, %s): %s",
                    attempt + 1, self._serial, e,
                )
                if attempt < 2:
                    time.sleep(0.5)
                    self._connect()
        raise RuntimeError(f"Capture failed after 3 attempts on {self._serial}")

    def _capture_png(self) -> np.ndarray:
        """Capture via screencap -p (PNG). Smaller over network."""
        data = self._device.shell("screencap -p", encoding=None, timeout=10)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode PNG screencap")
        return frame

    def _capture_raw(self) -> np.ndarray:
        """Capture via screencap (RAW RGBA). No compression overhead."""
        data = self._device.shell("screencap", encoding=None, timeout=10)
        # Header: 16 bytes on Android 14+ (w, h, format, colorSpace)
        if len(data) < 16:
            raise RuntimeError(f"Screencap too short: {len(data)} bytes")
        w, h, fmt, _ = struct.unpack_from("<IIII", data, 0)
        pixels = data[16:]
        expected = w * h * 4
        if len(pixels) < expected:
            raise RuntimeError(
                f"Screencap data truncated: {len(pixels)}/{expected}"
            )
        arr = np.frombuffer(pixels[:expected], dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    # Alias for backward compatibility (navigator uses .grab())
    grab = capture

    @property
    def serial(self) -> str:
        return self._serial

    def close(self) -> None:
        """Release resources."""
        self._device = None


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_capture(
    adb_serial: str = "localhost:5555",
    backend: str = "raw",
    **kwargs,
):
    """Create a capture instance for the given backend.

    Args:
        adb_serial: ADB device serial.
        backend: ``"raw"``, ``"png"``, ``"scrcpy"``, or ``"dashboard"``.
        **kwargs: Extra arguments forwarded to the backend constructor.

    Returns:
        An object with ``.capture()`` / ``.grab()`` / ``.close()`` API.
    """
    if backend == "dashboard":
        try:
            from hillclimb.dashboard_capture import DashboardCapture

            dash_kwargs = {}
            if "dashboard_url" in kwargs:
                dash_kwargs["dashboard_url"] = kwargs["dashboard_url"]
            return DashboardCapture(adb_serial=adb_serial, **dash_kwargs)
        except Exception as e:
            logger.warning(
                "dashboard capture failed on %s: %s — falling back to raw",
                adb_serial, e,
            )
            return ScreenCapture(adb_serial=adb_serial, backend="raw")
    if backend == "scrcpy":
        try:
            from hillclimb.scrcpy_capture import ScrcpyCapture

            # Filter out dashboard_url — not accepted by ScrcpyCapture
            scrcpy_kwargs = {
                k: v for k, v in kwargs.items() if k != "dashboard_url"
            }
            return ScrcpyCapture(adb_serial=adb_serial, **scrcpy_kwargs)
        except Exception as e:
            logger.warning(
                "scrcpy capture failed on %s: %s — falling back to raw",
                adb_serial, e,
            )
            return ScreenCapture(adb_serial=adb_serial, backend="raw")
    return ScreenCapture(adb_serial=adb_serial, backend=backend)


def main() -> None:
    """Smoke test: capture and save a frame from the first emulator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test screen capture")
    parser.add_argument(
        "--serial", default="localhost:5555", help="ADB serial"
    )
    parser.add_argument(
        "--backend", default="png", choices=["png", "raw", "scrcpy", "dashboard"],
    )
    parser.add_argument("--benchmark", type=int, default=0, help="Run N captures")
    args = parser.parse_args()

    cap = create_capture(adb_serial=args.serial, backend=args.backend)
    frame = cap.capture()
    print(f"Captured: {frame.shape} from {args.serial} (backend={args.backend})")
    cv2.imwrite("capture_test.png", frame)
    print("Saved to capture_test.png")

    if args.benchmark > 0:
        times = []
        for i in range(args.benchmark):
            t0 = time.time()
            cap.capture()
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        print(f"Benchmark: {avg*1000:.0f}ms avg, ~{1/avg:.1f} FPS ({args.benchmark} frames)")

    cap.close()


if __name__ == "__main__":
    main()
