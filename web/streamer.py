"""Emulator stream manager: scrcpy H.264 → JPEG frames for dashboard.

Uses ScrcpyCapture for near-instant frame grabs (~0 ms).
Falls back to ADB screencap if scrcpy fails to start.
"""

import subprocess
import threading
import time
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Capture settings
JPEG_QUALITY = 65
SCRCPY_FPS = 30
SCRCPY_INTERVAL = 1.0 / SCRCPY_FPS       # ~33 ms between JPEG encodes
SCREENCAP_INTERVAL = 0.5                   # fallback: 2 FPS
OFFLINE_RETRY_INTERVAL = 3.0


class EmulatorStream:
    """Captures frames from a single emulator in a background thread."""

    def __init__(self, emu_id: int):
        self.emu_id = emu_id
        self.name = f"hcr2-{emu_id}"
        self.adb_host = f"{self.name}:5555"

        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._scrcpy = None
        self._connected = False
        self._rotated = False
        self._capture_size: tuple[int, int] = (0, 0)

    @property
    def frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    @property
    def rotated(self) -> bool:
        return self._rotated

    @property
    def capture_size(self) -> tuple[int, int]:
        return self._capture_size

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True, name=f"stream-{self.name}",
        )
        self._thread.start()
        logger.info("Started stream for %s", self.name)

    def stop(self) -> None:
        self._running = False
        if self._scrcpy:
            try:
                self._scrcpy.close()
            except Exception:
                pass
            self._scrcpy = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Stopped stream for %s", self.name)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _stream_loop(self) -> None:
        """Background loop: grab frames, JPEG-encode, store."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        prev_img = None   # track numpy array identity to skip unchanged frames

        # Try scrcpy first
        self._try_init_scrcpy()

        while self._running:
            try:
                img = self._grab_frame()
                if img is None:
                    time.sleep(OFFLINE_RETRY_INTERVAL)
                    continue

                # Skip JPEG encode if frame hasn't changed (same numpy object).
                # On static screens scrcpy returns the same cached array,
                # so we avoid 30 FPS of redundant encodes + WS sends.
                if img is prev_img:
                    time.sleep(SCRCPY_INTERVAL)
                    continue
                prev_img = img

                # JPEG encode
                _, jpeg = cv2.imencode(".jpg", img, encode_params)
                with self._lock:
                    self._frame = jpeg.tobytes()

                interval = SCRCPY_INTERVAL if self._scrcpy else SCREENCAP_INTERVAL
                time.sleep(interval)

            except Exception as e:
                logger.debug("Stream error for %s: %s", self.name, e)
                # If scrcpy died, fall back to screencap
                if self._scrcpy:
                    try:
                        self._scrcpy.close()
                    except Exception:
                        pass
                    self._scrcpy = None
                time.sleep(OFFLINE_RETRY_INTERVAL)

    def _try_init_scrcpy(self) -> None:
        """Attempt to start ScrcpyCapture for this emulator."""
        try:
            from hillclimb.scrcpy_capture import ScrcpyCapture
            self._scrcpy = ScrcpyCapture(
                adb_serial=self.adb_host,
                max_fps=SCRCPY_FPS,
                max_size=800,
                bitrate=2_000_000,
                stale_timeout=0,  # static screen = last frame, no fallback
            )
            # ScrcpyCapture already rotates portrait → landscape
            self._rotated = True          # raw capture was portrait
            self._capture_size = (480, 800)  # raw (pre-rotation) size
            logger.info("ScrcpyCapture active for %s", self.name)
        except Exception as e:
            logger.warning("ScrcpyCapture failed for %s: %s — using screencap",
                           self.name, e)
            self._scrcpy = None

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Get a BGR numpy frame from scrcpy or screencap fallback."""
        if self._scrcpy:
            return self._scrcpy.capture()

        # Screencap fallback
        return self._screencap_frame()

    # ------------------------------------------------------------------
    # Screencap fallback
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        try:
            result = subprocess.run(
                ["adb", "connect", self.adb_host],
                capture_output=True, text=True, timeout=5,
            )
            self._connected = "connected" in result.stdout.lower()
            return self._connected
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._connected = False
            return False

    def _screencap_frame(self) -> Optional[np.ndarray]:
        """Capture via ADB screencap -p (PNG). Slow fallback."""
        if not self._connected:
            self._ensure_connected()
            if not self._connected:
                return None
        try:
            result = subprocess.run(
                ["adb", "-s", self.adb_host, "exec-out", "screencap", "-p"],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0 or not result.stdout:
                self._connected = False
                return None

            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return None

            h, w = img.shape[:2]
            self._capture_size = (w, h)
            if h > w:
                self._rotated = True
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                self._rotated = False
            return img
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug("Screencap failed for %s: %s", self.name, e)
            self._connected = False
            return None


class StreamManager:
    """Manages streams for all emulators."""

    def __init__(self):
        self._streams: dict[int, EmulatorStream] = {}
        self._lock = threading.Lock()

    def get_or_create(self, emu_id: int) -> EmulatorStream:
        with self._lock:
            if emu_id not in self._streams:
                stream = EmulatorStream(emu_id)
                stream.start()
                self._streams[emu_id] = stream
            return self._streams[emu_id]

    def start_streams(self, emu_ids: list[int]) -> None:
        for emu_id in emu_ids:
            self.get_or_create(emu_id)

    def stop_all(self) -> None:
        with self._lock:
            for stream in self._streams.values():
                stream.stop()
            self._streams.clear()

    def get_frame(self, emu_id: int) -> Optional[bytes]:
        stream = self._streams.get(emu_id)
        if stream:
            return stream.frame
        return None

    def get_stream_info(self, emu_id: int) -> tuple[bool, tuple[int, int]]:
        """Return (rotated, (capture_w, capture_h)) for coordinate mapping."""
        stream = self._streams.get(emu_id)
        if stream and stream.capture_size != (0, 0):
            return stream.rotated, stream.capture_size
        return True, (480, 800)
