"""MJPEG streaming from emulators via ADB screencap."""

import subprocess
import threading
import time
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Capture settings
CAPTURE_INTERVAL = 0.5  # seconds between frames
JPEG_QUALITY = 60
OFFLINE_FRAME_INTERVAL = 3.0  # slower polling when emulator is offline


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
        self._connected = False
        self._rotated = False
        self._capture_size: tuple[int, int] = (0, 0)  # (w, h) of raw capture

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
            target=self._capture_loop, daemon=True, name=f"stream-{self.name}"
        )
        self._thread.start()
        logger.info("Started stream for %s", self.name)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Stopped stream for %s", self.name)

    def _ensure_connected(self) -> bool:
        """Connect ADB to emulator."""
        try:
            result = subprocess.run(
                ["adb", "connect", self.adb_host],
                capture_output=True, text=True, timeout=5
            )
            self._connected = "connected" in result.stdout.lower()
            return self._connected
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._connected = False
            return False

    def _capture_frame(self) -> Optional[bytes]:
        """Capture a single frame via ADB screencap and encode as JPEG."""
        try:
            result = subprocess.run(
                ["adb", "-s", self.adb_host, "exec-out", "screencap", "-p"],
                capture_output=True, timeout=10
            )
            if result.returncode != 0 or not result.stdout:
                return None

            # Decode PNG from ADB
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return None

            # Track raw capture size and rotate portrait to landscape
            h, w = img.shape[:2]
            self._capture_size = (w, h)
            if h > w:
                self._rotated = True
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                self._rotated = False

            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            _, jpeg = cv2.imencode(".jpg", img, encode_params)
            return jpeg.tobytes()
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug("Capture failed for %s: %s", self.name, e)
            return None

    def _capture_loop(self) -> None:
        """Background capture loop."""
        while self._running:
            if not self._connected:
                self._ensure_connected()
                if not self._connected:
                    time.sleep(OFFLINE_FRAME_INTERVAL)
                    continue

            frame = self._capture_frame()
            if frame:
                with self._lock:
                    self._frame = frame
                time.sleep(CAPTURE_INTERVAL)
            else:
                # Lost connection — retry slower
                self._connected = False
                time.sleep(OFFLINE_FRAME_INTERVAL)


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
        """Start streams for the given emulator IDs."""
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
        return True, (480, 800)  # default: portrait, needs rotation
