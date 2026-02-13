"""Emulator stream manager: docker exec screencap → JPEG, round-robin.

Single background thread captures one emulator at a time via
`docker exec <container> screencap -p`, converts PNG→JPEG, and stores
the last frame per emulator. Each emulator refreshes every ~4s (8 emus × 0.5s).

No ADB dependency — uses docker.sock directly.
"""

import subprocess
import threading
import time
import logging
from itertools import cycle
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

JPEG_QUALITY = 65
CAPTURE_INTERVAL = 0.5   # seconds between emulators
CAPTURE_TIMEOUT = 5      # docker exec timeout


class StreamManager:
    """Captures frames from all emulators in a single round-robin thread."""

    def __init__(self):
        self._frames: dict[int, bytes] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._emu_ids: list[int] = []
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        # Capture metadata for coordinate mapping
        self._rotated: dict[int, bool] = {}
        self._capture_size: dict[int, tuple[int, int]] = {}

    def start_streams(self, emu_ids: list[int]) -> None:
        """Start the round-robin capture thread for given emulator IDs."""
        with self._lock:
            self._emu_ids = sorted(emu_ids)
        if self._running or not self._emu_ids:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="stream-roundrobin",
        )
        self._thread.start()
        logger.info("Started round-robin capture for emulators: %s", self._emu_ids)

    def ensure_emulator(self, emu_id: int) -> None:
        """Add an emulator to the round-robin cycle if not already tracked."""
        with self._lock:
            if emu_id not in self._emu_ids:
                self._emu_ids = sorted(self._emu_ids + [emu_id])
                logger.info("Added emulator %d to capture cycle", emu_id)

    def stop_all(self) -> None:
        """Stop the capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def get_frame(self, emu_id: int) -> Optional[bytes]:
        """Return last captured JPEG for the emulator, or None."""
        with self._lock:
            return self._frames.get(emu_id)

    def get_stream_info(self, emu_id: int) -> tuple[bool, tuple[int, int]]:
        """Return (rotated, (capture_w, capture_h)) for coordinate mapping."""
        with self._lock:
            rotated = self._rotated.get(emu_id, True)
            size = self._capture_size.get(emu_id, (480, 800))
        return rotated, size

    def _capture_loop(self) -> None:
        """Round-robin: capture one emulator at a time, sleep between."""
        idx = 0
        while self._running:
            with self._lock:
                ids = list(self._emu_ids)
            if not ids:
                time.sleep(CAPTURE_INTERVAL)
                continue
            idx = idx % len(ids)
            emu_id = ids[idx]
            idx += 1
            try:
                self._capture_one(emu_id)
            except Exception as e:
                logger.debug("Capture error for hcr2-%d: %s", emu_id, e)
            time.sleep(CAPTURE_INTERVAL)

    def _capture_one(self, emu_id: int) -> None:
        """Capture a single frame from one emulator via docker exec."""
        container = f"hcr2-{emu_id}"
        try:
            result = subprocess.run(
                ["docker", "exec", container, "screencap", "-p"],
                capture_output=True, timeout=CAPTURE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.debug("docker exec timeout for %s", container)
            return

        if result.returncode != 0 or not result.stdout:
            return

        # Decode PNG
        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return

        h, w = img.shape[:2]
        # Track raw capture size and rotation
        if h > w:
            rotated = True
            capture_size = (w, h)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = False
            capture_size = (w, h)

        # JPEG encode
        _, jpeg = cv2.imencode(".jpg", img, self._encode_params)
        jpeg_bytes = jpeg.tobytes()

        with self._lock:
            self._frames[emu_id] = jpeg_bytes
            self._rotated[emu_id] = rotated
            self._capture_size[emu_id] = capture_size
