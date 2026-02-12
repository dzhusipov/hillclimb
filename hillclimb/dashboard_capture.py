"""Capture frames from the dashboard WebSocket stream.

Connects to ``ws://HOST:8150/ws/stream/{emu_id}`` and buffers the latest
JPEG frame decoded to BGR numpy array.  ``capture()`` returns instantly
(~0 ms) from the buffer — no ADB or scrcpy interaction at all.

This avoids scrcpy-server conflicts: the dashboard owns scrcpy, and
training reads frames through this WebSocket relay.
"""

from __future__ import annotations

import logging
import struct
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DashboardCapture:
    """Capture frames via dashboard WebSocket (JPEG stream)."""

    def __init__(
        self,
        adb_serial: str = "localhost:5555",
        dashboard_url: str = "ws://localhost:8150",
    ) -> None:
        self._serial = adb_serial

        # Derive emu_id from port: localhost:5555 → 0, localhost:5557 → 2
        port = int(adb_serial.split(":")[-1])
        self._emu_id = port - 5555
        self._ws_url = f"{dashboard_url}/ws/stream/{self._emu_id}"

        self._frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._frame_time: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

        self._start()

    def _start(self) -> None:
        """Launch the WebSocket receiver thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True,
            name=f"dash-cap-{self._emu_id}",
        )
        self._thread.start()

        # Wait for first frame (up to 5 seconds)
        deadline = time.time() + 5.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.1)
        if self._frame is not None:
            logger.info(
                "DashboardCapture connected: emu=%d shape=%s",
                self._emu_id, self._frame.shape,
            )
        else:
            logger.warning(
                "DashboardCapture: no frame in 5s from %s — "
                "will keep retrying in background",
                self._ws_url,
            )

    def _recv_loop(self) -> None:
        """Background: connect to WS, recv JPEG bytes, decode, buffer."""
        from websockets.sync.client import connect, ClientConnection
        from websockets.exceptions import (
            ConnectionClosed,
            InvalidURI,
            InvalidHandshake,
        )

        backoff = 1.0
        max_backoff = 10.0

        while self._running:
            try:
                logger.debug("Connecting to %s", self._ws_url)
                ws: ClientConnection = connect(
                    self._ws_url,
                    open_timeout=5,
                    close_timeout=2,
                )
            except (OSError, InvalidURI, InvalidHandshake, Exception) as e:
                if self._running:
                    logger.debug(
                        "WS connect failed (%s): %s — retry in %.0fs",
                        self._ws_url, e, backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                continue

            backoff = 1.0  # reset on successful connect
            logger.debug("WS connected to %s", self._ws_url)

            try:
                for message in ws:
                    if not self._running:
                        break
                    if not isinstance(message, bytes):
                        continue
                    arr = np.frombuffer(message, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    with self._frame_lock:
                        self._frame = img
                        self._frame_time = time.time()
            except (ConnectionClosed, OSError) as e:
                if self._running:
                    logger.debug("WS disconnected (%s): %s", self._ws_url, e)
            except Exception as e:
                if self._running:
                    logger.warning("WS recv error (%s): %s", self._ws_url, e)
            finally:
                try:
                    ws.close()
                except Exception:
                    pass

            if self._running:
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Public API (same as ScrcpyCapture / ScreenCapture)
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Return latest buffered frame (BGR, H×W×3). ~0 ms.

        Falls back to ADB screencap if WS stream is stale (>5s).
        """
        with self._frame_lock:
            frame = self._frame
            age = time.time() - self._frame_time if self._frame_time else 999

        if frame is not None and age < 5.0:
            return frame

        logger.warning(
            "Dashboard frame stale (%.1fs) emu=%d — screencap fallback",
            age, self._emu_id,
        )
        return self._screencap_fallback()

    grab = capture

    @property
    def serial(self) -> str:
        return self._serial

    def close(self) -> None:
        """Stop receiver thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        logger.info("DashboardCapture closed for emu=%d", self._emu_id)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _screencap_fallback(self) -> np.ndarray:
        """ADB screencap fallback (slow but works without dashboard)."""
        subprocess.run(
            ["adb", "connect", self._serial],
            capture_output=True, timeout=5,
        )
        result = subprocess.run(
            ["adb", "-s", self._serial, "exec-out", "screencap"],
            capture_output=True, timeout=10,
        )
        data = result.stdout
        if len(data) < 16:
            raise RuntimeError(f"Screencap too short: {len(data)} bytes")
        w, h, fmt, _ = struct.unpack_from("<IIII", data, 0)
        pixels = data[16:]
        expected = w * h * 4
        if len(pixels) < expected:
            raise RuntimeError(
                f"Screencap truncated: {len(pixels)}/{expected}"
            )
        arr = np.frombuffer(pixels[:expected], dtype=np.uint8).reshape(h, w, 4)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        # Auto-rotate portrait → landscape
        rh, rw = bgr.shape[:2]
        if rh > rw:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        return bgr
