"""Screen capture via scrcpy-server: low-latency H.264 streaming.

Pushes scrcpy-server.jar to the device, starts it in raw_stream mode
(pure H.264 Annex B, no metadata), decodes with PyAV in a background
thread, and exposes the latest frame via capture() ≈ 0ms.

Replaces ~300ms ADB screencap with near-instant frame reads.

No dependency on ``adbutils`` — uses only the ``adb`` CLI binary,
so it works both in the main conda env and in the dashboard Docker
container.
"""

from __future__ import annotations

import logging
import os
import random
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JAR = str(_PROJECT_ROOT / "vendor" / "scrcpy-server.jar")
_DEVICE_JAR_PATH = "/data/local/tmp/scrcpy-server.jar"


class ScrcpyCapture:
    """Capture frames from scrcpy-server H.264 stream via PyAV."""

    def __init__(
        self,
        adb_serial: str = "localhost:5555",
        max_fps: int = 15,
        max_size: int = 800,
        bitrate: int = 2_000_000,
        server_jar: str = _DEFAULT_JAR,
        stale_timeout: float = 2.0,
    ) -> None:
        self._serial = adb_serial
        self._max_fps = max_fps
        self._max_size = max_size
        self._bitrate = bitrate
        self._stale_timeout = stale_timeout

        # Resolve relative jar path to project root
        jar_path = Path(server_jar)
        if not jar_path.is_absolute():
            jar_path = _PROJECT_ROOT / jar_path
        self._server_jar = str(jar_path)

        self._running = False
        self._frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._frame_time: float = 0.0
        self._decode_thread: threading.Thread | None = None
        self._server_proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._local_port: int = 0
        self._scid: str = ""
        self._restart_count = 0
        self._max_restarts = 3

        self._ensure_adb_connected()
        self._start()

    # ------------------------------------------------------------------
    # ADB helpers (subprocess only, no adbutils)
    # ------------------------------------------------------------------

    def _ensure_adb_connected(self) -> None:
        """Make sure the ADB server knows about this device."""
        subprocess.run(
            ["adb", "connect", self._serial],
            capture_output=True, timeout=5,
        )

    def _adb_shell(self, cmd: str, timeout: int = 10) -> str:
        """Run ``adb shell <cmd>`` and return stdout."""
        result = subprocess.run(
            ["adb", "-s", self._serial, "shell", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _push_jar(self) -> None:
        """Push scrcpy-server.jar to device (skip if same size)."""
        local_size = os.path.getsize(self._server_jar)
        try:
            out = self._adb_shell(f"wc -c < {_DEVICE_JAR_PATH}", timeout=5)
            remote_size = int(out.strip())
            if remote_size == local_size:
                logger.debug("JAR already on device (%d bytes)", local_size)
                return
        except Exception:
            pass
        logger.info("Pushing scrcpy-server.jar (%d bytes) to %s",
                     local_size, self._serial)
        subprocess.check_call(
            ["adb", "-s", self._serial, "push",
             self._server_jar, _DEVICE_JAR_PATH],
            timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _kill_existing_server(self) -> None:
        """Kill only our own scrcpy-server (by scid), not other instances.

        Avoids killing dashboard's scrcpy servers when training starts.
        """
        if self._scid:
            try:
                self._adb_shell(
                    f"pkill -f 'scid={self._scid}'", timeout=5,
                )
            except Exception:
                pass
            time.sleep(0.3)

    def _start(self) -> None:
        """Start scrcpy-server, set up tunnel, launch decode thread."""
        self._push_jar()
        self._kill_existing_server()

        # Unique session ID
        self._scid = f"{random.randint(0, 0x7FFFFFFF):08x}"
        socket_name = f"scrcpy_{self._scid}"

        # Start server via ADB shell (non-blocking subprocess)
        cmd = [
            "adb", "-s", self._serial, "shell",
            f"CLASSPATH={_DEVICE_JAR_PATH}",
            "app_process", "/", "com.genymobile.scrcpy.Server",
            "2.4",
            "tunnel_forward=true",
            "audio=false",
            "control=false",
            "cleanup=false",
            "power_on=false",
            "raw_stream=true",
            f"max_size={self._max_size}",
            f"max_fps={self._max_fps}",
            f"video_bit_rate={self._bitrate}",
            f"scid={self._scid}",
        ]
        logger.info("Starting scrcpy-server on %s (scid=%s)",
                     self._serial, self._scid)
        self._server_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # Wait for server to bind socket
        time.sleep(1.5)

        # Check server is still alive
        if self._server_proc.poll() is not None:
            stderr = self._server_proc.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"scrcpy-server exited immediately: {stderr[:500]}"
            )

        # Set up ADB forward tunnel
        result = subprocess.run(
            ["adb", "-s", self._serial, "forward",
             "tcp:0", f"localabstract:{socket_name}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"adb forward failed: {result.stderr}")
        self._local_port = int(result.stdout.strip())
        logger.info("ADB forward → localhost:%d", self._local_port)

        # Connect TCP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        self._sock.connect(("127.0.0.1", self._local_port))
        self._sock.settimeout(2.0)
        logger.info("Connected to scrcpy-server on %s", self._serial)

        # Start decode thread
        self._running = True
        self._restart_count = 0
        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True, name=f"scrcpy-{self._serial}",
        )
        self._decode_thread.start()

        # Wait for first frame (up to 5 seconds)
        deadline = time.time() + 5.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.1)
        if self._frame is None:
            raise RuntimeError(
                f"No frame received from scrcpy-server on {self._serial} "
                "within 5 seconds"
            )
        logger.info("First frame received from %s: %s",
                     self._serial, self._frame.shape)

    def _decode_loop(self) -> None:
        """Background thread: read socket → parse H.264 → decode → buffer."""
        import av

        codec = av.CodecContext.create("h264", "r")

        while self._running:
            try:
                data = self._sock.recv(65536)
                if not data:
                    logger.warning("scrcpy socket closed on %s", self._serial)
                    break

                packets = codec.parse(data)
                for pkt in packets:
                    frames = codec.decode(pkt)
                    for frame in frames:
                        arr = frame.to_ndarray(format="bgr24")
                        # Auto-rotate: ReDroid is 480x800 portrait,
                        # game is landscape
                        h, w = arr.shape[:2]
                        if h > w:
                            arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
                        with self._frame_lock:
                            self._frame = arr
                            self._frame_time = time.time()

            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    logger.warning("Socket error on %s: %s", self._serial, e)
                break
            except Exception as e:
                if self._running:
                    logger.error("Decode error on %s: %s", self._serial, e)
                break

        if self._running:
            logger.warning("Decode loop exited on %s, will attempt restart",
                           self._serial)
            self._try_restart()

    def _try_restart(self) -> None:
        """Attempt to restart scrcpy-server after failure."""
        self._restart_count += 1
        if self._restart_count > self._max_restarts:
            logger.error("Max restarts (%d) reached on %s",
                         self._max_restarts, self._serial)
            return

        logger.info("Restarting scrcpy-server on %s (attempt %d/%d)",
                     self._serial, self._restart_count, self._max_restarts)
        backoff = self._restart_count * 1.0
        time.sleep(backoff)

        try:
            self._cleanup_server()
            self._start()
        except Exception as e:
            logger.error("Restart failed on %s: %s", self._serial, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Return latest frame (BGR, H×W×3). Near-instant (~0ms).

        Falls back to ADB screencap if stream is stale (>stale_timeout).
        If stale_timeout <= 0, always returns the last buffered frame
        without fallback (useful for dashboard where stale == static screen).
        """
        with self._frame_lock:
            frame = self._frame
            age = time.time() - self._frame_time if self._frame_time else 999

        # No stale check — always return last frame if available
        if self._stale_timeout <= 0:
            if frame is not None:
                return frame
            # No frame yet — must fallback
            return self._screencap_fallback()

        if frame is not None and age < self._stale_timeout:
            return frame

        # Fallback to screencap
        logger.warning("scrcpy frame stale (%.1fs) on %s — screencap fallback",
                       age, self._serial)
        return self._screencap_fallback()

    # Alias for navigator compatibility
    grab = capture

    @property
    def serial(self) -> str:
        return self._serial

    def close(self) -> None:
        """Stop server, close socket, join thread, remove forward."""
        self._running = False
        self._cleanup_server()
        if self._decode_thread and self._decode_thread.is_alive():
            self._decode_thread.join(timeout=3.0)
        self._decode_thread = None
        logger.info("ScrcpyCapture closed for %s", self._serial)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cleanup_server(self) -> None:
        """Stop server process, close socket, remove ADB forward."""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        if self._server_proc:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=3)
            except Exception:
                try:
                    self._server_proc.kill()
                except Exception:
                    pass
            self._server_proc = None

        if self._local_port:
            try:
                subprocess.run(
                    ["adb", "-s", self._serial, "forward",
                     "--remove", f"tcp:{self._local_port}"],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
            self._local_port = 0

        self._kill_existing_server()

    def _screencap_fallback(self) -> np.ndarray:
        """Fallback: ADB screencap (slow but reliable)."""
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
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
