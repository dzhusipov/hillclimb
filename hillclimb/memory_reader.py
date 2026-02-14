"""
Memory reader for HCR2 — reads car position directly from game process memory.

Uses `nodefinder` C tool that scans the largest scudo:primary region (~22MB)
to find the car's Cocos2d-x Node via structural pattern + delta filtering,
then streams pos_x/pos_y readings via binary protocol.

Structural pattern: scale [1,1,1] + rotation sin²+cos²=1 + pos_x copy at +108
+ car body markers (±0.707 at +96/+100) + pos_Y duplicate at [-36]=[-32].
Typically finds ~2-5 exact car body Nodes out of 22MB, then delta filter picks
the live (moving) one.

Anti-cheat safe: reads ~22MB via process_vm_readv (limit ~70MB).

Usage:
    reader = MemoryReader(container="hcr2-0")
    if reader.scan():        # find car position (~2-3s when car is moving)
        while racing:
            state = reader.read()  # ~1ms per read
            print(state.pos_x, state.distance)
    reader.stop()
"""

import struct
import subprocess
import logging
import time
import select
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CarState:
    """Car physics state read from memory."""
    pos_x: float = 0.0
    pos_y: float = 0.0
    vel_x: float = 0.0    # derived from delta pos_x / delta_t
    vel_y: float = 0.0    # derived from delta pos_y / delta_t
    timestamp: float = 0.0
    valid: bool = False

    # Initial position (set at scan time)
    _initial_x: float = field(default=0.0, repr=False)

    @property
    def distance(self) -> float:
        """Distance traveled since race start (meters)."""
        return self.pos_x - self._initial_x if self.valid else 0.0

    @property
    def speed(self) -> float:
        """Speed magnitude (m/s)."""
        return (self.vel_x ** 2 + self.vel_y ** 2) ** 0.5 if self.valid else 0.0


class MemoryReader:
    """Reads HCR2 car position from process memory via nodefinder.

    nodefinder runs inside the Docker container, scans the largest
    scudo:primary region (~22MB) for car body Nodes, and streams binary
    pos_x/pos_y readings via stdout pipe.
    """

    def __init__(self, container: str = "hcr2-0",
                 package: str = "com.fingersoft.hcr2",
                 interval_ms: int = 20,
                 max_sec: int = 120):
        self.container = container
        self.package = package
        self.interval_ms = interval_ms
        self.max_sec = max_sec

        self._proc: Optional[subprocess.Popen] = None
        self._pid: Optional[int] = None
        self._initial_x: float = 0.0
        self._initial_y: float = 0.0
        self._last_x: float = 0.0
        self._last_y: float = 0.0
        self._last_time: float = 0.0
        self._scanned: bool = False

    @property
    def is_active(self) -> bool:
        """True if nodefinder is running and producing data."""
        return self._scanned and self._proc is not None and self._proc.poll() is None

    def _get_pid(self) -> Optional[int]:
        """Get HCR2 game PID inside the container."""
        try:
            result = subprocess.run(
                ["docker", "exec", self.container, "pidof", self.package],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError) as e:
            logger.error("Failed to get PID on %s: %s", self.container, e)
        return None

    def scan(self, timeout: float = 10.0) -> bool:
        """Scan memory to find car position address and start streaming.

        Call this at the START of each race, after the car begins moving.
        Takes ~3s (cocos2d scan + 2s wait + delta filter + validation).

        Returns True if car position was found and streaming started.
        """
        self.stop()

        # Get PID
        self._pid = self._get_pid()
        if not self._pid:
            logger.warning("Game not running on %s", self.container)
            return False

        # Launch nodefinder
        try:
            self._proc = subprocess.Popen(
                ["docker", "exec", self.container,
                 "/data/local/tmp/nodefinder",
                 str(self._pid),
                 str(self.interval_ms),
                 str(self.max_sec),
                 "--wait=2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as e:
            logger.error("Failed to launch nodefinder: %s", e)
            return False

        # Read header: "OK\n" + 2 floats (initial_x, initial_y)
        try:
            header_line = b""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                ready, _, _ = select.select([self._proc.stdout], [], [], 0.5)
                if ready:
                    b = self._proc.stdout.read(1)
                    if not b:
                        break
                    header_line += b
                    if b == b'\n':
                        break

            header_str = header_line.decode('utf-8', errors='replace').strip()

            if header_str.startswith("ERR:"):
                logger.warning("nodefinder error on %s: %s", self.container, header_str)
                self._kill()
                return False

            if header_str != "OK":
                logger.warning("Unexpected nodefinder header on %s: %r", self.container, header_str)
                self._kill()
                return False

            # Read initial position (2 floats = 8 bytes)
            initial_data = self._read_exact(8, timeout=2.0)
            if initial_data is None or len(initial_data) < 8:
                logger.warning("Failed to read initial position from nodefinder")
                self._kill()
                return False

            self._initial_x, self._initial_y = struct.unpack('<ff', initial_data)
            self._last_x = self._initial_x
            self._last_y = self._initial_y
            self._last_time = time.monotonic()
            self._scanned = True

            logger.info("MemoryReader attached: %s PID=%d initial=(%.1f, %.1f)",
                        self.container, self._pid, self._initial_x, self._initial_y)
            return True

        except Exception as e:
            logger.error("nodefinder header read failed: %s", e)
            self._kill()
            return False

    def read(self) -> CarState:
        """Read current car position from the nodefinder pipe.

        Returns CarState with valid=False if no data available or stream ended.
        Non-blocking: returns the latest available sample, or stale data if none ready.
        """
        if not self.is_active:
            return CarState()

        # Read all available samples, keep the latest
        # Protocol: normal frame = 8 bytes [pos_x, pos_y]
        #           switch marker = 12 bytes [NaN, new_initial_x, new_initial_y]
        pos_x = None
        pos_y = None
        now = time.monotonic()

        while True:
            ready, _, _ = select.select([self._proc.stdout], [], [], 0)
            if not ready:
                break
            data = self._proc.stdout.read(8)
            if not data or len(data) < 8:
                self._scanned = False
                return CarState()
            x, y = struct.unpack('<ff', data)

            # NaN marker = address switch, next 4 bytes = new_initial_y
            if x != x:  # NaN check
                extra = self._read_exact(4, timeout=1.0)
                if extra and len(extra) == 4:
                    new_initial_y = struct.unpack('<f', extra)[0]
                    old_initial = self._initial_x
                    self._initial_x = y  # "y" field carries new_initial_x
                    self._initial_y = new_initial_y
                    logger.info("MemoryReader: address switch, new initial=(%.1f, %.1f) old=%.1f",
                                self._initial_x, self._initial_y, old_initial)
                continue

            pos_x, pos_y = x, y

        if pos_x is None:
            # No new data — return last known state
            if self._last_time > 0:
                return CarState(
                    pos_x=self._last_x,
                    pos_y=self._last_y,
                    vel_x=0.0, vel_y=0.0,
                    timestamp=self._last_time,
                    valid=True,
                    _initial_x=self._initial_x,
                )
            return CarState()

        # Compute velocity from position delta
        dt = now - self._last_time if self._last_time > 0 else 0.02
        dt = max(dt, 0.001)  # avoid division by zero
        vel_x = (pos_x - self._last_x) / dt
        vel_y = (pos_y - self._last_y) / dt

        self._last_x = pos_x
        self._last_y = pos_y
        self._last_time = now

        return CarState(
            pos_x=pos_x,
            pos_y=pos_y,
            vel_x=vel_x,
            vel_y=vel_y,
            timestamp=now,
            valid=True,
            _initial_x=self._initial_x,
        )

    def stop(self):
        """Stop nodefinder and reset state."""
        self._kill()
        self._scanned = False
        self._pid = None
        self._initial_x = 0.0
        self._initial_y = 0.0
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_time = 0.0

    def _kill(self):
        """Kill the nodefinder subprocess."""
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._proc.kill()
                except OSError:
                    pass
            self._proc = None

    def _read_exact(self, n: int, timeout: float = 2.0) -> Optional[bytes]:
        """Read exactly n bytes from nodefinder stdout with timeout."""
        data = b""
        deadline = time.monotonic() + timeout
        while len(data) < n and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 0.5))
            if ready:
                chunk = self._proc.stdout.read(n - len(data))
                if not chunk:
                    return None
                data += chunk
        return data if len(data) == n else None

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    container = sys.argv[1] if len(sys.argv) > 1 else "hcr2-test"
    reader = MemoryReader(container=container, interval_ms=100)

    print(f"Scanning memory on {container}...")
    if not reader.scan(timeout=10):
        print("Failed to scan. Is the game in RACING state with car moving?")
        sys.exit(1)

    print(f"Streaming! initial=({reader._initial_x:.1f}, {reader._initial_y:.1f})")
    print(f"{'time':>8s} {'pos_x':>10s} {'pos_y':>10s} {'vel_x':>8s} {'vel_y':>8s} {'dist':>8s} {'speed':>8s}")

    try:
        t0 = time.monotonic()
        while reader.is_active:
            state = reader.read()
            if state.valid:
                elapsed = time.monotonic() - t0
                print(f"{elapsed:8.2f} {state.pos_x:10.2f} {state.pos_y:10.2f} "
                      f"{state.vel_x:8.2f} {state.vel_y:8.2f} "
                      f"{state.distance:8.1f} {state.speed:8.2f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.stop()
