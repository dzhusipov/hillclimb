"""
Memory reader for HCR2 — reads car physics data directly from game process memory.

Uses the car body signature (70.0, 35.5, 0.5, 0.5, 140.0, 71.0) to locate
the b2Body struct in heap, then reads position, velocity at known offsets.

Requires `safescan` and `safememserver` compiled and deployed at
/data/local/tmp/ inside the emulator container.

Anti-cheat safe: never holds /proc/PID/mem fd for more than ~1ms.
- safescan: opens/closes fd per memory region during scan
- safememserver: opens/closes fd per read request

Usage:
    reader = MemoryReader(container="hcr2-0")
    reader.attach()       # find game PID + body address via safescan
    data = reader.read()  # fast pread via safememserver (open→read→close)
    print(data.pos_x, data.vel_x)
    reader.detach()
"""

import os
import subprocess
import struct
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Car body struct offsets from body_x address
OFFSET_POS_X = 0
OFFSET_POS_Y = -4
OFFSET_VEL_X = 56
OFFSET_VEL_Y = 60
OFFSET_ROT_SIN = -16
OFFSET_ROT_COS = -12

# AABB signature: 6 consecutive floats unique to car body
BODY_SIGNATURE = struct.pack('<6f', 70.0, 35.5, 0.5, 0.5, 140.0, 71.0)
SIG_OFFSET_FROM_BODY_X = 28  # signature starts at body_x + 28

# Starting X position offset (displayed_distance ≈ pos_x - START_X_OFFSET)
START_X_OFFSET = 83.0

# Read span: from body_x - 16 to body_x + 64 = 80 bytes
READ_BASE_OFFSET = -16
READ_SIZE = 80


@dataclass
class CarState:
    """Car physics state read from memory."""
    pos_x: float = 0.0
    pos_y: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    rot_sin: float = 0.0
    rot_cos: float = 0.0
    timestamp: float = 0.0
    valid: bool = False

    @property
    def distance(self) -> float:
        """Approximate displayed distance in meters."""
        return self.pos_x - START_X_OFFSET if self.valid else 0.0

    @property
    def speed(self) -> float:
        """Speed magnitude."""
        return (self.vel_x ** 2 + self.vel_y ** 2) ** 0.5 if self.valid else 0.0

    @property
    def angle_deg(self) -> float:
        """Car angle in degrees (0 = level, positive = tilted back)."""
        import math
        if not self.valid or (self.rot_sin == 0 and self.rot_cos == 0):
            return 0.0
        return math.degrees(math.atan2(self.rot_sin, self.rot_cos))


class MemoryReader:
    """Reads HCR2 car body physics from process memory.

    Uses a persistent `memserver` process inside the container for fast reads
    (~1ms per read vs ~50ms per docker exec subprocess).
    """

    def __init__(self, container: str = "hcr2-0", package: str = "com.fingersoft.hcr2"):
        self.container = container
        self.package = package
        self.pid: Optional[int] = None
        self.body_x_addr: Optional[int] = None
        self._server: Optional[subprocess.Popen] = None
        self._attached = False

    def attach(self) -> bool:
        """Find game PID, locate car body via signature scan, start memserver.

        Returns True if body was found and memserver started.
        Call this at the start of each race (heap address changes between races).
        """
        self.detach()

        # Get PID
        try:
            result = subprocess.run(
                ["docker", "exec", self.container, "pidof", self.package],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning("Game not running on %s", self.container)
                return False
            self.pid = int(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError) as e:
            logger.error("Failed to get PID: %s", e)
            return False

        # Run safescan to find body address (anti-cheat safe: open/close per region)
        try:
            result = subprocess.run(
                ["docker", "exec", self.container,
                 "/data/local/tmp/safescan", str(self.pid)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                logger.warning("safescan failed: %s", result.stderr.strip())
                return False

            # safescan outputs hex address on stdout (e.g. "0x7a12345678\n")
            addr_str = result.stdout.strip()
            if addr_str:
                self.body_x_addr = int(addr_str, 16)
        except subprocess.TimeoutExpired:
            logger.error("safescan timed out")
            return False

        if not self.body_x_addr:
            logger.warning("Body signature not found (not in RACING state?)")
            return False

        # Start safememserver (anti-cheat safe: open/close per read)
        try:
            self._server = subprocess.Popen(
                ["docker", "exec", "-i", self.container,
                 "/data/local/tmp/safememserver", str(self.pid)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            # Quick test read
            state = self._read_raw()
            if state is None:
                logger.error("safememserver test read failed")
                self._kill_server()
                return False
        except OSError as e:
            logger.error("Failed to start safememserver: %s", e)
            return False

        self._attached = True
        logger.info("Attached: PID=%d body_x=0x%x", self.pid, self.body_x_addr)
        return True

    def detach(self):
        """Stop memserver and reset state."""
        self._kill_server()
        self.pid = None
        self.body_x_addr = None
        self._attached = False

    def _kill_server(self):
        if self._server:
            try:
                self._server.stdin.write(b"Q\n")
                self._server.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._server.terminate()
                self._server.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._server.kill()
                except OSError:
                    pass
            self._server = None

    def _read_raw(self) -> Optional[bytes]:
        """Read READ_SIZE bytes from memserver at body_x + READ_BASE_OFFSET."""
        if not self._server or self._server.poll() is not None:
            return None

        addr = self.body_x_addr + READ_BASE_OFFSET
        cmd = f"R {addr:x} {READ_SIZE}\n".encode()

        try:
            self._server.stdin.write(cmd)
            self._server.stdin.flush()

            fd = self._server.stdout.fileno()
            data = b""
            while len(data) < READ_SIZE:
                chunk = os.read(fd, READ_SIZE - len(data))
                if not chunk:
                    return None
                data += chunk
                # Check for error response (memserver sends "E\n")
                if len(data) == 2 and data == b"E\n":
                    return None

            return data
        except (BrokenPipeError, OSError):
            return None

    def read(self) -> CarState:
        """Read current car state from memory.

        Returns CarState with valid=False if read fails.
        """
        if not self._attached:
            return CarState()

        data = self._read_raw()
        if data is None or len(data) < READ_SIZE:
            self._attached = False
            return CarState()

        try:
            # Offsets in data buffer (buffer starts at body_x - 16):
            # body_x - 16 → buf[0]
            # body_x - 12 → buf[4]
            # body_x - 4  → buf[12]
            # body_x + 0  → buf[16]
            # body_x + 56 → buf[72]
            # body_x + 60 → buf[76]
            rot_sin = struct.unpack_from('<f', data, 0)[0]
            rot_cos = struct.unpack_from('<f', data, 4)[0]
            pos_y = struct.unpack_from('<f', data, 12)[0]
            pos_x = struct.unpack_from('<f', data, 16)[0]
            vel_x = struct.unpack_from('<f', data, 72)[0]
            vel_y = struct.unpack_from('<f', data, 76)[0]

            return CarState(
                pos_x=pos_x, pos_y=pos_y,
                vel_x=vel_x, vel_y=vel_y,
                rot_sin=rot_sin, rot_cos=rot_cos,
                timestamp=time.monotonic(),
                valid=True
            )
        except struct.error:
            self._attached = False
            return CarState()

    @property
    def is_attached(self) -> bool:
        return self._attached


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    container = sys.argv[1] if len(sys.argv) > 1 else "hcr2-0"
    reader = MemoryReader(container=container)

    print(f"Attaching to {container}...")
    if not reader.attach():
        print("Failed to attach. Is the game running in RACING state?")
        sys.exit(1)

    print(f"Attached! PID={reader.pid} body_x=0x{reader.body_x_addr:x}")
    print("Reading car state every 100ms (Ctrl+C to stop)...")
    print(f"{'time':>8s} {'pos_x':>10s} {'pos_y':>10s} {'vel_x':>10s} {'vel_y':>10s} {'dist':>8s} {'speed':>8s} {'angle':>8s}")

    try:
        t0 = time.monotonic()
        while True:
            state = reader.read()
            if not state.valid:
                print("READ FAILED - game may have exited")
                break
            elapsed = time.monotonic() - t0
            print(f"{elapsed:8.2f} {state.pos_x:10.3f} {state.pos_y:10.3f} "
                  f"{state.vel_x:10.4f} {state.vel_y:10.4f} "
                  f"{state.distance:8.1f} {state.speed:8.4f} {state.angle_deg:8.2f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.detach()
