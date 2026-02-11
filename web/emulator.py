"""Emulator discovery, status, and management via ADB + Docker CLI."""

import subprocess
import logging
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

MAX_EMULATORS = 8
HCR2_PACKAGE = "com.fingersoft.hcr2"
HCR2_ACTIVITY = "com.fingersoft.hcr2/.AppActivity"


@dataclass
class EmulatorStatus:
    id: int
    name: str           # "hcr2-0"
    adb_host: str       # "hcr2-0:5555"
    docker_status: str  # "running" | "exited" | "not found"
    boot_status: str    # "ready" | "booting" | "offline"

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    """Run a command and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug("Command failed: %s -> %s", cmd, e)
        return -1, ""


def _docker_status(name: str) -> str:
    """Get container status via docker inspect."""
    rc, out = _run(["docker", "inspect", "--format", "{{.State.Status}}", name])
    if rc != 0 or not out:
        return "not found"
    return out.strip()


def _adb_cmd(host: str, *args: str, timeout: int = 10) -> tuple[int, str]:
    """Run an ADB command against a specific host."""
    return _run(["adb", "-s", host, *args], timeout=timeout)


def _ensure_connected(host: str) -> None:
    """Connect ADB to the emulator (idempotent)."""
    _run(["adb", "connect", host], timeout=5)


def _boot_status(host: str) -> str:
    """Check if emulator has finished booting."""
    _ensure_connected(host)
    rc, out = _adb_cmd(host, "shell", "getprop", "sys.boot_completed")
    if rc == 0 and out.strip() == "1":
        return "ready"
    # Check if ADB responds at all
    rc2, _ = _adb_cmd(host, "get-state")
    if rc2 == 0:
        return "booting"
    return "offline"


def get_emulator_status(emu_id: int) -> EmulatorStatus:
    """Get status of a single emulator."""
    name = f"hcr2-{emu_id}"
    adb_host = f"{name}:5555"

    docker_st = _docker_status(name)
    if docker_st == "running":
        boot_st = _boot_status(adb_host)
    else:
        boot_st = "offline"

    return EmulatorStatus(
        id=emu_id,
        name=name,
        adb_host=adb_host,
        docker_status=docker_st,
        boot_status=boot_st,
    )


def get_all_status() -> list[EmulatorStatus]:
    """Get status of all emulators (only those that exist as containers)."""
    results = []
    for i in range(MAX_EMULATORS):
        status = get_emulator_status(i)
        # Include if container exists (running or stopped)
        if status.docker_status != "not found":
            results.append(status)
    return results


def restart_emulator(emu_id: int) -> tuple[bool, str]:
    """Restart an emulator container."""
    name = f"hcr2-{emu_id}"
    rc, out = _run(["docker", "restart", name], timeout=30)
    if rc == 0:
        return True, f"{name} restarted"
    return False, f"Failed to restart {name}: {out}"


def start_emulator(emu_id: int) -> tuple[bool, str]:
    """Start a stopped emulator container."""
    name = f"hcr2-{emu_id}"
    rc, out = _run(["docker", "start", name], timeout=30)
    if rc == 0:
        return True, f"{name} started"
    return False, f"Failed to start {name}: {out}"


def stop_emulator(emu_id: int) -> tuple[bool, str]:
    """Stop an emulator container."""
    name = f"hcr2-{emu_id}"
    rc, out = _run(["docker", "stop", name], timeout=30)
    if rc == 0:
        return True, f"{name} stopped"
    return False, f"Failed to stop {name}: {out}"


def start_game(emu_id: int) -> tuple[bool, str]:
    """Launch HCR2 on the emulator."""
    host = f"hcr2-{emu_id}:5555"
    _ensure_connected(host)
    rc, out = _adb_cmd(host, "shell", "am", "start", "-n", HCR2_ACTIVITY)
    if rc == 0:
        return True, "Game launched"
    return False, f"Failed to launch game: {out}"


def stop_game(emu_id: int) -> tuple[bool, str]:
    """Force-stop HCR2 on the emulator."""
    host = f"hcr2-{emu_id}:5555"
    _ensure_connected(host)
    rc, out = _adb_cmd(host, "shell", "am", "force-stop", HCR2_PACKAGE)
    if rc == 0:
        return True, "Game stopped"
    return False, f"Failed to stop game: {out}"


# --- Touch input ---

_active_touches: dict[int, subprocess.Popen] = {}
_touch_lock = threading.Lock()


def _map_coords(
    norm_x: float, norm_y: float,
    rotated: bool, cap_w: int, cap_h: int,
) -> tuple[int, int]:
    """Map normalized landscape coords to emulator screen coords."""
    if rotated:
        # Stream was rotated 90° CW from portrait (cap_w x cap_h).
        # Reverse: screen_x = norm_y * cap_w, screen_y = (1-norm_x) * cap_h
        screen_x = int(norm_y * cap_w)
        screen_y = int((1.0 - norm_x) * cap_h)
    else:
        # Landscape capture displayed directly
        screen_x = int(norm_x * cap_w)
        screen_y = int(norm_y * cap_h)
    return max(0, screen_x), max(0, screen_y)


def touch_down(
    emu_id: int, norm_x: float, norm_y: float,
    rotated: bool, cap_w: int, cap_h: int,
) -> tuple[bool, str]:
    """Start a touch (press) at normalized coordinates."""
    host = f"hcr2-{emu_id}:5555"
    _ensure_connected(host)
    x, y = _map_coords(norm_x, norm_y, rotated, cap_w, cap_h)

    # Kill any existing touch first
    touch_up(emu_id)

    # Start a 30s swipe-in-place (acts as a sustained press)
    proc = subprocess.Popen(
        ["adb", "-s", host, "shell", "input", "swipe",
         str(x), str(y), str(x), str(y), "30000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with _touch_lock:
        _active_touches[emu_id] = proc
    return True, f"Touch down ({x}, {y})"


def touch_up(emu_id: int) -> tuple[bool, str]:
    """Release an active touch."""
    with _touch_lock:
        proc = _active_touches.pop(emu_id, None)
    if proc:
        proc.kill()
        proc.wait()
        return True, "Touch up"
    return True, "No active touch"
