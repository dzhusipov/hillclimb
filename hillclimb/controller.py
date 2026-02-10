"""ADB input controller: send gas / brake / tap commands to Android."""

from __future__ import annotations

import subprocess
import time
from enum import IntEnum

from hillclimb.config import cfg


class Action(IntEnum):
    NOTHING = 0
    GAS = 1
    BRAKE = 2


class ADBController:
    """Send touch events to Android via ADB."""

    # Physical screen height in portrait (used for coordinate conversion)
    _PORTRAIT_H = 2340

    def __init__(self) -> None:
        self._device_args: list[str] = (
            ["-s", cfg.adb_device] if cfg.adb_device else []
        )
        self._landscape_input: bool = True
        self._last_mode_check: float = 0.0
        self._verify_connection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action: Action | int, duration_ms: int | None = None) -> None:
        """Execute an action on the device.

        Args:
            action: Action enum or int (0=nothing, 1=gas, 2=brake).
            duration_ms: Hold duration in ms. Defaults to cfg.action_hold_ms.
        """
        action = Action(action)
        duration_ms = duration_ms or cfg.action_hold_ms

        if action == Action.NOTHING:
            return
        elif action == Action.GAS:
            self._hold(cfg.gas_button.x, cfg.gas_button.y, duration_ms)
        elif action == Action.BRAKE:
            self._hold(cfg.brake_button.x, cfg.brake_button.y, duration_ms)

    def tap(self, x: int, y: int) -> None:
        """Single tap at (x, y) via short swipe (обход проблемы input tap в HCR2)."""
        self._hold(x, y, 50)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        """Long press at (x, y) for duration_ms."""
        self._hold(x, y, duration_ms)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hold(self, x: int, y: int, duration_ms: int) -> None:
        """Simulate long press via `input swipe`.

        Coordinates are in landscape (2340x1080). Automatically converts
        to portrait if scrcpy is not handling input forwarding.
        """
        tx, ty = self._to_input_coords(x, y)
        self._adb(
            "input", "swipe",
            str(tx), str(ty), str(tx), str(ty), str(duration_ms),
        )

    def _to_input_coords(self, lx: int, ly: int) -> tuple[int, int]:
        """Convert landscape coords to what `input` expects.

        When scrcpy server is running on device, `input` accepts landscape.
        Otherwise, `input` uses portrait coords (orientation=3 rotation).
        Caches the check for 10 seconds to avoid overhead.
        """
        now = time.monotonic()
        if now - self._last_mode_check > 10.0:
            self._last_mode_check = now
            try:
                ps_out = self._adb("ps", "-A")
                self._landscape_input = "scrcpy" in ps_out
            except Exception:
                self._landscape_input = True
        if self._landscape_input:
            return (lx, ly)
        # No scrcpy → portrait: px = ly, py = 2340 - lx
        return (ly, self._PORTRAIT_H - lx)

    def _adb(self, *args: str) -> str:
        cmd = [cfg.adb_path] + self._device_args + ["shell"] + list(args)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ADB command failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def _verify_connection(self) -> None:
        """Check that ADB can reach the device."""
        try:
            out = self._adb("echo", "ok")
            if "ok" not in out:
                raise RuntimeError("Unexpected ADB response")
        except FileNotFoundError:
            raise RuntimeError(
                "adb not found. Install via: brew install android-platform-tools"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ADB command timed out — is the device connected?")


def main() -> None:
    """Test: gas 2s → brake 1s → gas 3s."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ADB controller")
    parser.add_argument("--test", action="store_true", help="Run gas/brake test sequence")
    args = parser.parse_args()

    ctrl = ADBController()

    if args.test:
        print("Gas 2s ...")
        ctrl.execute(Action.GAS, duration_ms=2000)
        time.sleep(0.2)

        print("Brake 1s ...")
        ctrl.execute(Action.BRAKE, duration_ms=1000)
        time.sleep(0.2)

        print("Gas 3s ...")
        ctrl.execute(Action.GAS, duration_ms=3000)
        print("Done.")
    else:
        print("Use --test to run gas/brake sequence")


if __name__ == "__main__":
    main()
