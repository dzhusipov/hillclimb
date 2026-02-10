"""ADB input controller: send gas / brake / tap commands to Android."""

from __future__ import annotations

import subprocess
import time
from enum import IntEnum

from hillclimb.config import cfg


class ADBConnectionError(RuntimeError):
    """ADB lost connection or INJECT_EVENTS permission."""


class Action(IntEnum):
    NOTHING = 0
    GAS = 1
    BRAKE = 2


class ADBController:
    """Send touch events to Android via ADB."""

    def __init__(self) -> None:
        self._device_args: list[str] = (
            ["-s", cfg.adb_device] if cfg.adb_device else []
        )
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
        """Simulate long press via `input swipe` with same start/end coords."""
        self._adb(
            "input", "swipe",
            str(x), str(y), str(x), str(y), str(duration_ms),
        )

    def _adb(self, *args: str) -> str:
        cmd = [cfg.adb_path] + self._device_args + ["shell"] + list(args)
        for attempt in range(3):
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            err = result.stderr.strip()
            if "INJECT_EVENTS" in err or "SecurityException" in err:
                if attempt < 2:
                    time.sleep(2.0)
                    continue
                raise ADBConnectionError(
                    f"ADB lost INJECT_EVENTS permission (scrcpy disconnected?): {err}"
                )
            raise RuntimeError(f"ADB command failed: {err}")
        return ""  # unreachable

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
