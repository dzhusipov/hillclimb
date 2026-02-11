"""ADB input controller: send gas / brake / tap commands to Android.

Each ADBController instance maintains a persistent connection to a specific
device via adbutils.
"""

from __future__ import annotations

import logging
import time
from enum import IntEnum

logger = logging.getLogger(__name__)


class Action(IntEnum):
    NOTHING = 0
    GAS = 1
    BRAKE = 2


class ADBController:
    """Send touch events to a specific Android device via ADB."""

    def __init__(
        self,
        adb_serial: str = "localhost:5555",
        gas_x: int = 700,
        gas_y: int = 400,
        brake_x: int = 100,
        brake_y: int = 400,
        action_hold_ms: int = 100,
    ) -> None:
        """
        Args:
            adb_serial: ADB device serial.
            gas_x, gas_y: Gas button coordinates (landscape).
            brake_x, brake_y: Brake button coordinates (landscape).
            action_hold_ms: Default hold duration for actions.
        """
        self._serial = adb_serial
        self._gas_x = gas_x
        self._gas_y = gas_y
        self._brake_x = brake_x
        self._brake_y = brake_y
        self._action_hold_ms = action_hold_ms
        self._device = None
        self._connect()

    def _connect(self) -> None:
        """Establish persistent ADB connection."""
        import adbutils
        client = adbutils.AdbClient()
        try:
            client.connect(self._serial, timeout=5)
        except Exception:
            pass
        self._device = client.device(self._serial)
        # Verify connection
        try:
            self._device.shell("echo ok")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to {self._serial}: {e}")
        logger.info("Controller connected: %s", self._serial)

    def execute(self, action: Action | int, duration_ms: int | None = None) -> None:
        """Execute an action.

        Args:
            action: 0=nothing, 1=gas, 2=brake.
            duration_ms: Hold duration in ms. Defaults to action_hold_ms.
        """
        action = Action(action)
        duration_ms = duration_ms or self._action_hold_ms

        if action == Action.NOTHING:
            return
        elif action == Action.GAS:
            self._hold(self._gas_x, self._gas_y, duration_ms)
        elif action == Action.BRAKE:
            self._hold(self._brake_x, self._brake_y, duration_ms)

    def tap(self, x: int, y: int) -> None:
        """Single tap at (x, y)."""
        self._hold(x, y, 50)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        """Long press at (x, y) for duration_ms."""
        self._hold(x, y, duration_ms)

    def _hold(self, x: int, y: int, duration_ms: int) -> None:
        """Simulate press via `input swipe` with same start/end."""
        cmd = f"input swipe {x} {y} {x} {y} {duration_ms}"
        for attempt in range(3):
            try:
                self._device.shell(cmd)
                return
            except Exception as e:
                logger.warning(
                    "ADB command failed (attempt %d/3, %s): %s",
                    attempt + 1, self._serial, e,
                )
                if attempt < 2:
                    time.sleep(1.0)
                    self._connect()
        raise RuntimeError(f"ADB input failed after 3 attempts on {self._serial}")

    @property
    def serial(self) -> str:
        return self._serial

    def close(self) -> None:
        """Release resources."""
        self._device = None


def main() -> None:
    """Test: gas 2s -> brake 1s -> gas 3s."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ADB controller")
    parser.add_argument("--serial", default="localhost:5555", help="ADB serial")
    parser.add_argument("--test", action="store_true", help="Run gas/brake test")
    args = parser.parse_args()

    ctrl = ADBController(adb_serial=args.serial)

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
        print(f"Connected to {args.serial}. Use --test to run gas/brake sequence.")


if __name__ == "__main__":
    main()
