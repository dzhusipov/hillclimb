"""Menu navigation: auto-start races and restart after crashes."""

from __future__ import annotations

import time

from hillclimb.capture import ScreenCapture
from hillclimb.config import cfg
from hillclimb.controller import ADBController
from hillclimb.vision import GameState, VisionAnalyzer


class Navigator:
    """Handles menu interactions: starting games and restarting after crash."""

    def __init__(
        self,
        controller: ADBController,
        capture: ScreenCapture,
        vision: VisionAnalyzer,
    ) -> None:
        self._ctrl = controller
        self._cap = capture
        self._vision = vision

    def ensure_racing(self, timeout: float = 15.0) -> bool:
        """Navigate from any state to RACING. Returns True if successful."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            frame = self._cap.grab()
            state = self._vision.analyze(frame)

            if state.game_state == GameState.RACING:
                return True

            if state.game_state == GameState.MENU:
                self._ctrl.tap(cfg.play_button.x, cfg.play_button.y)
                time.sleep(2.0)  # wait for game to load

            elif state.game_state in (GameState.CRASHED, GameState.RESULTS):
                self._ctrl.tap(cfg.retry_button.x, cfg.retry_button.y)
                time.sleep(2.0)

            elif state.game_state == GameState.UNKNOWN:
                # Tap centre of screen to dismiss any dialog
                frame_h, frame_w = frame.shape[:2]
                self._ctrl.tap(frame_w // 2, frame_h // 2)
                time.sleep(1.0)

        return False

    def restart_game(self, timeout: float = 15.0) -> bool:
        """Restart after crash/results screen. Returns True if back to RACING."""
        return self.ensure_racing(timeout)


def main() -> None:
    ctrl = ADBController()
    cap = ScreenCapture()
    vis = VisionAnalyzer()
    nav = Navigator(ctrl, cap, vis)

    print("Attempting to navigate to RACING state...")
    ok = nav.ensure_racing()
    print(f"Result: {'success' if ok else 'failed'}")


if __name__ == "__main__":
    main()
