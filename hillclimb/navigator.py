"""Menu navigation: full state machine for HCR2 game cycle.

Handles transitions between all 8 game states:
    MAIN_MENU → VEHICLE_SELECT → RACING → DRIVER_DOWN →
    TOUCH_TO_CONTINUE → RESULTS → (retry) → VEHICLE_SELECT
Plus: DOUBLE_COINS_POPUP (skip), UNKNOWN (fallback tap).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from hillclimb.config import cfg
from hillclimb.vision import GameState, VisionState

if TYPE_CHECKING:
    from hillclimb.capture import ScreenCapture
    from hillclimb.controller import ADBController
    from hillclimb.vision import VisionAnalyzer


class Navigator:
    """Handles menu interactions via a state-machine transition table."""

    def __init__(
        self,
        controller: ADBController,
        capture: ScreenCapture,
        vision: VisionAnalyzer,
    ) -> None:
        self._ctrl = controller
        self._cap = capture
        self._vision = vision
        self._last_results: VisionState | None = None
        self._same_state_count = 0
        self._prev_state: GameState | None = None
        self._racing_stuck_count = 0

    def _save_debug_frame(self, frame: np.ndarray, label: str) -> None:
        """Save a debug frame to logs/nav_debug/ for later analysis."""
        import cv2
        from pathlib import Path
        debug_dir = Path(cfg.log_dir) / "nav_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = debug_dir / f"{ts}_{label}.png"
        cv2.imwrite(str(path), frame)
        print(f"  [NAV] Saved debug frame: {path}")

    @property
    def last_results(self) -> VisionState | None:
        """Last VisionState captured on the RESULTS screen (for logging)."""
        return self._last_results

    def ensure_racing(self, timeout: float = 20.0) -> bool:
        """Navigate from any state to RACING. Returns True if successful."""
        deadline = time.time() + timeout
        self._same_state_count = 0
        self._prev_state = None
        nav_start = time.time()

        while time.time() < deadline:
            frame = self._cap.capture()

            # Portrait frame = not in game (home screen, other app)
            h, w = frame.shape[:2]
            if h > w:
                print(f"  [NAV] Portrait frame ({w}x{h}) — not in game, relaunching...")
                self._save_debug_frame(frame, "not_in_game")
                self._relaunch_game()
                continue

            state = self._vision.analyze(frame)
            gs = state.game_state
            elapsed = time.time() - nav_start

            print(f"  [NAV] {elapsed:.1f}s | state={gs.name} | stuck={self._same_state_count}")

            # Stuck detection: same state 3 cycles in a row → fallback tap
            if gs == self._prev_state:
                self._same_state_count += 1
            else:
                self._same_state_count = 0
            self._prev_state = gs

            if self._same_state_count >= 3 and gs != GameState.RACING:
                print(f"  [NAV] STUCK on {gs.name} for {self._same_state_count} cycles — fallback tap center")
                self._save_debug_frame(frame, f"stuck_{gs.name}")
                self._ctrl.tap(cfg.center_screen.x, cfg.center_screen.y)
                time.sleep(1.0)
                self._same_state_count = 0
                continue

            # Already racing — done
            if gs == GameState.RACING:
                # Проверяем "fling" промо: рука на экране, машина стоит
                # Если speed ≈ 0 несколько циклов подряд → свайп (оттянуть и отпустить)
                try:
                    speed = float(state.speed_estimate)
                except (TypeError, ValueError):
                    speed = 1.0
                if speed < 0.02:
                    self._racing_stuck_count += 1
                else:
                    self._racing_stuck_count = 0
                if self._racing_stuck_count >= 2:
                    self._fling()
                    self._racing_stuck_count = 0
                return True

            # Transition table
            if gs == GameState.MAIN_MENU:
                print(f"  [NAV] → tap RACE button ({cfg.race_button.x}, {cfg.race_button.y})")
                self._ctrl.tap(cfg.race_button.x, cfg.race_button.y)
                time.sleep(2.0)

            elif gs == GameState.VEHICLE_SELECT:
                print(f"  [NAV] → tap START ({cfg.start_button.x}, {cfg.start_button.y})")
                self._ctrl.tap(cfg.start_button.x, cfg.start_button.y)
                time.sleep(2.0)
                self._dismiss_popups()
                time.sleep(1.5)

            elif gs == GameState.DOUBLE_COINS_POPUP:
                print(f"  [NAV] → dismiss DOUBLE_COINS popup")
                self._dismiss_popups()
                time.sleep(2.0)

            elif gs == GameState.DRIVER_DOWN:
                print(f"  [NAV] → DRIVER_DOWN — BACK to skip second chance")
                self._ctrl.tap(cfg.center_screen.x, cfg.center_screen.y)
                time.sleep(0.3)
                # BACK скипает экран "second chance" (машина висит в воздухе)
                self._ctrl.keyevent("KEYCODE_BACK")
                time.sleep(0.5)

            elif gs == GameState.TOUCH_TO_CONTINUE:
                print(f"  [NAV] → tap center (TOUCH_TO_CONTINUE)")
                self._ctrl.tap(cfg.center_screen.x, cfg.center_screen.y)
                time.sleep(1.5)

            elif gs == GameState.RESULTS:
                self._last_results = state
                print(f"  [NAV] → tap RETRY ({cfg.retry_button.x}, {cfg.retry_button.y})")
                self._ctrl.tap(cfg.retry_button.x, cfg.retry_button.y)
                time.sleep(2.0)

            elif gs == GameState.CAPTCHA:
                print(f"  [NAV] → solving CAPTCHA...")
                self._save_debug_frame(frame, "captcha")
                self._solve_captcha(frame)
                time.sleep(2.0)

            elif gs == GameState.UNKNOWN:
                print(f"  [NAV] → UNKNOWN state — BACK + tap center")
                self._save_debug_frame(frame, "unknown")
                # BACK может скипнуть second chance или другой попап
                self._ctrl.keyevent("KEYCODE_BACK")
                time.sleep(0.5)
                self._ctrl.tap(cfg.center_screen.x, cfg.center_screen.y)
                time.sleep(0.5)

        return False

    def _relaunch_game(self) -> None:
        """Force-stop and relaunch HCR2."""
        print("  [NAV] Relaunching HCR2...")
        self._ctrl.shell("am force-stop com.fingersoft.hcr2")
        time.sleep(1.0)
        self._ctrl.shell("am start -n com.fingersoft.hcr2/.AppActivity")
        time.sleep(5.0)

    def _solve_captcha(self, frame: np.ndarray) -> None:
        """Обойти проверку 'ARE YOU A ROBOT?'.

        Капча — overlay от другого процесса, ADB input tap не может до неё
        достучаться (INJECT_EVENTS permission). Решение: BACK → перезапуск игры.
        """
        print("  [CAPTCHA] Captcha detected — pressing BACK to dismiss")
        self._ctrl.keyevent("KEYCODE_BACK")
        time.sleep(1.0)

        # Проверяем — ушла ли капча?
        frame2 = self._cap.capture()
        state = self._vision.analyze(frame2)
        if state.game_state != GameState.CAPTCHA:
            print(f"  [CAPTCHA] BACK worked! state: {state.game_state.name}")
            return

        # BACK не помог — HOME + перезапуск игры
        print("  [CAPTCHA] BACK didn't work — HOME + relaunch")
        self._ctrl.keyevent("KEYCODE_HOME")
        time.sleep(2.0)

        # Перезапускаем HCR2
        self._relaunch_game()
        print("  [CAPTCHA] Game relaunched")

    def _fling(self) -> None:
        """Свайп вниз-вверх для 'fling' промо в Adventures (оттянуть и отпустить)."""
        # Свайп вниз от центра машины (оттягиваем)
        self._ctrl.swipe(1170, 400, 1170, 700, 300)
        time.sleep(0.1)
        # Свайп вверх (отпускаем — запуск)
        self._ctrl.swipe(1170, 700, 1170, 300, 150)

    def _dismiss_popups(self) -> None:
        """Закрыть попапы SKIP в обеих известных позициях."""
        # DOUBLE COINS — синяя SKIP (центр попапа)
        self._ctrl.tap(967, 869)
        time.sleep(0.3)
        # Резерв: жёлтый SKIP (cfg) + вторая позиция
        self._ctrl.tap(cfg.skip_button.x, cfg.skip_button.y)
        time.sleep(0.3)
        self._ctrl.tap(990, 830)

    def restart_game(self, timeout: float = 20.0) -> bool:
        """Restart after crash/results screen. Returns True if back to RACING."""
        return self.ensure_racing(timeout)


def main() -> None:
    from hillclimb.capture import ScreenCapture
    from hillclimb.controller import ADBController
    from hillclimb.vision import VisionAnalyzer

    import argparse
    parser = argparse.ArgumentParser(description="Test navigator")
    parser.add_argument("--serial", default="localhost:5555", help="ADB serial")
    args = parser.parse_args()

    cap = ScreenCapture(adb_serial=args.serial)
    ctrl = ADBController(adb_serial=args.serial)
    vis = VisionAnalyzer()
    nav = Navigator(ctrl, cap, vis)

    print(f"Attempting to navigate to RACING state on {args.serial}...")
    ok = nav.ensure_racing()
    print(f"Result: {'success' if ok else 'failed'}")


if __name__ == "__main__":
    main()
