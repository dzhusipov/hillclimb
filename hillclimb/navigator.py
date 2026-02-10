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

    @property
    def last_results(self) -> VisionState | None:
        """Last VisionState captured on the RESULTS screen (for logging)."""
        return self._last_results

    def ensure_racing(self, timeout: float = 20.0) -> bool:
        """Navigate from any state to RACING. Returns True if successful."""
        deadline = time.time() + timeout
        self._same_state_count = 0
        self._prev_state = None

        while time.time() < deadline:
            frame = self._cap.grab()
            state = self._vision.analyze(frame)
            gs = state.game_state

            # Stuck detection: same state 3 cycles in a row → fallback tap
            if gs == self._prev_state:
                self._same_state_count += 1
            else:
                self._same_state_count = 0
            self._prev_state = gs

            if self._same_state_count >= 3 and gs != GameState.RACING:
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
                # RACE кнопка на панели Countryside — фиксированная позиция
                self._ctrl.tap(1860, 570)
                time.sleep(2.0)

            elif gs == GameState.VEHICLE_SELECT:
                self._ctrl.tap(cfg.start_button.x, cfg.start_button.y)
                time.sleep(2.0)
                # После START часто появляется попап (DOUBLE TOKENS/COINS)
                # который классификатор может не распознать — закрываем превентивно
                self._dismiss_popups()
                time.sleep(1.5)

            elif gs == GameState.DOUBLE_COINS_POPUP:
                self._dismiss_popups()
                time.sleep(2.0)

            elif gs == GameState.DRIVER_DOWN:
                self._ctrl.tap(cfg.center_screen.x, cfg.center_screen.y)
                time.sleep(1.0)

            elif gs == GameState.TOUCH_TO_CONTINUE:
                self._ctrl.tap(cfg.center_screen.x, cfg.center_screen.y)
                time.sleep(1.5)

            elif gs == GameState.RESULTS:
                # Capture results data before dismissing
                self._last_results = state
                self._ctrl.tap(cfg.retry_button.x, cfg.retry_button.y)
                time.sleep(2.0)

            elif gs == GameState.CAPTCHA:
                self._solve_captcha(frame)
                time.sleep(2.0)

            elif gs == GameState.UNKNOWN:
                # Пробуем закрыть попап: X в правом верхнем углу (Special Offer и т.п.)
                self._ctrl.tap(1790, 55)
                time.sleep(0.5)
                # Затем центр экрана (универсальный fallback)
                self._ctrl.tap(cfg.center_screen.x, cfg.center_screen.y)
                time.sleep(1.0)

        return False

    def _solve_captcha(self, frame: np.ndarray) -> None:
        """Пройти проверку 'ARE YOU A ROBOT?': найти чекбокс → тап → ждём OK → тап."""
        import cv2
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ищем чекбокс: белый квадрат ~30-60px на тёмном фоне
        # Чекбокс — яркий прямоугольник с чёткими краями
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # Ищем контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        checkbox_pos = None
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Чекбокс: квадратный, 20-80px, в верхней половине экрана
            if 15 < cw < 100 and 15 < ch < 100 and 0.6 < cw / ch < 1.6 and y < h * 0.6:
                # Проверяем что это не просто шум — внутри должно быть пусто (тёмное)
                inner = gray[y + 3 : y + ch - 3, x + 3 : x + cw - 3]
                if inner.size > 0 and np.mean(inner) < 150:
                    checkbox_pos = (x + cw // 2, y + ch // 2)
                    break

        if checkbox_pos:
            # Тапаем чекбокс
            self._ctrl.tap(checkbox_pos[0], checkbox_pos[1])
            time.sleep(1.5)
            # После чекбокса появляется зелёная кнопка OK — ищем её
            frame2 = self._cap.grab()
            state2 = self._vision.analyze(frame2)
            # Ищем зелёную кнопку в нижней части
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv2,
                                      np.array([35, 80, 80], dtype=np.uint8),
                                      np.array([85, 255, 255], dtype=np.uint8))
            # Ищем зелёный блоб
            contours_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours_g, key=cv2.contourArea, reverse=True):
                x, y, gw, gh = cv2.boundingRect(cnt)
                if gw > 50 and gh > 20:
                    self._ctrl.tap(x + gw // 2, y + gh // 2)
                    break
        else:
            # Не нашли чекбокс — тапаем примерное место (центр-верх)
            self._ctrl.tap(w // 2, int(h * 0.3))
            time.sleep(1.5)
            # Тапаем центр (OK кнопка)
            self._ctrl.tap(cfg.center_screen.x, int(h * 0.7))

    def _fling(self) -> None:
        """Свайп вниз-вверх для 'fling' промо в Adventures (оттянуть и отпустить)."""
        import subprocess
        # Свайп вниз от центра машины (оттягиваем)
        subprocess.run(
            [cfg.adb_path, "shell", "input", "swipe",
             "1170", "400", "1170", "700", "300"],
            capture_output=True, timeout=5,
        )
        time.sleep(0.1)
        # Свайп вверх (отпускаем — запуск)
        subprocess.run(
            [cfg.adb_path, "shell", "input", "swipe",
             "1170", "700", "1170", "300", "150"],
            capture_output=True, timeout=5,
        )

    def _dismiss_popups(self) -> None:
        """Закрыть попапы SKIP в обеих известных позициях."""
        # DOUBLE COINS — жёлтый SKIP
        self._ctrl.tap(cfg.skip_button.x, cfg.skip_button.y)
        time.sleep(0.3)
        # DOUBLE TOKENS — синий SKIP (другая позиция)
        self._ctrl.tap(990, 830)

    def restart_game(self, timeout: float = 20.0) -> bool:
        """Restart after crash/results screen. Returns True if back to RACING."""
        return self.ensure_racing(timeout)


def main() -> None:
    from hillclimb.capture import ScreenCapture
    from hillclimb.controller import ADBController
    from hillclimb.vision import VisionAnalyzer

    ctrl = ADBController()
    cap = ScreenCapture()
    vis = VisionAnalyzer()
    nav = Navigator(ctrl, cap, vis)

    print("Attempting to navigate to RACING state...")
    ok = nav.ensure_racing()
    print(f"Result: {'success' if ok else 'failed'}")


if __name__ == "__main__":
    main()
