"""Main game loop: capture -> vision -> agent -> controller."""

from __future__ import annotations

import argparse
import signal
import sys
import time

from hillclimb.agent_rules import RuleBasedAgent
from hillclimb.capture import ScreenCapture
from hillclimb.config import cfg
from hillclimb.controller import ADBController, Action
from hillclimb.logger import Logger
from hillclimb.navigator import Navigator
from hillclimb.vision import GameState, VisionAnalyzer


class GameLoop:
    """Orchestrates the capture -> CV -> decision -> input cycle."""

    def __init__(
        self,
        agent: object,
        adb_serial: str = "localhost:5555",
        headless: bool = False,
    ) -> None:
        self._capture = ScreenCapture(adb_serial=adb_serial)
        self._vision = VisionAnalyzer()
        self._controller = ADBController(
            adb_serial=adb_serial,
            gas_x=cfg.gas_button.x,
            gas_y=cfg.gas_button.y,
            brake_x=cfg.brake_button.x,
            brake_y=cfg.brake_button.y,
            action_hold_ms=cfg.action_hold_ms,
        )
        self._navigator = Navigator(
            self._controller, self._capture, self._vision,
        )
        self._agent = agent
        self._serial = adb_serial
        self._headless = headless
        self._running = False
        self._logger: Logger | None = None

    def run(self, max_episodes: int = 0, max_steps: int = 0) -> None:
        """Run the game loop.

        Args:
            max_episodes: Stop after this many episodes (0 = unlimited).
            max_steps: Stop after this many total steps (0 = unlimited).
        """
        self._running = True
        episode = 0
        total_steps = 0

        with Logger() as logger:
            self._logger = logger

            while self._running:
                episode += 1
                if max_episodes and episode > max_episodes:
                    break

                print(f"\n=== Episode {episode} [{self._serial}] ===")
                if not self._navigator.ensure_racing():
                    print("Failed to start race, retrying...")
                    time.sleep(2.0)
                    continue

                steps = self._run_episode(logger, episode)
                total_steps += steps

                # Log episode summary with results data
                last_results = self._navigator.last_results
                if last_results is not None:
                    logger.log_episode_summary(
                        episode=episode,
                        steps=steps,
                        results_coins=last_results.results_coins,
                        results_distance_m=last_results.results_distance_m,
                    )
                    print(
                        f"Episode {episode}: {steps} steps, "
                        f"dist={last_results.results_distance_m:.0f}m, "
                        f"coins={last_results.results_coins}"
                    )
                else:
                    logger.log_episode_summary(episode=episode, steps=steps)
                    print(f"Episode {episode} ended after {steps} steps")

                if max_steps and total_steps >= max_steps:
                    break

        print(f"Finished: {episode} episodes, {total_steps} total steps")

    def _run_episode(self, logger: Logger, episode: int) -> int:
        """Run a single episode until crash/results/fuel empty. Returns step count."""
        step = 0
        while self._running:
            t0 = time.time()

            frame = self._capture.capture()
            state = self._vision.analyze(frame)
            gs = state.game_state

            # Episode end conditions
            if gs == GameState.DRIVER_DOWN:
                logger.log(state, Action.NOTHING, frame)
                print(f"  DRIVER DOWN at step {step}")
                return step

            if gs == GameState.TOUCH_TO_CONTINUE:
                logger.log(state, Action.NOTHING, frame)
                print(f"  TOUCH TO CONTINUE at step {step}")
                return step

            if gs == GameState.RESULTS:
                logger.log(state, Action.NOTHING, frame)
                print(f"  RESULTS at step {step}")
                return step

            if gs != GameState.RACING:
                # Unexpected state — try to recover
                self._navigator.ensure_racing(timeout=5.0)
                continue

            # Agent decision
            action = self._agent.decide(state)

            # Execute
            self._controller.execute(action)

            # Log
            logger.log(state, action, frame)
            step += 1

            # Debug display
            if not self._headless:
                import cv2
                debug = self._vision.draw_debug(frame, state)
                action_name = Action(action).name
                cv2.putText(debug, f"Action: {action_name}", (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("hillclimb", debug)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

            # Timing
            elapsed = time.time() - t0
            sleep_time = cfg.loop_interval_sec - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return step

    def stop(self) -> None:
        self._running = False


def _load_agent(name: str) -> object:
    if name == "rules":
        return RuleBasedAgent()
    elif name == "rl":
        from hillclimb.agent_rl import RLAgent
        return RLAgent()
    else:
        raise ValueError(f"Unknown agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hill Climb Racing AI game loop")
    parser.add_argument("--agent", default="rules", choices=["rules", "rl"],
                        help="Agent type (default: rules)")
    parser.add_argument("--serial", default="localhost:5555",
                        help="ADB serial (default: localhost:5555)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Max episodes (0=unlimited)")
    parser.add_argument("--headless", action="store_true",
                        help="Disable debug window")
    args = parser.parse_args()

    agent = _load_agent(args.agent)
    loop = GameLoop(agent, adb_serial=args.serial, headless=args.headless)

    # Graceful shutdown on Ctrl+C
    def _signal_handler(sig: int, frame: object) -> None:
        print("\nShutting down...")
        loop.stop()

    signal.signal(signal.SIGINT, _signal_handler)

    loop.run(max_episodes=args.episodes)


if __name__ == "__main__":
    main()
