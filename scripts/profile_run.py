#!/usr/bin/env python3
"""Profile 5 episodes with detailed timing on every step.

Tests pipelined step: async controller + RAW capture overlap.
"""

import statistics
import sys
import time

sys.path.insert(0, ".")

from hillclimb.capture import ScreenCapture
from hillclimb.config import cfg
from hillclimb.controller import ADBConnectionError, ADBController, Action
from hillclimb.navigator import Navigator
from hillclimb.vision import GameState, VisionAnalyzer, extract_game_field

SERIAL = "localhost:5555"
NUM_EPISODES = 5


def main():
    print(f"capture_backend={cfg.capture_backend}, action_hold_ms={cfg.action_hold_ms}")
    print()

    capture = ScreenCapture(adb_serial=SERIAL, backend=cfg.capture_backend)
    vision = VisionAnalyzer()
    controller = ADBController(
        adb_serial=SERIAL,
        gas_x=cfg.gas_button.x,
        gas_y=cfg.gas_button.y,
        brake_x=cfg.brake_button.x,
        brake_y=cfg.brake_button.y,
        action_hold_ms=cfg.action_hold_ms,
    )
    navigator = Navigator(controller, capture, vision)

    # Timing accumulators
    all_capture = []
    all_vision = []
    all_extract = []
    all_fire = []
    all_step_total = []
    all_reset = []

    for ep in range(1, NUM_EPISODES + 1):
        print(f"\n{'='*70}")
        print(f"EPISODE {ep}/{NUM_EPISODES}")
        print(f"{'='*70}")

        # --- RESET ---
        t_reset_start = time.time()
        ok = navigator.restart_game(timeout=25.0)
        t_reset_end = time.time()
        reset_time = t_reset_end - t_reset_start
        all_reset.append(reset_time)
        print(f"  RESET: {reset_time:.2f}s (ok={ok})")

        time.sleep(0.5)

        step = 0
        ep_capture = []
        ep_vision = []
        ep_extract = []
        ep_fire = []
        ep_step_total = []
        ep_start = time.time()
        prev_dist = 0.0
        max_dist = 0.0

        while True:
            t_step_start = time.time()

            # --- FIRE ACTION (async, non-blocking) ---
            action = Action.GAS
            t0 = time.time()
            controller.execute_async(action, duration_ms=cfg.action_hold_ms)
            t_fire = time.time() - t0
            ep_fire.append(t_fire)

            # --- CAPTURE (overlaps with action execution) ---
            t0 = time.time()
            try:
                frame = capture.capture()
            except Exception as e:
                print(f"  CAPTURE ERROR at step {step}: {e}")
                break
            t_capture = time.time() - t0
            ep_capture.append(t_capture)

            # --- VISION ---
            t0 = time.time()
            state = vision.analyze(frame)
            t_vision = time.time() - t0
            ep_vision.append(t_vision)

            # --- EXTRACT GAME FIELD ---
            t0 = time.time()
            game_field = extract_game_field(frame)
            t_extract = time.time() - t0
            ep_extract.append(t_extract)

            gs = state.game_state

            t_step = time.time() - t_step_start
            ep_step_total.append(t_step)

            # Episode end
            if gs in (GameState.DRIVER_DOWN, GameState.TOUCH_TO_CONTINUE, GameState.RESULTS):
                controller.wait_action()
                print(f"  step={step:4d}  {gs.name:20s}  "
                      f"fire={t_fire*1000:4.0f}ms  cap={t_capture*1000:5.0f}ms  "
                      f"vis={t_vision*1000:4.0f}ms  ext={t_extract*1000:4.1f}ms  "
                      f"total={t_step*1000:5.0f}ms  dist={state.distance_m:.0f}m")
                break

            # Navigate popups
            if gs in (GameState.CAPTCHA, GameState.DOUBLE_COINS_POPUP, GameState.UNKNOWN,
                      GameState.MAIN_MENU, GameState.VEHICLE_SELECT):
                controller.wait_action()
                print(f"  step={step:4d}  {gs.name:20s}  — navigating...")
                navigator.ensure_racing(timeout=10.0)
                continue

            if state.distance_m > max_dist:
                max_dist = state.distance_m

            # Log every 10th step + first 5
            if step < 5 or step % 10 == 0:
                print(f"  step={step:4d}  {gs.name:20s}  "
                      f"fire={t_fire*1000:4.0f}ms  cap={t_capture*1000:5.0f}ms  "
                      f"vis={t_vision*1000:4.0f}ms  ext={t_extract*1000:4.1f}ms  "
                      f"total={t_step*1000:5.0f}ms  "
                      f"fuel={state.fuel:.0%}  rpm={state.rpm:.0%}  dist={state.distance_m:.0f}m")

            step += 1

            # Safety limit
            if step > 500:
                print(f"  SAFETY LIMIT: 500 steps reached")
                break

        ep_time = time.time() - ep_start
        print(f"\n  Episode {ep} summary: {step} steps, {ep_time:.1f}s, max_dist={max_dist:.0f}m")

        if ep_capture:
            print(f"    capture:    min={min(ep_capture)*1000:.0f}ms  "
                  f"med={statistics.median(ep_capture)*1000:.0f}ms  "
                  f"avg={statistics.mean(ep_capture)*1000:.0f}ms  "
                  f"max={max(ep_capture)*1000:.0f}ms  "
                  f"p95={sorted(ep_capture)[int(len(ep_capture)*0.95)]*1000:.0f}ms")
        if ep_fire:
            print(f"    fire_async: min={min(ep_fire)*1000:.1f}ms  "
                  f"med={statistics.median(ep_fire)*1000:.1f}ms  "
                  f"avg={statistics.mean(ep_fire)*1000:.1f}ms  "
                  f"max={max(ep_fire)*1000:.1f}ms")
        if ep_vision:
            print(f"    vision:     min={min(ep_vision)*1000:.0f}ms  "
                  f"med={statistics.median(ep_vision)*1000:.0f}ms  "
                  f"avg={statistics.mean(ep_vision)*1000:.0f}ms  "
                  f"max={max(ep_vision)*1000:.0f}ms")
        if ep_step_total:
            print(f"    step_total: min={min(ep_step_total)*1000:.0f}ms  "
                  f"med={statistics.median(ep_step_total)*1000:.0f}ms  "
                  f"avg={statistics.mean(ep_step_total)*1000:.0f}ms  "
                  f"max={max(ep_step_total)*1000:.0f}ms  "
                  f"steps/sec={1.0/statistics.mean(ep_step_total):.1f}")

        all_capture.extend(ep_capture)
        all_vision.extend(ep_vision)
        all_extract.extend(ep_extract)
        all_fire.extend(ep_fire)
        all_step_total.extend(ep_step_total)

    # --- GLOBAL SUMMARY ---
    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY ({NUM_EPISODES} episodes, {len(all_step_total)} total steps)")
    print(f"{'='*70}")
    print(f"  resets:       avg={statistics.mean(all_reset):.1f}s  "
          f"min={min(all_reset):.1f}s  max={max(all_reset):.1f}s")

    for name, data in [
        ("fire_async", all_fire),
        ("capture", all_capture),
        ("vision", all_vision),
        ("extract_gf", all_extract),
        ("step_total", all_step_total),
    ]:
        if data:
            s = sorted(data)
            print(f"  {name:12s}  "
                  f"avg={statistics.mean(data)*1000:6.1f}ms  "
                  f"med={statistics.median(data)*1000:6.1f}ms  "
                  f"p95={s[int(len(s)*0.95)]*1000:6.1f}ms  "
                  f"max={max(data)*1000:6.1f}ms")

    if all_step_total:
        avg_step = statistics.mean(all_step_total)
        print(f"\n  Avg step: {avg_step*1000:.0f}ms = {1/avg_step:.1f} steps/sec")
        print(f"  Breakdown:")
        avg_fire = statistics.mean(all_fire)
        avg_cap = statistics.mean(all_capture)
        avg_vis = statistics.mean(all_vision)
        avg_ext = statistics.mean(all_extract)
        overhead = avg_step - avg_fire - avg_cap - avg_vis - avg_ext
        total = avg_step
        print(f"    fire_async:  {avg_fire*1000:6.1f}ms  ({avg_fire/total*100:4.1f}%)")
        print(f"    capture:     {avg_cap*1000:6.0f}ms  ({avg_cap/total*100:4.1f}%)")
        print(f"    vision:      {avg_vis*1000:6.0f}ms  ({avg_vis/total*100:4.1f}%)")
        print(f"    extract_gf:  {avg_ext*1000:6.1f}ms  ({avg_ext/total*100:4.1f}%)")
        print(f"    overhead:    {overhead*1000:6.0f}ms  ({overhead/total*100:4.1f}%)")

    capture.close()
    controller.close()


if __name__ == "__main__":
    main()
