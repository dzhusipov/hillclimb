"""Training script: train a PPO agent on Hill Climb Racing."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from hillclimb.config import cfg

LOG_DIR = Path(cfg.log_dir)
EPISODES_JSONL = LOG_DIR / "train_episodes.jsonl"
STATUS_JSON = LOG_DIR / "training_status.json"


def _write_status(data: dict) -> None:
    """Atomically write training_status.json via temp + os.replace."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, STATUS_JSON)


def _append_episode(record: dict) -> None:
    """Append one JSON line to train_episodes.jsonl (O_APPEND safe)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(EPISODES_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


class EpisodeLogCallback:
    """Callback для логирования каждого эпизода в консоль + JSONL (multi-env aware)."""

    def __init__(self, num_envs: int = 1, total_timesteps: int = 0):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Inner(BaseCallback):
            def __init__(inner_self):
                super().__init__()
                inner_self._num_envs = num_envs
                inner_self._total_target = total_timesteps
                inner_self._episode_count = 0
                inner_self._episode_rewards = [0.0] * num_envs
                inner_self._episode_steps = [0] * num_envs
                inner_self._episode_starts = [time.time()] * num_envs
                inner_self._best_distance = 0.0
                inner_self._train_start = time.time()
                inner_self._recent_distances: list[float] = []

            def _on_step(inner_self) -> bool:
                infos = inner_self.locals.get("infos", [])
                rewards = inner_self.locals.get("rewards", [])
                dones = inner_self.locals.get("dones", [])

                for i in range(min(len(infos), inner_self._num_envs)):
                    inner_self._episode_steps[i] += 1
                    if i < len(rewards):
                        inner_self._episode_rewards[i] += float(rewards[i])

                    info = infos[i]
                    done = dones[i] if i < len(dones) else False
                    if done or "episode" in info:
                        inner_self._episode_count += 1
                        dt = time.time() - inner_self._episode_starts[i]
                        dist = info.get("max_distance_m", info.get("distance_m", 0))
                        if dist > inner_self._best_distance:
                            inner_self._best_distance = dist
                        total_t = time.time() - inner_self._train_start
                        gs = info.get("game_state", "?")
                        print(
                            f"  EP {inner_self._episode_count:3d} | "
                            f"env={i} | "
                            f"{inner_self._episode_steps[i]:4d} steps | "
                            f"{dt:5.1f}s | "
                            f"R={inner_self._episode_rewards[i]:+7.1f} | "
                            f"dist={dist:.0f}m | "
                            f"best={inner_self._best_distance:.0f}m | "
                            f"{inner_self.num_timesteps}/{inner_self.locals.get('total_timesteps', '?')} | "
                            f"{total_t/60:.1f}min | {gs}"
                        )

                        # JSONL episode log
                        _append_episode({
                            "episode": inner_self._episode_count,
                            "env_idx": i,
                            "timestamp": time.time(),
                            "distance": dist,
                            "reward": round(inner_self._episode_rewards[i], 1),
                            "steps": inner_self._episode_steps[i],
                            "duration_s": round(dt, 1),
                            "state_final": gs,
                        })

                        # Atomic status update every 10 episodes
                        inner_self._recent_distances.append(dist)
                        if len(inner_self._recent_distances) > 10:
                            inner_self._recent_distances = inner_self._recent_distances[-10:]
                        if inner_self._episode_count % 10 == 0:
                            elapsed_h = total_t / 3600
                            _write_status({
                                "training_active": True,
                                "current_episode": inner_self._episode_count,
                                "total_timesteps": inner_self.num_timesteps,
                                "total_target": inner_self._total_target,
                                "best_distance": inner_self._best_distance,
                                "avg_distance_10": round(
                                    sum(inner_self._recent_distances) / len(inner_self._recent_distances), 1
                                ),
                                "episodes_per_hour": round(inner_self._episode_count / max(elapsed_h, 0.001)),
                                "last_update": time.time(),
                            })

                        inner_self._episode_rewards[i] = 0.0
                        inner_self._episode_steps[i] = 0
                        inner_self._episode_starts[i] = time.time()
                return True

        self._cls = _Inner

    def create(self):
        return self._cls()


def make_env(adb_serial: str, render_mode: str | None = None):
    """Factory function for SubprocVecEnv (applies TimeLimit wrapper)."""
    def _init():
        from gymnasium.wrappers import TimeLimit

        from hillclimb.env import HillClimbEnv
        env = HillClimbEnv(adb_serial=adb_serial, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        return env
    return _init


def train(
    total_timesteps: int | None = None,
    num_envs: int = 1,
    render: bool = False,
    resume: str | None = None,
    skip_envs: set[int] | None = None,
) -> None:
    """Train a PPO agent with CNN+MLP (MultiInputPolicy)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv,
        VecFrameStack,
        VecMonitor,
        VecNormalize,
        VecTransposeImage,
    )

    skip_envs = skip_envs or set()
    total_timesteps = total_timesteps or cfg.total_timesteps

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Device: {device}")

    # n_steps per env; with SubprocVecEnv effective rollout = n_steps * num_envs
    n_steps = min(cfg.n_steps, max(total_timesteps // num_envs, 1))

    # Environment(s)
    render_mode = "human" if render else None
    serials = [cfg.emulator_serial(i) for i in range(num_envs) if i not in skip_envs]
    actual_num_envs = len(serials)
    print(f"Emulators ({actual_num_envs}): {', '.join(serials)}")
    if skip_envs:
        print(f"Skipped envs: {sorted(skip_envs)}")
    num_envs = actual_num_envs

    env = SubprocVecEnv([make_env(s, render_mode) for s in serials])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    env = VecFrameStack(env, n_stack=cfg.n_stack)
    env = VecTransposeImage(env)

    # Model directory
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # batch_size must be <= n_steps * num_envs
    effective_rollout = n_steps * num_envs
    batch_size = min(cfg.batch_size, effective_rollout)

    # Create or resume model
    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=env, device=device)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=cfg.learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            verbose=1,
            device=device,
            tensorboard_log=str(Path(cfg.log_dir) / "tensorboard"),
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(n_steps * 5, 1000),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_hillclimb",
    )
    episode_cb = EpisodeLogCallback(num_envs=num_envs, total_timesteps=total_timesteps).create()

    print(f"Training for {total_timesteps} timesteps (n_steps={n_steps}, batch={batch_size})")
    print(f"Config: lr={cfg.learning_rate}, envs={num_envs}, rollout={effective_rollout}")
    print(f"PPO: n_epochs={cfg.n_epochs}, gamma={cfg.gamma}, clip={cfg.clip_range}, ent={cfg.ent_coef}")
    print(f"Obs: image(84,84,{cfg.n_stack}) + vector({6 * cfg.n_stack},) | Policy: MultiInputPolicy")
    print(f"Actions: Discrete(3) — 0=nothing, 1=gas, 2=brake | hold={cfg.action_hold_ms}ms")
    print("-" * 70)

    save_path = model_dir / "ppo_hillclimb"

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([checkpoint_cb, episode_cb]),
            progress_bar=True,
        )
        print("\nTraining complete.")
    except KeyboardInterrupt:
        print("\n\nCtrl+C — saving model before exit...")
    except Exception as exc:
        print(f"\n\nTraining crashed: {exc}")
        print("Saving model before exit...")

    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    # Save VecNormalize statistics (reward running mean/std)
    vec_norm_path = save_path.with_suffix(".vecnorm.pkl")
    env.save(str(vec_norm_path))
    print(f"VecNormalize stats saved to {vec_norm_path}")

    # Mark training as inactive
    _write_status({"training_active": False, "last_update": time.time()})

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent for Hill Climb Racing")
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"Total timesteps (default: {cfg.total_timesteps})")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Approximate episodes (converted to timesteps, ~200 steps/episode)")
    parser.add_argument("--num-envs", type=int, default=cfg.num_emulators,
                        help=f"Number of parallel environments (default: {cfg.num_emulators})")
    parser.add_argument("--render", action="store_true",
                        help="Show debug window during training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume training from")
    parser.add_argument("--skip-envs", type=str, default=None,
                        help="Comma-separated env indices to skip (e.g. '0' or '0,3')")
    args = parser.parse_args()

    timesteps = args.timesteps
    if timesteps is None and args.episodes is not None:
        timesteps = args.episodes * 200

    skip = set()
    if args.skip_envs:
        skip = {int(x.strip()) for x in args.skip_envs.split(",")}

    train(
        total_timesteps=timesteps,
        num_envs=args.num_envs,
        render=args.render,
        resume=args.resume,
        skip_envs=skip,
    )


if __name__ == "__main__":
    main()
