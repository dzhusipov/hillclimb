"""Training script: train a PPO agent on Hill Climb Racing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hillclimb.config import cfg


class EpisodeLogCallback:
    """Callback для логирования каждого эпизода в консоль (multi-env aware)."""

    def __init__(self, num_envs: int = 1):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Inner(BaseCallback):
            def __init__(inner_self):
                super().__init__()
                inner_self._num_envs = num_envs
                inner_self._episode_count = 0
                inner_self._episode_rewards = [0.0] * num_envs
                inner_self._episode_steps = [0] * num_envs
                inner_self._episode_starts = [time.time()] * num_envs
                inner_self._best_distance = 0.0
                inner_self._train_start = time.time()

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
                        inner_self._episode_rewards[i] = 0.0
                        inner_self._episode_steps[i] = 0
                        inner_self._episode_starts[i] = time.time()
                return True

        self._cls = _Inner

    def create(self):
        return self._cls()


def make_env(adb_serial: str, render_mode: str | None = None):
    """Factory function for SubprocVecEnv."""
    def _init():
        from hillclimb.env import HillClimbEnv
        return HillClimbEnv(adb_serial=adb_serial, render_mode=render_mode)
    return _init


def train(
    total_timesteps: int | None = None,
    num_envs: int = 1,
    render: bool = False,
    resume: str | None = None,
) -> None:
    """Train a PPO agent."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from hillclimb.env import HillClimbEnv

    total_timesteps = total_timesteps or cfg.total_timesteps

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Device: {device}")

    # n_steps per env; with SubprocVecEnv effective rollout = n_steps * num_envs
    n_steps = min(cfg.n_steps, max(total_timesteps // num_envs, 1))

    # Environment(s)
    render_mode = "human" if render else None
    serials = [cfg.emulator_serial(i) for i in range(num_envs)]
    print(f"Emulators ({num_envs}): {', '.join(serials)}")

    if num_envs == 1:
        env = HillClimbEnv(adb_serial=serials[0], render_mode=render_mode)
    else:
        env = SubprocVecEnv([make_env(s, render_mode) for s in serials])

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
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
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
    episode_cb = EpisodeLogCallback(num_envs=num_envs).create()

    print(f"Training for {total_timesteps} timesteps (n_steps={n_steps}, batch={batch_size})")
    print(f"Config: lr={cfg.learning_rate}, envs={num_envs}, rollout={effective_rollout}")
    print(f"Actions: Discrete(3) — 0=nothing, 1=gas, 2=brake")
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
    args = parser.parse_args()

    timesteps = args.timesteps
    if timesteps is None and args.episodes is not None:
        timesteps = args.episodes * 200

    train(
        total_timesteps=timesteps,
        num_envs=args.num_envs,
        render=args.render,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
