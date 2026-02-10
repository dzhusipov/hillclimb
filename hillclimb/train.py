"""Training script: train a PPO agent on Hill Climb Racing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hillclimb.config import cfg


class EpisodeLogCallback:
    """Callback для логирования каждого эпизода в консоль."""

    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Inner(BaseCallback):
            def __init__(inner_self):
                super().__init__()
                inner_self._episode_count = 0
                inner_self._episode_start = time.time()
                inner_self._episode_rewards = 0.0
                inner_self._episode_steps = 0
                inner_self._best_distance = 0.0
                inner_self._train_start = time.time()

            def _on_step(inner_self) -> bool:
                inner_self._episode_steps += 1
                infos = inner_self.locals.get("infos", [])
                rewards = inner_self.locals.get("rewards", [0])
                inner_self._episode_rewards += float(rewards[0])

                for info in infos:
                    if "episode" in info or info.get("game_state") == "RESULTS":
                        inner_self._episode_count += 1
                        dt = time.time() - inner_self._episode_start
                        # max_distance_m из RACING шагов (надёжнее results OCR)
                        dist = info.get("max_distance_m", info.get("distance_m", 0))
                        if dist > inner_self._best_distance:
                            inner_self._best_distance = dist
                        total_t = time.time() - inner_self._train_start
                        gs = info.get("game_state", "?")
                        print(
                            f"  EP {inner_self._episode_count:3d} | "
                            f"{inner_self._episode_steps:4d} steps | "
                            f"{dt:5.1f}s | "
                            f"R={inner_self._episode_rewards:+7.1f} | "
                            f"dist={dist:.0f}m | "
                            f"best={inner_self._best_distance:.0f}m | "
                            f"{inner_self.num_timesteps}/{inner_self.locals.get('total_timesteps', '?')} | "
                            f"{total_t/60:.1f}min | {gs}"
                        )
                        inner_self._episode_rewards = 0.0
                        inner_self._episode_steps = 0
                        inner_self._episode_start = time.time()
                return True

        self._cls = _Inner

    def create(self):
        return self._cls()


def train(
    total_timesteps: int | None = None,
    render: bool = False,
    resume: str | None = None,
) -> None:
    """Train a PPO agent."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from hillclimb.env import HillClimbEnv

    total_timesteps = total_timesteps or cfg.total_timesteps

    # MlpPolicy быстрее на CPU чем на MPS (нет CNN — нет смысла в GPU)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Device: {device}")

    # n_steps для коротких прогонов: min(2048, total_timesteps)
    n_steps = min(cfg.n_steps, total_timesteps)

    # Environment
    render_mode = "human" if render else None
    env = HillClimbEnv(render_mode=render_mode)

    # Model directory
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create or resume model
    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            batch_size=min(cfg.batch_size, n_steps),
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
    episode_cb = EpisodeLogCallback().create()

    print(f"Training for {total_timesteps} timesteps (n_steps={n_steps})")
    print(f"Config: lr={cfg.learning_rate}, batch={min(cfg.batch_size, n_steps)}")
    print(f"Actions: MultiDiscrete([3, 5]) — тип × длительность [100,200,400,700,1100]мс")
    print("-" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_cb, episode_cb]),
        progress_bar=True,
    )

    # Save final model
    save_path = model_dir / "ppo_hillclimb"
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent for Hill Climb Racing")
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"Total timesteps (default: {cfg.total_timesteps})")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Approximate episodes (converted to timesteps, ~200 steps/episode)")
    parser.add_argument("--render", action="store_true",
                        help="Show debug window during training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume training from")
    args = parser.parse_args()

    timesteps = args.timesteps
    if timesteps is None and args.episodes is not None:
        timesteps = args.episodes * 200

    train(total_timesteps=timesteps, render=args.render, resume=args.resume)


if __name__ == "__main__":
    main()
