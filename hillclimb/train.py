"""Training script: train a PPO agent on Hill Climb Racing."""

from __future__ import annotations

import argparse
from pathlib import Path

from hillclimb.config import cfg


def train(
    total_timesteps: int | None = None,
    render: bool = False,
    resume: str | None = None,
) -> None:
    """Train a PPO agent.

    Args:
        total_timesteps: Number of environment steps. Defaults to cfg value.
        render: Show debug window during training.
        resume: Path to existing model to continue training from.
    """
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from hillclimb.env import HillClimbEnv

    total_timesteps = total_timesteps or cfg.total_timesteps

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    # Create environment
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
            batch_size=cfg.batch_size,
            n_steps=cfg.n_steps,
            verbose=1,
            device=device,
            tensorboard_log=str(Path(cfg.log_dir) / "tensorboard"),
        )

    # Checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.n_steps * 10,  # save every ~10 rollouts
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_hillclimb",
    )

    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # Save final model
    save_path = model_dir / "ppo_hillclimb"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

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
        timesteps = args.episodes * 200  # rough estimate

    train(total_timesteps=timesteps, render=args.render, resume=args.resume)


if __name__ == "__main__":
    main()
