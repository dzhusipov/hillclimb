"""Evaluate a trained RL model: run episodes and report statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hillclimb.config import cfg


def evaluate(
    model_path: str | None = None,
    n_episodes: int = 10,
    render: bool = True,
) -> None:
    """Run the trained model for n_episodes and print stats."""
    from stable_baselines3 import PPO

    from hillclimb.env import HillClimbEnv

    model_path = model_path or str(Path(cfg.model_dir) / "ppo_hillclimb")
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    render_mode = "human" if render else None
    env = HillClimbEnv(render_mode=render_mode)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}, "
              f"final_fuel={info.get('fuel', '?')}")

    env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.1f} +/- {np.std(episode_rewards):.1f}")
    print(f"Mean length: {np.mean(episode_lengths):.0f} +/- {np.std(episode_lengths):.0f}")
    print(f"Best reward: {max(episode_rewards):.1f}")
    print(f"Worst reward: {min(episode_rewards):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model (default: models/ppo_hillclimb)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable debug window")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
