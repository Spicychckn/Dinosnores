"""
PPO training script for the Dinosnores RL agent.

Uses MaskablePPO from sb3-contrib so the agent only samples from
valid actions at each step — critical for this environment since most
actions are invalid most of the time.

Usage
-----
    python3 train.py                                         # train from scratch
    python3 train.py --timesteps 5000000                     # longer run
    python3 train.py --envs 16                               # more parallel envs
    python3 train.py --resume models/dinosnores_ppo_final    # continue training
    python3 train.py --resume models/best/best_model         # continue from best
"""

import argparse
import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from dinosnores.env import DinosnoresEnv, N_ACTIONS, _OBS_DIM


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def _progress_bar_available() -> bool:
    try:
        import tqdm, rich  # noqa: F401
        return True
    except ImportError:
        return False


def make_env(seed: int = 0):
    """Factory used by make_vec_env."""
    def _init():
        return DinosnoresEnv(seed=seed)
    return _init


def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 0,
    resume: str | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Observation dim : {_OBS_DIM}")
    print(f"Action count    : {N_ACTIONS}")
    print(f"Parallel envs   : {n_envs}")
    print(f"Total timesteps : {total_timesteps:,}")
    if resume:
        print(f"Resuming from   : {resume}")
    print()

    # Parallel training environments
    vec_env = make_vec_env(DinosnoresEnv, n_envs=n_envs, seed=seed)

    # Separate eval environment (single, deterministic seed)
    eval_env = DummyVecEnv([make_env(seed=999)])

    # Save a checkpoint every 100k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="dinosnores_ppo",
    )

    # Evaluate and save the best model every 50k steps
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=log_dir,
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    if resume:
        model = MaskablePPO.load(
            resume,
            env=vec_env,
            verbose=1,
            tensorboard_log=log_dir if _tensorboard_available() else None,
        )
        print(f"Loaded model — resuming from {model.num_timesteps:,} timesteps\n")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir if _tensorboard_available() else None,
            seed=seed,
            # PPO hyperparameters — reasonable defaults for a discrete game env
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,       # high gamma: rewards are sparse and long-horizon
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,     # encourage exploration early on
            learning_rate=3e-4,
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        use_masking=True,
        progress_bar=_progress_bar_available(),
        reset_num_timesteps=resume is None,
    )

    final_path = os.path.join(save_dir, "dinosnores_ppo_final")
    model.save(final_path)
    print(f"\nModel saved to {final_path}.zip")

    # Quick evaluation of the final model
    print("\nEvaluating final model (20 episodes)...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, use_masking=True
    )
    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs",      type=int, default=8)
    parser.add_argument("--save-dir",  type=str, default="models")
    parser.add_argument("--log-dir",   type=str, default="logs")
    parser.add_argument("--seed",      type=int,  default=0)
    parser.add_argument("--resume",    type=str,  default=None,
                        help="Path to a saved model to continue training from")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        resume=args.resume,
    )
