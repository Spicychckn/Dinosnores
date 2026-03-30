"""
PPO training script for the Dinosnores RL agent.

Uses MaskablePPO from sb3-contrib so the agent only samples from
valid actions at each step — critical for this environment since most
actions are invalid most of the time.

Before PPO begins, a Behavioural Cloning (BC) pre-training phase runs the
greedy heuristic for a number of episodes, then trains the policy network
via cross-entropy loss to imitate those demonstrations.  This warm-starts
the policy so PPO begins from a sensible baseline rather than random noise.

Usage
-----
    python3 train.py                                         # train from scratch (with BC pre-training)
    python3 train.py --pretrain-episodes 20                  # fewer/more BC demo episodes
    python3 train.py --no-pretrain                           # skip BC, pure RL from scratch
    python3 train.py --timesteps 5000000                     # longer PPO run
    python3 train.py --envs 16                               # more parallel envs
    python3 train.py --resume models/dinosnores_ppo_final    # continue training (skips BC)
    python3 train.py --resume models/best/best_model         # continue from best
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from dinosnores.env import DinosnoresEnv, ACTION_TO_IDX, N_ACTIONS, _OBS_DIM
from dinosnores.heuristic import GreedyHeuristic


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def _progress_bar_available() -> bool:
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def collect_heuristic_demos(
    n_episodes: int = 50,
    base_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the greedy heuristic for n_episodes and record every
    (observation, action_idx) pair.  Returns two arrays:
        obs     : float32 (N, obs_dim)
        actions : int64   (N,)

    Each episode uses a distinct seed (base_seed + ep) so the random
    plant-level and egg-type outcomes vary, giving BC diverse observations
    rather than near-identical copies of the same trajectory.
    """
    heuristic = GreedyHeuristic()
    obs_list, action_list = [], []
    total_reward = 0.0

    print(f"Collecting heuristic demonstrations ({n_episodes} episodes, seeds {base_seed}–{base_seed + n_episodes - 1})...")
    for ep in range(n_episodes):
        env = DinosnoresEnv(seed=base_seed + ep)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            valid = env.sim.get_valid_actions(env._state)
            action = heuristic.choose_action(env._state, valid)
            action_idx = ACTION_TO_IDX[action]

            obs_list.append(obs.copy())
            action_list.append(action_idx)

            obs, reward, terminated, truncated, _ = env.step(action_idx)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        print(
            f"  Episode {ep + 1:>3}/{n_episodes} (seed={base_seed + ep}) — "
            f"reward: {ep_reward:8.1f}  "
            f"score: {env._state.score:6}  "
            f"wake-ups: {env._state.wake_ups}"
        )

    print(
        f"\nHeuristic mean reward : {total_reward / n_episodes:.1f} "
        f"({len(obs_list):,} total steps)\n"
    )
    return (
        np.array(obs_list,    dtype=np.float32),
        np.array(action_list, dtype=np.int64),
    )


def pretrain_bc(
    model: MaskablePPO,
    obs: np.ndarray,
    actions: np.ndarray,
    n_epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3,
    early_stop_delta: float = 0.01,
) -> None:
    """
    Behavioural cloning: minimise cross-entropy between the policy's action
    distribution and the heuristic's demonstrated actions.

    Uses the policy's own evaluate_actions() so the same network weights
    that PPO will later fine-tune are the ones being pre-trained.

    Intentionally kept short (low n_epochs, early stopping on plateau) so
    the policy is warm-started near the heuristic without collapsing onto it.
    PPO's entropy bonus then keeps exploration alive for the fine-tuning phase.

    early_stop_delta : stop if relative loss improvement < this threshold,
                       e.g. 0.01 = stop when improvement drops below 1%.
    """
    obs_t    = torch.as_tensor(obs).to(model.device)
    action_t = torch.as_tensor(actions).to(model.device)

    dataset  = TensorDataset(obs_t, action_t)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    print(f"Behavioural cloning pre-training — up to {n_epochs} epochs, {len(obs):,} samples")
    prev_loss = float("inf")
    for epoch in range(n_epochs):
        total_loss = 0.0
        for obs_batch, action_batch in loader:
            # evaluate_actions returns (values, log_probs, entropy)
            _, log_probs, _ = model.policy.evaluate_actions(obs_batch, action_batch)
            loss = -log_probs.mean()  # maximise log-likelihood = minimise cross-entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(obs_batch)

        epoch_loss = total_loss / len(obs)
        improvement = (prev_loss - epoch_loss) / (prev_loss + 1e-8)
        print(f"  Epoch {epoch + 1}/{n_epochs} — loss: {epoch_loss:.4f}  (Δ {improvement:+.1%})")

        if epoch > 0 and improvement < early_stop_delta:
            print(f"  Early stop: improvement {improvement:.2%} < {early_stop_delta:.0%} threshold")
            break
        prev_loss = epoch_loss

    print("Pre-training complete.\n")


def make_env(seed: int = 0):
    """Factory used by make_vec_env."""
    def _init():
        return DinosnoresEnv(seed=seed)
    return _init


def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    n_steps: int = 8192,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 0,
    resume: str | None = None,
    pretrain_episodes: int = 50,
    bc_epochs: int = 3,
    bc_lr: float = 1e-3,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Observation dim : {_OBS_DIM}")
    print(f"Action count    : {N_ACTIONS}")
    print(f"Parallel envs   : {n_envs}")
    print(f"n_steps         : {n_steps}")
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
        # Override ent_coef when resuming to counteract entropy collapse,
        # which causes numerical instability (probs don't sum to 1) after
        # long training runs.
        model = MaskablePPO.load(
            resume,
            env=vec_env,
            verbose=1,
            tensorboard_log=log_dir if _tensorboard_available() else None,
            custom_objects={
                "ent_coef": 0.05,        # re-inject entropy after collapse
                "learning_rate": 1e-4,   # smaller lr for fine-tuning
                "max_grad_norm": 0.5,    # gradient clipping for stability
            },
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
            n_steps=n_steps,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,       # high gamma: rewards are sparse and long-horizon
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,     # higher entropy to prevent early collapse with large masked action space
            learning_rate=1e-4,
            max_grad_norm=0.5,
        )

    # BC pre-training: only on a fresh run (not when resuming)
    if resume is None and pretrain_episodes > 0:
        obs_demo, action_demo = collect_heuristic_demos(
            n_episodes=pretrain_episodes, base_seed=seed
        )
        pretrain_bc(model, obs_demo, action_demo, n_epochs=bc_epochs, lr=bc_lr)

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
    parser.add_argument("--timesteps",         type=int,  default=2_000_000)
    parser.add_argument("--envs",              type=int,  default=8)
    parser.add_argument("--n-steps",           type=int,  default=8192,
                        help="PPO rollout length per env before each update (default 8192)")
    parser.add_argument("--save-dir",          type=str,  default="models")
    parser.add_argument("--log-dir",           type=str,  default="logs")
    parser.add_argument("--seed",              type=int,  default=0)
    parser.add_argument("--resume",            type=str,  default=None,
                        help="Path to a saved model to continue training from")
    parser.add_argument("--pretrain-episodes", type=int,  default=50,
                        help="Heuristic episodes to collect for BC pre-training (0 to skip)")
    parser.add_argument("--bc-epochs",          type=int,  default=3,
                        help="Max BC training epochs (early stop if loss plateaus)")
    parser.add_argument("--bc-lr",              type=float, default=1e-3,
                        help="Learning rate for BC pre-training")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        n_steps=args.n_steps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        resume=args.resume,
        pretrain_episodes=args.pretrain_episodes,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
    )
