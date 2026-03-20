"""
Evaluate a trained Dinosnores PPO agent and explain its strategy.

Outputs two things:
  1. A step-by-step trace of one full episode (what the agent does and why
     the state looks interesting at that moment).
  2. An action frequency table across N episodes showing which actions the
     agent relies on most.

Usage
-----
    python3 evaluate.py                              # uses models/best/best_model.zip
    python3 evaluate.py --model models/dinosnores_ppo_final.zip
    python3 evaluate.py --model models/best/best_model.zip --episodes 50
    python3 evaluate.py --trace-only                 # skip frequency table
"""

import argparse
import collections
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from dinosnores.env import DinosnoresEnv, ALL_ACTIONS
from dinosnores.constants import GAME_DURATION_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_action(model, obs, masks) -> int:
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    return int(action)


def _is_noteworthy(info: dict, reward: float, prev_wake_ups: int, state) -> bool:
    """Return True if this step is worth printing in the trace."""
    return (
        reward > 0
        or info.get("woke_trex")
        or info.get("spawned")
        or info.get("grew")
        or info.get("merged_egg")
        or info.get("merged_plant")
        or info.get("beacon_used")
        or info.get("fed_meteor")
        or info.get("fed")
        or info.get("summoned")
        or state.wake_ups != prev_wake_ups
    )


# ---------------------------------------------------------------------------
# Single episode trace
# ---------------------------------------------------------------------------

def run_trace(model, seed: int = 0, max_duration_seconds: float = GAME_DURATION_SECONDS):
    env = DinosnoresEnv(seed=seed, max_duration_seconds=max_duration_seconds)
    obs, _ = env.reset()
    state = env._state

    print("=" * 70)
    print("STRATEGY TRACE  (deterministic rollout, seed={})".format(seed))
    print("=" * 70)
    print(f"{'Turn':>6}  {'Time':>7}  {'Action':<28}  {'T-Rex HP':>10}  {'Score':>6}  Notes")
    print("-" * 90)

    total_reward = 0.0
    action_counts: dict[str, int] = collections.defaultdict(int)

    while True:
        masks = env.action_masks()
        action_idx = _pick_action(model, obs, masks)
        action_name = ALL_ACTIONS[action_idx].value
        action_counts[action_name] += 1

        prev_wake_ups = state.wake_ups
        obs, reward, terminated, truncated, info = env.step(action_idx)
        state = env._state
        total_reward += reward

        # Build a short notes string from info
        notes = []
        if info.get("woke_trex"):
            notes.append(f"WAKE! +{info.get('score_earned', 0)} score")
        if info.get("spawned"):
            notes.append(f"spawned {info['spawned']}")
        if info.get("grew"):
            notes.append(f"grew {info['grew']}")
        if info.get("merged_egg"):
            notes.append(f"egg→baby ({info['merged_egg']})")
        if info.get("merged_plant"):
            notes.append(f"plant lvl{info['merged_plant']}→{info['merged_plant']+1}")
        if info.get("beacon_used"):
            notes.append(f"beacon → HP={info['hp_after']}")
        if info.get("fed_meteor"):
            notes.append("meteor→soup")
        if info.get("fed"):
            ct, lvl, val = info["fed"]
            notes.append(f"fed {ct} lvl{lvl} (+{val})")
        if info.get("summoned"):
            notes.append(f"summoned {info['summoned']}")

        if _is_noteworthy(info, reward, prev_wake_ups, state):
            hours, rem = divmod(state.elapsed_seconds, 3600)
            mins = rem // 60
            hp_str = f"{state.trex_hp}/{state.trex_max_hp}"
            print(
                f"{state.turn:>6}  {hours}h{mins:02d}m  "
                f"{action_name:<28}  {hp_str:>10}  {state.score:>6}  "
                + ", ".join(notes)
            )

        if terminated or truncated:
            break

    print("-" * 90)
    print(f"Episode finished — {state.turn} turns | Score: {state.score} | Wake-ups: {state.wake_ups}")
    print()
    return action_counts, state.score


# ---------------------------------------------------------------------------
# Multi-episode frequency table
# ---------------------------------------------------------------------------

def run_frequency_table(model, n_episodes: int = 20, seed_offset: int = 100):
    print("=" * 70)
    print(f"ACTION FREQUENCY TABLE  ({n_episodes} episodes)")
    print("=" * 70)

    total_counts: dict[str, int] = collections.defaultdict(int)
    scores = []

    for ep in range(n_episodes):
        env = DinosnoresEnv(seed=seed_offset + ep)
        obs, _ = env.reset()
        ep_counts: dict[str, int] = collections.defaultdict(int)

        while True:
            masks = env.action_masks()
            action_idx = _pick_action(model, obs, masks)
            ep_counts[ALL_ACTIONS[action_idx].value] += 1
            obs, _, terminated, truncated, _ = env.step(action_idx)
            if terminated or truncated:
                break

        for k, v in ep_counts.items():
            total_counts[k] += v
        scores.append(env._state.score)
        print(f"  Episode {ep+1:2d}: score={env._state.score:6d}  wake-ups={env._state.wake_ups}")

    total_steps = sum(total_counts.values())
    print()
    print(f"Mean score: {np.mean(scores):.0f} ± {np.std(scores):.0f}   "
          f"(min {min(scores)}, max {max(scores)})")
    print()
    print(f"{'Action':<30}  {'Count':>8}  {'% of steps':>10}")
    print("-" * 55)
    for action, count in sorted(total_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_steps
        if pct >= 0.1:  # skip negligible actions
            print(f"{action:<30}  {count:>8}  {pct:>9.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="models/best/best_model.zip",
        help="Path to saved MaskablePPO model (.zip)"
    )
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes for frequency table")
    parser.add_argument("--trace-seed", type=int, default=0,
                        help="RNG seed for the strategy trace episode")
    parser.add_argument("--trace-only", action="store_true",
                        help="Skip the frequency table")
    parser.add_argument("--freq-only", action="store_true",
                        help="Skip the strategy trace")
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...\n")
    model = MaskablePPO.load(args.model)

    if not args.freq_only:
        run_trace(model, seed=args.trace_seed)

    if not args.trace_only:
        run_frequency_table(model, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
