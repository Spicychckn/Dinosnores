"""
Run the greedy heuristic and produce a trace / frequency table.

Mirrors evaluate.py but uses GreedyHeuristic instead of a trained model,
making it easy to debug the heuristic's decisions without needing a saved model.

Usage
-----
    python3 run_heuristic.py                    # trace + frequency table
    python3 run_heuristic.py --trace-only       # skip frequency table
    python3 run_heuristic.py --freq-only        # skip trace
    python3 run_heuristic.py --verbose          # print every step, not just noteworthy ones
    python3 run_heuristic.py --episodes 10      # run 10 episodes for frequency table
    python3 run_heuristic.py --seed 42          # set RNG seed for trace episode
"""

import argparse
import collections

import numpy as np

from dinosnores.constants import GAME_DURATION_SECONDS, HerbivoreType
from dinosnores.heuristic import _SOUP_PER_STEGO, FREE_SPACES_BUFFER, GreedyHeuristic
from dinosnores.simulator import DinosnoresSimulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_noteworthy(info: dict, reward: float, prev_wake_ups: int, state) -> bool:
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
        or info.get("damage_dealt")
        or state.wake_ups != prev_wake_ups
    )


def _format_notes(info: dict) -> str:
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
        notes.append(f"plant lvl{info['merged_plant']}→{info['merged_plant'] + 1}")
    if info.get("beacon_used"):
        notes.append(f"beacon → HP={info['hp_after']}")
    if info.get("fed_meteor"):
        notes.append("meteor→soup")
    if info.get("fed"):
        ct, lvl, val = info["fed"]
        notes.append(f"fed {ct} lvl{lvl} (+{val})")
    if info.get("summoned"):
        notes.append(f"summoned {info['summoned']}")
    if info.get("damage_dealt"):
        notes.append(f"dmg={info['damage_dealt']}")
    return ", ".join(notes)


# ---------------------------------------------------------------------------
# Single episode trace
# ---------------------------------------------------------------------------


def run_trace(
    seed: int = 0,
    max_duration_seconds: float = GAME_DURATION_SECONDS,
    verbose: bool = False,
):
    sim = DinosnoresSimulator(max_duration_seconds=max_duration_seconds, seed=seed)
    state = sim.reset()
    heuristic = GreedyHeuristic()

    print("=" * 110)
    print(f"HEURISTIC TRACE  (seed={seed}{'  verbose' if verbose else ''})")
    print("=" * 110)
    print(
        f"{'Turn':>6}  {'Time':>7}  {'Action':<28}  "
        f"{'T-Rex HP':>12}  {'Grid':>6}  {'Soup':>9}  "
        f"{'Stegos':>6}  {'Score':>6}  Notes"
    )
    print("-" * 110)

    total_reward = 0.0
    action_counts: dict[str, int] = collections.defaultdict(int)

    while True:
        valid = sim.get_valid_actions(state)
        action = heuristic.choose_action(state, valid)
        action_name = action.value
        action_counts[action_name] += 1

        prev_wake_ups = state.wake_ups
        next_state, reward, done, info = sim.step(state, action)
        state = next_state
        total_reward += reward

        notes = _format_notes(info)

        # In verbose mode print everything; otherwise only noteworthy steps
        if verbose or _is_noteworthy(info, reward, prev_wake_ups, state):
            hours, rem = divmod(state.elapsed_seconds, 3600)
            mins = rem // 60
            hp_str = f"{state.trex_hp}/{state.trex_max_hp}"
            grid_str = f"{state.grid_available()}/32"
            soup_str = f"{state.primordial_soup:,}"
            n_stegos = state.adult_herbivores[HerbivoreType.STEGOSAURUS]

            # In verbose mode, annotate why the attack phase hasn't triggered yet
            if verbose and action_name == "wait" and n_stegos > 0:
                needed = n_stegos * _SOUP_PER_STEGO
                free = state.grid_available()
                if free > FREE_SPACES_BUFFER:
                    notes = notes or f"grid not full yet (free={free}, buffer={FREE_SPACES_BUFFER})"
                elif state.primordial_soup < needed:
                    notes = notes or f"waiting for soup ({state.primordial_soup:,}/{needed:,})"

            print(
                f"{state.turn:>6}  {hours}h{mins:02d}m  "
                f"{action_name:<28}  {hp_str:>12}  {grid_str:>6}  {soup_str:>9}  "
                f"{n_stegos:>6}  {state.score:>6}  {notes}"
            )

        if done:
            break

    print("-" * 110)
    print(
        f"Episode finished — {state.turn} turns | Score: {state.score} | Wake-ups: {state.wake_ups} | Total reward: {total_reward:.1f}"
    )
    print()
    return action_counts, state.score


# ---------------------------------------------------------------------------
# Multi-episode frequency table
# ---------------------------------------------------------------------------


def run_frequency_table(n_episodes: int = 20, seed_offset: int = 100):
    print("=" * 70)
    print(f"ACTION FREQUENCY TABLE  ({n_episodes} episodes)")
    print("=" * 70)

    total_counts: dict[str, int] = collections.defaultdict(int)
    scores = []

    for ep in range(n_episodes):
        sim = DinosnoresSimulator(seed=seed_offset + ep)
        state = sim.reset()
        heuristic = GreedyHeuristic()
        ep_counts: dict[str, int] = collections.defaultdict(int)

        while True:
            valid = sim.get_valid_actions(state)
            action = heuristic.choose_action(state, valid)
            ep_counts[action.value] += 1
            state, _, done, _ = sim.step(state, action)
            if done:
                break

        for k, v in ep_counts.items():
            total_counts[k] += v
        scores.append(state.score)
        print(f"  Episode {ep + 1:2d}: score={state.score:6d}  wake-ups={state.wake_ups}")

    total_steps = sum(total_counts.values())
    print()
    print(
        f"Mean score: {np.mean(scores):.0f} ± {np.std(scores):.0f}   "
        f"(min {min(scores)}, max {max(scores)})"
    )
    print()
    print(f"{'Action':<30}  {'Count':>8}  {'% of steps':>10}")
    print("-" * 55)
    for action, count in sorted(total_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_steps
        if pct >= 0.1:
            print(f"{action:<30}  {count:>8}  {pct:>9.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20, help="Episodes for frequency table")
    parser.add_argument("--trace-only", action="store_true", help="Skip the frequency table")
    parser.add_argument("--freq-only", action="store_true", help="Skip the strategy trace")
    parser.add_argument(
        "--verbose", action="store_true", help="Print every step, not just noteworthy ones"
    )
    args = parser.parse_args()

    if not args.freq_only:
        run_trace(seed=args.seed, verbose=args.verbose)

    if not args.trace_only:
        run_frequency_table(n_episodes=args.episodes)


if __name__ == "__main__":
    main()
