"""
Demo script: runs the Dinosnores simulator with a simple greedy heuristic agent.

The heuristic follows a fixed priority order to exercise all game mechanics:
  feed meteor → feed currency → attack → grow → merge eggs → merge plants
  → spawn egg → spawn plant → wait
"""

from dinosnores import ActionType, DinosnoresSimulator

ATTACK_PRIORITY = [
    ActionType.ATTACK_SABER_TOOTH,
    ActionType.ATTACK_RAPTOR,
    ActionType.ATTACK_MAMMOTH,
    ActionType.ATTACK_PTERODACTYL,
    ActionType.ATTACK_TRICERATOPS,
    ActionType.ATTACK_BRONTOSAURUS,
    ActionType.ATTACK_STEGOSAURUS,
]

GROW_PRIORITY = [
    ActionType.GROW_RAPTOR,
    ActionType.GROW_PTERODACTYL,
    ActionType.GROW_TRICERATOPS,
    ActionType.GROW_BRONTOSAURUS,
    ActionType.GROW_STEGOSAURUS,
]

MERGE_EGG_PRIORITY = [
    ActionType.MERGE_RAPTOR_EGG,
    ActionType.MERGE_PTERODACTYL_EGG,
    ActionType.MERGE_BRONTOSAURUS_EGG,
    ActionType.MERGE_TRICERATOPS_EGG,
    ActionType.MERGE_STEGOSAURUS_EGG,
]

FEED_CURRENCY_PRIORITY = [
    ActionType.FEED_FANGS,
    ActionType.FEED_HORNS,
    ActionType.FEED_BONES,
]


def heuristic_agent(valid_actions, state):
    valid_set = set(valid_actions)

    if ActionType.USE_BEACON in valid_set:
        return ActionType.USE_BEACON
    if ActionType.FEED_METEOR in valid_set:
        return ActionType.FEED_METEOR
    for a in FEED_CURRENCY_PRIORITY:
        if a in valid_set:
            return a
    for a in ATTACK_PRIORITY:
        if a in valid_set:
            return a
    for a in GROW_PRIORITY:
        if a in valid_set:
            return a

    # Merge plants whenever possible to build toward higher levels
    if ActionType.MERGE_PLANT in valid_set:
        return ActionType.MERGE_PLANT

    # Decide whether to spawn plant or egg next.
    # Spawn eggs only if we have at least as many plants in the pipeline
    # as babies waiting to be grown (each baby needs 8 lvl1 plant-equivalents).
    total_plants = sum(count * (2 ** (lvl - 1)) for lvl, count in state.plants.items() if count > 0)
    total_babies = sum(state.baby_herbivores.values()) + sum(state.baby_carnivores.values())
    plants_needed = total_babies * 8  # rough lvl1-equivalent cost per baby steg

    if ActionType.SPAWN_PLANT in valid_set and total_plants < plants_needed + 8:
        return ActionType.SPAWN_PLANT

    for a in MERGE_EGG_PRIORITY:
        if a in valid_set:
            return a
    if ActionType.SPAWN_HERBIVORE_EGG in valid_set:
        return ActionType.SPAWN_HERBIVORE_EGG
    if ActionType.SPAWN_CARNIVORE_EGG in valid_set:
        return ActionType.SPAWN_CARNIVORE_EGG
    if ActionType.SPAWN_PLANT in valid_set:
        return ActionType.SPAWN_PLANT
    return ActionType.WAIT


def main():
    sim = DinosnoresSimulator(seed=0, max_duration_seconds=10 * 3600)
    state = sim.reset()

    print("=== Dinosnores Simulator Demo ===")
    print(state)
    print()

    total_reward = 0.0
    while True:
        valid = sim.get_valid_actions(state)
        action = heuristic_agent(valid, state)

        state, reward, done, info = sim.step(state, action)
        total_reward += reward

        if reward > 0 or "woke_trex" in info:
            hours, rem = divmod(state.elapsed_seconds, 3600)
            mins = rem // 60
            print(
                f"Turn {state.turn:6d} ({hours}h {mins:02d}m) | Action: {action.value:25s} | "
                f"Reward: {reward:5.0f} | Score: {state.score:6d} | "
                f"Wake-ups: {state.wake_ups} | T-Rex HP: {state.trex_hp}/{state.trex_max_hp}"
            )

        if done:
            break

    hours, rem = divmod(state.elapsed_seconds, 3600)
    mins = rem // 60
    print()
    print(f"Episode finished after {state.turn} turns ({hours}h {mins:02d}m).")
    print(f"Final score: {state.score}  |  Total reward: {total_reward:.0f}")
    print()
    print(state)


if __name__ == "__main__":
    main()
