"""
Greedy heuristic for the Dinosnores simulator.

Deterministic three-phase strategy:

Phase 0 – OPENING (runs once):
  1. USE_BEACON          → halves T-Rex HP: 100 → 50
  2. ATTACK_TRICERATOPS  → 50 dmg kills T-Rex (1st wake-up); drops 1 lvl-1 horn
  3. USE_BEACON          → halves new T-Rex HP: 125 → 62  (handled by always-beacon rule)
  4. ATTACK_TRICERATOPS  → 50 dmg, drops 1 lvl-1 horn (T-Rex survives at 12 HP)
  5. MERGE_HORNS         → 2× lvl-1 → 1× lvl-2 horn

Phase 1 – FILL (build stegos one at a time until grid is full):
  - Wait until soup ≥ STEGO_SOUP_COST (8,000) to commit to each stego.
  - Plant-first: fully build the lvl-4 plant before spawning any eggs.
  - Repeat until grid_available() ≤ FREE_SPACES_BUFFER.

Phase 2 – ATTACK LOOP (once board is full):
  - Wait until soup ≥ ATTACK_BATCH × STEGO_SOUP_COST (64,000).
  - Attack exactly ATTACK_BATCH stegos; each drops a lvl-1 bone.
    (Bone merges are interleaved between attacks via the always-merge rule.)
  - After all attacks: remaining bones merge up to lvl-4 and auto-feed.
  - Refill the ATTACK_BATCH used slots with new stegos (plant-first, one at a time).
  - Repeat.

Soup cost per adult Stego from scratch (at HN/VP lvl-1):
  - 2 eggs : 2 × 2,000 = 4,000 soup
  - 8 lvl-1 plants → 1 lvl-4 plant : 8 × 500 = 4,000 soup
  - Total : 8,000 soup
"""

from typing import List, Optional

from .actions import ActionType
from .state import GameState
from .constants import HerbivoreType, MAX_CURRENCY_LEVEL


ATTACK_BATCH       = 8    # stegos to attack per wave
FREE_SPACES_BUFFER = 4    # grid spaces kept free for merging headroom

_SOUP_PER_STEGO    = 8_000
_ATTACK_THRESHOLD  = ATTACK_BATCH * _SOUP_PER_STEGO  # 64,000


class GreedyHeuristic:
    """
    Priority-based heuristic with a small piece of state (_batch_remaining)
    to track how many attacks remain in the current wave, ensuring exactly
    ATTACK_BATCH stegos are attacked before refilling begins.
    """

    def __init__(self):
        self._batch_remaining: int = 0  # attacks still owed in the current wave

    def choose_action(
        self, state: GameState, valid_actions: List[ActionType]
    ) -> ActionType:
        va = set(valid_actions)

        def has(a: ActionType) -> bool:
            return a in va

        free         = state.grid_available()
        grid_full    = free <= FREE_SPACES_BUFFER
        n_stegos     = state.adult_herbivores[HerbivoreType.STEGOSAURUS]
        stego_eggs   = state.herbivore_eggs[HerbivoreType.STEGOSAURUS]
        stego_babies = state.baby_herbivores[HerbivoreType.STEGOSAURUS]
        lvl4_plants  = state.plants.get(4, 0)
        trices       = state.adult_herbivores[HerbivoreType.TRICERATOPS]

        # ----------------------------------------------------------------
        # ALWAYS: merge bones before feeding — maximises currency value.
        # These fire between attacks to keep bones flowing toward lvl-4.
        # ----------------------------------------------------------------
        if has(ActionType.MERGE_BONES):
            return ActionType.MERGE_BONES

        if has(ActionType.FEED_BONES) and _has_max_level_bone(state):
            return ActionType.FEED_BONES

        # ----------------------------------------------------------------
        # ALWAYS: use beacon whenever T-Rex is at full HP.
        # Handles opening steps 1 & 3 automatically.
        # ----------------------------------------------------------------
        if has(ActionType.USE_BEACON) and state.trex_hp == state.trex_max_hp:
            return ActionType.USE_BEACON

        # ----------------------------------------------------------------
        # ALWAYS: feed meteors for free soup
        # ----------------------------------------------------------------
        if has(ActionType.FEED_METEOR):
            return ActionType.FEED_METEOR

        # ----------------------------------------------------------------
        # PHASE 0 — OPENING
        # Complete once both Triceratops have attacked and their horns merged.
        # ----------------------------------------------------------------
        in_opening = trices > 0 or _has_mergeable_horn_pair(state)

        if in_opening:
            # Steps 2 & 4: attack Triceratops (beacon fires above if T-Rex is full)
            if trices > 0 and has(ActionType.ATTACK_TRICERATOPS):
                return ActionType.ATTACK_TRICERATOPS

            # Step 5: merge the two lvl-1 horns into a lvl-2
            if has(ActionType.MERGE_HORNS):
                return ActionType.MERGE_HORNS

        # ----------------------------------------------------------------
        # PHASE 2 — ATTACK LOOP
        #
        # Checked before the build phase so mid-attack the heuristic never
        # starts building new stegos instead of continuing the wave.
        #
        # _batch_remaining tracks how many attacks are still owed.  It is
        # set when the trigger condition is first met and decremented each
        # time an attack action is returned.  The always-rules above handle
        # bone merges between attacks automatically.
        # ----------------------------------------------------------------

        # Trigger a new wave when the board is full and soup is sufficient
        if (self._batch_remaining == 0
                and grid_full
                and n_stegos >= ATTACK_BATCH
                and state.primordial_soup >= _ATTACK_THRESHOLD):
            self._batch_remaining = ATTACK_BATCH

        # Execute the next attack in the current wave
        if self._batch_remaining > 0:
            if has(ActionType.ATTACK_STEGOSAURUS):
                self._batch_remaining -= 1
                return ActionType.ATTACK_STEGOSAURUS

        # ----------------------------------------------------------------
        # PHASE 1 / REFILL
        # Runs both during initial board fill and after each attack wave.
        # Only active when no attack wave is in progress (_batch_remaining
        # == 0) and bones have been fully processed (no pending merges).
        # ----------------------------------------------------------------
        if self._batch_remaining == 0 and not _any_bones(state):
            action = self._build_one_stego(
                state, has, lvl4_plants, stego_eggs, stego_babies, grid_full, free
            )
            if action is not None:
                return action

        return ActionType.WAIT

    # ------------------------------------------------------------------
    # Build helper
    # ------------------------------------------------------------------

    def _build_one_stego(
        self,
        state: GameState,
        has,
        lvl4_plants: int,
        stego_eggs: int,
        stego_babies: int,
        grid_full: bool,
        free: int,
    ) -> Optional[ActionType]:
        """
        Return the next action to advance one Stego through the pipeline,
        or None if the caller should WAIT.

        Grow/merge actions are always taken (they free grid space).
        New stego starts are blocked when the grid is full.
        In-progress pipelines are allowed to spawn (plant or egg) as long as
        at least 1 grid space is free — this prevents a deadlock where the
        4-plant peak of the merge tree exactly fills the FREE_SPACES_BUFFER.
        A new stego is only started when soup ≥ _SOUP_PER_STEGO;
        an in-progress stego continues regardless.
        """
        in_progress = (
            stego_eggs > 0
            or stego_babies > 0
            or sum(state.plants.values()) > 0
        )
        can_build = in_progress or state.primordial_soup >= _SOUP_PER_STEGO

        if not can_build:
            return None

        # Grow/merge always — these free space and never block on grid
        if has(ActionType.GROW_STEGOSAURUS):
            return ActionType.GROW_STEGOSAURUS

        if has(ActionType.MERGE_STEGOSAURUS_EGG):
            return ActionType.MERGE_STEGOSAURUS_EGG

        if has(ActionType.MERGE_PLANT):
            return ActionType.MERGE_PLANT

        has_unclaimed_plant = lvl4_plants > stego_babies

        # An in-progress pipeline may temporarily need to spawn when the board
        # is at the buffer limit (e.g. 4 plants occupying all 4 buffer spaces).
        # Allow the spawn as long as at least 1 space exists — merges will
        # immediately free it back up.
        if in_progress and free > 0:
            if has_unclaimed_plant and has(ActionType.SPAWN_HERBIVORE_EGG):
                return ActionType.SPAWN_HERBIVORE_EGG
            if not has_unclaimed_plant and has(ActionType.SPAWN_PLANT):
                return ActionType.SPAWN_PLANT

        # For starting a brand-new stego, require the full buffer headroom
        if grid_full:
            return None

        if has_unclaimed_plant and has(ActionType.SPAWN_HERBIVORE_EGG):
            return ActionType.SPAWN_HERBIVORE_EGG

        if has(ActionType.SPAWN_PLANT):
            return ActionType.SPAWN_PLANT

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_max_level_bone(state: GameState) -> bool:
    return state.bone_items.get(MAX_CURRENCY_LEVEL, 0) >= 1


def _any_bones(state: GameState) -> bool:
    return any(state.bone_items.get(lvl, 0) > 0 for lvl in range(1, MAX_CURRENCY_LEVEL + 1))


def _has_mergeable_horn_pair(state: GameState) -> bool:
    """True if the opening's dropped horns still need merging."""
    return any(state.horn_items.get(lvl, 0) >= 2 for lvl in range(1, MAX_CURRENCY_LEVEL))
