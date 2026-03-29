"""
Greedy heuristic for the Dinosnores simulator.

Deterministic three-phase strategy:

Phase 0 – OPENING (runs once):
  1. USE_BEACON          → halves T-Rex HP: 100 → 50
  2. ATTACK_TRICERATOPS  → 50 dmg kills T-Rex (1st wake-up); drops 1 lvl-1 horn
  3. USE_BEACON          → halves new T-Rex HP: 125 → 62  (handled by always-beacon rule)
  4. ATTACK_TRICERATOPS  → 50 dmg, drops 1 lvl-1 horn (T-Rex survives at 12 HP)
  5. MERGE_HORNS         → 2× lvl-1 → 1× lvl-2 horn  (handled by always-currency rule)

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

Event Shop integration:
  - Ad rewards (free items) are claimed as soon as they can be used efficiently:
      Day 0 Mammoth: claim only when T-Rex HP ≤ effective Mammoth damage so it
        can be attacked immediately — no wasted grid space holding an idle beast.
      Day 1 Saber-Tooth: same threshold with STT damage.
      Day 2 Lvl-4 Horn item: always claim (no beast timing needed).
  - Paid shop slots are bought opportunistically when affordable, except the
    Day 2 Alarm Clock (SHOP_SLOT_2) which is only bought when T-Rex HP is
    at full health so it can be used immediately on the following turn.
  - Beast attacks (Mammoth, STT) and USE_ALARM_CLOCK are only fired when
    T-Rex HP ≤ effective damage (beasts) or HP == max HP (alarm clock) to
    minimise wasted damage, mirroring the beacon's full-HP rule.

Soup cost per adult Stego from scratch (at HN/VP lvl-1):
  - 2 eggs : 2 × 2,000 = 4,000 soup
  - 8 lvl-1 plants → 1 lvl-4 plant : 8 × 500 = 4,000 soup
  - Total : 8,000 soup
"""

from typing import Dict, List, Optional

from .actions import ActionType
from .state import GameState
from .constants import (
    HerbivoreType, BeastType,
    BEAST_STATS, BRUTISH_BEASTS_BONUS,
    MAX_CURRENCY_LEVEL,
    SHOP_DAY_TURNS,
    GAME_DURATION_SECONDS, SECONDS_PER_TURN,
)


ATTACK_BATCH       = 8    # stegos to attack per wave
FREE_SPACES_BUFFER = 4    # grid spaces kept free for merging headroom

_MAX_TURNS     = GAME_DURATION_SECONDS // SECONDS_PER_TURN  # 25920
END_GAME_TURNS = 3_000   # last ~8.3 hours = sprint window

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

        mammoth_dmg  = _effective_beast_dmg(state, BeastType.MAMMOTH)
        saber_dmg    = _effective_beast_dmg(state, BeastType.SABER_TOOTH)
        shop_day     = min(state.turn // SHOP_DAY_TURNS, 2)

        # ----------------------------------------------------------------
        # ALWAYS: process currency items — merge up, then feed at max level
        # to avoid grid clutter and maximise currency value per item.
        # ----------------------------------------------------------------
        if has(ActionType.MERGE_BONES):
            return ActionType.MERGE_BONES
        if has(ActionType.FEED_BONES) and _has_max_level_item(state.bone_items):
            return ActionType.FEED_BONES

        if has(ActionType.MERGE_HORNS):
            return ActionType.MERGE_HORNS
        if has(ActionType.FEED_HORNS) and _has_item_at_or_above(state.horn_items, 3):
            return ActionType.FEED_HORNS

        if has(ActionType.MERGE_FANGS):
            return ActionType.MERGE_FANGS
        if has(ActionType.FEED_FANGS) and _has_item_at_or_above(state.fang_items, 3):
            return ActionType.FEED_FANGS

        # ----------------------------------------------------------------
        # SPRINT PHASE — last END_GAME_TURNS turns: dump everything.
        # Feed all items (level restriction lifted), attack all creatures
        # with no HP threshold, summon beasts while affordable.
        # Alarm clock fires LAST — after all creature/beast attacks are
        # exhausted — to avoid pushing T-Rex HP beyond dino attack range
        # before we've spent our creatures.
        # ----------------------------------------------------------------
        if _in_sprint(state):
            if has(ActionType.FEED_BONES): return ActionType.FEED_BONES
            if has(ActionType.FEED_HORNS): return ActionType.FEED_HORNS
            if has(ActionType.FEED_FANGS): return ActionType.FEED_FANGS

            if has(ActionType.FEED_METEOR): return ActionType.FEED_METEOR

            if has(ActionType.USE_BEACON) and state.trex_hp == state.trex_max_hp:
                return ActionType.USE_BEACON

            for attack in (
                ActionType.ATTACK_SABER_TOOTH,
                ActionType.ATTACK_MAMMOTH,
                ActionType.ATTACK_RAPTOR,
                ActionType.ATTACK_PTERODACTYL,
                ActionType.ATTACK_BRONTOSAURUS,
                ActionType.ATTACK_TRICERATOPS,
                ActionType.ATTACK_STEGOSAURUS,
            ):
                if has(attack): return attack

            if has(ActionType.SUMMON_SABER_TOOTH): return ActionType.SUMMON_SABER_TOOTH
            if has(ActionType.SUMMON_MAMMOTH):     return ActionType.SUMMON_MAMMOTH

            if has(ActionType.USE_ALARM_CLOCK): return ActionType.USE_ALARM_CLOCK

            return ActionType.WAIT

        # ----------------------------------------------------------------
        # ALWAYS: use beacon when T-Rex is at full HP.
        # Handles opening steps 1 & 3 automatically, and primes the T-Rex
        # for efficient beast / alarm-clock kills.
        # ----------------------------------------------------------------
        if has(ActionType.USE_BEACON) and state.trex_hp == state.trex_max_hp:
            return ActionType.USE_BEACON

        # ----------------------------------------------------------------
        # ALWAYS: feed meteors for free soup.
        # ----------------------------------------------------------------
        if has(ActionType.FEED_METEOR):
            return ActionType.FEED_METEOR

        # ----------------------------------------------------------------
        # ALWAYS: attack beasts only when they can finish the T-Rex.
        # Minimises wasted damage — the beacon rule above naturally halves
        # HP to within kill range before these fire.
        # ----------------------------------------------------------------
        if has(ActionType.ATTACK_SABER_TOOTH) and state.trex_hp <= saber_dmg:
            return ActionType.ATTACK_SABER_TOOTH
        if has(ActionType.ATTACK_MAMMOTH) and state.trex_hp <= mammoth_dmg:
            return ActionType.ATTACK_MAMMOTH

        # ----------------------------------------------------------------
        # EVENT SHOP — free ad claim (timing varies by ad type)
        #
        # Day 0 Mammoth / Day 1 STT: claim only when T-Rex HP is already
        #   within the beast's kill range, so the beast attacks immediately
        #   next turn.  This avoids holding a grid-occupying beast that
        #   can't be used yet, and ensures the currency drop (horn/fang)
        #   is produced at a time when it can be processed promptly.
        # Day 2 Lvl-4 Horn item: always claim — no beast timing needed.
        # ----------------------------------------------------------------
        if has(ActionType.SHOP_CLAIM_AD):
            if shop_day == 2:
                return ActionType.SHOP_CLAIM_AD
            elif shop_day == 0 and state.trex_hp <= mammoth_dmg:
                return ActionType.SHOP_CLAIM_AD
            elif shop_day == 1 and state.trex_hp <= saber_dmg:
                return ActionType.SHOP_CLAIM_AD

        # ----------------------------------------------------------------
        # EVENT SHOP — paid items (bought opportunistically when affordable)
        #
        # Day 2 slot 2 is the Alarm Clock: only buy during sprint so it
        # doesn't occupy a grid space for thousands of idle turns.
        # Days 0/1 slot 2 (HN lvl-2, CN) and slots 0/1 are bought
        # immediately.  Priority: Slot 2 > Slot 1 > Slot 0.
        # ----------------------------------------------------------------
        if has(ActionType.SHOP_SLOT_2) and (shop_day != 2 or _in_sprint(state)):
            return ActionType.SHOP_SLOT_2
        if has(ActionType.SHOP_SLOT_1):
            return ActionType.SHOP_SLOT_1
        if has(ActionType.SHOP_SLOT_0):
            return ActionType.SHOP_SLOT_0

        # ----------------------------------------------------------------
        # PHASE 0 — OPENING
        # Complete once both Triceratops have attacked and their horns
        # have been merged (the always-currency rule above handles the
        # actual merge; in_opening keeps us in this phase until done).
        # ----------------------------------------------------------------
        in_opening = trices > 0 or _has_mergeable_horn_pair(state)

        if in_opening:
            # Steps 2 & 4: attack Triceratops (beacon fires above when full)
            if trices > 0 and has(ActionType.ATTACK_TRICERATOPS):
                return ActionType.ATTACK_TRICERATOPS

        # ----------------------------------------------------------------
        # PHASE 2 — ATTACK LOOP
        #
        # Checked before the build phase so mid-attack the heuristic never
        # starts building new stegos instead of continuing the wave.
        #
        # _batch_remaining tracks how many attacks are still owed.  It is
        # set when the trigger condition is first met and decremented each
        # time an attack action is returned.  The always-rules above handle
        # currency merges between attacks automatically.
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

def _effective_beast_dmg(state: GameState, beast_type: BeastType) -> int:
    """Damage a beast deals accounting for the Brutish Beasts upgrade level."""
    base = BEAST_STATS[beast_type].base_damage
    mult = 1.0 + BRUTISH_BEASTS_BONUS[state.brutish_beasts_level]
    return int(base * mult)


def _has_max_level_item(item_dict: Dict[int, int]) -> bool:
    """True if at least one currency item at MAX_CURRENCY_LEVEL exists."""
    return item_dict.get(MAX_CURRENCY_LEVEL, 0) >= 1


def _has_item_at_or_above(item_dict: Dict[int, int], min_level: int) -> bool:
    """True if at least one currency item at or above min_level exists."""
    return any(item_dict.get(lvl, 0) >= 1 for lvl in range(min_level, MAX_CURRENCY_LEVEL + 1))


def _any_bones(state: GameState) -> bool:
    """True if any bone items remain on the grid (gates stego build/refill phase)."""
    return any(state.bone_items.get(lvl, 0) > 0 for lvl in range(1, MAX_CURRENCY_LEVEL + 1))


def _in_sprint(state: GameState) -> bool:
    """True when we are in the final END_GAME_TURNS turns of the event."""
    return (_MAX_TURNS - state.turn) <= END_GAME_TURNS


def _has_mergeable_horn_pair(state: GameState) -> bool:
    """True if the opening's dropped horns still need merging."""
    return any(state.horn_items.get(lvl, 0) >= 2 for lvl in range(1, MAX_CURRENCY_LEVEL))
