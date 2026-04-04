"""
Advanced greedy heuristic for the Dinosnores simulator — v2.

Implements the 3-day strategy from heuristic_plan.txt in phases:

Phase 1 – OPENING + STEGO FILL  (implemented)
  Buy Soup Stores lvl1+2 with starting 20 bones + 10 horns.
  Fill board with stegos, run 8-stego attack waves.
  Buy VP2 from Day 0 shop slot 1 (25 bones) opportunistically.
  Stego fill continues until bones >= 120 AND soup >= 300k (HN4 push trigger).

Phase 2 – HN4 PUSH  (TODO)
Phase 3 – MIGRATION  (TODO)
Phase 4 – PTERO WAVES  (TODO)
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


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

ATTACK_BATCH        = 8      # stegos to attack per wave
FREE_SPACES_BUFFER  = 4      # grid spaces kept free for merging headroom

_MAX_TURNS          = GAME_DURATION_SECONDS // SECONDS_PER_TURN
END_GAME_TURNS      = 3_000

_SOUP_PER_STEGO     = 8_000
_STEGO_WAVE_SOUP    = ATTACK_BATCH * _SOUP_PER_STEGO  # 64k

# Trigger for HN4 push — accumulated in stego fill phase
_HN4_PUSH_SOUP      = 300_000
_HN4_PUSH_BONES     = 120   # first tranche: 3x HN1 purchases


class GreedyHeuristicV2:
    """
    Multi-phase heuristic implementing the 3-day Dinosnores strategy.
    """

    def __init__(self):
        self._batch_remaining: int = 0  # stego attacks still owed in current wave

    def choose_action(
        self, state: GameState, valid_actions: List[ActionType]
    ) -> ActionType:
        va = set(valid_actions)

        def has(a: ActionType) -> bool:
            return a in va

        free        = state.grid_available()
        grid_full   = free <= FREE_SPACES_BUFFER
        shop_day    = min(state.turn // SHOP_DAY_TURNS, 2)
        mammoth_dmg = _effective_beast_dmg(state, BeastType.MAMMOTH)
        saber_dmg   = _effective_beast_dmg(state, BeastType.SABER_TOOTH)

        # ----------------------------------------------------------------
        # ALWAYS: merge currency items up, then feed at max level.
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
        # SPRINT — last END_GAME_TURNS turns: dump everything.
        # ----------------------------------------------------------------
        if _in_sprint(state):
            return self._sprint(state, has)

        # ----------------------------------------------------------------
        # ALWAYS: beacon on full HP.
        # ----------------------------------------------------------------
        if has(ActionType.USE_BEACON) and state.trex_hp == state.trex_max_hp:
            return ActionType.USE_BEACON

        # ----------------------------------------------------------------
        # ALWAYS: feed meteors for free soup.
        # ----------------------------------------------------------------
        if has(ActionType.FEED_METEOR):
            return ActionType.FEED_METEOR

        # ----------------------------------------------------------------
        # ALWAYS: beast attacks only when within kill range.
        # ----------------------------------------------------------------
        if has(ActionType.ATTACK_SABER_TOOTH) and state.trex_hp <= saber_dmg:
            return ActionType.ATTACK_SABER_TOOTH
        if has(ActionType.ATTACK_MAMMOTH) and state.trex_hp <= mammoth_dmg:
            return ActionType.ATTACK_MAMMOTH

        # ----------------------------------------------------------------
        # SOUP STORES lvl1+2 — buy with starting 20 bones + 10 horns
        # before anything else spends them.
        # ----------------------------------------------------------------
        if state.soup_stores_level < 2 and has(ActionType.BUY_SOUP_STORES):
            return ActionType.BUY_SOUP_STORES

        # ----------------------------------------------------------------
        # SHOP ADS — timing mirrors v1.
        # ----------------------------------------------------------------
        if has(ActionType.SHOP_CLAIM_AD):
            if shop_day == 2:
                return ActionType.SHOP_CLAIM_AD
            elif shop_day == 0 and state.trex_hp <= mammoth_dmg:
                return ActionType.SHOP_CLAIM_AD
            elif shop_day == 1 and state.trex_hp <= saber_dmg:
                return ActionType.SHOP_CLAIM_AD

        # ----------------------------------------------------------------
        # SHOP PAID — slot 0 + 1 opportunistic; slot 2 gated per phase.
        # Slot 2 on Day 1 = CN1 (70 bones) — bought in migration phase.
        # Slot 2 on Day 2 = alarm clock    — bought during sprint only.
        # ----------------------------------------------------------------
        if has(ActionType.SHOP_SLOT_0):
            return ActionType.SHOP_SLOT_0
        if has(ActionType.SHOP_SLOT_1):
            return ActionType.SHOP_SLOT_1
        if has(ActionType.SHOP_SLOT_2) and shop_day == 0:
            # Day 0 slot 2 = HN2 (30 horns); buy once HN4 push is underway
            # so we don't waste a grid space on an early idle HN2.
            if state.primordial_soup >= _HN4_PUSH_SOUP:
                return ActionType.SHOP_SLOT_2

        # ----------------------------------------------------------------
        # OPENING — buy SS1+SS2 then attack the two starting trices.
        # ----------------------------------------------------------------
        in_opening = (
            state.adult_herbivores[HerbivoreType.TRICERATOPS] > 0
            or _has_mergeable_horn_pair(state)
        )

        if in_opening:
            action = self._do_opening(state, has)
            if action is not None:
                return action

        # ----------------------------------------------------------------
        # HN4 PUSH TRIGGER — once bones + soup thresholds are met,
        # hand off to the (not yet implemented) push phase.
        # Until Phase 2 is coded the stego fill loop continues below.
        # ----------------------------------------------------------------
        _hn4_push_ready = (
            state.big_bones >= _HN4_PUSH_BONES
            and state.primordial_soup >= _HN4_PUSH_SOUP
        )
        _hn4_built = state.herbivore_nests.get(4, 0) >= 1

        # TODO (Phase 2): if _hn4_push_ready and not _hn4_built: ...

        # ----------------------------------------------------------------
        # STEGO FILL / ATTACK LOOP — same wave logic as v1.
        # Runs during opening fill and continues until HN4 is built.
        # ----------------------------------------------------------------
        if not _hn4_built:
            action = self._do_stego_wave(state, has, free, grid_full)
            if action is not None:
                return action

        return ActionType.WAIT

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _do_opening(self, state: GameState, has) -> Optional[ActionType]:
        """
        Attack the two starting trices (SS1+SS2 handled above beacon section).
        Beacon fires in the always-rules when T-Rex is at full HP.
        """
        # Attack the two starting trices.
        if (state.adult_herbivores[HerbivoreType.TRICERATOPS] > 0
                and has(ActionType.ATTACK_TRICERATOPS)):
            return ActionType.ATTACK_TRICERATOPS

        return None

    def _do_stego_wave(
        self,
        state: GameState,
        has,
        free: int,
        grid_full: bool,
    ) -> Optional[ActionType]:
        """
        8-stego attack waves identical to v1, with one addition:
        VP2 is obtained from Day 0 shop slot 1 (25 bones) and merged
        automatically with the existing VP1 via MERGE_VOLCANIC_PATCH.
        """
        n_stegos     = state.adult_herbivores[HerbivoreType.STEGOSAURUS]
        stego_eggs   = state.herbivore_eggs[HerbivoreType.STEGOSAURUS]
        stego_babies = state.baby_herbivores[HerbivoreType.STEGOSAURUS]
        lvl4_plants  = state.plants.get(4, 0)

        # Merge VP stations as soon as a pair exists (VP1+VP1 → VP2, etc.).
        if has(ActionType.MERGE_VOLCANIC_PATCH):
            return ActionType.MERGE_VOLCANIC_PATCH

        # Trigger a new attack wave when conditions are met.
        if (self._batch_remaining == 0
                and grid_full
                and n_stegos >= ATTACK_BATCH
                and state.primordial_soup >= _STEGO_WAVE_SOUP):
            self._batch_remaining = ATTACK_BATCH

        # Execute the next attack in the current wave.
        if self._batch_remaining > 0:
            if has(ActionType.ATTACK_STEGOSAURUS):
                self._batch_remaining -= 1
                return ActionType.ATTACK_STEGOSAURUS

        # Refill / initial fill: build one stego at a time.
        if self._batch_remaining == 0 and not _any_bones(state):
            return self._build_one_stego(
                state, has, lvl4_plants, stego_eggs, stego_babies, grid_full, free
            )

        return None

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
        Identical to v1: return the next action to advance one stego through
        the plant+egg pipeline, or None if we should WAIT.
        """
        in_progress = (
            stego_eggs > 0
            or stego_babies > 0
            or sum(state.plants.values()) > 0
        )
        can_build = in_progress or state.primordial_soup >= _SOUP_PER_STEGO

        if not can_build:
            return None

        if has(ActionType.GROW_STEGOSAURUS):
            return ActionType.GROW_STEGOSAURUS
        if has(ActionType.MERGE_STEGOSAURUS_EGG):
            return ActionType.MERGE_STEGOSAURUS_EGG
        if has(ActionType.MERGE_PLANT):
            return ActionType.MERGE_PLANT

        has_unclaimed_plant = lvl4_plants > stego_babies

        if in_progress and free > 0:
            if has_unclaimed_plant and has(ActionType.SPAWN_HERBIVORE_EGG):
                return ActionType.SPAWN_HERBIVORE_EGG
            if not has_unclaimed_plant and has(ActionType.SPAWN_PLANT):
                return ActionType.SPAWN_PLANT

        if grid_full:
            return None

        if has_unclaimed_plant and has(ActionType.SPAWN_HERBIVORE_EGG):
            return ActionType.SPAWN_HERBIVORE_EGG
        if has(ActionType.SPAWN_PLANT):
            return ActionType.SPAWN_PLANT

        return None

    def _sprint(self, state: GameState, has) -> ActionType:
        """End-game sprint: feed everything, attack everything, buy clocks."""
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

        if has(ActionType.BUY_ALARM_CLOCK):    return ActionType.BUY_ALARM_CLOCK
        if has(ActionType.SUMMON_SABER_TOOTH): return ActionType.SUMMON_SABER_TOOTH
        if has(ActionType.SUMMON_MAMMOTH):     return ActionType.SUMMON_MAMMOTH
        if has(ActionType.USE_ALARM_CLOCK):    return ActionType.USE_ALARM_CLOCK

        return ActionType.WAIT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_beast_dmg(state: GameState, beast_type: BeastType) -> int:
    base = BEAST_STATS[beast_type].base_damage
    mult = 1.0 + BRUTISH_BEASTS_BONUS[state.brutish_beasts_level]
    return int(base * mult)


def _has_max_level_item(item_dict: Dict[int, int]) -> bool:
    return item_dict.get(MAX_CURRENCY_LEVEL, 0) >= 1


def _has_item_at_or_above(item_dict: Dict[int, int], min_level: int) -> bool:
    return any(item_dict.get(lvl, 0) >= 1 for lvl in range(min_level, MAX_CURRENCY_LEVEL + 1))


def _any_bones(state: GameState) -> bool:
    return any(state.bone_items.get(lvl, 0) > 0 for lvl in range(1, MAX_CURRENCY_LEVEL + 1))


def _in_sprint(state: GameState) -> bool:
    return (_MAX_TURNS - state.turn) <= END_GAME_TURNS


def _has_mergeable_horn_pair(state: GameState) -> bool:
    return any(state.horn_items.get(lvl, 0) >= 2 for lvl in range(1, MAX_CURRENCY_LEVEL))
