"""
DinosnoresSimulator — core game simulator for the Dinosnores event.

Each call to step(action) advances the simulation by one turn:
  1. The chosen action is executed (if valid).
  2. Passive generation runs (Primordial Crater soup + beacon recharge).

Key design points
-----------------
- Plants and eggs are NOT generated passively.  The player must take a
  SPAWN_PLANT / SPAWN_*_EGG action, which costs Primordial Soup and places
  the entity on the grid.
- The grid has 32 spaces.  The Alien Beacon occupies 1 permanent space;
  every station instance, plant, egg, adult creature, and beast each occupies
  1 additional space.
- Stations are bought at level 1 (spending Big Bones/Horns/Fangs) and merged
  two-at-a-time to produce a higher-level instance.
- Spawn actions use the cheapest (lowest-level) available station of the
  required type; the soup cost scales with that station's level.
- Each Primordial Crater instance generates soup passively every turn based
  on its level.

Interface
---------
sim = DinosnoresSimulator(seed=42)
state = sim.reset()
valid = sim.get_valid_actions(state)
state, reward, done, info = sim.step(state, action)
"""

import math
import random as _random
from typing import Any

from .actions import ActionType
from .constants import (
    ALARM_CLOCK_BUY_COST,
    # Unlocks
    ALARM_CLOCK_UNLOCK_WAKE_UPS,
    BASE_SOUP_CAPACITY,
    # Beacon
    BEACON_MAX_CHARGES,
    BEACON_RECHARGE_TURNS,
    BEACON_SOUP_FRACTION,
    BEAST_STATS,
    BRUTISH_BEASTS_BONUS,
    BYE_BYE_PLANET_REDUCTION,
    CARNIVORE_NEST_EGG_PROBS,
    CARNIVORE_NEST_SPAWN_COST,
    CARNIVORE_NEST_UNLOCK_WAKE_UPS,
    CARNIVORE_STATS,
    CREATURE_CURRENCY_DROP,
    # Currency items
    CURRENCY_ITEM_VALUE,
    GAME_DURATION_SECONDS,
    GREATER_CRATERS_BONUS,
    HERBIVORE_NEST_EGG_PROBS,
    HERBIVORE_NEST_SPAWN_COST,
    HERBIVORE_NEST_UNLOCK_WAKE_UPS,
    # Stat tables
    HERBIVORE_STATS,
    MAX_BRUTISH_BEASTS_LEVEL,
    MAX_BYE_BYE_PLANET_LEVEL,
    MAX_CN_LEVEL,
    MAX_CURRENCY_LEVEL,
    MAX_GREATER_CRATERS_LEVEL,
    MAX_HN_LEVEL,
    # Upgrade max levels
    MAX_MORE_SCORE_LEVEL,
    MAX_PC_LEVEL,
    # Plant
    MAX_PLANT_LEVEL,
    MAX_SHARPER_FANGS_LEVEL,
    MAX_SOUP_STORES_LEVEL,
    # Station constants
    MAX_VP_LEVEL,
    # Upgrade effects
    MORE_SCORE_BONUS,
    PRIMORDIAL_CRATER_SOUP_PER_TURN,
    PRIMORDIAL_CRATER_UNLOCK_WAKE_UPS,
    SCORE_BASE_INCREMENT,
    SCORE_BASE_INITIAL,
    # Time model
    SECONDS_PER_TURN,
    SHARPER_FANGS_BONUS,
    SHOP_CATALOG,
    # Event Shop
    SHOP_DAY_TURNS,
    SHOP_ITEMS_PER_DAY,
    SHOP_NUM_DAYS,
    SOUP_STORES_BONUS,
    STATION_BUY_COST,
    TREX_HP_PER_WAKEUP,
    # T-Rex
    TREX_INITIAL_HP,
    # Upgrade costs
    UPGRADE_COSTS,
    VOLCANIC_PATCH_PLANT_PROBS,
    VOLCANIC_PATCH_SPAWN_COST,
    VOLCANIC_PATCH_UNLOCK_WAKE_UPS,
    BeastType,
    CarnivoreType,
    # Enums
    HerbivoreType,
)
from .state import GameState


class DinosnoresSimulator:
    """
    Discrete-turn simulator for the Dinosnores mini-game.

    One step = one player action + one round of passive generation.
    One turn represents 10 seconds of real game time (SECONDS_PER_TURN).

    Parameters
    ----------
    max_duration_seconds : float
        Episode ends after this many simulated seconds (default: 72 hours).
        Internally converted to turns: max_turns = max_duration_seconds / SECONDS_PER_TURN.
    score_target : int or None
        If set, episode ends when score >= score_target.
    """

    def __init__(self, max_duration_seconds: float = GAME_DURATION_SECONDS, score_target: int | None = None, seed: int | None = None):
        self.max_turns = int(max_duration_seconds / SECONDS_PER_TURN)
        self.score_target = score_target
        self.seed = seed
        self.rng = _random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> GameState:
        """Return a fresh initial GameState and reset the RNG."""
        self.rng = _random.Random(self.seed)
        state = GameState()
        # Starting conditions
        state.primordial_soup = 50_000
        state.big_bones = 20
        state.horns = 10
        state.adult_herbivores[HerbivoreType.STEGOSAURUS] = 1
        state.adult_herbivores[HerbivoreType.BRONTOSAURUS] = 1
        state.adult_herbivores[HerbivoreType.TRICERATOPS] = 2
        state.plants = {3: 2}
        state.primordial_craters = {2: 1}
        return state

    def get_valid_actions(self, state: GameState) -> list[ActionType]:
        """
        Return every action that can legally be executed in the given state.
        WAIT is valid only when there is something worth waiting for — beacon
        recharging or passive soup generating.  When neither holds, WAIT would
        skip to the end of the game with no benefit, so it is omitted.
        """
        wait_useful = (
            state.beacon_charges < BEACON_MAX_CHARGES or
            self._soup_rate(state) > 0
        )
        valid: list[ActionType] = [ActionType.WAIT] if wait_useful else []
        free = state.grid_available()

        # --- Spawn plant (Volcanic Patch required, 1 free grid space, soup cost) ---
        if (_has_station(state.volcanic_patches) and
                free >= 1 and
                state.primordial_soup >= self.spawn_cost_plant(state)):
            valid.append(ActionType.SPAWN_PLANT)

        # --- Spawn herbivore egg (Herbivore Nest required; egg type is probabilistic) ---
        if _has_station(state.herbivore_nests) and free >= 1:
            cost = self.spawn_cost_herbivore_egg(state)
            if state.primordial_soup >= cost:
                valid.append(ActionType.SPAWN_HERBIVORE_EGG)

        # --- Spawn carnivore egg (Carnivore Nest required; egg type is probabilistic) ---
        # No wake-up gate here: having a CN on the grid (however obtained) is sufficient.
        if (_has_station(state.carnivore_nests) and free >= 1):
            cost = self.spawn_cost_carnivore_egg(state)
            if state.primordial_soup >= cost:
                valid.append(ActionType.SPAWN_CARNIVORE_EGG)

        # --- Merge herbivore eggs (2× egg → 1 baby) ---
        for h_type in HerbivoreType:
            if state.herbivore_eggs[h_type] >= 2:
                valid.append(_MERGE_HERBIVORE_EGG_ACTION[h_type])

        # --- Merge carnivore eggs (2× egg → 1 baby) ---
        for c_type in CarnivoreType:
            if state.carnivore_eggs[c_type] >= 2:
                valid.append(_MERGE_CARNIVORE_EGG_ACTION[c_type])

        # --- Grow herbivores (baby + plant → adult) ---
        for h_type in HerbivoreType:
            stats = HERBIVORE_STATS[h_type]
            if (state.baby_herbivores[h_type] >= 1 and
                    state.plants.get(stats.plant_level_required, 0) >= 1):
                valid.append(_GROW_HERBIVORE_ACTION[h_type])

        # --- Grow carnivores (baby + adult herbivore → adult) ---
        for c_type in CarnivoreType:
            stats = CARNIVORE_STATS[c_type]
            if (state.baby_carnivores[c_type] >= 1 and
                    state.adult_herbivores[stats.herbivore_food] >= 1):
                valid.append(_GROW_CARNIVORE_ACTION[c_type])

        # --- Merge plants (2× lvl N → 1× lvl N+1; always lowest pair) ---
        if any(state.plants.get(lvl, 0) >= 2 for lvl in range(1, MAX_PLANT_LEVEL)):
            valid.append(ActionType.MERGE_PLANT)

        # --- Summon beasts (currency cost + wake-up unlock + 1 free grid space) ---
        for b_type in BeastType:
            stats = BEAST_STATS[b_type]
            if (state.wake_ups >= stats.unlock_wake_ups and
                    free >= 1 and
                    _can_afford(state, stats.summon_cost)):
                valid.append(_SUMMON_BEAST_ACTION[b_type])

        # --- Attack with herbivores ---
        for h_type in HerbivoreType:
            if state.adult_herbivores[h_type] >= 1:
                valid.append(_ATTACK_HERBIVORE_ACTION[h_type])

        # --- Attack with carnivores ---
        for c_type in CarnivoreType:
            if state.adult_carnivores[c_type] >= 1:
                valid.append(_ATTACK_CARNIVORE_ACTION[c_type])

        # --- Attack with beasts ---
        for b_type in BeastType:
            if state.beasts[b_type] >= 1:
                valid.append(_ATTACK_BEAST_ACTION[b_type])

        # --- Special abilities ---
        if state.beacon_charges >= 1:
            valid.append(ActionType.USE_BEACON)

        if state.meteors >= 1:
            valid.append(ActionType.FEED_METEOR)

        if state.alarm_clocks >= 1 or state.wake_ups >= ALARM_CLOCK_UNLOCK_WAKE_UPS:
            valid.append(ActionType.USE_ALARM_CLOCK)

        if (state.wake_ups >= ALARM_CLOCK_UNLOCK_WAKE_UPS and
                free >= 1 and
                _can_afford(state, ALARM_CLOCK_BUY_COST)):
            valid.append(ActionType.BUY_ALARM_CLOCK)

        # --- Buy stations (requires wake-up threshold, currency, 1 free grid space) ---
        if (state.wake_ups >= HERBIVORE_NEST_UNLOCK_WAKE_UPS and
                free >= 1 and
                _can_afford(state, STATION_BUY_COST["herbivore_nest"])):
            valid.append(ActionType.BUY_HERBIVORE_NEST)

        if (state.wake_ups >= VOLCANIC_PATCH_UNLOCK_WAKE_UPS and
                free >= 1 and
                _can_afford(state, STATION_BUY_COST["volcanic_patch"])):
            valid.append(ActionType.BUY_VOLCANIC_PATCH)

        if (state.wake_ups >= CARNIVORE_NEST_UNLOCK_WAKE_UPS and
                free >= 1 and
                _can_afford(state, STATION_BUY_COST["carnivore_nest"])):
            valid.append(ActionType.BUY_CARNIVORE_NEST)

        if (state.wake_ups >= PRIMORDIAL_CRATER_UNLOCK_WAKE_UPS and
                free >= 1 and
                _can_afford(state, STATION_BUY_COST["primordial_crater"])):
            valid.append(ActionType.BUY_PRIMORDIAL_CRATER)

        # --- Merge stations (one action per type; always merges lowest pair) ---
        if any(state.volcanic_patches.get(lvl, 0) >= 2 for lvl in range(1, MAX_VP_LEVEL)):
            valid.append(ActionType.MERGE_VOLCANIC_PATCH)
        if any(state.herbivore_nests.get(lvl, 0) >= 2 for lvl in range(1, MAX_HN_LEVEL)):
            valid.append(ActionType.MERGE_HERBIVORE_NEST)
        if any(state.carnivore_nests.get(lvl, 0) >= 2 for lvl in range(1, MAX_CN_LEVEL)):
            valid.append(ActionType.MERGE_CARNIVORE_NEST)
        if any(state.primordial_craters.get(lvl, 0) >= 2 for lvl in range(1, MAX_PC_LEVEL)):
            valid.append(ActionType.MERGE_PRIMORDIAL_CRATER)

        # --- Feed currency items (feeds highest available level) ---
        if any(state.bone_items.get(lvl, 0) >= 1 for lvl in range(1, MAX_CURRENCY_LEVEL + 1)):
            valid.append(ActionType.FEED_BONES)
        if any(state.horn_items.get(lvl, 0) >= 1 for lvl in range(1, MAX_CURRENCY_LEVEL + 1)):
            valid.append(ActionType.FEED_HORNS)
        if any(state.fang_items.get(lvl, 0) >= 1 for lvl in range(1, MAX_CURRENCY_LEVEL + 1)):
            valid.append(ActionType.FEED_FANGS)

        # --- Merge currency items (merges lowest pair; auto-feeds if result is lvl 4) ---
        if any(state.bone_items.get(lvl, 0) >= 2 for lvl in range(1, MAX_CURRENCY_LEVEL)):
            valid.append(ActionType.MERGE_BONES)
        if any(state.horn_items.get(lvl, 0) >= 2 for lvl in range(1, MAX_CURRENCY_LEVEL)):
            valid.append(ActionType.MERGE_HORNS)
        if any(state.fang_items.get(lvl, 0) >= 2 for lvl in range(1, MAX_CURRENCY_LEVEL)):
            valid.append(ActionType.MERGE_FANGS)

        # --- Purchase upgrades ---
        if (state.more_score_level < MAX_MORE_SCORE_LEVEL and
                _can_afford(state, UPGRADE_COSTS["more_score"][state.more_score_level])):
            valid.append(ActionType.BUY_MORE_SCORE)

        if (state.bye_bye_planet_level < MAX_BYE_BYE_PLANET_LEVEL and
                _can_afford(state, UPGRADE_COSTS["bye_bye_planet"][state.bye_bye_planet_level])):
            valid.append(ActionType.BUY_BYE_BYE_PLANET)

        if (state.sharper_fangs_level < MAX_SHARPER_FANGS_LEVEL and
                _can_afford(state, UPGRADE_COSTS["sharper_fangs"][state.sharper_fangs_level])):
            valid.append(ActionType.BUY_SHARPER_FANGS)

        if (state.brutish_beasts_level < MAX_BRUTISH_BEASTS_LEVEL and
                _can_afford(state, UPGRADE_COSTS["brutish_beasts"][state.brutish_beasts_level])):
            valid.append(ActionType.BUY_BRUTISH_BEASTS)

        if (state.greater_craters_level < MAX_GREATER_CRATERS_LEVEL and
                _can_afford(state, UPGRADE_COSTS["greater_craters"][state.greater_craters_level])):
            valid.append(ActionType.BUY_GREATER_CRATERS)

        if (state.soup_stores_level < MAX_SOUP_STORES_LEVEL and
                _can_afford(state, UPGRADE_COSTS["soup_stores"][state.soup_stores_level])):
            valid.append(ActionType.BUY_SOUP_STORES)

        # --- Event Shop ---
        current_day = min(state.turn // SHOP_DAY_TURNS, SHOP_NUM_DAYS - 1)
        shop_slot_actions = [
            ActionType.SHOP_SLOT_0,
            ActionType.SHOP_SLOT_1,
            ActionType.SHOP_SLOT_2,
            ActionType.SHOP_CLAIM_AD,
        ]
        for slot_idx, shop_action in enumerate(shop_slot_actions):
            catalog_idx = current_day * SHOP_ITEMS_PER_DAY + slot_idx
            if state.shop_items_claimed[catalog_idx]:
                continue
            if free < 1:
                continue
            _, _, cost, _ = SHOP_CATALOG[catalog_idx]
            if not _can_afford(state, cost):  # always True for (0,0,0) ad slots
                continue
            valid.append(shop_action)

        return valid

    def step(
        self, state: GameState, action: ActionType
    ) -> tuple[GameState, float, bool, dict[str, Any]]:
        """
        Apply *action* to *state* and return:
            (next_state, reward, done, info)

        next_state  — new GameState after the action and passive generation
        reward      — score delta earned this step (float)
        done        — True if the episode has ended
        info        — dict with extra details (e.g. 'woke_trex', 'damage_dealt')

        The input state is NOT modified; a copy is returned.
        """
        state = state.copy()
        info: dict[str, Any] = {}
        score_before = state.score

        valid = self.get_valid_actions(state)
        if action not in valid:
            raise ValueError(f"Action {action} is not valid in the current state.")

        if action == ActionType.WAIT:
            turns = self._compute_wait_skip(state)
            self._fast_forward(state, turns)
            info["wait_turns"] = turns
        else:
            self._execute_action(state, action, info)
            self._passive_generation(state)
            state.turn += 1

        reward = float(state.score - score_before)
        done = (state.turn >= self.max_turns or
                (self.score_target is not None and state.score >= self.score_target))

        return state, reward, done, info

    # ------------------------------------------------------------------
    # Spawn cost helpers  (use cheapest/lowest-level available station)
    # ------------------------------------------------------------------

    @staticmethod
    def _cheapest_level(station_dict: dict[int, int]) -> int:
        """Return the lowest level that has at least one instance."""
        return min(lvl for lvl, cnt in station_dict.items() if cnt > 0)

    def spawn_cost_plant(self, state: GameState) -> int:
        return VOLCANIC_PATCH_SPAWN_COST[self._cheapest_level(state.volcanic_patches)]

    def spawn_cost_herbivore_egg(self, state: GameState) -> int:
        return HERBIVORE_NEST_SPAWN_COST[self._cheapest_level(state.herbivore_nests)]

    def spawn_cost_carnivore_egg(self, state: GameState) -> int:
        return CARNIVORE_NEST_SPAWN_COST[self._cheapest_level(state.carnivore_nests)]

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, state: GameState, action: ActionType,
                        info: dict[str, Any]) -> None:

        if action == ActionType.WAIT:
            return

        # --- Spawn plant ---
        if action == ActionType.SPAWN_PLANT:
            level = self._cheapest_level(state.volcanic_patches)
            cost = VOLCANIC_PATCH_SPAWN_COST[level]
            state.primordial_soup -= cost
            plant_lvl = self._sample_plant(level)
            state.plants[plant_lvl] = state.plants.get(plant_lvl, 0) + 1
            info["spawned"] = f"plant_lvl{plant_lvl}"
            info["soup_cost"] = cost
            return

        # --- Spawn herbivore egg ---
        if action == ActionType.SPAWN_HERBIVORE_EGG:
            level = self._cheapest_level(state.herbivore_nests)
            cost = HERBIVORE_NEST_SPAWN_COST[level]
            state.primordial_soup -= cost
            h_type = self._sample_herbivore_egg(level)
            state.herbivore_eggs[h_type] += 1
            info["spawned"] = f"{h_type.value}_egg"
            info["soup_cost"] = cost
            return

        # --- Spawn carnivore egg ---
        if action == ActionType.SPAWN_CARNIVORE_EGG:
            level = self._cheapest_level(state.carnivore_nests)
            cost = CARNIVORE_NEST_SPAWN_COST[level]
            state.primordial_soup -= cost
            c_type = self._sample_carnivore_egg(level)
            state.carnivore_eggs[c_type] += 1
            info["spawned"] = f"{c_type.value}_egg"
            info["soup_cost"] = cost
            return

        # --- Merge herbivore eggs (2 eggs → 1 baby) ---
        if action in _MERGE_HERBIVORE_EGG_ACTION.values():
            h_type = _ACTION_TO_HERBIVORE_EGG_MERGE[action]
            state.herbivore_eggs[h_type] -= 2
            state.baby_herbivores[h_type] += 1
            info["merged_egg"] = h_type.value
            return

        # --- Merge carnivore eggs (2 eggs → 1 baby) ---
        if action in _MERGE_CARNIVORE_EGG_ACTION.values():
            c_type = _ACTION_TO_CARNIVORE_EGG_MERGE[action]
            state.carnivore_eggs[c_type] -= 2
            state.baby_carnivores[c_type] += 1
            info["merged_egg"] = c_type.value
            return

        # --- Grow herbivores (baby + plant → adult) ---
        if action in _GROW_HERBIVORE_ACTION.values():
            h_type = _ACTION_TO_HERBIVORE_GROW[action]
            stats = HERBIVORE_STATS[h_type]
            state.baby_herbivores[h_type] -= 1
            state.plants[stats.plant_level_required] -= 1
            state.adult_herbivores[h_type] += 1
            info["grew"] = h_type.value
            return

        # --- Grow carnivores (baby + adult herbivore → adult) ---
        if action in _GROW_CARNIVORE_ACTION.values():
            c_type = _ACTION_TO_CARNIVORE_GROW[action]
            stats = CARNIVORE_STATS[c_type]
            state.baby_carnivores[c_type] -= 1
            state.adult_herbivores[stats.herbivore_food] -= 1
            state.adult_carnivores[c_type] += 1
            info["grew"] = c_type.value
            return

        # --- Merge plants (lowest pair) ---
        if action == ActionType.MERGE_PLANT:
            lvl = _lowest_mergeable_level(state.plants, MAX_PLANT_LEVEL)
            state.plants[lvl] -= 2
            state.plants[lvl + 1] = state.plants.get(lvl + 1, 0) + 1
            info["merged_plant"] = lvl
            return

        # --- Summon beasts ---
        if action in _SUMMON_BEAST_ACTION.values():
            b_type = _ACTION_TO_BEAST_SUMMON[action]
            stats = BEAST_STATS[b_type]
            _spend_currency(state, stats.summon_cost)
            state.beasts[b_type] += 1
            info["summoned"] = b_type.value
            return

        # --- Attack with herbivore ---
        if action in _ATTACK_HERBIVORE_ACTION.values():
            h_type = _ACTION_TO_HERBIVORE_ATTACK[action]
            stats = HERBIVORE_STATS[h_type]
            damage = int(stats.base_damage * (1.0 + SHARPER_FANGS_BONUS[state.sharper_fangs_level]))
            state.adult_herbivores[h_type] -= 1
            _drop_currency(state, h_type)
            self._deal_damage(state, damage, info)
            info["attacker"] = h_type.value
            info["damage"] = damage
            return

        # --- Attack with carnivore ---
        if action in _ATTACK_CARNIVORE_ACTION.values():
            c_type = _ACTION_TO_CARNIVORE_ATTACK[action]
            stats = CARNIVORE_STATS[c_type]
            damage = int(stats.base_damage * (1.0 + SHARPER_FANGS_BONUS[state.sharper_fangs_level]))
            state.adult_carnivores[c_type] -= 1
            _drop_currency(state, c_type)
            self._deal_damage(state, damage, info)
            info["attacker"] = c_type.value
            info["damage"] = damage
            return

        # --- Attack with beast ---
        if action in _ATTACK_BEAST_ACTION.values():
            b_type = _ACTION_TO_BEAST_ATTACK[action]
            stats = BEAST_STATS[b_type]
            damage = int(stats.base_damage * (1.0 + BRUTISH_BEASTS_BONUS[state.brutish_beasts_level]))
            state.beasts[b_type] -= 1
            _drop_currency(state, b_type)
            self._deal_damage(state, damage, info)
            info["attacker"] = b_type.value
            info["damage"] = damage
            return

        # --- Alien Beacon ---
        if action == ActionType.USE_BEACON:
            state.beacon_charges -= 1
            if state.beacon_charges < BEACON_MAX_CHARGES and state.beacon_recharge_counter == 0:
                state.beacon_recharge_counter = 1
            state.trex_hp = max(1, state.trex_hp // 2)  # beacon alone cannot fully wake T-Rex
            state.meteors += 1
            info["beacon_used"] = True
            info["hp_after"] = state.trex_hp
            return

        # --- Feed Meteor ---
        if action == ActionType.FEED_METEOR:
            state.meteors -= 1
            self._add_soup(state, int(state.soup_capacity * BEACON_SOUP_FRACTION))
            info["fed_meteor"] = True
            return

        # --- Alarm Clock ---
        if action == ActionType.USE_ALARM_CLOCK:
            if state.alarm_clocks >= 1:
                state.alarm_clocks -= 1
                info["alarm_clock_consumed"] = True
            self._deal_damage(state, state.trex_hp, info)
            info["alarm_clock"] = True
            return

        # --- Buy alarm clock from build menu ---
        if action == ActionType.BUY_ALARM_CLOCK:
            _spend_currency(state, ALARM_CLOCK_BUY_COST)
            state.alarm_clocks += 1
            return

        # --- Feed currency items (highest available level) ---
        if action in (ActionType.FEED_BONES, ActionType.FEED_HORNS, ActionType.FEED_FANGS):
            currency_type = _FEED_ACTION_TO_TYPE[action]
            item_dict = _item_dict(state, currency_type)
            lvl = _highest_item_level(item_dict)
            item_dict[lvl] -= 1
            value = CURRENCY_ITEM_VALUE[lvl]
            _credit_currency(state, currency_type, value)
            info["fed"] = (currency_type, lvl, value)
            return

        # --- Merge currency items (lowest pair; auto-feed if result is lvl 4) ---
        if action in (ActionType.MERGE_BONES, ActionType.MERGE_HORNS, ActionType.MERGE_FANGS):
            currency_type = _MERGE_CURRENCY_ACTION_TO_TYPE[action]
            item_dict = _item_dict(state, currency_type)
            lvl = _lowest_mergeable_level(item_dict, MAX_CURRENCY_LEVEL)
            item_dict[lvl] -= 2
            new_lvl = lvl + 1
            item_dict[new_lvl] = item_dict.get(new_lvl, 0) + 1
            info["merged"] = (currency_type, lvl)
            if new_lvl == MAX_CURRENCY_LEVEL:
                item_dict[new_lvl] -= 1
                value = CURRENCY_ITEM_VALUE[new_lvl]
                _credit_currency(state, currency_type, value)
                info["fed"] = (currency_type, new_lvl, value)
            return

        # --- Buy stations ---
        if action == ActionType.BUY_HERBIVORE_NEST:
            _spend_currency(state, STATION_BUY_COST["herbivore_nest"])
            state.herbivore_nests[1] = state.herbivore_nests.get(1, 0) + 1
            return

        if action == ActionType.BUY_VOLCANIC_PATCH:
            _spend_currency(state, STATION_BUY_COST["volcanic_patch"])
            state.volcanic_patches[1] = state.volcanic_patches.get(1, 0) + 1
            return

        if action == ActionType.BUY_CARNIVORE_NEST:
            _spend_currency(state, STATION_BUY_COST["carnivore_nest"])
            state.carnivore_nests[1] = state.carnivore_nests.get(1, 0) + 1
            return

        if action == ActionType.BUY_PRIMORDIAL_CRATER:
            _spend_currency(state, STATION_BUY_COST["primordial_crater"])
            state.primordial_craters[1] = state.primordial_craters.get(1, 0) + 1
            return

        # --- Merge stations (lowest available pair) ---
        if action == ActionType.MERGE_VOLCANIC_PATCH:
            _merge_station(state.volcanic_patches, _lowest_mergeable_level(state.volcanic_patches, MAX_VP_LEVEL))
            return

        if action == ActionType.MERGE_HERBIVORE_NEST:
            _merge_station(state.herbivore_nests, _lowest_mergeable_level(state.herbivore_nests, MAX_HN_LEVEL))
            return

        if action == ActionType.MERGE_CARNIVORE_NEST:
            _merge_station(state.carnivore_nests, _lowest_mergeable_level(state.carnivore_nests, MAX_CN_LEVEL))
            return

        if action == ActionType.MERGE_PRIMORDIAL_CRATER:
            _merge_station(state.primordial_craters, _lowest_mergeable_level(state.primordial_craters, MAX_PC_LEVEL))
            return

        # --- Purchase upgrades ---
        if action == ActionType.BUY_MORE_SCORE:
            _spend_currency(state, UPGRADE_COSTS["more_score"][state.more_score_level])
            state.more_score_level += 1
            return

        if action == ActionType.BUY_BYE_BYE_PLANET:
            _spend_currency(state, UPGRADE_COSTS["bye_bye_planet"][state.bye_bye_planet_level])
            state.bye_bye_planet_level += 1
            return

        if action == ActionType.BUY_SHARPER_FANGS:
            _spend_currency(state, UPGRADE_COSTS["sharper_fangs"][state.sharper_fangs_level])
            state.sharper_fangs_level += 1
            return

        if action == ActionType.BUY_BRUTISH_BEASTS:
            _spend_currency(state, UPGRADE_COSTS["brutish_beasts"][state.brutish_beasts_level])
            state.brutish_beasts_level += 1
            return

        if action == ActionType.BUY_GREATER_CRATERS:
            _spend_currency(state, UPGRADE_COSTS["greater_craters"][state.greater_craters_level])
            state.greater_craters_level += 1
            return

        if action == ActionType.BUY_SOUP_STORES:
            _spend_currency(state, UPGRADE_COSTS["soup_stores"][state.soup_stores_level])
            state.soup_stores_level += 1
            state.soup_capacity = BASE_SOUP_CAPACITY + SOUP_STORES_BONUS[state.soup_stores_level]
            return

        # --- Event Shop ---
        shop_slot_actions = [
            ActionType.SHOP_SLOT_0,
            ActionType.SHOP_SLOT_1,
            ActionType.SHOP_SLOT_2,
            ActionType.SHOP_CLAIM_AD,
        ]
        if action in shop_slot_actions:
            slot_idx = shop_slot_actions.index(action)
            current_day = min(state.turn // SHOP_DAY_TURNS, SHOP_NUM_DAYS - 1)
            catalog_idx = current_day * SHOP_ITEMS_PER_DAY + slot_idx
            _, _, cost, label = SHOP_CATALOG[catalog_idx]
            _spend_currency(state, cost)
            state.shop_items_claimed[catalog_idx] = True
            self._apply_shop_effect(state, label)
            info["shop_claimed"] = label
            return

    def _apply_shop_effect(self, state: GameState, label: str) -> None:
        """Apply the item effect for the given SHOP_CATALOG label."""
        if label == "horn_item_lvl2":
            state.horn_items[2] = state.horn_items.get(2, 0) + 1
        elif label == "volcanic_patch_lvl1":
            state.volcanic_patches[1] = state.volcanic_patches.get(1, 0) + 1
        elif label == "herbivore_nest_lvl2":
            state.herbivore_nests[2] = state.herbivore_nests.get(2, 0) + 1
        elif label == "mammoth_ad":
            state.beasts[BeastType.MAMMOTH] += 1
        elif label == "volcanic_patch_lvl2":
            state.volcanic_patches[2] = state.volcanic_patches.get(2, 0) + 1
        elif label == "primordial_crater_lvl1":
            state.primordial_craters[1] = state.primordial_craters.get(1, 0) + 1
        elif label == "carnivore_nest_lvl1":
            state.carnivore_nests[1] = state.carnivore_nests.get(1, 0) + 1
        elif label == "saber_tooth_ad":
            state.beasts[BeastType.SABER_TOOTH] += 1
        elif label == "volcanic_patch_lvl3":
            state.volcanic_patches[3] = state.volcanic_patches.get(3, 0) + 1
        elif label == "primordial_crater_lvl3":
            state.primordial_craters[3] = state.primordial_craters.get(3, 0) + 1
        elif label == "alarm_clock":
            state.alarm_clocks += 1
        elif label == "horn_item_lvl4_ad":
            state.horn_items[4] = state.horn_items.get(4, 0) + 1

    # ------------------------------------------------------------------
    # Passive generation (Primordial Crater soup + beacon recharge)
    # ------------------------------------------------------------------

    def _passive_generation(self, state: GameState) -> None:
        # Each Primordial Crater instance generates soup based on its level
        gc_bonus = GREATER_CRATERS_BONUS[state.greater_craters_level]
        for level, count in state.primordial_craters.items():
            if count > 0:
                self._add_soup(state, (PRIMORDIAL_CRATER_SOUP_PER_TURN[level] + gc_bonus) * count)

        # Adult herbivores passively generate soup each turn
        for h_type in HerbivoreType:
            count = state.adult_herbivores[h_type]
            if count > 0:
                stats = HERBIVORE_STATS[h_type]
                self._add_soup(state, stats.soup_production * count)

        self._tick_beacon(state)

    def _tick_beacon(self, state: GameState) -> None:
        if state.beacon_charges >= BEACON_MAX_CHARGES:
            state.beacon_recharge_counter = 0
            return

        recharge_turns = max(
            1,
            BEACON_RECHARGE_TURNS - BYE_BYE_PLANET_REDUCTION[state.bye_bye_planet_level],
        )
        state.beacon_recharge_counter += 1
        if state.beacon_recharge_counter >= recharge_turns:
            state.beacon_charges = min(state.beacon_charges + 1, BEACON_MAX_CHARGES)
            state.beacon_recharge_counter = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deal_damage(self, state: GameState, damage: int,
                     info: dict[str, Any]) -> None:
        effective = min(damage, state.trex_hp)
        state.trex_hp -= effective
        info.setdefault("damage_dealt", 0)
        info["damage_dealt"] += effective

        if state.trex_hp <= 0:
            self._wake_trex(state, info)

    def _wake_trex(self, state: GameState, info: dict[str, Any]) -> None:
        base_score = SCORE_BASE_INITIAL + state.wake_ups * SCORE_BASE_INCREMENT
        multiplier = 1.0 + MORE_SCORE_BONUS[state.more_score_level]
        awarded = int(base_score * multiplier)
        state.score += awarded

        state.wake_ups += 1
        state.trex_max_hp = TREX_INITIAL_HP + TREX_HP_PER_WAKEUP * state.wake_ups
        state.trex_hp     = state.trex_max_hp

        info["woke_trex"]    = True
        info["score_earned"] = awarded

    def _sample_plant(self, vp_level: int) -> int:
        """Sample a plant level from the probability table for the given VP level."""
        probs = VOLCANIC_PATCH_PLANT_PROBS[vp_level]
        r = self.rng.random()
        cumulative = 0.0
        for plant_lvl, prob in probs:
            cumulative += prob
            if r < cumulative:
                return plant_lvl
        return probs[-1][0]

    def _sample_carnivore_egg(self, nest_level: int) -> CarnivoreType:
        """Sample an egg type from the probability table for the given nest level."""
        probs = CARNIVORE_NEST_EGG_PROBS[nest_level]
        r = self.rng.random()
        cumulative = 0.0
        for c_type, prob in probs:
            cumulative += prob
            if r < cumulative:
                return c_type
        return probs[-1][0]

    def _sample_herbivore_egg(self, nest_level: int) -> HerbivoreType:
        """Sample an egg type from the probability table for the given nest level."""
        probs = HERBIVORE_NEST_EGG_PROBS[nest_level]
        r = self.rng.random()
        cumulative = 0.0
        for h_type, prob in probs:
            cumulative += prob
            if r < cumulative:
                return h_type
        return probs[-1][0]  # floating-point safety fallback

    def _add_soup(self, state: GameState, amount: int) -> None:
        state.primordial_soup = min(state.primordial_soup + amount, state.soup_capacity)

    def _soup_rate(self, state: GameState) -> int:
        """Passive soup generation per turn from craters and adult herbivores."""
        gc_bonus = GREATER_CRATERS_BONUS[state.greater_craters_level]
        rate = sum(
            (PRIMORDIAL_CRATER_SOUP_PER_TURN[lvl] + gc_bonus) * cnt
            for lvl, cnt in state.primordial_craters.items()
            if cnt > 0
        )
        for h_type in HerbivoreType:
            rate += HERBIVORE_STATS[h_type].soup_production * state.adult_herbivores[h_type]
        return rate

    def _compute_wait_skip(self, state: GameState) -> int:
        """Return the number of turns to skip on a WAIT action.

        Advances to the earliest of:
          - turns until the next beacon charge is gained
          - turns until soup reaches the threshold for the cheapest spawn action
          - one beacon recharge cycle (BEACON_RECHARGE_TURNS) as a safety cap
          - end of episode
        The beacon-cycle cap ensures that strategies waiting for soup thresholds
        above the cheapest spawn cost (e.g. an attack-wave threshold) still get
        called regularly even when the beacon is fully charged.
        Falls back to 1 if no meaningful event can be calculated.
        """
        remaining = self.max_turns - state.turn
        if remaining <= 0:
            return 1

        # Cap at one beacon cycle so the caller always re-evaluates state
        # within 3 real-time hours, even when beacon is at max charges and
        # soup already exceeds all spawn-cost thresholds.
        skip = min(remaining, BEACON_RECHARGE_TURNS)

        # Turns until next beacon charge
        if state.beacon_charges < BEACON_MAX_CHARGES:
            recharge_turns = max(
                1,
                BEACON_RECHARGE_TURNS - BYE_BYE_PLANET_REDUCTION[state.bye_bye_planet_level],
            )
            turns_to_charge = recharge_turns - state.beacon_recharge_counter
            skip = min(skip, turns_to_charge)

        # Turns until soup is sufficient for the cheapest spawn action
        soup_rate = self._soup_rate(state)
        if soup_rate > 0 and state.grid_available() > 0:
            targets = []
            if _has_station(state.volcanic_patches):
                targets.append(self.spawn_cost_plant(state))
            if _has_station(state.herbivore_nests):
                targets.append(self.spawn_cost_herbivore_egg(state))
            if (_has_station(state.carnivore_nests) and
                    state.wake_ups >= CARNIVORE_NEST_UNLOCK_WAKE_UPS):
                targets.append(self.spawn_cost_carnivore_egg(state))
            for target in targets:
                if state.primordial_soup < target:
                    skip = min(skip, math.ceil((target - state.primordial_soup) / soup_rate))

        return max(1, skip)

    def _fast_forward(self, state: GameState, turns: int) -> None:
        """Apply `turns` worth of passive generation in one step."""
        self._add_soup(state, self._soup_rate(state) * turns)

        # Beacon recharge — compute charges gained over the full skip
        if state.beacon_charges < BEACON_MAX_CHARGES:
            recharge_turns = max(
                1,
                BEACON_RECHARGE_TURNS - BYE_BYE_PLANET_REDUCTION[state.bye_bye_planet_level],
            )
            counter = state.beacon_recharge_counter + turns
            charges_gained = counter // recharge_turns
            state.beacon_charges = min(state.beacon_charges + charges_gained, BEACON_MAX_CHARGES)
            if state.beacon_charges >= BEACON_MAX_CHARGES:
                state.beacon_recharge_counter = 0
            else:
                state.beacon_recharge_counter = counter % recharge_turns

        state.turn += turns


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _lowest_mergeable_level(d: dict[int, int], max_level: int) -> int:
    """Return the lowest level in d that has ≥2 items and is below max_level."""
    for lvl in range(1, max_level):
        if d.get(lvl, 0) >= 2:
            return lvl
    raise ValueError("No mergeable level found")  # should never happen if validity check passed


def _highest_item_level(d: dict[int, int]) -> int:
    """Return the highest level in d that has ≥1 item."""
    for lvl in range(MAX_CURRENCY_LEVEL, 0, -1):
        if d.get(lvl, 0) >= 1:
            return lvl
    raise ValueError("No item found")  # should never happen if validity check passed


def _credit_currency(state: "GameState", currency_type: str, value: int) -> None:
    """Add value to the spendable balance for the given currency type."""
    if currency_type == "bones":
        state.big_bones += value
    elif currency_type == "horns":
        state.horns += value
    else:
        state.fangs += value


def _item_dict(state: "GameState", currency_type: str) -> dict[int, int]:
    """Return the mutable item dict for a given currency type string."""
    if currency_type == "bones":
        return state.bone_items
    elif currency_type == "horns":
        return state.horn_items
    else:
        return state.fang_items


def _drop_currency(state: "GameState", creature_type) -> None:
    """Place the currency item dropped by a creature attack onto the grid."""
    currency_type, drop_level = CREATURE_CURRENCY_DROP[creature_type]
    d = _item_dict(state, currency_type)
    d[drop_level] = d.get(drop_level, 0) + 1


def _has_station(station_dict: dict[int, int]) -> bool:
    """Return True if there is at least one instance of this station type."""
    return any(cnt > 0 for cnt in station_dict.values())


def _merge_station(station_dict: dict[int, int], from_level: int) -> None:
    """Merge two instances at from_level into one at from_level+1."""
    station_dict[from_level] -= 2
    station_dict[from_level + 1] = station_dict.get(from_level + 1, 0) + 1


def _can_afford(state: GameState, cost: tuple) -> bool:
    bb, h, f = cost
    return state.big_bones >= bb and state.horns >= h and state.fangs >= f


def _spend_currency(state: GameState, cost: tuple) -> None:
    bb, h, f = cost
    state.big_bones -= bb
    state.horns     -= h
    state.fangs     -= f


# ------------------------------------------------------------------
# Module-level lookup tables (built once for speed)
# ------------------------------------------------------------------

_MERGE_HERBIVORE_EGG_ACTION = {
    HerbivoreType.STEGOSAURUS:  ActionType.MERGE_STEGOSAURUS_EGG,
    HerbivoreType.TRICERATOPS:  ActionType.MERGE_TRICERATOPS_EGG,
    HerbivoreType.BRONTOSAURUS: ActionType.MERGE_BRONTOSAURUS_EGG,
}
_ACTION_TO_HERBIVORE_EGG_MERGE = {v: k for k, v in _MERGE_HERBIVORE_EGG_ACTION.items()}

_MERGE_CARNIVORE_EGG_ACTION = {
    CarnivoreType.PTERODACTYL: ActionType.MERGE_PTERODACTYL_EGG,
    CarnivoreType.RAPTOR:      ActionType.MERGE_RAPTOR_EGG,
}
_ACTION_TO_CARNIVORE_EGG_MERGE = {v: k for k, v in _MERGE_CARNIVORE_EGG_ACTION.items()}

_GROW_HERBIVORE_ACTION = {
    HerbivoreType.STEGOSAURUS:  ActionType.GROW_STEGOSAURUS,
    HerbivoreType.TRICERATOPS:  ActionType.GROW_TRICERATOPS,
    HerbivoreType.BRONTOSAURUS: ActionType.GROW_BRONTOSAURUS,
}
_ACTION_TO_HERBIVORE_GROW = {v: k for k, v in _GROW_HERBIVORE_ACTION.items()}

_GROW_CARNIVORE_ACTION = {
    CarnivoreType.PTERODACTYL: ActionType.GROW_PTERODACTYL,
    CarnivoreType.RAPTOR:      ActionType.GROW_RAPTOR,
}
_ACTION_TO_CARNIVORE_GROW = {v: k for k, v in _GROW_CARNIVORE_ACTION.items()}


_SUMMON_BEAST_ACTION = {
    BeastType.MAMMOTH:     ActionType.SUMMON_MAMMOTH,
    BeastType.SABER_TOOTH: ActionType.SUMMON_SABER_TOOTH,
}
_ACTION_TO_BEAST_SUMMON = {v: k for k, v in _SUMMON_BEAST_ACTION.items()}

_ATTACK_HERBIVORE_ACTION = {
    HerbivoreType.STEGOSAURUS:  ActionType.ATTACK_STEGOSAURUS,
    HerbivoreType.TRICERATOPS:  ActionType.ATTACK_TRICERATOPS,
    HerbivoreType.BRONTOSAURUS: ActionType.ATTACK_BRONTOSAURUS,
}
_ACTION_TO_HERBIVORE_ATTACK = {v: k for k, v in _ATTACK_HERBIVORE_ACTION.items()}

_ATTACK_CARNIVORE_ACTION = {
    CarnivoreType.PTERODACTYL: ActionType.ATTACK_PTERODACTYL,
    CarnivoreType.RAPTOR:      ActionType.ATTACK_RAPTOR,
}
_ACTION_TO_CARNIVORE_ATTACK = {v: k for k, v in _ATTACK_CARNIVORE_ACTION.items()}

_ATTACK_BEAST_ACTION = {
    BeastType.MAMMOTH:     ActionType.ATTACK_MAMMOTH,
    BeastType.SABER_TOOTH: ActionType.ATTACK_SABER_TOOTH,
}
_ACTION_TO_BEAST_ATTACK = {v: k for k, v in _ATTACK_BEAST_ACTION.items()}

# Feed / merge currency dispatch maps
_FEED_ACTION_TO_TYPE: dict = {
    ActionType.FEED_BONES: "bones",
    ActionType.FEED_HORNS: "horns",
    ActionType.FEED_FANGS: "fangs",
}
_MERGE_CURRENCY_ACTION_TO_TYPE: dict = {
    ActionType.MERGE_BONES: "bones",
    ActionType.MERGE_HORNS: "horns",
    ActionType.MERGE_FANGS: "fangs",
}
