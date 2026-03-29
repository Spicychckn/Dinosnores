"""
Gymnasium environment wrapper for the Dinosnores simulator.

Observation space : flat float32 vector (~72 features), all normalised to [0, 1].
Action space      : Discrete over all ActionType values.
Action masking    : action_masks() returns a bool array compatible with
                    MaskablePPO from sb3-contrib.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator import DinosnoresSimulator
from .actions import ActionType
from .constants import (
    HerbivoreType, CarnivoreType, BeastType,
    GRID_SIZE, BEACON_MAX_CHARGES, BEACON_RECHARGE_TURNS,
    BASE_SOUP_CAPACITY, SOUP_STORES_BONUS,
    MAX_PLANT_LEVEL,
    MAX_VP_LEVEL, MAX_HN_LEVEL, MAX_CN_LEVEL, MAX_PC_LEVEL,
    MAX_MORE_SCORE_LEVEL, MAX_BYE_BYE_PLANET_LEVEL,
    MAX_SHARPER_FANGS_LEVEL, MAX_BRUTISH_BEASTS_LEVEL,
    MAX_GREATER_CRATERS_LEVEL, MAX_SOUP_STORES_LEVEL,
    MAX_CURRENCY_LEVEL,
    GAME_DURATION_SECONDS,
    HERBIVORE_STATS,
    SHOP_DAY_TURNS, SHOP_NUM_DAYS, SHOP_TOTAL_ITEMS,
)

# Ordered list of all actions — index = action integer passed to step()
ALL_ACTIONS: list[ActionType] = list(ActionType)
N_ACTIONS: int = len(ALL_ACTIONS)
ACTION_TO_IDX: dict[ActionType, int] = {a: i for i, a in enumerate(ALL_ACTIONS)}

# Upper bounds used for normalisation (values are clipped then divided)
_MAX_SOUP_CAPACITY = BASE_SOUP_CAPACITY + max(SOUP_STORES_BONUS)  # 1 000 000
_MAX_TREX_HP       = 2_100   # ~83 wake-ups worth
_MAX_WAKE_UPS      = 100
_MAX_SCORE         = 100_000
_MAX_CURRENCY      = 500
_MAX_COUNT_HERB    = 10
_MAX_COUNT_CARN    = 5
_MAX_COUNT_BEAST   = 5
_MAX_COUNT_PLANT   = 20
_MAX_COUNT_STATION = 5
_MAX_COUNT_ITEM    = 10
_MAX_METEORS       = 5
_MAX_ALARM_CLOCKS  = 3
_MAX_SHOP_DAY      = SHOP_NUM_DAYS - 1  # 2


def _clip01(x: float, max_val: float) -> float:
    return float(np.clip(x / max_val, 0.0, 1.0))


class DinosnoresEnv(gym.Env):
    """
    Single-episode Gymnasium environment wrapping DinosnoresSimulator.

    Parameters
    ----------
    max_duration_seconds : float
        Simulated seconds per episode (default: full 72-hour event).
    score_target : int or None
        Optional early-termination score threshold.
    seed : int or None
        RNG seed forwarded to the simulator.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_duration_seconds: float = GAME_DURATION_SECONDS,
        score_target: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.sim = DinosnoresSimulator(
            max_duration_seconds=max_duration_seconds,
            score_target=score_target,
            seed=seed,
        )
        self._state = None
        self._max_turns = self.sim.max_turns

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.sim.reset()
        return self._obs(), {}

    def step(self, action_idx: int):
        action = ALL_ACTIONS[action_idx]
        # Fall back to WAIT for invalid actions (only happens outside masked training)
        valid = self.sim.get_valid_actions(self._state)
        if action not in valid:
            action = ActionType.WAIT
        self._state, reward, terminated, info = self.sim.step(self._state, action)
        reward += self._shaped_reward(info)
        truncated = False
        return self._obs(), float(reward), terminated, truncated, info

    def _shaped_reward(self, info: dict) -> float:
        """Small intermediate rewards to guide the agent through the pipeline.

        Design intent:
        - Per-turn survival cost discourages wasting turns on idle loops.
        - Only egg spawns (not plant spawns) are rewarded — plants are a cost,
          not a goal; rewarding plant spawns caused the agent to spam them.
        - Grow reward is scaled by the creature's soup_production: stego/bronto
          produce soup every turn, making them more valuable to grow early.
        - Damage reward directly incentivises attacking instead of just waiting
          for beacon recharges.
        """
        r = 0.0
        r -= 0.01                                      # per-turn wait tax

        spawned = info.get("spawned", "")
        if spawned.endswith("_egg"):
            r += 2.0                                   # egg spawned (plants give no reward)

        if info.get("merged_egg"):
            r += 2.0                                   # 2 eggs → 1 baby

        grew = info.get("grew")
        if grew:
            try:
                h_type = HerbivoreType(grew)
                # Scale by soup_production: stego=8, bronto=10, trice=5
                r += 5.0 + HERBIVORE_STATS[h_type].soup_production
            except ValueError:
                r += 6.0                               # carnivore — strong attacker, no soup

        r += info.get("damage_dealt", 0) * 0.01        # reward attacking

        if info.get("fed_meteor"):
            r += 0.2                                   # converted meteor to soup
        if info.get("fed"):
            r += 0.5                                   # fed currency item to T-Rex
        shop_label = info.get("shop_claimed", "")
        if shop_label:
            r += 3.0 if shop_label.endswith("_ad") else 1.5
        return r

    def action_masks(self) -> np.ndarray:
        """Return a bool mask (length N_ACTIONS) of currently valid actions."""
        valid = set(self.sim.get_valid_actions(self._state))
        return np.array([a in valid for a in ALL_ACTIONS], dtype=bool)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        s = self._state
        feats: list[float] = []

        # --- T-Rex ---
        feats.append(_clip01(s.trex_hp, max(s.trex_max_hp, 1)))
        feats.append(_clip01(s.trex_max_hp, _MAX_TREX_HP))
        feats.append(_clip01(s.wake_ups, _MAX_WAKE_UPS))

        # --- Score ---
        feats.append(_clip01(s.score, _MAX_SCORE))

        # --- Soup ---
        feats.append(_clip01(s.primordial_soup, max(s.soup_capacity, 1)))
        feats.append(_clip01(s.soup_capacity, _MAX_SOUP_CAPACITY))

        # --- Plants by level (1–MAX_PLANT_LEVEL) ---
        for lvl in range(1, MAX_PLANT_LEVEL + 1):
            feats.append(_clip01(s.plants.get(lvl, 0), _MAX_COUNT_PLANT))

        # --- Spendable currency ---
        feats.append(_clip01(s.big_bones, _MAX_CURRENCY))
        feats.append(_clip01(s.horns,     _MAX_CURRENCY))
        feats.append(_clip01(s.fangs,     _MAX_CURRENCY))

        # --- Alien Beacon ---
        feats.append(_clip01(s.beacon_charges,          BEACON_MAX_CHARGES))
        feats.append(_clip01(s.beacon_recharge_counter, BEACON_RECHARGE_TURNS))
        feats.append(_clip01(s.meteors,       _MAX_METEORS))
        feats.append(_clip01(s.alarm_clocks,  _MAX_ALARM_CLOCKS))

        # --- Herbivores ---
        for t in HerbivoreType:
            feats.append(_clip01(s.herbivore_eggs[t],   _MAX_COUNT_HERB))
        for t in HerbivoreType:
            feats.append(_clip01(s.baby_herbivores[t],  _MAX_COUNT_HERB))
        for t in HerbivoreType:
            feats.append(_clip01(s.adult_herbivores[t], _MAX_COUNT_HERB))

        # --- Carnivores ---
        for t in CarnivoreType:
            feats.append(_clip01(s.carnivore_eggs[t],   _MAX_COUNT_CARN))
        for t in CarnivoreType:
            feats.append(_clip01(s.baby_carnivores[t],  _MAX_COUNT_CARN))
        for t in CarnivoreType:
            feats.append(_clip01(s.adult_carnivores[t], _MAX_COUNT_CARN))

        # --- Beasts ---
        for t in BeastType:
            feats.append(_clip01(s.beasts[t], _MAX_COUNT_BEAST))

        # --- Stations (count per level) ---
        for lvl in range(1, MAX_VP_LEVEL + 1):
            feats.append(_clip01(s.volcanic_patches.get(lvl, 0),   _MAX_COUNT_STATION))
        for lvl in range(1, MAX_HN_LEVEL + 1):
            feats.append(_clip01(s.herbivore_nests.get(lvl, 0),    _MAX_COUNT_STATION))
        for lvl in range(1, MAX_CN_LEVEL + 1):
            feats.append(_clip01(s.carnivore_nests.get(lvl, 0),    _MAX_COUNT_STATION))
        for lvl in range(1, MAX_PC_LEVEL + 1):
            feats.append(_clip01(s.primordial_craters.get(lvl, 0), _MAX_COUNT_STATION))

        # --- Upgrade levels ---
        feats.append(_clip01(s.more_score_level,      MAX_MORE_SCORE_LEVEL))
        feats.append(_clip01(s.bye_bye_planet_level,  MAX_BYE_BYE_PLANET_LEVEL))
        feats.append(_clip01(s.sharper_fangs_level,   MAX_SHARPER_FANGS_LEVEL))
        feats.append(_clip01(s.brutish_beasts_level,  MAX_BRUTISH_BEASTS_LEVEL))
        feats.append(_clip01(s.greater_craters_level, MAX_GREATER_CRATERS_LEVEL))
        feats.append(_clip01(s.soup_stores_level,     MAX_SOUP_STORES_LEVEL))

        # --- Currency items (dropped by creatures, not yet fed) ---
        for lvl in range(1, MAX_CURRENCY_LEVEL + 1):
            feats.append(_clip01(s.bone_items.get(lvl, 0), _MAX_COUNT_ITEM))
        for lvl in range(1, MAX_CURRENCY_LEVEL + 1):
            feats.append(_clip01(s.horn_items.get(lvl, 0), _MAX_COUNT_ITEM))
        for lvl in range(1, MAX_CURRENCY_LEVEL + 1):
            feats.append(_clip01(s.fang_items.get(lvl, 0), _MAX_COUNT_ITEM))

        # --- Event Shop ---
        shop_day = min(s.turn // SHOP_DAY_TURNS, _MAX_SHOP_DAY)
        feats.append(float(shop_day) / _MAX_SHOP_DAY)
        for claimed in s.shop_items_claimed:
            feats.append(1.0 if claimed else 0.0)

        # --- Time / grid ---
        feats.append(_clip01(s.turn,              self._max_turns))
        feats.append(_clip01(s.grid_available(),  GRID_SIZE))

        return np.array(feats, dtype=np.float32)


# Compute observation dimension by running _obs() on a dummy state once
def _compute_obs_dim() -> int:
    env = DinosnoresEnv.__new__(DinosnoresEnv)
    env.sim = DinosnoresSimulator()
    env._state = env.sim.reset()
    env._max_turns = env.sim.max_turns
    return len(env._obs())


_OBS_DIM: int = _compute_obs_dim()
