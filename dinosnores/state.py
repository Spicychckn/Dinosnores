"""
GameState dataclass for the Dinosnores simulator.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict

from .constants import (
    HerbivoreType,
    CarnivoreType,
    BeastType,
    TREX_INITIAL_HP,
    BEACON_MAX_CHARGES,
    BASE_SOUP_CAPACITY,
    GRID_SIZE,
)



@dataclass
class GameState:
    # -----------------------------------------------------------------------
    # T-Rex
    # -----------------------------------------------------------------------
    trex_hp:     int = TREX_INITIAL_HP   # current HP
    trex_max_hp: int = TREX_INITIAL_HP   # resets to this value after each wake-up
    wake_ups:    int = 0                 # total number of times T-Rex has been woken

    # -----------------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------------
    score: int = 0

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------
    primordial_soup: int = 0
    soup_capacity:   int = BASE_SOUP_CAPACITY
    plants:          Dict[int, int] = field(default_factory=dict)  # level -> count

    # Currency used for upgrades and station improvements
    big_bones: int = 0
    horns:     int = 0
    fangs:     int = 0

    # -----------------------------------------------------------------------
    # Alien Beacon
    # -----------------------------------------------------------------------
    beacon_charges:          int = BEACON_MAX_CHARGES
    beacon_recharge_counter: int = 0  # turns since last charge started regenerating
    meteors:                 int = 0  # meteor grid items (spawned by beacon, feed for soup)

    # -----------------------------------------------------------------------
    # Creatures
    # adult_* counts are creatures ready to be deployed against the T-Rex.
    # -----------------------------------------------------------------------
    herbivore_eggs:    Dict[HerbivoreType, int] = field(default_factory=lambda: {t: 0 for t in HerbivoreType})
    baby_herbivores:   Dict[HerbivoreType, int] = field(default_factory=lambda: {t: 0 for t in HerbivoreType})
    adult_herbivores:  Dict[HerbivoreType, int] = field(default_factory=lambda: {t: 0 for t in HerbivoreType})

    carnivore_eggs:    Dict[CarnivoreType, int] = field(default_factory=lambda: {t: 0 for t in CarnivoreType})
    baby_carnivores:   Dict[CarnivoreType, int] = field(default_factory=lambda: {t: 0 for t in CarnivoreType})
    adult_carnivores:  Dict[CarnivoreType, int] = field(default_factory=lambda: {t: 0 for t in CarnivoreType})

    beasts: Dict[BeastType, int] = field(default_factory=lambda: {t: 0 for t in BeastType})

    # -----------------------------------------------------------------------
    # Currency items on the grid  (Dict maps level -> count)
    # Dropped by creatures on attack; must be fed to T-Rex to become spendable.
    # -----------------------------------------------------------------------
    bone_items: Dict[int, int] = field(default_factory=dict)
    horn_items: Dict[int, int] = field(default_factory=dict)
    fang_items: Dict[int, int] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Stations  (Dict maps level -> count of instances at that level)
    # Stations are bought at level 1 and merged up: lvl1+lvl1 -> lvl2, etc.
    # Each instance occupies 1 grid space.
    # -----------------------------------------------------------------------
    volcanic_patches:    Dict[int, int] = field(default_factory=lambda: {1: 1})
    herbivore_nests:     Dict[int, int] = field(default_factory=lambda: {1: 1})
    carnivore_nests:     Dict[int, int] = field(default_factory=dict)
    primordial_craters:  Dict[int, int] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Upgrade levels  (0 = not purchased)
    # -----------------------------------------------------------------------
    more_score_level:      int = 0
    bye_bye_planet_level:  int = 0
    sharper_fangs_level:   int = 0
    brutish_beasts_level:  int = 0
    greater_craters_level: int = 0
    soup_stores_level:     int = 0

    # -----------------------------------------------------------------------
    # Simulation bookkeeping
    # -----------------------------------------------------------------------
    turn: int = 0  # discrete turn counter

    # -----------------------------------------------------------------------
    # Grid helpers
    # -----------------------------------------------------------------------
    def grid_permanent(self) -> int:
        """Grid spaces occupied by the Alien Beacon (always 1)."""
        return 1

    def grid_occupancy(self) -> int:
        """Total grid spaces currently in use."""
        stations = (
            sum(self.volcanic_patches.values())
            + sum(self.herbivore_nests.values())
            + sum(self.carnivore_nests.values())
            + sum(self.primordial_craters.values())
        )
        variable = (
            sum(self.plants.values())
            + sum(self.herbivore_eggs.values())
            + sum(self.baby_herbivores.values())
            + sum(self.adult_herbivores.values())
            + sum(self.carnivore_eggs.values())
            + sum(self.baby_carnivores.values())
            + sum(self.adult_carnivores.values())
            + sum(self.beasts.values())
            + sum(self.bone_items.values())
            + sum(self.horn_items.values())
            + sum(self.fang_items.values())
            + self.meteors
        )
        return 1 + stations + variable  # 1 = Alien Beacon

    def grid_available(self) -> int:
        """Free grid spaces available for new entities."""
        return GRID_SIZE - self.grid_occupancy()

    # -----------------------------------------------------------------------
    # Station level helpers
    # -----------------------------------------------------------------------
    def max_vp_level(self) -> int:
        """Highest Volcanic Patch level currently on the grid (0 if none)."""
        return max((lvl for lvl, cnt in self.volcanic_patches.items() if cnt > 0), default=0)

    def max_hn_level(self) -> int:
        """Highest Herbivore Nest level currently on the grid (0 if none)."""
        return max((lvl for lvl, cnt in self.herbivore_nests.items() if cnt > 0), default=0)

    def max_cn_level(self) -> int:
        """Highest Carnivore Nest level currently on the grid (0 if none)."""
        return max((lvl for lvl, cnt in self.carnivore_nests.items() if cnt > 0), default=0)

    def max_pc_level(self) -> int:
        """Highest Primordial Crater level currently on the grid (0 if none)."""
        return max((lvl for lvl, cnt in self.primordial_craters.items() if cnt > 0), default=0)

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> int:
        from .constants import SECONDS_PER_TURN
        return self.turn * SECONDS_PER_TURN

    def copy(self) -> "GameState":
        """Return a deep copy of this state."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        hours, rem = divmod(self.elapsed_seconds, 3600)
        mins = rem // 60
        lines = [
            f"Turn {self.turn} ({hours}h {mins:02d}m) | Score {self.score} | Wake-ups {self.wake_ups}",
            f"  T-Rex HP: {self.trex_hp}/{self.trex_max_hp}",
            f"  Grid: {self.grid_occupancy()}/{GRID_SIZE} used  ({self.grid_available()} free)",
            f"  Soup: {self.primordial_soup:,}/{self.soup_capacity:,}  Plants: {dict(self.plants)}",
            f"  Currency — Bones: {self.big_bones}  Horns: {self.horns}  Fangs: {self.fangs}",
            f"  Beacon charges: {self.beacon_charges}/{self.beacon_recharge_counter}t recharge"
            + (f"  Meteors: {self.meteors}" if self.meteors else ""),
            f"  Herbivore eggs:    { {t.value: v for t,v in self.herbivore_eggs.items() if v} }",
            f"  Baby herbivores:   { {t.value: v for t,v in self.baby_herbivores.items() if v} }",
            f"  Adult herbivores:  { {t.value: v for t,v in self.adult_herbivores.items() if v} }",
            f"  Carnivore eggs:    { {t.value: v for t,v in self.carnivore_eggs.items() if v} }",
            f"  Baby carnivores:   { {t.value: v for t,v in self.baby_carnivores.items() if v} }",
            f"  Adult carnivores:  { {t.value: v for t,v in self.adult_carnivores.items() if v} }",
            f"  Beasts:           { {t.value: v for t,v in self.beasts.items() if v} }",
            f"  Stations — VP:{dict(self.volcanic_patches)} HN:{dict(self.herbivore_nests)} "
            f"CN:{dict(self.carnivore_nests)} PC:{dict(self.primordial_craters)}",
            f"  Upgrades — score:{self.more_score_level} fangs:{self.sharper_fangs_level} "
            f"beasts:{self.brutish_beasts_level} craters:{self.greater_craters_level} "
            f"soup:{self.soup_stores_level} beacon:{self.bye_bye_planet_level}",
        ]
        if self.bone_items:
            lines.append(f"  Bone items:  {dict(self.bone_items)}")
        if self.horn_items:
            lines.append(f"  Horn items:  {dict(self.horn_items)}")
        if self.fang_items:
            lines.append(f"  Fang items:  {dict(self.fang_items)}")
        return "\n".join(lines)
