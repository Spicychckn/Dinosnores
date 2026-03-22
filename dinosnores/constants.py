"""
Game constants for Dinosnores simulator.

Notes on design assumptions (wiki is sparse on some values):
  - Plants required per herbivore type: scaled by relative damage output
  - Station/upgrade costs: approximate, using Big Bones / Horns / Fangs
  - Beast summoning: costs Primordial Soup (exact mechanic not specified in wiki)
  - Beacon recharge: 1,080 turns per charge (3 hours × 360 turns/hour; 1 turn = 10 s)
  - Currency per wake-up: approximate escalating values
  - Soup costs for spawning plants/eggs: approximate; station level halves the cost as an
    upgrade incentive
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

# ---------------------------------------------------------------------------
# Time model
# ---------------------------------------------------------------------------

SECONDS_PER_TURN      = 10            # one passive tick = 10 real seconds
GAME_DURATION_SECONDS = 72 * 3600     # 259,200 s — full event duration


# ---------------------------------------------------------------------------
# Creature type enums
# ---------------------------------------------------------------------------

class HerbivoreType(Enum):
    STEGOSAURUS  = "stegosaurus"
    TRICERATOPS  = "triceratops"
    BRONTOSAURUS = "brontosaurus"


class CarnivoreType(Enum):
    PTERODACTYL = "pterodactyl"
    RAPTOR      = "raptor"


class BeastType(Enum):
    MAMMOTH     = "mammoth"
    SABER_TOOTH = "saber_tooth"


# ---------------------------------------------------------------------------
# Creature stat definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HerbivoreStats:
    base_damage:         int   # damage dealt to T-Rex when deployed
    soup_production:     int   # Primordial Soup generated per turn (passive, while adult)
    plant_level_required: int  # level of plant consumed to grow baby -> adult


@dataclass(frozen=True)
class CarnivoreStats:
    base_damage:    int            # damage dealt to T-Rex when deployed
    herbivore_food: HerbivoreType  # adult herbivore consumed to grow egg -> adult


@dataclass(frozen=True)
class BeastStats:
    base_damage:      int    # damage dealt to T-Rex when deployed
    unlock_wake_ups:  int    # minimum wake-up count needed to unlock
    summon_cost:      tuple  # (big_bones, horns, fangs) spent to summon one beast


HERBIVORE_STATS: Dict[HerbivoreType, HerbivoreStats] = {
    HerbivoreType.STEGOSAURUS:  HerbivoreStats(base_damage=10,  soup_production=3, plant_level_required=4),
    HerbivoreType.TRICERATOPS:  HerbivoreStats(base_damage=50,  soup_production=0, plant_level_required=5),
    HerbivoreType.BRONTOSAURUS: HerbivoreStats(base_damage=20,  soup_production=5, plant_level_required=6),
}

CARNIVORE_STATS: Dict[CarnivoreType, CarnivoreStats] = {
    CarnivoreType.PTERODACTYL: CarnivoreStats(base_damage=100, herbivore_food=HerbivoreType.STEGOSAURUS),
    CarnivoreType.RAPTOR:      CarnivoreStats(base_damage=400, herbivore_food=HerbivoreType.BRONTOSAURUS),
}

BEAST_STATS: Dict[BeastType, BeastStats] = {
    BeastType.MAMMOTH:     BeastStats(base_damage=200, unlock_wake_ups=10, summon_cost=(30,  0, 0)),
    BeastType.SABER_TOOTH: BeastStats(base_damage=500, unlock_wake_ups=40, summon_cost=( 0, 30, 0)),
}


# ---------------------------------------------------------------------------
# T-Rex progression
# ---------------------------------------------------------------------------

TREX_INITIAL_HP       = 100  # starting HP
TREX_HP_PER_WAKEUP    = 25   # HP added to max after each wake-up
SCORE_BASE_INITIAL    = 50   # score awarded for the very first wake-up
SCORE_BASE_INCREMENT  = 10   # additional score per subsequent wake-up


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

GRID_SIZE = 32  # total positions on the game board

# Permanent grid occupants (always present from the start):
#   Volcanic Patch, Herbivore Nest, Alien Beacon = 3 spaces.
# The Carnivore Nest (+1) appears when unlocked; Primordial Crater (+1) when built.
# Use GameState.grid_permanent() to get the current fixed count.


# ---------------------------------------------------------------------------
# Station constants
# ---------------------------------------------------------------------------

# Per-station max levels (stations are merged up, not linearly upgraded)
MAX_VP_LEVEL = 5   # Volcanic Patch
MAX_HN_LEVEL = 5   # Herbivore Nest
MAX_CN_LEVEL = 3   # Carnivore Nest
MAX_PC_LEVEL = 4   # Primordial Crater

# Cost (big_bones, horns, fangs) to buy one level-1 station instance.
# Higher levels are obtained by merging two same-level instances.
STATION_BUY_COST: Dict[str, tuple] = {
    "herbivore_nest":    (40,  0,  0),
    "volcanic_patch":    (20, 20,  0),
    "carnivore_nest":    (50, 50,  0),
    "primordial_crater": ( 0,  0, 40),  # 40 Big Fangs (confirmed)
}

# Wake-up thresholds for each station to become purchasable
VOLCANIC_PATCH_UNLOCK_WAKE_UPS    = 5
HERBIVORE_NEST_UNLOCK_WAKE_UPS    = 0
# CARNIVORE_NEST_UNLOCK_WAKE_UPS  = 20  (defined below with other unlock thresholds)
PRIMORDIAL_CRATER_UNLOCK_WAKE_UPS = 30

# Herbivore egg type probabilities by Herbivore Nest level
# Each entry is a list of (HerbivoreType, probability) pairs summing to 1.0
from typing import Tuple as _Tuple  # avoid shadowing the module-level List import
HERBIVORE_NEST_EGG_PROBS: Dict[int, List[_Tuple[HerbivoreType, float]]] = {
    1: [(HerbivoreType.STEGOSAURUS, 1.00)],
    2: [(HerbivoreType.STEGOSAURUS, 0.70), (HerbivoreType.TRICERATOPS, 0.30)],
    3: [(HerbivoreType.STEGOSAURUS, 0.30), (HerbivoreType.TRICERATOPS, 0.70)],
    4: [(HerbivoreType.STEGOSAURUS, 0.20), (HerbivoreType.TRICERATOPS, 0.50), (HerbivoreType.BRONTOSAURUS, 0.30)],
    5: [(HerbivoreType.TRICERATOPS, 0.30), (HerbivoreType.BRONTOSAURUS, 0.70)],
}

MAX_PLANT_LEVEL = 6  # highest plant level; merged up from what Volcanic Patch produces

# Plant level probabilities by Volcanic Patch level
VOLCANIC_PATCH_PLANT_PROBS: Dict[int, List[_Tuple[int, float]]] = {
    1: [(1, 1.00)],
    2: [(1, 0.70), (2, 0.30)],
    3: [(2, 1.00)],
    4: [(2, 0.70), (3, 0.30)],
    5: [(3, 1.00)],
}

CARNIVORE_NEST_EGG_PROBS: Dict[int, List[_Tuple[CarnivoreType, float]]] = {
    1: [(CarnivoreType.PTERODACTYL, 1.00)],
    2: [(CarnivoreType.PTERODACTYL, 0.50), (CarnivoreType.RAPTOR, 0.50)],
    3: [(CarnivoreType.RAPTOR, 1.00)],
}

# Soup cost to activate (use) a station of each level — index = station level.
# Higher-level stations cost more soup per use.
VOLCANIC_PATCH_SPAWN_COST: List[int] = [0,   500,  600,  700,  800, 1_000]
HERBIVORE_NEST_SPAWN_COST: List[int] = [0, 2_000, 2_250, 2_500, 2_750, 3_000]
CARNIVORE_NEST_SPAWN_COST: List[int] = [0, 4_000, 4_500, 5_000]

# Primordial Soup generated per turn by each Primordial Crater instance at each level
# (wiki: +7, +12, +19, +28)
PRIMORDIAL_CRATER_SOUP_PER_TURN: List[int] = [0, 7, 12, 19, 28]


# ---------------------------------------------------------------------------
# Alien Beacon
# ---------------------------------------------------------------------------

BEACON_MAX_CHARGES        = 5
BEACON_RECHARGE_TURNS     = 1_080  # turns per charge (3 hours @ 1 turn = 10 s)
BEACON_SOUP_FRACTION      = 1/20 # soup reward = 1/20th of soup_capacity


# ---------------------------------------------------------------------------
# Feature unlock thresholds
# ---------------------------------------------------------------------------

CARNIVORE_NEST_UNLOCK_WAKE_UPS = 20
ALARM_CLOCK_UNLOCK_WAKE_UPS    = 50
# VOLCANIC_PATCH_UNLOCK_WAKE_UPS and PRIMORDIAL_CRATER_UNLOCK_WAKE_UPS
# are defined in the Station constants section above.


# ---------------------------------------------------------------------------
# Upgrades  (level 0 = not purchased)
# ---------------------------------------------------------------------------

# More Score: +10% per level up to +50% (5 levels)
MAX_MORE_SCORE_LEVEL   = 5
MORE_SCORE_BONUS: List[float] = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

# Bye Bye Planet: reduces beacon recharge by N turns per level (3 levels)
MAX_BYE_BYE_PLANET_LEVEL = 3
BYE_BYE_PLANET_REDUCTION: List[int] = [0, 120, 240, 360]  # turns removed from recharge (20/40/60 real-time minutes)

# Sharper Fangs: +25/+50/+100% to dino damage (herbivores + carnivores), 3 levels
MAX_SHARPER_FANGS_LEVEL  = 3
SHARPER_FANGS_BONUS: List[float] = [0.0, 0.25, 0.50, 1.00]

# Brutish Beasts: +25/+50/+100% to beast damage, 3 levels
MAX_BRUTISH_BEASTS_LEVEL = 3
BRUTISH_BEASTS_BONUS: List[float] = [0.0, 0.25, 0.50, 1.00]

# Greater Craters: +1 or +2 extra soup per Primordial Crater per tick, 2 levels
MAX_GREATER_CRATERS_LEVEL = 2
GREATER_CRATERS_BONUS: List[int] = [0, 1, 2]

# Soup Stores: base capacity 100k, each level adds 100k (5 levels)
BASE_SOUP_CAPACITY      = 100_000
MAX_SOUP_STORES_LEVEL   = 5
SOUP_STORES_BONUS: List[int] = [0, 100_000, 200_000, 300_000, 400_000, 500_000]


# ---------------------------------------------------------------------------
# Upgrade purchase costs: (big_bones, horns, fangs) per level
# Index i = cost to go from level i to level i+1
# ---------------------------------------------------------------------------

UPGRADE_COSTS: Dict[str, List[tuple]] = {
    #                      0→1         1→2         2→3         3→4         4→5
    "more_score":      [(20, 10, 0), (60, 0, 0), (0, 0, 50), (0, 75, 0), (0, 0, 100)],
    "bye_bye_planet":  [(20,  0, 0), ( 0,40, 0), (0, 0, 20)],
    "sharper_fangs":   [(20,  0, 0), (30,30, 0), (0, 0, 50)],
    "brutish_beasts":  [( 0, 20, 0), (25, 0,10), (0, 0, 25)],
    "greater_craters": [(50,  0, 0), ( 0,50, 0)],
    "soup_stores":     [( 0, 10, 0), (20, 0, 0), (0,30, 0), (0,25,25), (75, 0, 0)],
}

# Currency items dropped by creatures on attack
# Each item has a level; values below are spendable currency earned when fed to the T-Rex.
CURRENCY_ITEM_VALUE: List[int] = [0, 2, 5, 12, 30]  # index = item level
MAX_CURRENCY_LEVEL = 4

# Maps creature type → (currency_type_str, drop_level)
CREATURE_CURRENCY_DROP = {
    HerbivoreType.STEGOSAURUS:  ("bones", 1),
    HerbivoreType.BRONTOSAURUS: ("bones", 2),
    HerbivoreType.TRICERATOPS:  ("horns", 1),
    CarnivoreType.PTERODACTYL:  ("fangs", 1),
    CarnivoreType.RAPTOR:       ("fangs", 2),
    BeastType.MAMMOTH:          ("horns", 2),
    BeastType.SABER_TOOTH:      ("fangs", 3),
}
