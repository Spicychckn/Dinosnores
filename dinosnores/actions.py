"""
Action definitions for the Dinosnores simulator.

Each ActionType value is a string label used for logging and debugging.
The simulator's get_valid_actions() returns a subset based on current state.
"""

from enum import Enum


class ActionType(Enum):
    # -----------------------------------------------------------------------
    # Do nothing — advance the turn (Primordial Crater still generates soup)
    # -----------------------------------------------------------------------
    WAIT = "wait"

    # -----------------------------------------------------------------------
    # Spawn entities onto the grid  (each costs Primordial Soup and uses 1
    # grid space per entity created; requires the relevant station to exist)
    # -----------------------------------------------------------------------
    SPAWN_PLANT            = "spawn_plant"             # Volcanic Patch required

    SPAWN_HERBIVORE_EGG    = "spawn_herbivore_egg"      # Herbivore Nest required; egg type determined by nest level

    SPAWN_CARNIVORE_EGG    = "spawn_carnivore_egg"      # Carnivore Nest required; egg type determined by nest level

    # -----------------------------------------------------------------------
    # Merge eggs (2× same egg → 1 baby of that type)
    # -----------------------------------------------------------------------
    MERGE_STEGOSAURUS_EGG  = "merge_steg_egg"
    MERGE_TRICERATOPS_EGG  = "merge_tri_egg"
    MERGE_BRONTOSAURUS_EGG = "merge_bront_egg"

    MERGE_PTERODACTYL_EGG  = "merge_ptero_egg"
    MERGE_RAPTOR_EGG       = "merge_raptor_egg"

    # -----------------------------------------------------------------------
    # Grow creatures (baby + food → adult)
    # Herbivores:  consume 1 plant of the required level
    # Carnivores:  consume 1 adult herbivore of the matching food type
    # -----------------------------------------------------------------------
    GROW_STEGOSAURUS  = "grow_stegosaurus"   # baby steg + lvl 4 plant
    GROW_TRICERATOPS  = "grow_triceratops"   # baby tri  + lvl 5 plant
    GROW_BRONTOSAURUS = "grow_brontosaurus"  # baby bront + lvl 6 plant

    GROW_PTERODACTYL  = "grow_pterodactyl"   # baby ptero + adult stegosaurus
    GROW_RAPTOR       = "grow_raptor"        # baby raptor + adult brontosaurus

    # -----------------------------------------------------------------------
    # Merge plants (2× lvl N → 1× lvl N+1)
    # -----------------------------------------------------------------------
    MERGE_PLANT_LVL1 = "merge_plant_1"
    MERGE_PLANT_LVL2 = "merge_plant_2"
    MERGE_PLANT_LVL3 = "merge_plant_3"
    MERGE_PLANT_LVL4 = "merge_plant_4"
    MERGE_PLANT_LVL5 = "merge_plant_5"

    # -----------------------------------------------------------------------
    # Summon beasts
    # Costs Primordial Soup; creature is immediately ready to attack.
    # Requires wake_ups >= unlock_wake_ups.
    # -----------------------------------------------------------------------
    SUMMON_MAMMOTH     = "summon_mammoth"
    SUMMON_SABER_TOOTH = "summon_saber_tooth"

    # -----------------------------------------------------------------------
    # Attack T-Rex
    # Deploys one ready adult creature; deals damage and, for herbivores,
    # gains Primordial Soup.  Consumes that creature.
    # -----------------------------------------------------------------------
    ATTACK_STEGOSAURUS  = "attack_stegosaurus"
    ATTACK_TRICERATOPS  = "attack_triceratops"
    ATTACK_BRONTOSAURUS = "attack_brontosaurus"

    ATTACK_PTERODACTYL  = "attack_pterodactyl"
    ATTACK_RAPTOR       = "attack_raptor"

    ATTACK_MAMMOTH      = "attack_mammoth"
    ATTACK_SABER_TOOTH  = "attack_saber_tooth"

    # -----------------------------------------------------------------------
    # Special abilities
    # -----------------------------------------------------------------------
    # Alien Beacon: halve current T-Rex HP; spawn 1 Meteor grid item.
    # Requires beacon_charges >= 1.
    USE_BEACON      = "use_beacon"

    # Feed Meteor to T-Rex: remove 1 meteor from grid; gain soup_capacity/20 soup.
    FEED_METEOR     = "feed_meteor"

    # Alarm Clock: deal damage equal to 100% of current T-Rex HP (instant kill).
    # Unlocked at wake_ups >= ALARM_CLOCK_UNLOCK_WAKE_UPS.
    USE_ALARM_CLOCK = "use_alarm_clock"

    # -----------------------------------------------------------------------
    # Buy stations
    # Places one level-1 instance on the grid; costs (big_bones, horns, fangs)
    # from STATION_BUY_COST and requires 1 free grid space.
    # Each station type has a wake-up unlock threshold.
    # -----------------------------------------------------------------------
    BUY_VOLCANIC_PATCH    = "buy_volcanic_patch"
    BUY_HERBIVORE_NEST    = "buy_herbivore_nest"
    BUY_CARNIVORE_NEST    = "buy_carnivore_nest"
    BUY_PRIMORDIAL_CRATER = "buy_primordial_crater"

    # -----------------------------------------------------------------------
    # Merge stations
    # Combines two instances of level N into one instance of level N+1.
    # Frees 1 grid space net.  Named as MERGE_<STATION>_<FROM_LEVEL>.
    # -----------------------------------------------------------------------
    MERGE_VOLCANIC_PATCH_1 = "merge_vp_1"   # 2× lvl1 → 1× lvl2
    MERGE_VOLCANIC_PATCH_2 = "merge_vp_2"   # 2× lvl2 → 1× lvl3
    MERGE_VOLCANIC_PATCH_3 = "merge_vp_3"
    MERGE_VOLCANIC_PATCH_4 = "merge_vp_4"   # 2× lvl4 → 1× lvl5 (max)

    MERGE_HERBIVORE_NEST_1 = "merge_hn_1"
    MERGE_HERBIVORE_NEST_2 = "merge_hn_2"
    MERGE_HERBIVORE_NEST_3 = "merge_hn_3"
    MERGE_HERBIVORE_NEST_4 = "merge_hn_4"   # 2× lvl4 → 1× lvl5 (max)

    MERGE_CARNIVORE_NEST_1 = "merge_cn_1"
    MERGE_CARNIVORE_NEST_2 = "merge_cn_2"   # 2× lvl2 → 1× lvl3 (max)

    MERGE_PRIMORDIAL_CRATER_1 = "merge_pc_1"
    MERGE_PRIMORDIAL_CRATER_2 = "merge_pc_2"
    MERGE_PRIMORDIAL_CRATER_3 = "merge_pc_3"  # 2× lvl3 → 1× lvl4 (max)

    # -----------------------------------------------------------------------
    # Feed currency items to T-Rex (0 damage; item removed from grid,
    # its value added to the matching spendable currency balance)
    # -----------------------------------------------------------------------
    FEED_BONES_LVL1 = "feed_bones_1"
    FEED_BONES_LVL2 = "feed_bones_2"
    FEED_BONES_LVL3 = "feed_bones_3"
    FEED_BONES_LVL4 = "feed_bones_4"

    FEED_HORNS_LVL1 = "feed_horns_1"
    FEED_HORNS_LVL2 = "feed_horns_2"
    FEED_HORNS_LVL3 = "feed_horns_3"
    FEED_HORNS_LVL4 = "feed_horns_4"

    FEED_FANGS_LVL1 = "feed_fangs_1"
    FEED_FANGS_LVL2 = "feed_fangs_2"
    FEED_FANGS_LVL3 = "feed_fangs_3"
    FEED_FANGS_LVL4 = "feed_fangs_4"

    # -----------------------------------------------------------------------
    # Merge currency items (2× LvlN → 1× LvlN+1; frees 1 grid space)
    # -----------------------------------------------------------------------
    MERGE_BONES_LVL1 = "merge_bones_1"
    MERGE_BONES_LVL2 = "merge_bones_2"
    MERGE_BONES_LVL3 = "merge_bones_3"

    MERGE_HORNS_LVL1 = "merge_horns_1"
    MERGE_HORNS_LVL2 = "merge_horns_2"
    MERGE_HORNS_LVL3 = "merge_horns_3"

    MERGE_FANGS_LVL1 = "merge_fangs_1"
    MERGE_FANGS_LVL2 = "merge_fangs_2"
    MERGE_FANGS_LVL3 = "merge_fangs_3"

    # -----------------------------------------------------------------------
    # Purchase upgrades
    # Each costs (big_bones, horns, fangs) from UPGRADE_COSTS.
    # -----------------------------------------------------------------------
    BUY_MORE_SCORE      = "buy_more_score"
    BUY_BYE_BYE_PLANET  = "buy_bye_bye_planet"
    BUY_SHARPER_FANGS   = "buy_sharper_fangs"
    BUY_BRUTISH_BEASTS  = "buy_brutish_beasts"
    BUY_GREATER_CRATERS = "buy_greater_craters"
    BUY_SOUP_STORES     = "buy_soup_stores"
