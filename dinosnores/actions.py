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
    SPAWN_PLANT = "spawn_plant"  # Volcanic Patch required

    SPAWN_HERBIVORE_EGG = (
        "spawn_herbivore_egg"  # Herbivore Nest required; egg type determined by nest level
    )

    SPAWN_CARNIVORE_EGG = (
        "spawn_carnivore_egg"  # Carnivore Nest required; egg type determined by nest level
    )

    # -----------------------------------------------------------------------
    # Merge eggs (2× same egg → 1 baby of that type)
    # -----------------------------------------------------------------------
    MERGE_STEGOSAURUS_EGG = "merge_steg_egg"
    MERGE_TRICERATOPS_EGG = "merge_tri_egg"
    MERGE_BRONTOSAURUS_EGG = "merge_bront_egg"

    MERGE_PTERODACTYL_EGG = "merge_ptero_egg"
    MERGE_RAPTOR_EGG = "merge_raptor_egg"

    # -----------------------------------------------------------------------
    # Grow creatures (baby + food → adult)
    # Herbivores:  consume 1 plant of the required level
    # Carnivores:  consume 1 adult herbivore of the matching food type
    # -----------------------------------------------------------------------
    GROW_STEGOSAURUS = "grow_stegosaurus"  # baby steg + lvl 4 plant
    GROW_TRICERATOPS = "grow_triceratops"  # baby tri  + lvl 5 plant
    GROW_BRONTOSAURUS = "grow_brontosaurus"  # baby bront + lvl 6 plant

    GROW_PTERODACTYL = "grow_pterodactyl"  # baby ptero + adult stegosaurus
    GROW_RAPTOR = "grow_raptor"  # baby raptor + adult brontosaurus

    # -----------------------------------------------------------------------
    # Merge plants (2× lvl N → 1× lvl N+1; always merges lowest available pair)
    # -----------------------------------------------------------------------
    MERGE_PLANT = "merge_plant"

    # -----------------------------------------------------------------------
    # Summon beasts
    # Costs Primordial Soup; creature is immediately ready to attack.
    # Requires wake_ups >= unlock_wake_ups.
    # -----------------------------------------------------------------------
    SUMMON_MAMMOTH = "summon_mammoth"
    SUMMON_SABER_TOOTH = "summon_saber_tooth"

    # -----------------------------------------------------------------------
    # Attack T-Rex
    # Deploys one ready adult creature; deals damage and, for herbivores,
    # gains Primordial Soup.  Consumes that creature.
    # -----------------------------------------------------------------------
    ATTACK_STEGOSAURUS = "attack_stegosaurus"
    ATTACK_TRICERATOPS = "attack_triceratops"
    ATTACK_BRONTOSAURUS = "attack_brontosaurus"

    ATTACK_PTERODACTYL = "attack_pterodactyl"
    ATTACK_RAPTOR = "attack_raptor"

    ATTACK_MAMMOTH = "attack_mammoth"
    ATTACK_SABER_TOOTH = "attack_saber_tooth"

    # -----------------------------------------------------------------------
    # Special abilities
    # -----------------------------------------------------------------------
    # Alien Beacon: halve current T-Rex HP; spawn 1 Meteor grid item.
    # Requires beacon_charges >= 1.
    USE_BEACON = "use_beacon"

    # Feed Meteor to T-Rex: remove 1 meteor from grid; gain soup_capacity/20 soup.
    FEED_METEOR = "feed_meteor"

    # Alarm Clock: deal damage equal to 100% of current T-Rex HP (instant kill).
    # Unlocked at wake_ups >= ALARM_CLOCK_UNLOCK_WAKE_UPS.
    USE_ALARM_CLOCK = "use_alarm_clock"

    # -----------------------------------------------------------------------
    # Buy stations
    # Places one level-1 instance on the grid; costs (big_bones, horns, fangs)
    # from STATION_BUY_COST and requires 1 free grid space.
    # Each station type has a wake-up unlock threshold.
    # -----------------------------------------------------------------------
    BUY_VOLCANIC_PATCH = "buy_volcanic_patch"
    BUY_HERBIVORE_NEST = "buy_herbivore_nest"
    BUY_CARNIVORE_NEST = "buy_carnivore_nest"
    BUY_PRIMORDIAL_CRATER = "buy_primordial_crater"

    # -----------------------------------------------------------------------
    # Buy alarm clock from build menu
    # Costs 100 fangs each; unlimited purchases; unlocked at 50 wake-ups.
    # Places 1 alarm clock grid item (each USE_ALARM_CLOCK consumes 1).
    # -----------------------------------------------------------------------
    BUY_ALARM_CLOCK = "buy_alarm_clock"

    # -----------------------------------------------------------------------
    # Merge stations
    # Combines two instances of level N into one instance of level N+1.
    # Frees 1 grid space net.  Named as MERGE_<STATION>_<FROM_LEVEL>.
    # -----------------------------------------------------------------------
    MERGE_VOLCANIC_PATCH = "merge_vp"  # always merges lowest available pair
    MERGE_HERBIVORE_NEST = "merge_hn"
    MERGE_CARNIVORE_NEST = "merge_cn"
    MERGE_PRIMORDIAL_CRATER = "merge_pc"

    # -----------------------------------------------------------------------
    # Feed currency items to T-Rex (feeds highest available level of that type)
    # -----------------------------------------------------------------------
    FEED_BONES = "feed_bones"
    FEED_HORNS = "feed_horns"
    FEED_FANGS = "feed_fangs"

    # -----------------------------------------------------------------------
    # Merge currency items (merges lowest available pair; auto-feeds if result
    # is lvl 4, since that is always the correct follow-up action)
    # -----------------------------------------------------------------------
    MERGE_BONES = "merge_bones"
    MERGE_HORNS = "merge_horns"
    MERGE_FANGS = "merge_fangs"

    # -----------------------------------------------------------------------
    # Purchase upgrades
    # Each costs (big_bones, horns, fangs) from UPGRADE_COSTS.
    # -----------------------------------------------------------------------
    BUY_MORE_SCORE = "buy_more_score"
    BUY_BYE_BYE_PLANET = "buy_bye_bye_planet"
    BUY_SHARPER_FANGS = "buy_sharper_fangs"
    BUY_BRUTISH_BEASTS = "buy_brutish_beasts"
    BUY_GREATER_CRATERS = "buy_greater_craters"
    BUY_SOUP_STORES = "buy_soup_stores"

    # -----------------------------------------------------------------------
    # Event Shop — day-agnostic slots (simulator resolves the concrete item
    # based on the current shop day at execution time)
    # -----------------------------------------------------------------------
    SHOP_SLOT_0 = "shop_slot_0"  # today's first paid item
    SHOP_SLOT_1 = "shop_slot_1"  # today's second paid item
    SHOP_SLOT_2 = "shop_slot_2"  # today's third paid item
    SHOP_CLAIM_AD = "shop_claim_ad"  # today's free ad reward
