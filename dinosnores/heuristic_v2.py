"""
Advanced greedy heuristic for the Dinosnores simulator — v2.

Implements the 3-day strategy from heuristic_plan.txt in phases:

Phase 1 – OPENING + STEGO FILL  (implemented)
  Buy Soup Stores lvl1+2 with starting 20 bones + 10 horns.
  Fill board with stegos, run 8-stego attack waves.
  Buy VP2 from Day 0 shop slot 1 (25 bones) opportunistically.
  Stego fill continues until bones >= 120 AND soup >= 300k (HN4 push trigger).

Phase 2 – HN4 PUSH  (implemented)
  Part A: buy 3× HN1 (120 bones), merge 4× HN1 → HN3.
  Horn pile: spawn trices from HN3, attack for horn_item_lvl1; claim
  mammoth ad (horn_item_lvl2 drop); buy Day 0 shop horn item (5 bones).
  Merge all horn items → lvl4, feed → 30 horns → buy shop HN2 (30h).
  Part B: buy 2× HN1 (80 bones), merge HN3+HN2+2xHN1 → HN4.
  Stego waves continue throughout to keep bones flowing.

Phase 3 – MIGRATION  (implemented)
  Grow brontos from HN4 eggs; keep them on the board for passive soup.
  Attack trices immediately for horns. Stego waves continue for bones.
  Buy CN1 (Day 1 shop slot 2, 70 bones) once brontos ≥ 2, bones ≥ 110,
  grid ≥ 5 free. Buy extra HN1 (40 bones) after CN1 for guaranteed stego
  eggs to use as ptero food in Phase 4.

Phase 4 – PTERO WAVES  (implemented)
  Triggered once CN1 and the extra HN1 are both on the grid.
  HN1 guarantees 100% stego eggs (ptero food); CN1 generates ptero eggs.
  Build up 8 pteros, attack in a wave, repeat. Brontos from Phase 3 stay
  on the board for passive soup. Attack trices/raptors immediately.
  Merge CN1+CN1 → CN2 when affordable (50 bones + 50 horns from build menu).
"""


from .actions import ActionType
from .constants import (
    BEAST_STATS,
    BRUTISH_BEASTS_BONUS,
    GAME_DURATION_SECONDS,
    MAX_CURRENCY_LEVEL,
    SECONDS_PER_TURN,
    SHOP_DAY_TURNS,
    BeastType,
    CarnivoreType,
    HerbivoreType,
)
from .state import GameState

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

# Trigger for CN1 purchase (Day 1 shop slot 2)
_CN_BONES            = 70   # CN1 shop cost
_EXTRA_HN1_BONES     = 40   # extra HN1 from build menu, bought after CN1 for Phase 4 stego supply
_CN_TRIGGER_BONES    = _CN_BONES + _EXTRA_HN1_BONES  # 110 — afford both in one window
_MIN_BRONTOS_FOR_CN  = 2    # bronto soup engine must be running before buying CN1
_GRID_RESERVE_FOR_CN = 5    # grid slots needed: CN1 + buffer for egg/plant pipeline

# Migration phase soup floors — prevent draining soup to 0 while growing brontos
_MIGRATION_PLANT_SOUP_FLOOR = 2_000   # don't spawn plants below this level
_MIGRATION_EGG_SOUP_FLOOR   = 5_000   # don't spawn eggs below this level
_MIGRATION_STEGO_WAVE_MIN   = 4       # attack stegos in Phase 3 when >= this many adults
_PTERO_PHASE_SOUP_MIN       = 150_000 # minimum soup before buying the extra HN1 that
                                       # triggers Phase 4; seeds that hit CN1 earlier
                                       # accumulate less soup — this prevents them from
                                       # entering Phase 4 with only 74-90k (enough for
                                       # ~5 pteros before starvation).

# Ptero wave constants
_PTERO_WAVE_SIZE        = 8     # attack pteros in batches of this size
_PTERO_EGG_SOUP_FLOOR   = 2_000 # minimum soup before spawning eggs in Phase 4
_PTERO_PLANT_SOUP_FLOOR = 700   # minimum soup before spawning plants in Phase 4
                                 # (plants cost 500-700; separate from egg floor)


class GreedyHeuristicV2:
    """
    Multi-phase heuristic implementing the 3-day Dinosnores strategy.
    """

    def __init__(self):
        self._batch_remaining: int = 0  # stego attacks still owed in current wave

    def choose_action(
        self, state: GameState, valid_actions: list[ActionType]
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
        # SHOP ADS — timing gated per day.
        # Day 0 = mammoth_ad (drops horn_item_lvl2): wait until HN3 is built
        #   so the item lands during the horn pile phase, not idle in stego fill.
        # Day 1 = saber_tooth_ad: wait until within kill range.
        # Day 2 = horn_item_lvl4_ad: free, always claim.
        # ----------------------------------------------------------------
        _has_hn3_for_shop = state.herbivore_nests.get(3, 0) >= 1
        if has(ActionType.SHOP_CLAIM_AD):
            if shop_day == 2:
                return ActionType.SHOP_CLAIM_AD
            elif shop_day == 0 and _has_hn3_for_shop:
                return ActionType.SHOP_CLAIM_AD
            elif shop_day == 1 and state.trex_hp <= saber_dmg:
                return ActionType.SHOP_CLAIM_AD

        # ----------------------------------------------------------------
        # SHOP PAID — slot 0 + 1 opportunistic; slot 2 gated per phase.
        # Day 0 slot 0 = horn_item_lvl2 (5 bones): wait until HN3 is built
        #   so the horn item arrives in the horn pile phase, not wasting a
        #   grid space during stego fill.
        # Slot 2 on Day 1 = CN1 (70 bones) — bought in migration phase.
        # Slot 2 on Day 2 = alarm clock    — bought during sprint only.
        # ----------------------------------------------------------------
        if has(ActionType.SHOP_SLOT_0) and (shop_day != 0 or _has_hn3_for_shop):
            return ActionType.SHOP_SLOT_0
        if has(ActionType.SHOP_SLOT_1):
            return ActionType.SHOP_SLOT_1
        if has(ActionType.SHOP_SLOT_2) and shop_day == 0:
            # Day 0 slot 2 = HN2 (30 horns). The simulator gates availability
            # on affording 30 horns, which doesn't happen until the horn pile
            # is complete — buy immediately once affordable so we don't miss
            # the Day 0 window (soup never hits 300k before Day 0 ends).
            return ActionType.SHOP_SLOT_2
        if has(ActionType.SHOP_SLOT_2) and shop_day == 1:
            # Day 1 slot 2 = CN1 (70 bones) — buy once HN4 migration is
            # underway: bronto soup engine running and enough bones to also
            # afford the extra HN1 right after.
            # Note: stego count is NOT required — with 2 brontos (10 soup/turn)
            # + crater (12/turn), the soup engine is self-sustaining regardless
            # of stego count.  Requiring stegos >= 8 was too strict since HN4
            # produces only 20% stego eggs and phase-3 attacks push them below 8.
            if (state.herbivore_nests.get(4, 0) >= 1
                    and state.big_bones >= _CN_TRIGGER_BONES
                    and state.adult_herbivores[HerbivoreType.BRONTOSAURUS] >= _MIN_BRONTOS_FOR_CN
                    and free >= _GRID_RESERVE_FOR_CN):
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

        _hn4_built       = state.herbivore_nests.get(4, 0) >= 1
        _bones_ready     = state.big_bones >= _HN4_PUSH_BONES
        _soup_ready      = state.primordial_soup >= _HN4_PUSH_SOUP
        _hn4_push_active = _bones_ready and not _hn4_built

        # ----------------------------------------------------------------
        # HN4 PUSH — once bones ≥ 120, start buying HN1s and building
        # toward HN4. Stego refill is suppressed (not the attack wave) so
        # soup can accumulate passively to 300k for the migration buffer.
        # ----------------------------------------------------------------
        if _hn4_push_active:
            action = self._do_hn4_push(state, has, free, grid_full)
            if action is not None:
                return action

        # ----------------------------------------------------------------
        # STEGO FILL / ATTACK LOOP
        # During stego fill (bones < 120): full wave cycle.
        # Soup accumulation (bones ≥ 120, soup < 300k): suppress both
        # attacks AND refill — keep the full stego board intact so passive
        # soup generation is maximised while soup climbs to 300k.
        # During HN4 push (bones ≥ 120, soup ≥ 300k): full wave cycle
        # resumes so bones keep flowing for HN1 purchases.
        # ----------------------------------------------------------------
        if not _hn4_built:
            accumulating_soup = _bones_ready and not _soup_ready
            action = self._do_stego_wave(
                state, has, free, grid_full,
                suppress_wave=accumulating_soup,
                suppress_refill=accumulating_soup,
            )
            if action is not None:
                return action

        # ----------------------------------------------------------------
        # MIGRATION / PTERO WAVES — HN4 is built.
        # Phase 3 (Migration): grow brontos, attack trices, stego waves.
        # Phase 4 (Ptero Waves): triggered once CN1 + extra HN1 are on
        #   the grid; coordinate stego food supply with ptero pipeline.
        # ----------------------------------------------------------------
        if _hn4_built:
            _ptero_phase = (
                sum(state.carnivore_nests.values()) > 0
                and state.herbivore_nests.get(1, 0) >= 1
            )
            if _ptero_phase:
                action = self._do_ptero_waves(state, has, free, grid_full)
            else:
                action = self._do_migration(state, has, free, grid_full)
            if action is not None:
                return action

        return ActionType.WAIT

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _do_opening(self, state: GameState, has) -> ActionType | None:
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
        suppress_grow_stego: bool = False,
        suppress_wave: bool = False,
        suppress_refill: bool = False,
    ) -> ActionType | None:
        """
        8-stego attack waves.

        suppress_grow_stego: skip GROW_STEGOSAURUS so lvl4 plants are not
          consumed before reaching lvl5 for a waiting trice baby.
        suppress_wave: skip triggering new attack waves (keeps the stego
          board intact while soup accumulates passively to 300k).
        suppress_refill: skip _build_one_stego (used together with
          suppress_wave during the soup accumulation phase).
        """
        n_stegos     = state.adult_herbivores[HerbivoreType.STEGOSAURUS]
        stego_eggs   = state.herbivore_eggs[HerbivoreType.STEGOSAURUS]
        stego_babies = state.baby_herbivores[HerbivoreType.STEGOSAURUS]
        lvl4_plants  = state.plants.get(4, 0)

        # Merge VP stations as soon as a pair exists (VP1+VP1 → VP2, etc.).
        if has(ActionType.MERGE_VOLCANIC_PATCH):
            return ActionType.MERGE_VOLCANIC_PATCH

        # Trigger a new attack wave when conditions are met.
        if (not suppress_wave
                and self._batch_remaining == 0
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
        if not suppress_refill and self._batch_remaining == 0 and not _any_bones(state):
            return self._build_one_stego(
                state, has, lvl4_plants, stego_eggs, stego_babies, grid_full, free,
                suppress_grow=suppress_grow_stego,
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
        suppress_grow: bool = False,
    ) -> ActionType | None:
        """
        Return the next action to advance one stego through the plant+egg
        pipeline, or None if we should WAIT.

        suppress_grow: skip GROW_STEGOSAURUS when a trice baby is waiting
        for a lvl5 plant so lvl4 plants aren't consumed prematurely.
        """
        in_progress = (
            stego_eggs > 0
            or stego_babies > 0
            or sum(state.plants.values()) > 0
        )
        can_build = in_progress or state.primordial_soup >= _SOUP_PER_STEGO

        if not can_build:
            return None

        if not suppress_grow and has(ActionType.GROW_STEGOSAURUS):
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

    def _do_hn4_push(
        self,
        state: GameState,
        has,
        free: int,
        grid_full: bool,
    ) -> ActionType | None:
        """
        Drive the HN1 → HN3 → HN4 merge path while running the horn pile
        to accumulate 30 spendable horns for the shop HN2 purchase.

        Part A (no HN3 yet):
          Buy HN1s until 4 HN1-equivalents exist, then let MERGE_HERBIVORE_NEST
          (called first) collapse them to HN3 automatically.

        Horn pile (HN3 exists, horns < 30):
          Spawn trice eggs from HN3, grow trices, attack for horn_item_lvl1.
          Mammoth ad and shop horn item are already bought by the always-rules.
          Merge all horn items to lvl4 → feed → 30 spendable horns.
          Shop HN2 (SHOP_SLOT_2, 30 horns) is then bought by the general
          shop section above (gated on soup >= _HN4_PUSH_SOUP).

        Part B (HN3 exists, horns >= 30 so HN2 purchase is imminent/done):
          Buy 2 more HN1s (80 bones total). Combined with the shop HN2,
          MERGE_HERBIVORE_NEST collapses HN3+HN2+2xHN1 → HN4.
        """
        has_hn3 = state.herbivore_nests.get(3, 0) >= 1

        # ----------------------------------------------------------------
        # Always: merge herbivore nests — highest priority so merges
        # happen immediately after each purchase (frees grid space).
        # ----------------------------------------------------------------
        if has(ActionType.MERGE_HERBIVORE_NEST):
            return ActionType.MERGE_HERBIVORE_NEST

        # ----------------------------------------------------------------
        # Count HN1-equivalents in the system (1×HN1=1, HN2=2, HN3=4, HN4=8)
        # ----------------------------------------------------------------
        hn_nodes = (
            state.herbivore_nests.get(1, 0) * 1
            + state.herbivore_nests.get(2, 0) * 2
            + state.herbivore_nests.get(3, 0) * 4
            + state.herbivore_nests.get(4, 0) * 8
        )

        # ----------------------------------------------------------------
        # Part A: buy HN1s until we have 4 nodes (1 start + 3 new → HN3).
        # ----------------------------------------------------------------
        if not has_hn3 and hn_nodes < 4:
            if has(ActionType.BUY_HERBIVORE_NEST) and free >= 1:
                return ActionType.BUY_HERBIVORE_NEST

        # ----------------------------------------------------------------
        # Part B: once HN3 exists and the shop HN2 is in the system
        # (either just purchased — horns dropped to 0 — or still pending
        # purchase), buy 2 more HN1s to reach 8 nodes → HN4.
        # Shop HN2 = 2 nodes; 2×HN1 = 2 nodes; combined with HN3 (4) = 8.
        # ----------------------------------------------------------------
        shop_hn2_in_system = state.herbivore_nests.get(2, 0) >= 1
        if has_hn3 and (shop_hn2_in_system or state.horns >= 30) and hn_nodes < 8:
            if has(ActionType.BUY_HERBIVORE_NEST) and free >= 1:
                return ActionType.BUY_HERBIVORE_NEST

        # ----------------------------------------------------------------
        # Horn pile: grow and immediately attack triceratops from HN3 eggs.
        # Each trice drops horn_item_lvl1; the always-rules merge + feed them.
        # ----------------------------------------------------------------
        if has(ActionType.ATTACK_TRICERATOPS):
            return ActionType.ATTACK_TRICERATOPS

        if has(ActionType.GROW_TRICERATOPS):
            return ActionType.GROW_TRICERATOPS

        if has(ActionType.MERGE_TRICERATOPS_EGG):
            return ActionType.MERGE_TRICERATOPS_EGG

        # Build trice pipeline when HN3 is ready and we still need horns.
        if has_hn3 and state.horns < 30:
            action = self._build_one_trice(state, has, free)
            if action is not None:
                return action

        # ----------------------------------------------------------------
        # Continue stego waves to keep bone generation flowing.
        # Suppress GROW_STEGOSAURUS if a trice baby is waiting for a plant
        # so lvl4 plants aren't consumed before reaching lvl5.
        # ----------------------------------------------------------------
        trice_baby_waiting = (
            state.baby_herbivores[HerbivoreType.TRICERATOPS] > 0
            and state.plants.get(5, 0) == 0
        )
        return self._do_stego_wave(
            state, has, free, grid_full,
            suppress_grow_stego=trice_baby_waiting,
            suppress_refill=False,  # HN4 push needs stego refill for bone gen
        )

    def _build_one_trice(
        self,
        state: GameState,
        has,
        free: int,
    ) -> ActionType | None:
        """
        Advance one trice through the pipeline:
          2× HN3 eggs → baby trice → lvl5 plant → adult trice.
        Only spawns a new pipeline if fewer than 2 are already in-flight
        (eggs + baby), so we don't flood the grid.
        """
        trice_eggs  = state.herbivore_eggs[HerbivoreType.TRICERATOPS]
        trice_baby  = state.baby_herbivores[HerbivoreType.TRICERATOPS]
        lvl5_plants = state.plants.get(5, 0)

        in_flight = trice_eggs + trice_baby

        # Merge plants upward if a trice baby is waiting for lvl5.
        if trice_baby > 0 and lvl5_plants == 0:
            if has(ActionType.MERGE_PLANT):
                return ActionType.MERGE_PLANT
            if free > 0 and has(ActionType.SPAWN_PLANT):
                return ActionType.SPAWN_PLANT

        # Spawn a new trice egg pair if pipeline is empty and grid has room.
        if in_flight < 2 and free > FREE_SPACES_BUFFER and has(ActionType.SPAWN_HERBIVORE_EGG):
            return ActionType.SPAWN_HERBIVORE_EGG

        return None

    def _do_migration(
        self,
        state: GameState,
        has,
        free: int,
        grid_full: bool,
    ) -> ActionType | None:
        """
        Phase 3: HN4 is built. Unified egg-spam loop: spawn from HN4 and
        handle all three egg types as they arrive from RNG.

        HN4 egg distribution: 20% stego, 50% trice, 30% bronto.
        - Trices: grow and attack immediately (0 soup/turn; take damage + horns)
        - Brontos: grow and keep on board (5 soup/turn passive income)
        - Stegos: grow and wave-attack in batches of 8 for bones

        Triggers the CN1 purchase (Day 1 shop slot 2, gated in shop section).
        After CN1, buys ONE extra HN1 from the build menu so Phase 4 has a
        guaranteed stego-egg source for ptero food (once HN1 is on the grid,
        SPAWN_HERBIVORE_EGG uses HN1 first since it is cheaper than HN4).
        """
        n_stegos    = state.adult_herbivores[HerbivoreType.STEGOSAURUS]
        bronto_baby = state.baby_herbivores[HerbivoreType.BRONTOSAURUS]
        trice_baby  = state.baby_herbivores[HerbivoreType.TRICERATOPS]

        # Merge VP stations (better plants = cheaper pipelines).
        if has(ActionType.MERGE_VOLCANIC_PATCH):
            return ActionType.MERGE_VOLCANIC_PATCH

        # Attack trices immediately — no soup generation, use them for
        # damage and horns then free the grid space.
        if has(ActionType.ATTACK_TRICERATOPS):
            return ActionType.ATTACK_TRICERATOPS

        # Stego attack: fire when we have a reasonable batch (≥4 in Phase 3
        # rather than the strict 8 from Phase 1) so we don't WAIT idle when
        # soup is too low to spawn more eggs.  Lower threshold keeps bones
        # flowing and the T-Rex HP dropping continuously.
        if (n_stegos >= _MIGRATION_STEGO_WAVE_MIN
                and has(ActionType.ATTACK_STEGOSAURUS)):
            return ActionType.ATTACK_STEGOSAURUS

        # Complete creature pipelines in priority order.
        # Brontos first (each grown bronto permanently upgrades soup engine).
        if has(ActionType.GROW_BRONTOSAURUS):
            return ActionType.GROW_BRONTOSAURUS

        # Trices next — unless a bronto baby is waiting for plants needed
        # to merge upward to lvl6 (don't steal those plants for trice lvl5).
        bronto_needs_plant = bronto_baby > 0 and state.plants.get(6, 0) == 0
        if not bronto_needs_plant and has(ActionType.GROW_TRICERATOPS):
            return ActionType.GROW_TRICERATOPS

        # Stegos last — suppress ONLY if trice baby needs the lvl5 plant.
        # Do NOT suppress for bronto: bronto needs lvl6 (merged from two lvl5s),
        # so consuming a lvl4 for stego doesn't actually steal bronto's plant
        # unless we already have a lvl5 ready to merge.  More importantly,
        # when soup is low we can't spawn more plants anyway, so we should
        # use whatever plant is on the board to grow a stego and keep T-Rex
        # damage flowing rather than leaving it idle.
        suppress_stego = trice_baby > 0 and state.plants.get(5, 0) == 0
        if not suppress_stego and has(ActionType.GROW_STEGOSAURUS):
            return ActionType.GROW_STEGOSAURUS

        # Merge all egg types (all three can pile up from HN4 RNG).
        if has(ActionType.MERGE_BRONTOSAURUS_EGG): return ActionType.MERGE_BRONTOSAURUS_EGG
        if has(ActionType.MERGE_TRICERATOPS_EGG):  return ActionType.MERGE_TRICERATOPS_EGG
        if has(ActionType.MERGE_STEGOSAURUS_EGG):  return ActionType.MERGE_STEGOSAURUS_EGG

        # Merge plants upward (all pipelines benefit).
        if has(ActionType.MERGE_PLANT):
            return ActionType.MERGE_PLANT

        # Buy ONE extra HN1 after CN1 is purchased.
        # Once HN1 is on the grid, SPAWN_HERBIVORE_EGG uses HN1 (cheaper,
        # 2000 soup vs HN4's 2750) → 100% stego eggs for ptero food in Phase 4.
        cn1_purchased = sum(state.carnivore_nests.values()) > 0
        if (cn1_purchased
                and state.herbivore_nests.get(1, 0) == 0
                and state.big_bones >= _EXTRA_HN1_BONES
                and has(ActionType.BUY_HERBIVORE_NEST)
                and free >= 1):
            return ActionType.BUY_HERBIVORE_NEST

        # Spawn plants for waiting babies — gated by a soup floor so we
        # don't drain soup to zero and get stuck in a WAIT spiral.
        soup = state.primordial_soup
        if (bronto_needs_plant
                and free > 0
                and soup >= _MIGRATION_PLANT_SOUP_FLOOR
                and has(ActionType.SPAWN_PLANT)):
            return ActionType.SPAWN_PLANT
        if (trice_baby > 0 and state.plants.get(5, 0) == 0
                and free > 0
                and soup >= _MIGRATION_PLANT_SOUP_FLOOR
                and has(ActionType.SPAWN_PLANT)):
            return ActionType.SPAWN_PLANT

        # Spawn eggs from HN4 when we have headroom and a soup buffer.
        if (free > FREE_SPACES_BUFFER
                and soup >= _MIGRATION_EGG_SOUP_FLOOR
                and has(ActionType.SPAWN_HERBIVORE_EGG)):
            return ActionType.SPAWN_HERBIVORE_EGG

        # Spawn plants to keep any in-flight pipeline moving.
        in_pipeline = (
            sum(state.herbivore_eggs.values()) > 0
            or sum(state.baby_herbivores.values()) > 0
            or sum(state.plants.values()) > 0
        )
        if (in_pipeline
                and free > 0
                and soup >= _MIGRATION_PLANT_SOUP_FLOOR
                and has(ActionType.SPAWN_PLANT)):
            return ActionType.SPAWN_PLANT

        return None

    def _do_ptero_waves(
        self,
        state: GameState,
        has,
        free: int,
        grid_full: bool,
    ) -> ActionType | None:
        """
        Phase 4: CN1 + HN1 are on the grid.

        HN1 spawns 100% stego eggs (2,000 soup each) — dedicated ptero food.
        CN1 spawns ptero eggs (4,000 soup each).
        Pipeline: 2 ptero eggs → baby ptero + 1 adult stego → adult ptero.
        Attack in waves of _PTERO_WAVE_SIZE (8).

        Brontos from Phase 3 remain on the board for passive soup.
        Trices are attacked immediately; raptors (if CN2 unlocked) likewise.
        Stego attack wave is suppressed — stegos are consumed as ptero food;
        only genuinely excess stegos (beyond immediate ptero demand) are attacked.

        MERGE_PLANT fires only when the lowest mergeable pair is strictly below
        the minimum plant level needed by any waiting baby — this prevents
        accidentally merging lvl4+lvl4→lvl5 and destroying the only plants
        stegos can use (they require exactly lvl4, not "at least lvl4").

        CN2 can be purchased from the build menu (50 bones + 50 horns) once
        affordable; MERGE_CARNIVORE_NEST fires automatically thereafter.
        """
        n_pteros   = state.adult_carnivores[CarnivoreType.PTERODACTYL]
        n_stegos   = state.adult_herbivores[HerbivoreType.STEGOSAURUS]

        ptero_eggs = state.carnivore_eggs[CarnivoreType.PTERODACTYL]
        ptero_baby = state.baby_carnivores[CarnivoreType.PTERODACTYL]
        stego_eggs = state.herbivore_eggs[HerbivoreType.STEGOSAURUS]
        stego_baby = state.baby_herbivores[HerbivoreType.STEGOSAURUS]

        bronto_baby = state.baby_herbivores[HerbivoreType.BRONTOSAURUS]
        trice_baby  = state.baby_herbivores[HerbivoreType.TRICERATOPS]

        soup = state.primordial_soup

        # Round-down counts: 1 orphaned egg ≠ 1 committed creature.
        # Using // 2 (not (n+1)//2) prevents treating a lone egg as a
        # complete pair and stalling the pipeline waiting for a match.
        ptero_committed = ptero_eggs // 2 + ptero_baby + n_pteros
        stego_supply    = stego_eggs // 2 + stego_baby + n_stegos

        # Merge stations whenever a pair is available.
        if has(ActionType.MERGE_VOLCANIC_PATCH): return ActionType.MERGE_VOLCANIC_PATCH
        if has(ActionType.MERGE_CARNIVORE_NEST): return ActionType.MERGE_CARNIVORE_NEST

        # Attack trices immediately — no soup value, take damage + horns.
        if has(ActionType.ATTACK_TRICERATOPS): return ActionType.ATTACK_TRICERATOPS

        # Attack raptors immediately — high damage (400), no reason to hold.
        if has(ActionType.ATTACK_RAPTOR): return ActionType.ATTACK_RAPTOR

        # Fire ptero wave at full target, OR early if soup is exhausted and
        # nothing more can be built (no eggs in flight, can't afford CN egg).
        # An early wave frees grid space and earns score while we wait for soup
        # to recover; remaining stegos become food for the next batch.
        # "Stalled" = ptero pipeline can't advance without more soup:
        #   no baby to grow, and < 2 eggs (can't merge a lone orphan egg).
        ptero_stalled  = ptero_baby == 0 and ptero_eggs < 2
        soup_exhausted = soup < _PTERO_EGG_SOUP_FLOOR
        fire_early     = ptero_stalled and soup_exhausted and n_pteros >= _PTERO_WAVE_SIZE // 2
        if (n_pteros >= _PTERO_WAVE_SIZE or fire_early) and has(ActionType.ATTACK_PTERODACTYL):
            return ActionType.ATTACK_PTERODACTYL

        # Complete creature pipelines: ptero first (needs stego food),
        # then bronto (passive soup), then stego (creates ptero food supply).
        if has(ActionType.GROW_PTERODACTYL):  return ActionType.GROW_PTERODACTYL
        if has(ActionType.GROW_BRONTOSAURUS): return ActionType.GROW_BRONTOSAURUS
        if has(ActionType.GROW_STEGOSAURUS):  return ActionType.GROW_STEGOSAURUS

        # Grow trices (horns/damage) — don't steal plants bronto is waiting for.
        bronto_needs_plant = bronto_baby > 0 and state.plants.get(6, 0) == 0
        if not bronto_needs_plant and has(ActionType.GROW_TRICERATOPS):
            return ActionType.GROW_TRICERATOPS

        # Note: stegos are NOT attacked for bones in Phase 4 — they are reserved
        # as ptero food.  Any stego on the board will eventually be consumed by
        # GROW_PTERODACTYL.  Attacking them early creates a treadmill that
        # prevents the ptero pipeline from ever advancing to a full wave.

        # Merge egg types.
        if has(ActionType.MERGE_PTERODACTYL_EGG):  return ActionType.MERGE_PTERODACTYL_EGG
        if has(ActionType.MERGE_RAPTOR_EGG):        return ActionType.MERGE_RAPTOR_EGG
        if has(ActionType.MERGE_STEGOSAURUS_EGG):   return ActionType.MERGE_STEGOSAURUS_EGG
        if has(ActionType.MERGE_BRONTOSAURUS_EGG):  return ActionType.MERGE_BRONTOSAURUS_EGG
        if has(ActionType.MERGE_TRICERATOPS_EGG):   return ActionType.MERGE_TRICERATOPS_EGG

        # Smart plant merge: herbivore babies require EXACTLY their plant level
        # (not "at least"), so never merge a pair at the level a baby currently
        # needs.  Only merge if the lowest pair is strictly BELOW min baby need.
        min_plant_need = _min_plant_need(state)
        lowest_pair    = _lowest_plant_pair(state)
        if (has(ActionType.MERGE_PLANT)
                and lowest_pair is not None
                and lowest_pair < min_plant_need):
            return ActionType.MERGE_PLANT

        # Buy a second CN1 from the build menu to merge toward CN2.
        # Shop CN1 is already on the grid; build-menu CN1 costs 50 bones + 50 horns.
        cn2_built = state.carnivore_nests.get(2, 0) >= 1 or state.carnivore_nests.get(3, 0) >= 1
        if (not cn2_built
                and state.carnivore_nests.get(1, 0) >= 1
                and state.big_bones >= 50 and state.horns >= 50
                and has(ActionType.BUY_CARNIVORE_NEST) and free >= 1):
            return ActionType.BUY_CARNIVORE_NEST

        # Spawn plants only when a baby is waiting AND no plant of the exact
        # required level exists on the board.  Avoid over-spawning: just one
        # plant per baby need; MERGE_PLANT will build it up from lower levels.
        if stego_baby > 0 and state.plants.get(4, 0) == 0:
            if free > 0 and soup >= _PTERO_PLANT_SOUP_FLOOR:
                if has(ActionType.SPAWN_PLANT): return ActionType.SPAWN_PLANT
        if trice_baby > 0 and state.plants.get(5, 0) == 0:
            if free > 0 and soup >= _PTERO_PLANT_SOUP_FLOOR:
                if has(ActionType.SPAWN_PLANT): return ActionType.SPAWN_PLANT
        if bronto_needs_plant:
            if free > 0 and soup >= _PTERO_PLANT_SOUP_FLOOR:
                if has(ActionType.SPAWN_PLANT): return ActionType.SPAWN_PLANT

        # Spawn eggs: build ptero pipeline and keep stego food supply matched.
        if free > FREE_SPACES_BUFFER and soup >= _PTERO_EGG_SOUP_FLOOR:
            # Stego eggs first when ptero pipeline is ahead of food supply.
            if ptero_committed > stego_supply and has(ActionType.SPAWN_HERBIVORE_EGG):
                return ActionType.SPAWN_HERBIVORE_EGG
            # Advance ptero pipeline toward wave target.
            if ptero_committed < _PTERO_WAVE_SIZE and has(ActionType.SPAWN_CARNIVORE_EGG):
                return ActionType.SPAWN_CARNIVORE_EGG
            # Ensure food supply matches committed pteros.
            if stego_supply < ptero_committed and has(ActionType.SPAWN_HERBIVORE_EGG):
                return ActionType.SPAWN_HERBIVORE_EGG

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


def _has_max_level_item(item_dict: dict[int, int]) -> bool:
    return item_dict.get(MAX_CURRENCY_LEVEL, 0) >= 1


def _has_item_at_or_above(item_dict: dict[int, int], min_level: int) -> bool:
    return any(item_dict.get(lvl, 0) >= 1 for lvl in range(min_level, MAX_CURRENCY_LEVEL + 1))


def _any_bones(state: GameState) -> bool:
    return any(state.bone_items.get(lvl, 0) > 0 for lvl in range(1, MAX_CURRENCY_LEVEL + 1))


def _in_sprint(state: GameState) -> bool:
    return (_MAX_TURNS - state.turn) <= END_GAME_TURNS


def _has_mergeable_horn_pair(state: GameState) -> bool:
    return any(state.horn_items.get(lvl, 0) >= 2 for lvl in range(1, MAX_CURRENCY_LEVEL))


def _min_plant_need(state: GameState) -> int:
    """
    Return the minimum plant level needed by any baby currently waiting.
    Returns 7 (sentinel above MAX_PLANT_LEVEL) if no babies need plants.

    Used by the smart MERGE_PLANT guard: only merge if the lowest pair is
    strictly below this value (merging toward a needed level, not past it).
    """
    need = 7  # sentinel — no baby waiting
    if state.baby_herbivores[HerbivoreType.STEGOSAURUS] > 0:
        need = min(need, 4)
    if state.baby_herbivores[HerbivoreType.TRICERATOPS] > 0:
        need = min(need, 5)
    if state.baby_herbivores[HerbivoreType.BRONTOSAURUS] > 0:
        need = min(need, 6)
    return need


def _lowest_plant_pair(state: GameState) -> int | None:
    """
    Return the level of the lowest plant pair available for merging, or None.
    MERGE_PLANT always merges the lowest available pair.
    """
    from .constants import MAX_PLANT_LEVEL
    for lvl in range(1, MAX_PLANT_LEVEL):  # can't merge beyond max level
        if state.plants.get(lvl, 0) >= 2:
            return lvl
    return None
