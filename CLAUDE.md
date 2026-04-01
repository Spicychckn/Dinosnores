# Dinosnores Simulator — Project Context

## Goal
Build a reinforcement learning project in two phases:
1. **Simulator** (current phase) — models the Dinosnores mini-game as a discrete-turn MDP with a clean `step(state, action) → (next_state, reward, done, info)` interface.
2. **RL Agent** (next phase) — train a model to play the game using the simulator.

Wiki reference: https://necromerger.wiki.gg/wiki/Dinosnores

---

## Project Structure

```
Dinosnores/
├── CLAUDE.md
├── main.py                      # entry point / scratch testing
└── dinosnores/
    ├── __init__.py
    ├── constants.py             # all game constants, enums, stat tables
    ├── actions.py               # ActionType enum (every possible player action)
    ├── state.py                 # GameState dataclass + grid/station helpers
    ├── simulator.py             # DinosnoresSimulator — step(), get_valid_actions()
    └── env.py                   # DinosnoresEnv — Gymnasium wrapper + reward shaping
```

---

## Game Rules (as modelled)

### Core Loop
- The player attacks a T-Rex to reduce its HP to 0, "waking" it.
- Each wake awards **score** only.
- The T-Rex resets with higher max HP each wake-up (+25 HP per wake).
- Score formula: `(50 + wake_ups × 10) × (1 + more_score_bonus)`

### Grid
- **32 total spaces.**
- The **Alien Beacon** permanently occupies 1 space.
- Every station instance, plant, egg, adult creature, and beast each occupies **1 space**.
- Spawn and summon actions require at least 1 free grid space.
- Grow and attack actions always free space (never blocked by grid).

### Resources
- **Primordial Soup** — spent to spawn plants/eggs and summon beasts. Capped at `soup_capacity` (base 100,000).
- **Big Bones / Horns / Fangs** — spent to buy stations and purchase upgrades. Earned by feeding currency items (dropped by creatures on attack) to the T-Rex.
- **Plants** — consumed to grow herbivore eggs into adults.

### Passive Generation (every turn)
- **Primordial Crater**: each instance generates soup based on its level (`[0, 7, 12, 19, 28]` per turn).
- **Beacon recharge**: 1 charge restored every 1,080 game turns (3 real-time hours at 10 s/turn; reduced by Bye Bye Planet upgrade).
- Plants and eggs are **NOT** passively generated — they require explicit player actions.

---

## Creatures

### Herbivores (pipeline: 2 matching eggs → baby; baby + plant → adult; attack: deal damage; passive: generate soup each turn)
| Type        | Damage | Soup/turn | Plant lvl required |
|-------------|--------|-----------|--------------------|
| Stegosaurus | 10     | 3         | 4                  |
| Brontosaurus| 20     | 5         | 6                  |
| Triceratops | 50     | 0         | 5                  |

### Carnivores (pipeline: 2 matching eggs → baby; baby + 1 adult herbivore → adult; attack: deal damage only)
| Type        | Damage | Food (herbivore consumed) |
|-------------|--------|---------------------------|
| Pterodactyl | 100    | Stegosaurus               |
| Raptor      | 400    | Brontosaurus              |

### Beasts (summoned directly for currency; attack: deal damage only)
| Type       | Damage | Summon Cost   | Unlock (wake-ups) |
|------------|--------|---------------|-------------------|
| Mammoth    | 200    | 30 Big Bones  | 10                |
| Saber Tooth| 500    | 30 Horns      | 40                |

Damage is multiplied by upgrade bonuses (`sharper_fangs` for dinos, `brutish_beasts` for beasts).

---

## Stations

Stations are **bought at level 1** (spending currency) and **merged** to reach higher levels.
`level 1 + level 1 → level 2`, `level 2 + level 2 → level 3`, etc. Each merge frees 1 grid space.

| Station          | Max Level | Buy Cost (bones, horns, fangs) | Unlock (wake-ups) |
|------------------|-----------|--------------------------------|-------------------|
| Herbivore Nest   | 5         | (40, 0, 0)                     | 0                 |
| Volcanic Patch   | 5         | (20, 20, 0)                    | 5                 |
| Carnivore Nest   | 3         | (50, 50, 0)                    | 20                |
| Primordial Crater| 4         | (0, 0, 40) *(approx)*          | 30                |

**Starting state**: 1× Volcanic Patch lvl 1, 1× Herbivore Nest lvl 1, 1× Primordial Crater lvl 2 (given for free); 20,000 primordial soup; 1× adult Brontosaurus; 2× adult Triceratops.

### Spawn costs (soup per use, indexed by station level)
- Volcanic Patch: `[0, 500, 600, 700, 800, 1000]` → spawns 1 plant (level determined probabilistically by VP level; see table below)
- Herbivore Nest: `[0, 2000, 2250, 2500, 2750, 3000]` → spawns 1 egg (type determined probabilistically by nest level; see table below)
- Carnivore Nest: `[0, 4000, 4500, 5000]` → spawns 1 egg (type determined probabilistically by nest level; see table below)

Spawn actions use the **cheapest (lowest-level)** available station of the required type.

### Volcanic Patch plant probabilities (by VP level used)
| Level | Lvl 1 Plant | Lvl 2 Plant | Lvl 3 Plant |
|-------|-------------|-------------|-------------|
| 1     | 100%        | —           | —           |
| 2     | 70%         | 30%         | —           |
| 3     | —           | 100%        | —           |
| 4     | —           | 70%         | 30%         |
| 5     | —           | —           | 100%        |

### Carnivore Nest egg probabilities (by nest level used)
| Level | Pterodactyl | Raptor |
|-------|-------------|--------|
| 1     | 100%        | —      |
| 2     | 50%         | 50%    |
| 3     | —           | 100%   |

### Herbivore Nest egg probabilities (by nest level used)
| Level | Stegosaurus | Triceratops | Brontosaurus |
|-------|-------------|-------------|--------------|
| 1     | 100%        | —           | —            |
| 2     | 70%         | 30%         | —            |
| 3     | 30%         | 70%         | —            |
| 4     | 20%         | 50%         | 30%          |
| 5     | —           | 30%         | 70%          |

### Alien Beacon
- Starts with 2 charges; max 5 charges; recharges 1 per 1,080 game turns (3 real-time hours).
- **USE_BEACON**: halves current T-Rex HP (cannot fully kill it alone) + spawns 1 Meteor grid item.
- **FEED_METEOR**: remove 1 Meteor from grid; gain `soup_capacity / 20` soup.

---

## Upgrades (purchased with currency)

All costs are `(big_bones, horns, fangs)` tuples, index = current level.

| Upgrade         | Levels | Effect                                      | Costs per level                                              |
|-----------------|--------|---------------------------------------------|--------------------------------------------------------------|
| More Score      | 5      | +10% score per level                        | (20,10,0) (60,0,0) (0,0,50) (0,75,0) (0,0,100)             |
| Bye Bye Planet  | 3      | −20/−40/−60 min beacon recharge (−120/−240/−360 turns) | (20,0,0) (0,40,0) (0,0,20)                         |
| Sharper Fangs   | 3      | +25/+50/+100% dino damage                   | (20,0,0) (30,30,0) (0,0,50)                                 |
| Brutish Beasts  | 3      | +25/+50/+100% beast damage                  | (0,20,0) (25,0,10) (0,0,25)                                 |
| Greater Craters | 2      | +1/+2 soup per crater per tick              | (50,0,0) (0,50,0)                                           |
| Soup Stores     | 5      | +100k soup capacity per level               | (0,10,0) (20,0,0) (0,30,0) (0,25,25) (75,0,0)              |

---

## Simulator Interface

```python
from dinosnores.simulator import DinosnoresSimulator
from dinosnores.actions import ActionType

sim = DinosnoresSimulator(max_duration_seconds=72*3600, score_target=None)
state = sim.reset()

valid_actions = sim.get_valid_actions(state)
next_state, reward, done, info = sim.step(state, ActionType.SPAWN_PLANT)
```

- `step()` does **not** mutate `state`; it returns a deep copy.
- `reward` = score delta for that turn (non-zero only on T-Rex wake-ups).
- `info` dict includes keys like `woke_trex`, `score_earned`, `damage_dealt`, `spawned`, `grew`, `soup_cost`.

---

## Reward Shaping Design (`dinosnores/env.py`)

The simulator's raw reward is score-delta only (non-zero on T-Rex wake-ups). `_shaped_reward`
adds dense intermediate signals to guide the agent through the pipeline. Key decisions:

| Signal | Value | Rationale |
|---|---|---|
| Per-turn survival cost | −0.005 | Penalises idle loops (e.g. spam-spawning plants while waiting for beacon recharge) |
| Egg spawned | +2.0 | Only eggs move the pipeline forward — **plant spawns give no reward** |
| Egg merged → baby | +2.0 | Reinforces completing the egg → baby step |
| Herbivore grown | +5 + `soup_production` | Stego=+8, Bronto=+10, Trice=+5 — soup producers are prioritised early because they compound into more spawns |
| Carnivore grown | +6.0 | Flat bonus; strong attacker unlocked after the herbivore economy is built |
| Damage dealt | `dmg × 0.005` | Directly incentivises attacking each turn, not just waiting for beacons |
| Meteor fed | +1.0 | Converts grid clutter to soup |
| Currency fed | +0.5 | Feeds currency items to advance the T-Rex wake |

**Why plant spawns are not rewarded**: early training showed the agent discovering that
cheap plant spawns (+500 soup) could farm shaped reward at 4× the rate of egg spawns.
This created a local optimum where 830+ turns were spent on plant spam vs. 10 grows,
with shaped reward (≈830) dwarfing actual score reward (350). Removing plant spawn
reward broke this loop.

### Reward Shaping History

**v2 changes (after run 1 — 10M steps, score 891 / 9 wake-ups):**

Run 1 revealed a beacon-wait loop: the model spent ~25% of all steps on beacon → meteor
→ wait cycles (8.6% beacon, 8.6% meteor, 8% WAIT). Root cause: `FEED_METEOR` at +1.0
nearly offset the wait cost (−0.005 × 1080 turns per recharge cycle = −5.4), making
the loop competitive with actually attacking. The model scored 891 vs. the heuristic's
4250.

| Change | Old | New | Reason |
|---|---|---|---|
| Survival cost | −0.005/turn | −0.01/turn | Stronger WAIT tax — any productive action earns it back; only idle WAITs pay the full cost with nothing in return |
| Damage multiplier | ×0.005 | ×0.01 | Stego attack wave: +0.4 → +0.8 per turn; makes sustained attacking clearly beat beacon-waiting |
| Meteor reward | +1.0 | +0.2 | Beacon cycle earns +1.0 total vs −5.4 wait cost — loop is no longer profitable |

**v3 changes (after run 2 — 10M steps, score 1325 / 11-13 wake-ups):**

Run 2 fixed the beacon-wait loop but the model still cycled through single stegos.
Root cause: no incentive to maintain a large army — the reward for growing 1 stego
and attacking it immediately was identical to growing 8 and batch-attacking. Added
a per-turn reward for passive soup generation rate to make army-building valuable.

| Change | Old | New | Reason |
|---|---|---|---|
| Passive soup reward | none | rate × 0.001/turn | Rewards the economic engine; 8-stego board earns +0.041/turn vs +0.020/turn for starting board — incentivises building and maintaining a large army before attacking |

**v4 changes (after run 3 — 10M steps, score 616 / 7 wake-ups):**

Run 3 regressed badly. The model discovered it could farm egg spawn (+2) and egg merge (+2)
rewards by spamming the pipeline without completing it. By turn 20595 the grid was full of
babies (0/32), the model was stuck, and score dropped from 1325 → 616. Same exploit as the
plant-spawn loop in run 1 — intermediate pipeline rewards can be farmed without completing
the pipeline.

| Change | Old | New | Reason |
|---|---|---|---|
| Egg spawn reward | +2.0 | removed | Exploited: model spammed eggs without growing adults |
| Egg merge reward | +2.0 | removed | Same exploit — farmed +4/baby without ever completing to adult |
| Stegosaurus grow reward | +8.0 (+5+3) | +15.0 | Compensates for lost intermediate rewards; can't be farmed (requires baby + plant) |
| Brontosaurus grow reward | +10.0 (+5+5) | +18.0 | Same rationale |
| Triceratops grow reward | +5.0 (+5+0) | +10.0 | Same rationale |
| Carnivore grow reward | +6.0 | +10.0 | Same rationale |

**Run 4 findings (10M+ steps, score 1610 / 14 wake-ups):**

v4 rewards fixed the baby-hoarding exploit. Pipeline completion improved dramatically
(89 grows/episode vs run 3's 1). Score beat run 2 (1610 vs 1325). However the model
still cycles — grow 1 stego, attack with it, repeat — rather than maintaining a
persistent army. Plant spawning and merging dominated at 57% of actions (the model
correctly identified these as necessary for grows, but spent far too much time on them).

Root cause of army-cycling: `n_steps=2048` with 8 envs means each PPO update only
sees 16,384 turns of experience — about 8% of a 25,920-turn episode and only ~2 beacon
cycles. The model never observes the compounding value of keeping an army alive across
multiple beacon cycles, so the passive soup reward (0.001/turn) is invisible on this
horizon. The run also plateaued: 4.3M checkpoint scored 1645 but best model (saved at
~8-9M steps) regressed to 1610.

**Run 5 changes:**

| Change | Old | New | Reason |
|---|---|---|---|
| n_steps | 2048 | 8192 | 4× larger rollouts; each env now sees ~32% of an episode per update, allowing GAE to estimate value across multiple beacon cycles and making the passive soup reward visible as a long-term signal |

Run 5 was also migrated from the original training machine (Intel i5-6500T, 4c/4t, 35W TDP) to
a dedicated desktop (AMD Ryzen 7 7700X 8c/16t + RTX 4070 Ti Super) running WSL2 on Windows.
The i5-6500T was thermal-throttling and crashing under sustained training load (3.5+ hours at
full CPU causing thermal shutdown at 100°C). The new machine provides significantly more headroom:
better cooling, 2× the cores, and GPU-accelerated network updates via CUDA. Training speed
improved from ~854 FPS to ~1,200 FPS on DummyVecEnv; switching to SubprocVecEnv (queued for
run 6) is expected to push this further by parallelising simulator workers across all 8 cores.

---

## Known Approximations / TODOs

**Run 5 findings (10M steps, score 260 / 4 wake-ups):**

Major regression. The longer n_steps horizon allowed the model to discover a new stable
policy: beacon the T-Rex to near-zero HP early (4 wake-ups), then spend the rest of the
episode farming passive soup reward with the T-Rex stuck at 1 HP. Root cause: meteor feed
reward (+0.2) made the beacon+meteor cycle competitive with attacking. The model learned
that consistently earning +0.2/meteor + passive soup was better shaped reward per turn than
risking waking the T-Rex to a higher HP tier.

**Run 6 changes:**

| Change | Old | New | Reason |
|---|---|---|---|
| Meteor feed reward | +0.2 | removed | Has caused beacon-farming loops in runs 1, 2, and 5. The soup from feeding a meteor is already intrinsically valuable — no shaped reward needed |
| Vec env | DummyVecEnv | SubprocVecEnv | Parallelises simulator workers across all CPU cores for faster data collection |

**Run 6 findings (10M steps, score 260 / 4 wake-ups):**

No improvement over run 5 despite removing the meteor reward. The model converged to a fully
deterministic beacon-snipe + grid-clog strategy: grow 1 stego early, snipe T-Rex HP via repeated
beacons, then spend the rest of the episode spawning plants/eggs without completing the pipeline.
By end-game the grid was full (0/32) with orphaned babies and unmerged plants, leaving the model
stuck doing beacon+wait forever. Score was identical across all 5 evaluation episodes (260 ± 0).

Root cause: `n_steps=8192` makes the beacon-snipe loop a stable attractor. With 65K turns per
update, a consistent long-horizon strategy is strongly reinforced once discovered. The model can
"plan" the full beacon-snipe sequence within a single rollout and optimise directly for it.

Comparison with run 4 (n_steps=2048, 1610/14 wake-ups):
- Run 4 grows: 445 (5.0%) — run 6 grows: 95 (3.4%)
- Run 4 attacks: 450 (5.0%) — run 6 attacks: 100 (3.5%)
- Run 4 beacon use: 1.5% — run 6 beacon use: 4.9%
- Run 4 grid at end: 21/32 (healthy) — run 6 grid at end: 0/32 (clogged)

**Run 7 changes:**

| Change | Old | New | Reason |
|---|---|---|---|
| n_steps | 8192 | 2048 | Revert to run 4 setting — shorter rollouts prevent beacon-snipe from becoming a stable attractor; more frequent updates allow faster correction of bad patterns |
| gae_lambda | 0.95 | 0.98 | Increases effective GAE horizon from ~20 steps to ~50 steps, allowing the advantage estimate to connect army-building actions to the eventual wave attack (a grow cycle is ~15-20 turns; 50 steps covers 3-4 cycles) without the instability of huge rollouts |

Hypothesis: n_steps=2048 + gae_lambda=0.98 gives the best of both worlds — frequent updates
that prevent loop convergence, combined with a longer advantage horizon that can see the value
of building a batch army before attacking.

**Run 7 findings (10M steps, score 260 / 4 wake-ups):**

Identical result to runs 5 and 6 — fully deterministic 260/4, zero variance across all 5 episodes.
The gae_lambda change had no effect. The model converges to the same beacon-snipe + grid-clog
local minimum regardless of the advantage horizon.

Root cause identified: PPO is catastrophically forgetting the BC initialisation. The BC pre-training
gives the model the heuristic strategy (4250/25), but PPO then corrupts it within the first few
million steps. The beacon-snipe loop is a stable attractor with lower variance than the grow-attack
chain — PPO gravitates toward it even though it earns far less total reward.

Reward source breakdown for the heuristic:
- Grow reward: 3615 (44%) — the dominant signal
- Score reward: 4250 (52%) — sparse, end-of-pipeline
- Soup rate: 295 (4%) — minor
- Attack reward: 67 (<1%)

The model is not close to earning these rewards — PPO is optimising a different objective (beacon
consistency) rather than improving on the BC baseline.

**Run 8 changes:**

| Change | Old | New | Reason |
|---|---|---|---|
| target_kl | None | 0.01 | Stop PPO updates early if KL divergence exceeds 0.01 — prevents any single update from throwing away the BC init; forces policy to improve incrementally rather than jumping to a new local minimum |
| ent_coef | 0.05 | 0.02 | Reduced exploration pressure: BC init already provides action diversity; high entropy was amplifying drift away from the heuristic strategy |

Goal: the model should treat the heuristic as a floor (≥4250) and explore from there, rather than
regressing to 260. target_kl keeps updates conservative so the value function can learn the true
value of the heuristic policy before the policy itself drifts away from it.

**Run 8 findings (10M steps, score 350 / 5 wake-ups):**

Marginal improvement over runs 5-7 (260/4) but still far below run 4 (1610/14). target_kl never
triggered — approx_kl stayed at 0.0015 throughout, meaning updates were already conservative
enough. The constraint didn't help because the problem is the DIRECTION of gradients (all pointing
toward beacon-snipe), not the SIZE of updates.

Key diagnostic: evaluated a BC-only model (50 episodes, 3 epochs, loss=0.4578) before any PPO.
It scored only 110/2 wake-ups. The BC pre-training is NOT sufficiently imitating the heuristic —
the model learns the rough shape (starts building the pipeline) but hasn't converged to the precise
heuristic strategy. PPO then starts from this weak foundation and drifts to beacon-snipe because:
1. The BC init isn't strong enough to anchor the policy
2. Every gradient step points toward beacon-snipe (consistent reward)
3. The pipeline requires a longer action chain before earning reward

BC loss comparison: loss=1.67→0.74→0.46 over 3 epochs. Still converging — needs more epochs.

**Run 9 changes:**

| Change | Old | New | Reason |
|---|---|---|---|
| pretrain_episodes | 50 | 200 | 4× more BC demonstrations — model sees more diverse game states and more complete wave cycles |
| bc_epochs | 3 | 10 | More passes over the BC data until loss converges; 3 epochs was clearly insufficient (loss still 0.46) |

Hypothesis: with a stronger BC foundation (BC model scoring closer to heuristic before PPO starts),
PPO will have a harder time drifting to beacon-snipe and should improve on the heuristic baseline
rather than regressing from it.
