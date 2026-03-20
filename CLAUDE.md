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
    └── simulator.py             # DinosnoresSimulator — step(), get_valid_actions()
```

---

## Game Rules (as modelled)

### Core Loop
- The player attacks a T-Rex to reduce its HP to 0, "waking" it.
- Each wake awards **score** and **currency** (Big Bones, Horns, Fangs).
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
- **Big Bones / Horns / Fangs** — spent to buy stations and purchase upgrades. Earned on each T-Rex wake-up.
- **Plants** — consumed to grow herbivore eggs into adults.

### Passive Generation (every turn)
- **Primordial Crater**: each instance generates soup based on its level (`[0, 7, 12, 19, 28]` per turn).
- **Beacon recharge**: 1 charge restored every 18 turns (reduced by Bye Bye Planet upgrade).
- Plants and eggs are **NOT** passively generated — they require explicit player actions.

---

## Creatures

### Herbivores (grow: egg + plants → adult; attack: deal damage; passive: generate soup each turn)
| Type        | Damage | Soup/turn | Plant lvl required |
|-------------|--------|-----------|--------------------|
| Stegosaurus | 10     | 3         | 4                  |
| Brontosaurus| 20     | 5         | 6                  |
| Triceratops | 50     | 0         | 5                  |

### Carnivores (grow: egg + 1 adult herbivore → adult; attack: deal damage only)
| Type        | Damage | Food (herbivore consumed) |
|-------------|--------|---------------------------|
| Pterodactyl | 100    | Stegosaurus               |
| Raptor      | 400    | Brontosaurus              |

### Beasts (summoned directly for soup; attack: deal damage only)
| Type       | Damage | Soup Cost | Unlock (wake-ups) |
|------------|--------|-----------|-------------------|
| Mammoth    | 200    | 5,000     | 10                |
| Saber Tooth| 500    | 15,000    | 40                |

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

**Starting state**: 1× Volcanic Patch lvl 1, 1× Herbivore Nest lvl 1 (given for free).

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
- Starts with 2 charges; max 2 charges; recharges 1 per 18 turns.
- **USE_BEACON**: halves current T-Rex HP (cannot fully kill it alone) + spawns 1 Meteor grid item.
- **FEED_METEOR**: remove 1 Meteor from grid; gain `soup_capacity / 20` soup.

---

## Upgrades (purchased with currency)

All costs are `(big_bones, horns, fangs)` tuples, index = current level.

| Upgrade         | Levels | Effect                                      | Costs per level                                              |
|-----------------|--------|---------------------------------------------|--------------------------------------------------------------|
| More Score      | 5      | +10% score per level                        | (20,10,0) (60,0,0) (0,0,50) (0,75,0) (0,0,100)             |
| Bye Bye Planet  | 3      | −3 turns beacon recharge per level          | (20,0,0) (0,40,0) (0,0,20)                                  |
| Sharper Fangs   | 3      | +25/+50/+100% dino damage                   | (20,0,0) (30,30,0) (0,0,50)                                 |
| Brutish Beasts  | 3      | +25/+50/+100% beast damage                  | (0,20,0) (25,0,10) (0,0,25)                                 |
| Greater Craters | 2      | +1/+2 soup per crater per tick              | (50,0,0) (0,50,0)                                           |
| Soup Stores     | 5      | +100k soup capacity per level               | (0,10,0) (20,0,0) (0,30,0) (0,25,25) (75,0,0)              |

---

## Simulator Interface

```python
from dinosnores.simulator import DinosnoresSimulator
from dinosnores.actions import ActionType

sim = DinosnoresSimulator(max_turns=2000, score_target=None)
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

---

## Known Approximations / TODOs

- **Next step**: continue training the RL agent with the updated reward shaping.
