# Autonomous Fleet Management under Uncertainty

A planning agent for autonomous vehicles on a city grid. Passengers move at random and traffic hazards drift along their routes. The agent models the task as a **Markov Decision Process (MDP)** and solves it with **value iteration**. It is compared with a greedy baseline over 1,000 random episodes per scenario.

## Results

`python3 benchmark.py` (seeds 0–999, reset disabled, score = 2 × pickups − collisions):

| Scenario | Agent | Score | Pickups | Collisions | Solve time |
| --- | --- | --- | --- | --- |------------|
| Maze 6×6 | Greedy | 0.00 | 0.00 / 3 | 0.00 | –          |
| | **MDP** | **6.00** | **3.00 / 3** | 0.00 | 0.5 s      |
| Rush hour 5×5 | Greedy | 3.34 | 2.00 / 2 | 0.65 | –          |
| | **MDP** | **3.94** | 1.97 / 2 | **0.00** | 13.2 s     |
| Two cars 4×4 | Greedy | 6.00 | 3.00 / 3 | 0.00 | –          |
| | MDP | 6.00 | 3.00 / 3 | 0.00 | 8.8 s      |

Solve time = value iteration at start-up, measured on a MacBook Air; it will differ on other machines.

- **Maze:** the car starts inside a U-shaped wall. Every greedy move increases the Manhattan distance, so the greedy agent waits all game. The MDP drives around the wall and serves all three passengers.
- **Rush hour:** both agents serve the passengers, but the MDP avoids every collision (18% higher score).
- **Two cars:** on an open map greedy is enough. The cost of planning grows fast: states and joint actions multiply with each vehicle.

## How it works

**Environment** (`environment.py`): a grid of roads (`P`) and buildings (`I`). Each step, the agent acts, then:
- each waiting passenger jumps with probability `prob_change_location` to one of their possible cells;
- each hazard moves one step back, stays, or moves forward along its route, uniformly.

Scoring: +2 per pickup (each passenger once), −1 when a vehicle ends its move on a hazard's cell, −2 per reset.

**Greedy agent** (`FleetAgent`): moves to the neighbouring cell closest to a waiting passenger (Manhattan distance), breaking ties by hazard risk.

**MDP agent** (`OptimalFleetController`):
- **State:** vehicle cells, passenger cells (or "served"), hazard cells.
- **Actions:** up, down, left, right, wait, pickup per vehicle (plus reset and terminate when enabled).
- **Transitions:** product of the independent passenger and hazard move probabilities.
- **Solver:** value iteration with γ = 0.95, stopping when the largest change in a sweep is below 1e-4:

  `V(s) = max_a [ R(s,a) + γ · Σ P(s'|s,a) · V(s') ]`

The policy is computed once at start-up and then looked up at every step.

## Run it

Python 3.8+, standard library only.

```bash
python3 main.py                      # animated run of one scenario in the terminal
python3 benchmark.py                 # results table above (~1.5 min)
python3 benchmark.py maze_6x6 -n 200 # one scenario, fewer episodes
```

## Files

| File | Contents |
| --- | --- |
| `agents.py` | Greedy agent and MDP agent (value iteration) |
| `environment.py` | Simulator: rules, random moves, terminal display |
| `scenarios.py` | Benchmark scenarios |
| `benchmark.py` | Runs both agents over many seeded episodes without animation |
| `main.py` | Example scenario with animation |
| `config.py` | Rewards, penalties, time limit, animation speed |

## Limitations

- **Planning, not learning:** the transition probabilities are given to the agent. In a real city they would have to be estimated from data (model-based RL) or learned directly (Q-learning).
- **One policy for a fixed-length episode:** the infinite-horizon policy ignores how much time is left. Backward induction would give the exact finite-horizon policy.
- **Scale:** exact value iteration grows exponentially with the number of vehicles. Larger fleets need decomposition per vehicle, online planning, or function approximation.
- **No vehicle-vehicle collisions** are modelled.

## Author

**Itay Oucherenko**
