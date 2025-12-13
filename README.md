# Autonomous Fleet Management System

This project simulates an intelligent control system for a fleet of autonomous vehicles (AVs) operating in a dynamic, stochastic urban environment. It implements an Optimal Control policy using Markov Decision Processes (MDP) to manage ride requests and navigate through traffic hazards.

## Purpose

The primary goal of this project is to model real-world urban logistics where uncertainty is a key factor. Unlike static pathfinding problems, this system deals with stochastic elements: passengers may cancel or move (location uncertainty), and traffic hazards like congestion or erratic drivers behave probabilistically.

The purpose is to fit a control model that maximizes fleet efficiency (passenger pickups) while minimizing safety risks (collisions). The end goal is to demonstrate how **Reinforcement Learning** concepts, specifically **Value Iteration**, can be applied to solve complex, multi agent navigation problems in real-time.

## System Requirements

To run this project locally, ensure that your environment includes the following:
* **Python 3.8** or higher.
* **Standard Libraries only**: No external `pip install` is required. The project uses `itertools`, `functools`, `random`, and `copy` which are built-in.

## Structure

* **agents.py**: Contains the core logic for the fleet controllers:
    * `FleetAgent`: A baseline heuristic agent.
    * `OptimalFleetController`: The advanced agent using MDP and Value Iteration.
* **environment.py**: The simulation engine that handles state transitions, stochastic movement of entities, and the ASCII-based visualization loop.
* **config.py**: Centralized configuration for rewards, penalties, and simulation timing.
* **main.py**: The entry point. Defines the scenario (grid, vehicles, passengers) and executes the simulation.

## General Steps

1.  **Scenario Initialization**: Define the city grid, vehicle starting positions, passenger requests, and traffic hazards in `main.py`.
2.  **Policy Optimization**: Upon boot, the `OptimalFleetController` runs the Value Iteration algorithm to compute the optimal policy ($\pi^*$) for every possible state.
3.  **Simulation Loop**: The environment steps forward in time.
4.  **Action & Reaction**: The agent executes the optimal action, and the environment updates stochastically (passengers/hazards move).
5.  **Visualization**: The state is rendered to the terminal in real-time to allow qualitative evaluation of the agent's behavior.

## Environment & Dynamics

Before the agent can act, the environment models the stochastic nature of the city:
* **City Grid**: Modeled as a grid where tiles can be Passable (`P`) or Impassable (`I` - buildings).
* **Stochastic Passengers**: Passengers are not static targets. Each turn, there is a probability (defined in the scenario) that a passenger might move to a nearby coordinate or change their status, simulating real-world GPS drift or ride cancellations.
* **Traffic Hazards**: Hazards (representing congestion or dangerous drivers) move along predefined trajectories but with stochastic deviations. They can speed up, slow down, or reverse direction probabilistically, forcing the AV to maintain safety buffers.

## Algorithmic Control Approaches

The core of the project is the transition from a naive approach to an optimal mathematical model.

### 1. Heuristic Approach (`FleetAgent`)
The baseline agent calculates actions based on a greedy heuristic: minimizing Manhattan distance to the nearest passenger while avoiding immediate threats. While computationally cheap, this approach often fails in complex scenarios where long-term planning is required (e.g., waiting for a hazard to pass).

### 2. Optimal Control via MDP (`OptimalFleetController`)
We modeled the problem as a **Markov Decision Process (MDP)** tuple (S, A, P, R, γ):

* **State Space (S)**: A composite tuple of (Vehicle Locations, Passenger Locations, Hazard States).
* **Action Space (A)**: `{Up, Down, Left, Right, Wait, Pickup}`.
* **Reward Function (R)**: +2 for Pickup, -1 for Collision, -2 for Reset.
* **Bellman Equation**: We solved for the optimal value function V*(s) using Value Iteration:

    `V(s) = max_a Σ P(s'|s,a) * [R(s,a,s') + γ * V(s')]`

The algorithm iterates until the maximum change in value (Delta) drops below a convergence threshold (1e-4). The result is a policy that anticipates hazard movement and sacrifices immediate rewards for long-term safety and efficiency.
## Testing & Visualization

We validate the model by running scenarios where the heuristic agent fails (e.g., getting trapped by stochastic hazards) while the optimal agent succeeds.

**Visual Output:**
The project includes a terminal-based visualization engine:
* `A`: Autonomous Vehicle
* `P`: Passenger
* `!`: Traffic Hazard
* `@`: Active Interaction (Pickup or Collision)

The simulation displays the **Time Remaining** and **Current Score** in real-time, allowing for immediate visual verification of the fleet's performance.

## Author
**Itay Oucherenko**