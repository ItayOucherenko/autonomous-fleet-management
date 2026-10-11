"""Compare the greedy agent and the MDP agent over many random episodes.

Usage:
    python3 benchmark.py                  # all scenarios, 1,000 episodes each
    python3 benchmark.py maze_6x6         # one scenario
    python3 benchmark.py maze_6x6 -n 200  # fewer episodes
"""
import argparse
import contextlib
import io
import random
import time
from copy import deepcopy

from agents import FleetAgent, OptimalFleetController
from environment import AutonomousCitySimulation
from scenarios import SCENARIOS


def run_episode(scenario, agent, seed):
    """One episode using the simulator's own rules (apply + environment_step), without animation."""
    random.seed(seed)
    sim = AutonomousCitySimulation.__new__(AutonomousCitySimulation)  # skip __init__: reuse the trained agent
    sim.initial_state = deepcopy(scenario)
    sim.state = deepcopy(scenario)
    sim.score = 0
    sim.agent = agent

    pickups = collisions = 0
    while sim.state['time_limit'] > 0:
        action = agent.act(deepcopy(sim.state))
        if action == "terminate":
            break
        served_before = sum(1 for p in sim.state['passengers'].values() if p.get('served'))
        sim.apply(action)
        if action == "reset":
            continue
        pickups += sum(1 for p in sim.state['passengers'].values() if p.get('served')) - served_before
        hazard_cells = [h['trajectory'][h['index']] for h in sim.state['hazards'].values()]
        collisions += sum(v['location'] in hazard_cells for v in sim.state['vehicles'].values())
        sim.environment_step()
    return sim.score, pickups, collisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("-n", "--episodes", type=int, default=1000)
    args = parser.parse_args()

    print(f"{'Scenario':15s} {'Agent':7s} {'Score':>6s} {'Pickups':>8s} {'Collisions':>11s} {'Solve time':>11s}")
    for name in args.scenarios:
        scenario = deepcopy(SCENARIOS[name])
        for label, agent_class in (("greedy", FleetAgent), ("MDP", OptimalFleetController)):
            start = time.time()
            with contextlib.redirect_stdout(io.StringIO()):  # hide the agent's start-up prints
                agent = agent_class(deepcopy(scenario))
            solve_time = time.time() - start

            results = [run_episode(scenario, agent, seed) for seed in range(args.episodes)]
            score, pickups, collisions = (sum(r[i] for r in results) / args.episodes for i in range(3))
            print(f"{name:15s} {label:7s} {score:6.2f} {pickups:5.2f}/{len(scenario['passengers'])} "
                  f"{collisions:11.2f} {solve_time:10.1f}s", flush=True)


if __name__ == "__main__":
    main()
