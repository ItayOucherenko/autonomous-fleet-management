import time
import random
import os
from copy import deepcopy
import config
from agents import FleetAgent, OptimalFleetController


class AutonomousCitySimulation:
    def __init__(self, initial_input):
        self.initial_state = deepcopy(initial_input)
        self.state = deepcopy(initial_input)
        self.score = 0

        print("System Boot: Initializing Fleet Controller...")
        if self.state['optimal']:
            self.agent = OptimalFleetController(self.state)
        else:
            self.agent = FleetAgent(self.state)

    def print_state(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = [row[:] for row in self.state['city_grid']]
        rows, cols = len(grid), len(grid[0])

        display = [['.' if cell == 'P' else '#' for cell in row] for row in grid]


        for name, info in self.state['passengers'].items():
            if info.get('served'):
                continue
            r, c = info['location']
            display[r][c] = 'P'

        for name, info in self.state['hazards'].items():
            r, c = info['trajectory'][info['index']]
            display[r][c] = '!'

        for name, info in self.state['vehicles'].items():
            r, c = info['location']
            display[r][c] = '@' if display[r][c] != '.' and display[r][c] != '#' else 'A'

        print(f"\n--- TIME REMAINING: {self.state['time_limit']} | SCORE: {self.score} ---")
        print("Legend: [A] Vehicle, [P] Passenger, [!] Hazard, [#] Obstacle, [.] Road\n")
        print("  " + " ".join(str(i) for i in range(cols)))
        for r in range(rows):
            print(f"{r} " + " ".join(display[r]))
        print("\n")
        time.sleep(config.ANIMATION_DELAY)

    def run(self):
        while self.state['time_limit'] > 0:
            self.print_state()
            action = self.agent.act(deepcopy(self.state))

            if action == "terminate":
                print("Controller requested termination.")
                break

            print(f"Controller Action: {action}")
            self.apply(action)
            if action != "reset":
                self.environment_step()

        print(f"Simulation Ended. Final Efficiency Score: {self.score}")

    def apply(self, action):
        if action == "reset":
            # FIX 2: reset restores positions, but the clock keeps running
            time_left = self.state['time_limit']
            self.state = deepcopy(self.initial_state)
            self.state['time_limit'] = time_left - 1
            self.score += config.RESET_COST
            return

        for atomic in action:
            act_type = atomic[0]
            if act_type == 'move':
                # FIX 4: ignore moves off the grid or into a building
                if self.is_passable(atomic[2]):
                    self.state['vehicles'][atomic[1]]['location'] = atomic[2]
            elif act_type == 'pickup':
                # FIX 1: a passenger can be picked up only once, at the vehicle's cell
                passenger = self.state['passengers'][atomic[2]]
                vehicle_loc = self.state['vehicles'][atomic[1]]['location']
                if not passenger.get('served') and passenger['location'] == vehicle_loc:
                    passenger['served'] = True
                    self.score += config.PASSENGER_PICKUP_REWARD

        v_locs = [v['location'] for v in self.state['vehicles'].values()]
        h_locs = [h['trajectory'][h['index']] for h in self.state['hazards'].values()]

        for v_loc in v_locs:
            if v_loc in h_locs:
                self.score += config.COLLISION_PENALTY

    def is_passable(self, position):
        grid = self.state['city_grid']
        r, c = position
        return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 'P'

    def environment_step(self):
        for p in self.state['passengers'].values():
            if random.random() < p['prob_change_location']:
                p['location'] = random.choice(p['possible_locations'])

        for h in self.state['hazards'].values():
            idx = h['index']
            path = h['trajectory']
            if len(path) > 1:
                options = [0, 1] if idx == 0 else ([idx, idx - 1] if idx == len(path) - 1 else [idx - 1, idx, idx + 1])
                h['index'] = random.choice(options)

        self.state['time_limit'] -= 1