from itertools import product
from functools import reduce
from operator import mul, add
import config


class FleetAgent:
    """
    Heuristic-based agent for controlling autonomous vehicles.
    """

    def __init__(self, initial_config):
        self.config = initial_config
        self.city_grid = initial_config["city_grid"]
        self.active_vehicles = list(initial_config['vehicles'].keys())
        self.available_passengers = [
            p_id for p_id, info in initial_config['passengers'].items()
            if any(self.city_grid[pos[0]][pos[1]] == 'P'
                   for pos in info.get('possible_locations', [info['location']]))
        ]
        self.hazards_list = list(initial_config['hazards'].keys())

    def explore_valid_trajectories(self, position):
        directions = {
            'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1), 'stay': (0, 0)
        }
        rows, cols = len(self.city_grid), len(self.city_grid[0])
        return [
            (position[0] + dx, position[1] + dy)
            for direction, (dx, dy) in directions.items()
            if (0 <= position[0] + dx < rows and
                0 <= position[1] + dy < cols and
                self.city_grid[position[0] + dx][position[1] + dy] != 'I')
        ]

    def assess_traffic_threat(self, current_state, position):
        def calculate_single_threat(hazard_data):
            future_positions = self.get_next_hazard_positions(
                hazard_data['index'],
                hazard_data['trajectory']
            )
            return sum(prob for pos, prob in future_positions if pos == position)

        return reduce(add, map(calculate_single_threat, current_state['hazards'].values()), 0)

    def get_next_hazard_positions(self, idx, route):
        path_length = len(route)
        if path_length == 1:
            return [(route[0], 1.0)]

        position_range = (
            range(idx, idx + 2) if idx == 0
            else range(idx - 1, idx + 2) if 0 < idx < len(route) - 1
            else range(idx - 1, idx + 1)
        )
        probability = 0.5 if idx in (0, len(route) - 1) else 1 / 3
        return [(route[i], probability) for i in position_range if 0 <= i < len(route)]

    def act(self, state):
        waiting = {p_id: p for p_id, p in state['passengers'].items() if not p.get('served')}
        def get_optimal_maneuver(available_moves):
            def calculate_move_score(move):
                min_passenger_dist = min(
                    (abs(move[0] - p['location'][0]) + abs(move[1] - p['location'][1])
                     for p in waiting.values()),
                    default=0
                )
                threat_level = self.assess_traffic_threat(state, move)
                return (-min_passenger_dist, -threat_level)

            return max(available_moves, key=calculate_move_score)

        def process_vehicle(vehicle_id, vehicle_info):
            current_pos = vehicle_info['location']
            for p_id, p_info in waiting.items():
                if current_pos == p_info['location']:
                    return ("pickup", vehicle_id, p_id)

            moves = self.explore_valid_trajectories(current_pos)
            best_move = get_optimal_maneuver(moves)
            return ("wait", vehicle_id) if best_move == current_pos else ("move", vehicle_id, best_move)

        return tuple(process_vehicle(name, info) for name, info in state['vehicles'].items())


class OptimalFleetController:
    """
    MDP-based controller using Value Iteration to solve the stochastic
    fleet management problem optimally.
    """

    def __init__(self, initial_config):
        self.city_map = initial_config["city_grid"]
        self.time_horizon = initial_config["time_limit"]
        self.vehicles_info = [(v_id, data["location"]) for v_id, data in initial_config["vehicles"].items()]

        self.passenger_info = [
            (p_id, details["location"], details["possible_locations"], details["prob_change_location"])
            for p_id, details in initial_config["passengers"].items()]

        self.hazards_info = [(h_id, details["index"], details["trajectory"])
                             for h_id, details in initial_config["hazards"].items()]

        self.options = ["up", "down", "left", "right", "wait", "pickup"]
        if self.time_horizon > 1 and self.hazards_info and initial_config.get("allow_reset", True):
            self.options.extend(["reset", "terminate"])

        self.all_vehicle_actions = list(product(*[[(v_id, action) for action in self.options]
                                                  for v_id, _ in self.vehicles_info]))

        self.start_state = self.extract_initial_state(initial_config)
        self.state_catalog = self.build_state_space()
        self.state_values = {state: 0 for state in self.state_catalog}
        self.policy_map = {}

        print("Initializing Value Iteration for Fleet Optimization...")
        self.perform_value_iteration()
        print("Optimization Complete.")

    def extract_initial_state(self, system_data):
        def extract_vehicle_positions():
            return tuple((v_id, system_data["vehicles"][v_id]["location"]) for v_id, _ in self.vehicles_info)

        def extract_passenger_positions():
            return tuple((p_id, system_data["passengers"][p_id]["location"]) for p_id, _, _, _ in self.passenger_info)

        def extract_hazard_positions():
            return tuple(
                (h_id, system_data["hazards"][h_id]["trajectory"][system_data["hazards"][h_id]["index"]]) for h_id, _, _
                in self.hazards_info)

        return (extract_vehicle_positions(), extract_passenger_positions(), extract_hazard_positions())

    def build_state_space(self):
        valid_positions = [(x, y) for x in range(len(self.city_map)) for y in range(len(self.city_map[0])) if
                           self.city_map[x][y] == 'P']
        vehicle_states = [tuple(zip([v[0] for v in self.vehicles_info], positions)) for positions in
                          product(valid_positions, repeat=len(self.vehicles_info))]

        passenger_combinations = [list(p[2]) + [None] for p in self.passenger_info]  # None = served
        passenger_states = [tuple(zip([p[0] for p in self.passenger_info], locations)) for locations in
                            product(*passenger_combinations)]

        hazard_combinations = [h[2] for h in self.hazards_info]
        hazard_states = [tuple(zip([h[0] for h in self.hazards_info], locs)) for locs in product(*hazard_combinations)]

        return list(product(vehicle_states, passenger_states, hazard_states))

    def passenger_next_location(self, current_loc, possible_spots, change_prob):
        if current_loc is None:
            return [(None, 1.0)]
        locations_prob = {
            spot: (1 - change_prob + change_prob / len(possible_spots)) if spot == current_loc
            else change_prob / len(possible_spots)
            for spot in possible_spots
        }
        return list(locations_prob.items())

    def perform_value_iteration(self):
        discount_rate = 0.95
        max_rounds = min(self.time_horizon * 10, 1000)

        for iteration in range(max_rounds):
            adjustment = 0
            new_value_set = {}
            for state in self.state_catalog:
                previous_value = self.state_values.get(state, 0)
                best_value = float('-inf')
                optimal_action = None

                for current_action in self.all_vehicle_actions:
                    if not self.verify_valid_action(state, current_action): continue

                    next_states, probabilities = self.calculate_next_state_distribution(state, current_action)
                    immediate_reward = self.calculate_reward(state, current_action)

                    if next_states is None:
                        total_expected = immediate_reward
                    else:
                        future_rewards = sum(prob * self.state_values.get(next_state, 0) for next_state, prob in
                                             zip(next_states, probabilities))
                        total_expected = immediate_reward + discount_rate * future_rewards

                    if total_expected > best_value:
                        best_value = total_expected
                        optimal_action = current_action

                new_value_set[state] = best_value
                self.policy_map[state] = optimal_action
                adjustment = max(adjustment, abs(best_value - previous_value))

            self.state_values = new_value_set
            if adjustment < 1e-4: break

    def verify_valid_action(self, state_info, action_set):
        vehicle_positions, passenger_statuses, _ = state_info
        active_vehicle_locations = dict(vehicle_positions)
        passenger_locations = dict(passenger_statuses)
        for v_id, chosen_action in action_set:
            if chosen_action == "pickup" and active_vehicle_locations[v_id] not in passenger_locations.values():
                return False
        if chosen_action in self.MOVES:
            row, col = active_vehicle_locations[v_id]
            d_row, d_col = self.MOVES[chosen_action]
            if not self.is_passable((row + d_row, col + d_col)):
                return False
        return True

    MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def is_passable(self, position):
        row, col = position
        return (0 <= row < len(self.city_map) and 0 <= col < len(self.city_map[0])
                and self.city_map[row][col] == 'P')

    def calculate_next_state_distribution(self, current_state, selected_action):
        if any(op == "reset" for _, op in selected_action): return [self.start_state], [1.0]
        if any(op == "terminate" for _, op in selected_action): return None, [1.0]

        updated_vehicles = self.get_new_vehicle_positions(current_state[0], selected_action)
        vehicle_locations = dict(current_state[0])
        passengers_after_pickup = list(current_state[1])
        for v_id, op in selected_action:
            if op == "pickup":
                for i, (p_id, loc) in enumerate(current_state[1]):
                    if loc == vehicle_locations[v_id]:
                        passengers_after_pickup[i] = (p_id, None)
                        break
        passenger_moves = [self.passenger_next_location(loc, self.passenger_info[idx][2], self.passenger_info[idx][3])
                           for idx, (_, loc) in enumerate(passengers_after_pickup)]

        hazard_moves = []
        for i, (_, current_loc) in enumerate(current_state[2]):
            h_path = self.hazards_info[i][2]
            try:
                idx = h_path.index(current_loc)
                n = len(h_path)
                if n == 1:
                    moves = [(h_path[0], 1.0)]
                elif idx == 0:
                    moves = [(h_path[0], 0.5), (h_path[1], 0.5)]
                elif idx == n - 1:
                    moves = [(h_path[n - 2], 0.5), (h_path[n - 1], 0.5)]
                else:
                    moves = [(h_path[idx - 1], 1 / 3), (h_path[idx], 1 / 3), (h_path[idx + 1], 1 / 3)]
                hazard_moves.append(moves)
            except ValueError:
                hazard_moves.append([(current_loc, 1.0)])

        combined_states = []
        state_probabilities = []
        for passenger_outcomes in product(*passenger_moves):
            for hazard_outcomes in product(*hazard_moves):
                pass_pos_set = tuple((id_, loc) for (id_, _), (loc, _) in zip(current_state[1], passenger_outcomes))
                haz_pos_set = tuple((id_, loc) for (id_, _), (loc, _) in zip(current_state[2], hazard_outcomes))
                prob = reduce(mul, [p for _, p in passenger_outcomes] + [p for _, p in hazard_outcomes], 1.0)
                combined_states.append((updated_vehicles, pass_pos_set, haz_pos_set))
                state_probabilities.append(prob)
        return combined_states, state_probabilities

    def get_new_vehicle_positions(self, vehicles_info, action_sequence):
        updated = []
        for v_id, position in vehicles_info:
            for name, directive in action_sequence:
                if v_id == name and directive in ["up", "down", "left", "right"]:
                    row, col = position
                    d = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
                    updated.append((v_id, (row + d[directive][0], col + d[directive][1])))
                elif v_id == name:
                    updated.append((v_id, position))
        return tuple(updated)

    def calculate_reward(self, state, vehicle_actions):
        ops = [op for _, op in vehicle_actions]
        if "reset" in ops:
            return config.RESET_COST
        if "terminate" in ops:
            return 0
        reward = config.PASSENGER_PICKUP_REWARD * ops.count("pickup")
        new_positions = dict(self.get_new_vehicle_positions(state[0], vehicle_actions))
        hazard_cells = set(dict(state[2]).values())
        reward += sum(config.COLLISION_PENALTY for pos in new_positions.values() if pos in hazard_cells)
        return reward

    def act(self, state):
        current_state = self.state_from_info(state)
        action = self.policy_map.get(current_state)
        if not action: return tuple(("wait", v_id) for v_id, _ in self.vehicles_info)

        if any(act in ["terminate", "reset"] for _, act in action): return action[0][1]  # Return string action

        game_actions = []
        v_pos = dict((n, state["vehicles"][n]["location"]) for n, _ in self.vehicles_info)
        p_pos = dict((n, state["passengers"][n]["location"]) for n, _, _, _ in self.passenger_info
                     if not state["passengers"][n].get("served"))
        for v_name, act in action:
            if act == "pickup":
                target = next((p for p, loc in p_pos.items() if loc == v_pos[v_name]), None)
                if target: game_actions.append(("pickup", v_name, target))
            elif act in ["up", "down", "left", "right"]:
                x, y = v_pos[v_name]
                d = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
                game_actions.append(("move", v_name, (x + d[act][0], y + d[act][1])))
            else:
                game_actions.append(("wait", v_name))
        return tuple(game_actions)

    def state_from_info(self, info):
        v = tuple((n, info["vehicles"][n]["location"]) for n, _ in self.vehicles_info)
        p = tuple((n, None if info["passengers"][n].get("served") else info["passengers"][n]["location"])
                  for n, _, _, _ in self.passenger_info)
        h = tuple((n, info["hazards"][n]["trajectory"][info["hazards"][n]["index"]]) for n, _, _ in self.hazards_info)
        return (v, p, h)