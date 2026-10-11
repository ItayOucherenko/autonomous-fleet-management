"""Benchmark scenarios. '#' = building, '.' = road."""


def grid(rows):
    return [['I' if ch == '#' else 'P' for ch in row] for row in rows]


def fixed(cell):
    """A passenger who never moves."""
    return {"location": cell, "possible_locations": (cell,), "prob_change_location": 0.0}


SCENARIOS = {
    # The car starts inside a U-shaped wall; all passengers are above it.
    # Greedy distance says "stay"; the only way out is down and around.
    "maze_6x6": {
        "time_limit": 30,
        "city_grid": grid(["......",
                           ".####.",
                           ".#..#.",
                           ".#..#.",
                           ".#..#.",
                           "......"]),
        "vehicles": {"AV": {"location": (2, 2)}},
        "passengers": {"R1": fixed((0, 0)), "R2": fixed((0, 2)), "R3": fixed((0, 5))},
        "hazards": {"Patrol": {"index": 0, "trajectory": [(5, 1), (5, 2), (5, 3), (5, 4)]}},
    },
    # Open grid, two moving passengers, two hazards crossing the centre.
    "rush_hour_5x5": {
        "time_limit": 25,
        "city_grid": grid([".....",
                           ".#.#.",
                           ".....",
                           ".#.#.",
                           "....."]),
        "vehicles": {"AV": {"location": (4, 0)}},
        "passengers": {
            "R1": {"location": (0, 4), "possible_locations": ((0, 4), (0, 2), (2, 4)), "prob_change_location": 0.4},
            "R2": {"location": (4, 4), "possible_locations": ((4, 4), (2, 2)), "prob_change_location": 0.3},
        },
        "hazards": {"Bus": {"index": 0, "trajectory": [(2, 0), (2, 1), (2, 2), (2, 3)]},
                    "Taxi": {"index": 0, "trajectory": [(0, 2), (1, 2), (2, 2)]}},
    },
    # Two cars: the slowest to solve (~30 s), because states and joint actions multiply per car.
    "two_cars_4x4": {
        "time_limit": 20,
        "city_grid": grid(["....",
                           ".#..",
                           "..#.",
                           "...."]),
        "vehicles": {"AV1": {"location": (3, 0)}, "AV2": {"location": (0, 3)}},
        "passengers": {"R1": fixed((0, 0)), "R2": fixed((3, 3)), "R3": fixed((1, 2))},
        "hazards": {"Jam": {"index": 0, "trajectory": [(2, 0), (2, 1), (1, 0)]}},
    },
}

# Option B: no reset/terminate in the benchmark, so score = pickups x 2 - collisions
for scenario in SCENARIOS.values():
    scenario["allow_reset"] = False
