from environment import AutonomousCitySimulation

if __name__ == "__main__":
    # Define Scenario
    scenario = {
        "optimal": True,
        "time_limit": 20,
        "city_grid": [
            ['I', 'I', 'P'],
            ['P', 'I', 'P'],
            ['I', 'P', 'P'],
            ['P', 'I', 'P']
        ],
        "vehicles": {'AV_Alpha': {"location": (2, 2)}},
        "passengers": {
            'Request_1': {"location": (0, 2),
                          "possible_locations": ((0, 2), (1, 2), (2, 2)),
                          "prob_change_location": 0.9},
            'Request_2': {"location": (0, 0),
                          "possible_locations": ((0, 0), (2, 0)),
                          "prob_change_location": 0.45}
        },
        "hazards": {
            'Congestion_A': {"index": 1, "trajectory": [(1, 0), (2, 1)]},
            'Erratic_Driver_B': {"index": 0, "trajectory": [(3, 2)]}
        },
    }

    # Run Simulation
    sim = AutonomousCitySimulation(scenario)
    sim.run()