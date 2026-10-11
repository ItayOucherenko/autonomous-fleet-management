from environment import AutonomousCitySimulation
from scenarios import SCENARIOS

if __name__ == "__main__":
    scenario = dict(SCENARIOS["maze_6x6"], optimal=True)   # optimal=False shows the greedy agent getting stuck
    AutonomousCitySimulation(scenario).run()