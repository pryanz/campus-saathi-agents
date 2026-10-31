import numpy as np

from agents.base_agent import BaseAgent
from algorithms.value_iteration import ValueIterationAgent
from core.problem import CampusProblem
from envi.simulator import CampusSimulator


class NearestRoomAgent(BaseAgent):
    def __init__(self, problem):
        super().__init__(problem, name="NearestRoomAgent")

    def select_action(self, state, metrics=None):
        room, completed = state
        if self.problem.goal_test(state):
            return None
        if room in self.problem.steps_config and room not in completed:
            return ("complete", room)
        graph = self.problem.graph
        pending = [r for r in self.problem.steps_config if r not in completed]
        target = min(pending, key=lambda r: graph.shortest_path_weight(room, r))
        neighbors = graph.get_neighbors(room)
        step = min(neighbors, key=lambda n: neighbors[n] + graph.shortest_path_weight(n, target))
        return ("move", step)


def evaluate(agent, seeds):
    rows = []
    for seed in seeds:
        simulator = CampusSimulator("data/campus_map.json", "data/steps.json", seed=seed)
        rows.append(simulator.run_simulation(agent, max_steps=500))
    return {key: (np.mean([r[key] for r in rows]), np.std([r[key] for r in rows])) for key in rows[0]}


def main():
    simulator = CampusSimulator("data/campus_map.json", "data/steps.json", seed=42)
    problem = CampusProblem(simulator.graph, simulator.steps_config)
    solver = ValueIterationAgent(problem)

    print(f"states: {solver.state_space_size}")
    print(f"sweeps: {solver.iterations}, final change: {solver.residual:.2e}, converged: {solver.converged}")
    print(f"model value at start: {solver.value(problem.initial_state()):.2f}")

    seeds = range(20)
    for agent in (solver, NearestRoomAgent(problem)):
        stats = evaluate(agent, seeds)
        print(f"\n{agent.name} over {len(seeds)} seeds")
        for key in ("travel_time", "wait_time", "service_time", "total_cost"):
            mean, std = stats[key]
            print(f"  {key}: {mean:.2f} +/- {std:.2f}")


if __name__ == "__main__":
    main()
