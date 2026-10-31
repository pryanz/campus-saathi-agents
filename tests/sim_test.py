from envi.campus_graph import CampusGraph
from envi.simulator import CampusSimulator
from agents.base_agent import BaseAgent
from core.problem import CampusProblem

class DummyAgent(BaseAgent):
    def __init__(self, problem):
        super().__init__(problem, name="DummyAgent")
        self.problem = problem

    def select_action(self, state, metrics):
        actions = self.problem.actions(state)
        if not actions:
            return None
        #Prefer completing a step if available, otherwise move to first neighbor
        for action in actions:
            if action[0] == "complete":
                return action
        return actions[0]

def test_simulation():
    print("Initializing CampusGraph...")
    graph = CampusGraph("data/campus_map.json")
    print(f"Loaded nodes: {list(graph.nodes.keys())}")

    print("\nInitializing CampusProblem & Simulator...")
    problem = CampusProblem(graph, steps_config={})
    simulator = CampusSimulator(
        campus_map_path="data/campus_map.json",
        steps_config_path="data/steps.json",
        seed=42
    )

    problem = CampusProblem(graph, steps_config=simulator.steps_config)
    agent = DummyAgent(problem)
    
    print("\nRunning simulation test...")
    summary = simulator.run_simulation(agent, max_steps=100)
    print("Simulation Summary Result:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    test_simulation()