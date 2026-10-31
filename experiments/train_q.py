from envi.simulator import CampusSimulator
from core.problem import CampusProblem
from agents.q_learning_agent import QLearningAgent

def run_training_experiment():
    simulator = CampusSimulator("data/campus_map.json", "data/steps.json", seed=42)
    problem = CampusProblem(simulator.graph, steps_config=simulator.steps_config)
    agent = QLearningAgent(problem)

    milestones = [10, 50, 60, 75, 90, 100, 200, 500]
    max_episodes = max(milestones)
    
    print(f"Starting Q-Learning training over {max_episodes} episodes...")

    for episode in range(1, max_episodes + 1):

        state = ("gate", frozenset())
        done = False
        step_count = 0
        max_steps = 300

        while not done and step_count < max_steps:
            step_count += 1
            action = agent.select_action(state)
            
            if not action:
                break

            action_type, target = action
            current_room, completed_steps = state
            
            if action_type == "move":
                next_state = (target, completed_steps)
                reward = -2.0 
            else:  # complete step
                next_state = (current_room, completed_steps | {target})
                reward = +50.0

            done = problem.goal_test(next_state)
            if done:
                reward += 100.0

            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()
        
        if episode in milestones:

            orig_eps = agent.epsilon
            agent.epsilon = 0.0
            
            eval_metrics = simulator.run_simulation(agent, max_steps=300)
            print(f"--- Milestone Episode {episode} ---")
            print(f"  Total Cost: {eval_metrics['total_cost']}")
            print(f"  Travel Time: {eval_metrics['travel_time']}")
            print(f"  Wait Time: {eval_metrics['wait_time']}")
            
            agent.epsilon = orig_eps

if __name__ == "__main__":
    run_training_experiment()