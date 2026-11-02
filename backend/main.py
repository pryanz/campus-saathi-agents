import json
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from envi.simulator import CampusSimulator
from core.problem import CampusProblem

from agents.q_learning_agent import QLearningAgent
from algorithms.bfs import BFSAgent
from algorithms.dfs import DFSAgent
from algorithms.greedy_nearest import GreedyNearestAgent
from algorithms.ids import IDSAgent
from algorithms.astar import AStarAgent, QueueAwareAStarAgent
from algorithms.ucs import UCSAgent, QueueAwareUCSAgent
from algorithms.value_iteration import ValueIterationAgent
from algorithms.simulated_annealing import SimulatedAnnealingAgent

app = FastAPI(title="Campus Saathi API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_Q_TABLE_CACHE = {}


def get_trained_q_agent(problem: CampusProblem, iterations: int = 100) -> QLearningAgent:
    global _Q_TABLE_CACHE
    agent = QLearningAgent(problem)

    if iterations in _Q_TABLE_CACHE:
        agent.q_table = dict(_Q_TABLE_CACHE[iterations])
        agent.epsilon = 0.0 if iterations >= 100 else max(0.05, 1.0 - (iterations / 75.0))
        agent.name = f"QLearningAgent ({iterations} iter)"
        return agent

    for _ in range(iterations):
        state = ("gate", frozenset())
        done = False
        step_count = 0
        while not done and step_count < 200:
            step_count += 1
            action = agent.select_action(state)
            if not action:
                break
            action_type, target = action
            room, completed = state
            if action_type == "move":
                next_state = (target, completed)
                reward = -2.0
            else:
                next_state = (room, completed | {target})
                reward = 50.0
            done = problem.goal_test(next_state)
            if done:
                reward += 100.0
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()

    _Q_TABLE_CACHE[iterations] = dict(agent.q_table)
    agent.epsilon = 0.0 if iterations >= 100 else max(0.05, 1.0 - (iterations / 75.0))
    agent.name = f"QLearningAgent ({iterations} iter)"
    return agent


@app.get("/api/map")
def get_campus_map():
    """Returns the campus nodes and edges layout from JSON."""
    try:
        with open("data/campus_map.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "campus_map.json not found in data directory."}


@app.get("/api/simulate/{agent_type}")
def run_simulation_api(
    agent_type: str,
    congestion: bool = Query(True),
    iterations: int = Query(100),
    start_location: str = Query("gate"),
    completed_steps: list[str] = Query(None),
    observation_level: int = Query(0),
):
    """
    Runs a simulation episode for the specified agent type 
    and returns the step-by-step trace and metrics.
    Supports congestion toggle, Q-learning iterations, custom starting location, completed steps,
    and observation_level (0: blind, 1: local, 2: global queues).
    """
    simulator = CampusSimulator("data/campus_map.json", "data/steps.json", seed=42, observation_level=observation_level)

    is_congestion = congestion if isinstance(congestion, bool) else True
    q_iterations = iterations if isinstance(iterations, int) else 100
    obs_level = observation_level if isinstance(observation_level, int) else 0
    start_room = start_location if isinstance(start_location, str) and start_location in simulator.graph.nodes else "gate"
    valid_completed = completed_steps if isinstance(completed_steps, (list, set, tuple, frozenset)) else None
    completed_fs = frozenset(valid_completed) if valid_completed else frozenset()

    problem = CampusProblem(simulator.graph, steps_config=simulator.steps_config, start_room=start_room)
    if valid_completed:
        problem.initial_state = lambda: (start_room, completed_fs)
    simulator.problem = problem

    agent_key = agent_type.lower().strip()

    q_iterations = iterations if isinstance(iterations, int) else 100
    if agent_key.startswith("q_learning"):
        parts = agent_key.replace("-", "_").split("_")
        if len(parts) > 2 and parts[-1].isdigit():
            q_iterations = int(parts[-1])
        agent_key = "q_learning"

    if agent_key == "q_learning":
        agent = get_trained_q_agent(problem, iterations=q_iterations)
    elif agent_key in ("queue_aware_a_star", "qa_a_star"):
        agent = QueueAwareAStarAgent(problem, observation_level=obs_level or 2)
    elif agent_key in ("queue_aware_ucs", "qa_ucs"):
        agent = QueueAwareUCSAgent(problem, observation_level=obs_level or 2)
    elif agent_key == "a_star":
        agent = QueueAwareAStarAgent(problem, observation_level=obs_level) if obs_level > 0 else AStarAgent(problem)
    elif agent_key == "ucs":
        agent = QueueAwareUCSAgent(problem, observation_level=obs_level) if obs_level > 0 else UCSAgent(problem)
    elif agent_key == "value_iteration":
        agent = ValueIterationAgent(problem)
    elif agent_key == "simulated_annealing":
        agent = SimulatedAnnealingAgent(problem)
    elif agent_key == "bfs":
        agent = BFSAgent(problem)
    elif agent_key == "dfs":
        agent = DFSAgent(problem)
    elif agent_key == "greedy_nearest":
        agent = GreedyNearestAgent(problem)
    elif agent_key == "ids":
        agent = IDSAgent(problem)
    else:
        return {
            "error": f"Agent type '{agent_type}' not supported yet. Choose from a_star, queue_aware_a_star, ucs, queue_aware_ucs, bfs, dfs, greedy_nearest, ids, simulated_annealing, q_learning, value_iteration."
        }

    return simulator.run_simulation(
        agent,
        max_steps=500,
        congestion=is_congestion,
        start_room=start_room,
        completed_steps=completed_fs,
        observation_level=obs_level
    )


from pydantic import BaseModel
from backend.groq_service import query_campus_helper


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


class PlannerRequest(BaseModel):
    arrival_time: str = "09:30 AM"
    has_heavy_luggage: bool = True
    category: str = "OPEN"


# arriving students ke liye 'what should i do first' planner route
@app.get("/api/planner/what-first")
def get_what_first(
    arrival_time: str = Query("09:30 AM"),
    has_heavy_luggage: bool = Query(True),
    category: str = Query("OPEN")
):
    from backend.task_planner import plan_first_steps
    return plan_first_steps(arrival_time=arrival_time, has_heavy_luggage=has_heavy_luggage, category=category)


@app.post("/api/planner/what-first")
def post_what_first(request: PlannerRequest):
    from backend.task_planner import plan_first_steps
    return plan_first_steps(
        arrival_time=request.arrival_time,
        has_heavy_luggage=request.has_heavy_luggage,
        category=request.category
    )


# student query aur conversation history handle karne ke liye route
@app.post("/api/chat")
def chat_campus_helper(request: ChatRequest):
    def simulation_callback(agent_type: str, congestion: bool = True, start_location: str = "gate", completed_steps: list = None):
        return run_simulation_api(
            agent_type=agent_type,
            congestion=congestion,
            iterations=100,
            start_location=start_location,
            completed_steps=completed_steps,
        )

    try:
        reply = query_campus_helper(request.query, request.history, simulation_callback)
        return {"response": reply}
    except Exception as e:
        return {"response": f"Error communicating with AI assistant: {str(e)}"}