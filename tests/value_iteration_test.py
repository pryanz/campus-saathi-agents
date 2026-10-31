import numpy as np
import pytest

from agents.base_agent import BaseAgent
from algorithms.value_iteration import ValueIterationAgent
from core.problem import CampusProblem
from envi.simulator import CampusSimulator


@pytest.fixture(scope="module")
def problem():
    simulator = CampusSimulator("data/campus_map.json", "data/steps.json", seed=42)
    return CampusProblem(simulator.graph, simulator.steps_config)


@pytest.fixture(scope="module")
def agent(problem):
    return ValueIterationAgent(problem)


def completed_from_mask(agent, mask):
    return frozenset(room for room in agent.rooms if mask >> agent.room_bit[room] & 1)


def test_agent_is_a_base_agent(agent):
    assert isinstance(agent, BaseAgent)


def test_state_space_size(agent):
    expected = len(agent.nodes) * 2 ** len(agent.rooms) * 3
    assert agent.values.shape == (len(agent.nodes), 2 ** len(agent.rooms), 3)
    assert agent.state_space_size == expected == 67584


def test_congestion_dynamics_are_valid(agent):
    for cluster, matrix in agent.tick_matrix.items():
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert np.isclose(agent.occupancy[cluster].sum(), 1.0)
        assert (agent.occupancy[cluster] > 0).all()
        assert np.all(np.diff(agent.level_load[cluster]) > 0)


def test_value_iteration_converges(agent):
    assert agent.converged
    assert agent.iterations < agent.max_iterations
    next_values, _ = agent._backup(agent.values)
    assert np.abs(next_values - agent.values).max() < agent.epsilon


def test_same_seed_gives_same_solution(problem, agent):
    other = ValueIterationAgent(problem)
    assert np.array_equal(other.values, agent.values)
    assert np.array_equal(other.policy, agent.policy)


def test_more_congestion_is_never_better(agent):
    start = agent.node_index["gate"]
    low, medium, high = agent.values[start, 0]
    assert low >= medium >= high
    assert low > high


def test_goal_state_has_zero_value_and_no_action(problem, agent):
    done = (agent.nodes[0], frozenset(problem.steps_config))
    assert agent.select_action(done) is None
    assert agent.value(done) == 0.0


def test_actions_are_always_legal(problem, agent):
    rng = np.random.default_rng(7)
    checked = 0
    while checked < 300:
        room = agent.nodes[rng.integers(len(agent.nodes))]
        mask = int(rng.integers(agent.num_masks))
        level = int(rng.integers(3))
        completed = completed_from_mask(agent, mask)
        if problem.goal_test((room, completed)):
            continue
        action = agent.select_action((room, completed, level))
        assert action in problem.actions((room, completed))
        checked += 1


def test_policy_reaches_goal_from_start(problem, agent):
    state = problem.initial_state()
    for _ in range(100):
        action = agent.select_action(state)
        if action is None:
            break
        state = problem.result(state, action)
    assert problem.goal_test(state)


def test_policy_completes_every_step_in_simulator(agent):
    simulator = CampusSimulator("data/campus_map.json", "data/steps.json", seed=42)
    summary = simulator.run_simulation(agent, max_steps=500)
    full_service = sum(cfg["mean_service_time"] for cfg in simulator.steps_config.values())
    assert summary["service_time"] == full_service
    assert summary["total_cost"] > full_service
