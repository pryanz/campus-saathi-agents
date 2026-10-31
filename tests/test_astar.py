import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithms.astar import AStarAgent, astar_search, action_cost
from algorithms.heuristics import (ClusterAwareHeuristic,
                                   FarthestRemainingRoomHeuristic,
                                   zero_heuristic)
from algorithms.ucs import UCSAgent, ucs_search
from core.metrics import MetricsTracker
from core.problem import CampusProblem
from envi.campus_graph import CampusGraph

ROMANIA_EDGES = [
    ("Arad", "Zerind", 75), ("Arad", "Sibiu", 140), ("Arad", "Timisoara", 118),
    ("Zerind", "Oradea", 71), ("Oradea", "Sibiu", 151),
    ("Sibiu", "Fagaras", 99), ("Sibiu", "RimnicuVilcea", 80),
    ("Timisoara", "Lugoj", 111), ("Lugoj", "Mehadia", 70),
    ("Mehadia", "Drobeta", 75), ("Drobeta", "Craiova", 120),
    ("Craiova", "RimnicuVilcea", 146), ("Craiova", "Pitesti", 138),
    ("RimnicuVilcea", "Pitesti", 97),
    ("Fagaras", "Bucharest", 211), ("Pitesti", "Bucharest", 101),
]


def _romania_problem():
    names = sorted({n for u, v, _ in ROMANIA_EDGES for n in (u, v)})
    data = {
        "nodes": {n: {"cluster": "ro", "type": "city"} for n in names},
        "edges": [{"from": u, "to": v, "weight": w} for u, v, w in ROMANIA_EDGES],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        graph = CampusGraph(path)
    finally:
        os.remove(path)
    steps = {"Bucharest": {"cluster": "ro", "mean_service_time": 0.0}}
    return CampusProblem(graph, steps, start_room="Arad")


def _campus_problem():
    graph = CampusGraph(os.path.join(ROOT, "data", "campus_map.json"))
    with open(os.path.join(ROOT, "data", "steps.json")) as f:
        steps = json.load(f)
    return CampusProblem(graph, steps, start_room="gate")


def _replay(problem, plan):
    state, cost = problem.initial_state(), 0.0
    for action in plan:
        travel, service = action_cost(problem, state, action)
        nxt = problem.result(state, action)
        cost = problem.path_cost(cost, state, action, nxt,
                                 travel_time=travel, service_time=service)
        state = nxt
    return state, cost

def test_ucs_textbook_graph_optimal_cost():
    problem = _romania_problem()
    result = ucs_search(problem)
    assert result.cost == 418
    moves = [a[1] for a in result.plan if a[0] == "move"]
    assert moves == ["Sibiu", "RimnicuVilcea", "Pitesti", "Bucharest"]
    assert result.plan[-1] == ("complete", "Bucharest")


def test_ucs_plan_replays_to_goal_with_reported_cost():
    problem = _campus_problem()
    result = ucs_search(problem)
    final_state, cost = _replay(problem, result.plan)
    assert problem.goal_test(final_state)
    assert cost == result.cost

def test_astar_matches_ucs_on_textbook_graph():
    problem = _romania_problem()
    ucs = ucs_search(problem)
    for h in (FarthestRemainingRoomHeuristic(problem), ClusterAwareHeuristic(problem)):
        a = astar_search(problem, h)
        assert a.cost == ucs.cost == 418
        assert a.nodes_expanded <= ucs.nodes_expanded


def test_astar_same_optimal_cost_as_ucs_but_fewer_expansions_on_campus():
    problem = _campus_problem()
    ucs = ucs_search(problem)
    simple = astar_search(problem, FarthestRemainingRoomHeuristic(problem))
    cluster = astar_search(problem, ClusterAwareHeuristic(problem))

    assert simple.cost == ucs.cost
    assert cluster.cost == ucs.cost
    assert simple.nodes_expanded < ucs.nodes_expanded
    assert cluster.nodes_expanded < ucs.nodes_expanded
    # cluster-aware heuristic dominates the simple one
    assert cluster.nodes_expanded <= simple.nodes_expanded

    for res in (simple, cluster):
        final_state, cost = _replay(problem, res.plan)
        assert problem.goal_test(final_state)
        assert cost == res.cost

def _all_transitions(problem):
    """Enumerate every reachable (state, action, next_state, step_cost)."""
    start = problem.initial_state()
    seen, stack = {start}, [start]
    while stack:
        s = stack.pop()
        for a in problem.actions(s):
            n = problem.result(s, a)
            travel, service = action_cost(problem, s, a)
            yield s, a, n, travel + service
            if n not in seen:
                seen.add(n)
                stack.append(n)


def test_heuristics_are_consistent_and_zero_at_goal():
    """Consistency h(s) <= c(s,a,s') + h(s') on EVERY transition + h(goal) = 0
    implies admissibility (see docs/algorithms/astar.md)."""
    problem = _campus_problem()
    heuristics = [zero_heuristic,
                  FarthestRemainingRoomHeuristic(problem),
                  ClusterAwareHeuristic(problem)]
    for h in heuristics:
        for s, a, n, c in _all_transitions(problem):
            assert h(s) <= c + h(n) + 1e-9, (s, a)
            if problem.goal_test(s):
                assert h(s) == 0
    # dominance: cluster >= simple everywhere
    simple, cluster = heuristics[1], heuristics[2]
    for s, _, _, _ in _all_transitions(problem):
        assert cluster(s) >= simple(s)


def test_heuristics_never_exceed_true_cost_from_start():
    problem = _campus_problem()
    optimal = ucs_search(problem).cost
    s0 = problem.initial_state()
    assert FarthestRemainingRoomHeuristic(problem)(s0) <= optimal
    assert ClusterAwareHeuristic(problem)(s0) <= optimal


def test_agents_execute_plan_and_log_expansions():
    problem = _campus_problem()
    for agent in (UCSAgent(problem), AStarAgent(problem)):
        metrics = MetricsTracker()
        state, cost = problem.initial_state(), 0.0
        for _ in range(200):
            action = agent.select_action(state, metrics)
            if action is None:
                break
            travel, service = action_cost(problem, state, action)
            nxt = problem.result(state, action)
            cost = problem.path_cost(cost, state, action, nxt,
                                     travel_time=travel, service_time=service)
            state = nxt
        assert problem.goal_test(state)
        assert cost == ucs_search(problem).cost
        assert metrics.nodes_expanded == agent.last_result.nodes_expanded > 0


def test_agent_reset_clears_plan():
    problem = _campus_problem()
    agent = AStarAgent(problem)
    agent.select_action(problem.initial_state(), MetricsTracker())
    agent.reset()
    assert agent._plan == [] and agent.last_result is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")c