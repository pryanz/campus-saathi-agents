import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import Callable, List, Optional, Tuple

from agents.base_agent import BaseAgent
from algorithms.heuristics import ClusterAwareHeuristic

INF = float("inf")


@dataclass
class SearchResult:
    plan: Optional[List[Tuple[str, str]]]
    cost: float
    nodes_expanded: int
    nodes_generated: int
    final_state: Optional[tuple] = field(default=None)


def action_cost(problem, state, action) -> Tuple[float, float]:
    kind, target = action
    if kind == "move":
        room, _ = state
        return problem.graph.get_neighbors(room)[target], 0.0
    if kind == "complete":
        return 0.0, problem.steps_config[target]["mean_service_time"]
    raise ValueError(f"Unknown action type: {kind!r}")


def _reconstruct(parent, state) -> List[Tuple[str, str]]:

    actions = []
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return actions


def astar_search(problem,
                 heuristic: Callable[[tuple], float],
                 initial_state=None,
                 cost_fn: Callable = action_cost) -> SearchResult:

    start = problem.initial_state() if initial_state is None else initial_state

    tie = count()                      
    h_start = heuristic(start)
    frontier = [(h_start, h_start, next(tie), 0.0, start)]
    g_best = {start: 0.0}             
    parent = {start: None}             
    closed = set()
    expanded = generated = 0

    while frontier:
        _f, _h, _t, g, state = heapq.heappop(frontier)

        if state in closed:            
            continue

        if problem.goal_test(state):  
            return SearchResult(_reconstruct(parent, state), g,
                                expanded, generated, state)

        closed.add(state)
        expanded += 1

        for action in problem.actions(state):
            nxt = problem.result(state, action)
            if nxt in closed:
                continue
            travel, service = cost_fn(problem, state, action)
            g2 = problem.path_cost(g, state, action, nxt,
                                   travel_time=travel, service_time=service)
            if g2 < g_best.get(nxt, INF):
                g_best[nxt] = g2
                parent[nxt] = (state, action)
                h2 = heuristic(nxt)
                heapq.heappush(frontier, (g2 + h2, h2, next(tie), g2, nxt))
                generated += 1

    return SearchResult(None, INF, expanded, generated)


class AStarAgent(BaseAgent):
    
    def __init__(self, problem, heuristic: Optional[Callable] = None,
                 name: str = "AStarAgent"):
        super().__init__(problem, name=name)
        self.heuristic = heuristic if heuristic is not None else ClusterAwareHeuristic(problem)
        self._plan: List[tuple] = []
        self._expected_state = None
        self.last_result: Optional[SearchResult] = None

    def _plan_from(self, state, metrics):
        result = astar_search(self.problem, self.heuristic, initial_state=state)
        self.last_result = result
        self._plan = list(result.plan) if result.plan else []
        if metrics is not None:
            metrics.increment_node_expansion(result.nodes_expanded)

    def select_action(self, state, metrics) -> Optional[tuple]:
        state = (state[0], frozenset(state[1]))     
        if self.problem.goal_test(state):
            return None
        if not self._plan or state != self._expected_state:
            self._plan_from(state, metrics)
        if not self._plan:
            return None
        action = self._plan.pop(0)
        self._expected_state = self.problem.result(state, action)
        return action

    def reset(self):
        self._plan = []
        self._expected_state = None
        self.last_result = None