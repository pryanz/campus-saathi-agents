"""
A* search for the campus-reporting problem.

Algorithm
---------
A* is best-first graph search that always expands the frontier node with the
smallest evaluation function

    f(n) = g(n) + h(n)

where g(n) is the cost of the path found so far and h(n) the heuristic
estimate of the remaining cost.  The frontier is a priority queue (binary heap
via ``heapq``); a *closed set* remembers already-expanded states so that no
state is expanded twice.  The goal test is applied when a node is *popped*
(not when it is generated), which is what makes the first goal popped optimal.

Properties
----------
* Completeness : yes.  The state space (room x subset of completed steps) is
  finite and every step cost is >= 0, so the search terminates and finds a goal
  whenever one is reachable.  (In this project all costs are >= 2, i.e. bounded
  below by epsilon > 0.)
* Optimality   : yes, provided h is *admissible* and *consistent*.  With a
  consistent h, f is non-decreasing along every path, so the first time a state
  is popped its g-value is already optimal and the closed set never discards a
  better path.  (Both heuristics in ``heuristics.py`` are consistent.)
* Time         : worst case O(b^d) in general; here bounded by
  O(|S| * b * log|S|) with |S| = |rooms| * 2^|steps| distinct states and
  b <= max_degree + 1 successors per state (heap operations cost log|S|).
  A good heuristic shrinks the *effective* branching factor b*.
* Space        : O(|S|) -- the frontier, closed set and parent map all grow
  with the number of generated states.

With ``h(n) = 0`` A* degenerates to Uniform Cost Search (see ``ucs.py``).
"""

import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import Callable, List, Optional, Tuple

from agents.base_agent import BaseAgent
from algorithms.heuristics import ClusterAwareHeuristic

INF = float("inf")


@dataclass
class SearchResult:
    """Outcome of one search run.

    plan            : list of actions from the initial state to a goal, or None
                      if no goal is reachable.
    cost            : total path cost g(goal) (``inf`` if no plan).
    nodes_expanded  : states popped from the frontier and expanded
                      (the goal state itself is not counted).
    nodes_generated : successor states pushed onto the frontier.
    """
    plan: Optional[List[Tuple[str, str]]]
    cost: float
    nodes_expanded: int
    nodes_generated: int
    final_state: Optional[tuple] = field(default=None)


def action_cost(problem, state, action) -> Tuple[float, float]:
    """Return ``(travel_time, service_time)`` of taking ``action`` in ``state``.

    * move     -> travel time = edge weight, no service
    * complete -> no travel, service time = mean_service_time of that step
    Waiting time is stochastic and is not part of deterministic search (0).
    """
    kind, target = action
    if kind == "move":
        room, _ = state
        return problem.graph.get_neighbors(room)[target], 0.0
    if kind == "complete":
        return 0.0, problem.steps_config[target]["mean_service_time"]
    raise ValueError(f"Unknown action type: {kind!r}")


def _reconstruct(parent, state) -> List[Tuple[str, str]]:
    """Walk parent pointers back from ``state`` and return the action list."""
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
    """Run A* graph search and return a :class:`SearchResult`.

    :param problem:       object with initial_state, actions, result, goal_test
                          and path_cost (e.g. ``core.problem.CampusProblem``).
    :param heuristic:     callable h(state) -> float.  Must be admissible AND
                          consistent for the result to be optimal.
    :param initial_state: start the search here instead of
                          ``problem.initial_state()`` (used for re-planning).
    :param cost_fn:       maps (problem, state, action) -> (travel, service).
    """
    start = problem.initial_state() if initial_state is None else initial_state

    tie = count()                      # unique tiebreaker: heap never compares states
    h_start = heuristic(start)
    # heap entry: (f, h, tiebreak, g, state).  On equal f, smaller h (deeper) first.
    frontier = [(h_start, h_start, next(tie), 0.0, start)]
    g_best = {start: 0.0}              # cheapest known g per state
    parent = {start: None}             # state -> (parent_state, action)
    closed = set()
    expanded = generated = 0

    while frontier:
        _f, _h, _t, g, state = heapq.heappop(frontier)

        if state in closed:            # stale duplicate entry
            continue

        if problem.goal_test(state):   # goal test on POP => optimal
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
    """Agent that plans once with A* and then executes the plan step by step.

    ``select_action(state, metrics)`` returns the next action of the plan.
    The plan is computed on the first call (or whenever the incoming state is
    not the one the plan expected, e.g. after ``reset()``); the number of
    expanded nodes of every planning run is logged with
    ``metrics.increment_node_expansion``.  Returns ``None`` once the goal is
    reached.
    """

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
        state = (state[0], frozenset(state[1]))      # make sure it is hashable
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