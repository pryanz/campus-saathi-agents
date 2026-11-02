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


def queue_aware_action_cost(problem, state, action, queue_waits: dict = None, observation_level: int = 2) -> Tuple[float, float, float]:
    # dynamic queue wait time add karenge
    kind, target = action
    if kind == "move":
        room, _ = state
        return problem.graph.get_neighbors(room)[target], 0.0, 0.0
    if kind == "complete":
        service = float(problem.steps_config[target]["mean_service_time"])
        wait = 0.0
        if observation_level > 0 and queue_waits:
            wait = float(queue_waits.get(target, 0.0))
        return 0.0, wait, service
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
            c_res = cost_fn(problem, state, action)
            if len(c_res) == 3:
                travel, wait, service = c_res
            else:
                travel, service = c_res
                wait = 0.0
            g2 = problem.path_cost(g, state, action, nxt,
                                   travel_time=travel, wait_time=wait, service_time=service)
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


class QueueAwareAStarAgent(BaseAgent):
    # queue-aware pathfinding agent jo live bheed dekh kar optimal order chunta hai
    def __init__(self, problem, observation_level: int = 2, name: str = "QueueAwareAStar"):
        super().__init__(problem, name=name)
        self.observation_level = observation_level
        self._target_sequence = []
        self._plan = []
        self.latest_obs = {}

    def _get_candidate_sequences(self, obs: dict):
        fork_acad = [('L-206', 'L-106')] if obs.get('L-206', 0) < obs.get('L-106', 0) else [('L-106', 'L-206')]
        acad_fwd = ['L-103', 'L-104', 'L-105', fork_acad[0][0], fork_acad[0][1]]
        acad_rev = [fork_acad[0][0], fork_acad[0][1], 'L-105', 'L-104', 'L-103']

        hostel_1 = ['panini_gate', 'office', 'room', 'shop']
        hostel_2 = ['panini_gate', 'shop', 'office', 'room']
        hostel_3 = ['panini_gate', 'room', 'office', 'shop']
        hostel_opts = [hostel_1, hostel_2, hostel_3]

        candidates = []
        for a in [acad_fwd, acad_rev]:
            for h in hostel_opts:
                candidates.append(['gate', 'lhtc_gate'] + a + h)

        for h in hostel_opts:
            for a in [acad_fwd, acad_rev]:
                candidates.append(['gate'] + h + ['lhtc_gate'] + a)

        return candidates

    def _score_sequence(self, seq: list, obs: dict) -> float:
        travel = 0.0
        wait = 0.0
        serv = 0.0
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i+1]
            travel += self.problem.graph.shortest_path_weight(u, v)
            serv += float(self.problem.steps_config[v]['mean_service_time'])
            if self.observation_level > 0:
                wait += obs.get(v, 0.0)
        return travel + wait + serv

    def select_action(self, state, metrics=None, observation: dict = None):
        if observation is not None:
            self.latest_obs = observation
        cur, comp = state
        comp_set = set(comp)
        goal_set = self.problem.goal_steps
        if goal_set.issubset(comp_set):
            return None

        # agar sequence nahi bani toh queue-aware sequence banayenge
        if not self._target_sequence:
            candidates = self._get_candidate_sequences(self.latest_obs)
            self._target_sequence = min(candidates, key=lambda s: self._score_sequence(s, self.latest_obs))

        # agla uncompleted room dhundenge
        rem = [r for r in self._target_sequence if r not in comp_set]
        if not rem:
            return None
        target = rem[0]

        if cur == target:
            return ('complete', cur)

        path = self._bfs(cur, target)
        if path:
            return ('move', path[0])
        return None

    def _bfs(self, start: str, goal: str) -> list:
        from collections import deque
        if start == goal:
            return []
        queue = deque([[start]])
        visited = {start}
        while queue:
            p = queue.popleft()
            node = p[-1]
            if node == goal:
                return p[1:]
            for nbr in self.problem.graph.get_neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(p + [nbr])
        return []

    def reset(self):
        self._target_sequence = []
        self._plan = []
        self.latest_obs = {}
