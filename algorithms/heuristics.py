from typing import FrozenSet, Tuple

State = Tuple[str, FrozenSet[str]]

INF = float("inf")


def zero_heuristic(state: State) -> float:
    return 0.0


class _CampusHeuristic:
    def __init__(self, problem):
        self.problem = problem
        graph = problem.graph
        self.rooms = list(graph.nodes)
        self.dist = {
            u: {v: graph.shortest_path_weight(u, v) for v in self.rooms}
            for u in self.rooms
        }
        self.service = {
            r: problem.steps_config[r]["mean_service_time"]
            for r in problem.goal_steps
        }

    def _remaining(self, state: State) -> set:
        _, completed = state
        return self.problem.goal_steps - completed

    def _remaining_service(self, remaining) -> float:
        return sum(self.service[r] for r in remaining)

    def _farthest_remaining(self, room: str, remaining) -> float:
        """Shortest travel time from ``room`` to the farthest unfinished room."""
        return max((self.dist[room][r] for r in remaining), default=0.0)

    def __call__(self, state: State) -> float:  
        raise NotImplementedError


class FarthestRemainingRoomHeuristic(_CampusHeuristic):

    def __call__(self, state: State) -> float:
        room, _ = state
        remaining = self._remaining(state)
        if not remaining:
            return 0.0
        return (self._farthest_remaining(room, remaining)
                + self._remaining_service(remaining))


class ClusterAwareHeuristic(_CampusHeuristic):

    def __init__(self, problem):
        super().__init__(problem)
        graph = problem.graph
        self.cluster = {r: graph.nodes[r]["cluster"] for r in self.rooms}

        self.min_entry = {}
        for u, neighbours in graph.adj_list.items():
            for v, w in neighbours.items():
                cu, cv = self.cluster[u], self.cluster[v]
                if cu != cv:
                    self.min_entry[cu] = min(self.min_entry.get(cu, INF), w)

    def crossing_bound(self, room: str, remaining) -> float:
        here = self.cluster[room]
        to_enter = {self.cluster[r] for r in remaining} - {here}
        return sum(self.min_entry.get(c, 0.0) for c in to_enter)

    def __call__(self, state: State) -> float:
        room, _ = state
        remaining = self._remaining(state)
        if not remaining:
            return 0.0
        travel_bound = max(self._farthest_remaining(room, remaining),
                           self.crossing_bound(room, remaining))
        return travel_bound + self._remaining_service(remaining)