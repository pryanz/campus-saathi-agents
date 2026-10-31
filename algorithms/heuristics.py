"""
Heuristic functions for A* search on the campus-reporting problem.

Problem recap
-------------
A state is ``(room, completed_steps)``.  Two kinds of actions exist:

* ``("move", n)``      costs the edge weight w(room, n)  (travel time)
* ``("complete", r)``  costs ``steps_config[r]["mean_service_time"]``

The goal is reached when every room in ``problem.goal_steps`` is completed.
Waiting time is random (queues) and therefore NOT part of the deterministic
search cost; it is treated as 0 here.

Every heuristic h(state) below estimates the cheapest cost-to-go to a goal.

Course terminology used in this file
------------------------------------
* Admissible : h(n) <= h*(n) for every state n (never overestimates).
* Consistent : h(n) <= c(n, a, n') + h(n') for every transition, and
               h(goal) = 0.  Consistency implies admissibility and guarantees
               that A* with a closed set (graph search) is optimal and never
               needs to re-open a node.
* Dominance  : if h2(n) >= h1(n) for all n and both are admissible, A* with h2
               expands no more nodes than A* with h1 (up to tie-breaking).

Formal proofs are in ``docs/algorithms/astar.md``.

Complexity of evaluating a heuristic
------------------------------------
Construction : O(V * (E log V)) once (all-pairs shortest paths via Dijkstra).
Evaluation   : O(|remaining|) time, O(1) extra space, per state.
"""

from typing import FrozenSet, Tuple

State = Tuple[str, FrozenSet[str]]

INF = float("inf")


def zero_heuristic(state: State) -> float:
    """h(n) = 0.  Trivially admissible and consistent.

    Plugging this into A* turns it into Uniform Cost Search (UCS).
    """
    return 0.0


class _CampusHeuristic:
    """Shared pre-computation for the two campus heuristics.

    Pre-computes, once per problem:

    * ``dist[u][v]``  shortest-path travel time between every pair of rooms.
    * ``service[r]``  mean service time of every step that must be completed.
    """

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

    # -- helpers shared by both heuristics ---------------------------------
    def _remaining(self, state: State) -> set:
        """Steps that are still to be completed in ``state``."""
        _, completed = state
        return self.problem.goal_steps - completed

    def _remaining_service(self, remaining) -> float:
        """Sum of service times still to be paid.

        Exact, not just a bound: every remaining step must be completed
        exactly once, and each completion costs exactly its service time.
        """
        return sum(self.service[r] for r in remaining)

    def _farthest_remaining(self, room: str, remaining) -> float:
        """Shortest travel time from ``room`` to the farthest unfinished room."""
        return max((self.dist[room][r] for r in remaining), default=0.0)

    def __call__(self, state: State) -> float:  # pragma: no cover
        raise NotImplementedError


class FarthestRemainingRoomHeuristic(_CampusHeuristic):
    """Simple heuristic: walk to the farthest unfinished room + all service.

        h(room, D) = max_{x in R} d(room, x)  +  sum_{x in R} service(x)

    where R = goal_steps \\ D is the set of unfinished steps and d is the
    shortest-path travel time.

    Properties (proved in docs/algorithms/astar.md)
        * Admissible : every plan must physically reach each x in R, so travel
          is at least the distance to the farthest one; service cost is paid
          exactly once per remaining step.
        * Consistent : triangle inequality on d for moves; completing the
          current room does not change the max term (d(room, room) = 0).
    """

    def __call__(self, state: State) -> float:
        room, _ = state
        remaining = self._remaining(state)
        if not remaining:
            return 0.0
        return (self._farthest_remaining(room, remaining)
                + self._remaining_service(remaining))


class ClusterAwareHeuristic(_CampusHeuristic):
    """Cluster-aware heuristic with inter-cluster crossing penalties.

    The campus is partitioned into clusters (neutral / academic / hostel).
    To finish, the agent must visit every cluster that still contains an
    unfinished room.  Each cluster other than the current one must be *entered*
    at least once, and each such first entry is a distinct crossing edge whose
    cost is at least ``min_entry[cluster]`` (the cheapest edge entering that
    cluster from a different cluster).  Hence

        X(room, D) = sum_{c in C_rem \\ {cluster(room)}} min_entry[c]

    is a lower bound on travel time, where C_rem is the set of clusters that
    contain at least one unfinished room.

    The final heuristic takes the stronger of the two travel bounds:

        h(room, D) = max( farthest(room, R), X(room, D) )  +  sum service(R)

    The crossing penalties are NOT added on top of the farthest-room distance:
    the shortest path to the farthest room may already contain those crossing
    edges, so adding both would double-count them and break admissibility.

    Properties (proved in docs/algorithms/astar.md)
        * Admissible and consistent (max of two consistent travel bounds).
        * Dominates FarthestRemainingRoomHeuristic: h_cluster >= h_simple, so
          A* with this heuristic expands no more nodes (up to ties).
    """

    def __init__(self, problem):
        super().__init__(problem)
        graph = problem.graph
        self.cluster = {r: graph.nodes[r]["cluster"] for r in self.rooms}

        # min_entry[c] = cheapest edge that enters cluster c from another one.
        # adj_list is symmetric, so scanning every directed pair (u, v) with
        # cluster(u) != cluster(v) covers entries into cluster(u) via (v -> u).
        self.min_entry = {}
        for u, neighbours in graph.adj_list.items():
            for v, w in neighbours.items():
                cu, cv = self.cluster[u], self.cluster[v]
                if cu != cv:
                    self.min_entry[cu] = min(self.min_entry.get(cu, INF), w)

    def crossing_bound(self, room: str, remaining) -> float:
        """X: sum of the cheapest entry edge of each cluster still to be entered."""
        here = self.cluster[room]
        to_enter = {self.cluster[r] for r in remaining} - {here}
        # .get(c, 0.0) keeps the bound safe (lower) for unreachable clusters.
        return sum(self.min_entry.get(c, 0.0) for c in to_enter)

    def __call__(self, state: State) -> float:
        room, _ = state
        remaining = self._remaining(state)
        if not remaining:
            return 0.0
        travel_bound = max(self._farthest_remaining(room, remaining),
                           self.crossing_bound(room, remaining))
        return travel_bound + self._remaining_service(remaining)