from typing import Optional

from algorithms.astar import AStarAgent, SearchResult, action_cost, astar_search
from algorithms.heuristics import zero_heuristic


def ucs_search(problem, initial_state=None, cost_fn=action_cost) -> SearchResult:
    """Uniform Cost Search = A* with h(n) = 0.  Returns a ``SearchResult``."""
    return astar_search(problem, zero_heuristic,
                        initial_state=initial_state, cost_fn=cost_fn)


class UCSAgent(AStarAgent):
    """Agent that plans with UCS (A* with h = 0) and executes the plan.

    Inherits from ``BaseAgent`` through ``AStarAgent`` and implements
    ``select_action(state, metrics)``.
    """

    def __init__(self, problem, name: str = "UCSAgent"):
        super().__init__(problem, heuristic=zero_heuristic, name=name)