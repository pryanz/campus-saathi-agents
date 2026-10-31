"""
Uniform Cost Search (UCS) for the campus-reporting problem.

UCS is A* with the zero heuristic h(n) = 0: the frontier is ordered purely by
the path cost g(n), so the cheapest unexpanded node is always expanded next.
This module therefore reuses the single search core in ``astar.py`` instead of
duplicating the loop.

Properties
----------
* Completeness : yes, when the state space is finite (it is: room x subset of
  completed steps) or when every step cost is >= epsilon > 0.
* Optimality   : yes.  h = 0 is trivially admissible and consistent, so the
  first goal popped from the priority queue has minimum path cost.
* Time         : O(b^(1 + floor(C*/epsilon))) in general, where C* is the optimal
  cost and epsilon the smallest step cost.  Bounded here by O(|S| b log|S|)
  with |S| = |rooms| * 2^|steps| distinct states.
* Space        : O(b^(1 + floor(C*/epsilon))), bounded here by O(|S|).

UCS explores in "cost contours" with no sense of direction toward the goal, so
it generally expands more nodes than A* with a good heuristic.
"""

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