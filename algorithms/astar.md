# Algorithm 1: Uniform Cost Search and A* with Heuristics

Files: `algorithms/ucs.py`, `algorithms/astar.py`, `algorithms/heuristics.py`
Tests: `tests/test_astar.py`

## 1. Problem formulation

| Component | Definition (from `core/problem.py`) |
|---|---|
| State | `s = (r, D)`: current room `r`, `D` = set of completed steps |
| Initial state | `(start_room, ∅)` |
| Actions | `("move", n)` for each neighbour `n` of `r`; `("complete", r)` if `r` has a step and `r ∉ D` |
| Transition | move: `(n, D)`; complete: `(r, D ∪ {r})` |
| Goal test | `G ⊆ D`, where `G = goal_steps` (every room that has a step) |
| Step cost | move: `w(r, n)` (travel); complete: `σ(r)` (mean service time); waiting time = 0 (random, not known to a deterministic planner) |

Notation: `R(s) = G \ D` are the unfinished steps; `S(s) = Σ_{x∈R(s)} σ(x)`; `d(u,v)` is the shortest-path travel time in the campus graph.
The state space has at most `|V| · 2^|G|` states (here 11 · 2^11 = 22 528).

**Lemma 0 (cost decomposition).** Every path from `s` to a goal costs `travel + S(s)`.
*Proof.* `complete` is only available for a step not yet in `D` and adds exactly that step, so each `x ∈ R(s)` is completed exactly once on the way to a goal and no other step can be completed. Service costs therefore sum to exactly `S(s)`. All other actions are moves. ∎
Consequently `h*(s) = T*(s) + S(s)`, where `T*(s)` is the minimum travel time needed to visit every room of `R(s)`.

## 2. Algorithm behaviour

**A\*** orders the frontier (a binary heap, `heapq`) by `f(n) = g(n) + h(n)`. Tie-break: smaller `h` first, then insertion order. A closed set holds expanded states. The goal test is done when a node is *popped*, not when it is generated; this is what makes the first goal returned optimal.

**UCS** is A\* with `h ≡ 0` (`ucs.py` calls the same search core), so it expands states in increasing order of `g`.

| Property | UCS | A\* (consistent `h`) |
|---|---|---|
| Complete | Yes: finite state space, step costs ≥ 0 (here ≥ 2) | Yes |
| Optimal | Yes | Yes (needs admissible **and** consistent `h` because a closed set is used) |
| Time | `O(b^(1+⌊C*/ε⌋))`; bounded by `O(|S| b log|S|)` | Same worst case; effective branching factor `b*` is smaller with a good `h` |
| Space | `O(b^(1+⌊C*/ε⌋))`; bounded by `O(|S|)` | `O(|S|)` |

`b` = branching factor (≤ max degree + 1), `C*` = optimal cost, `ε` = smallest step cost (2 on the campus map), `|S|` = number of states.

**Why the closed set is safe (standard result).** If `h` is consistent, then along any path `f` is non-decreasing: `f(n') = g(n) + c + h(n') ≥ g(n) + h(n) = f(n)`. Hence when A\* pops a state, no cheaper path to it can still be on the frontier, so its `g` is already optimal and it never needs to be re-opened. The first goal popped has `f = g = C*`.

## 3. Heuristics

### 3.1 Simple farthest-remaining-room heuristic

```
h_far(r, D) = max_{x ∈ R} d(r, x)  +  S
```
(with `h_far = 0` when `R = ∅`).

**Admissible.** Each `complete(x)` can only be executed while standing in room `x`, so any plan must walk to every `x ∈ R`. Its travel time is therefore at least `d(r, x)` for every `x ∈ R`, hence at least the maximum. By Lemma 0 the service part is exactly `S`. So `h_far(s) ≤ T*(s) + S(s) = h*(s)`. ∎

**Consistent.** Let `M(r, D) = max_{x∈R} d(r, x)`. Check every action `s → s'` with cost `c`:

* *move `r → n`* with `c = w(r,n)`. `D` and `S` are unchanged. For every `x ∈ R`, the triangle inequality gives `d(r,x) ≤ w(r,n) + d(n,x)`, so `d(r,x) ≤ w(r,n) + M(n,D)`. Taking the maximum over `x`: `M(r,D) ≤ w(r,n) + M(n,D)`. Hence `h(s) ≤ c + h(s')`.
* *complete `r`* with `c = σ(r)`. `R' = R \ {r}`, `S' = S − σ(r)`. The room `r` contributes `d(r,r) = 0`, the smallest possible value, to the maximum, so removing it leaves the maximum unchanged: `M' = M` (and if `R' = ∅`, then `M = M' = 0`). Therefore `h(s) = M + S = σ(r) + (M' + S') = c + h(s')`, i.e. the inequality holds with equality.

And `h(goal) = 0` because `R = ∅`. ∎ (Consistency + `h(goal)=0` implies admissibility.)

### 3.2 Cluster-aware heuristic

Rooms belong to clusters (`neutral`, `academic`, `hostel`). Let `cl(r)` be the cluster of room `r`, `C_rem = {cl(x) : x ∈ R}`, and `min_entry[c]` = the weight of the cheapest edge `(u,v)` with `cl(v) = c` and `cl(u) ≠ c`. On the campus map: `min_entry[neutral] = 15`, `min_entry[academic] = 15`, `min_entry[hostel] = 15`.

Inter-cluster crossing bound:

```
X(r, D) = Σ_{c ∈ C_rem \ {cl(r)}} min_entry[c]
```

Heuristic:

```
h_cl(r, D) = max( M(r, D), X(r, D) )  +  S
```

The crossing penalties are combined with the distance term using `max`, not `+`. The shortest path to the farthest room may already contain crossing edges; adding `X` on top of `M` would count them twice and could overestimate.

**Admissible.** The walk must visit every cluster in `C_rem`. For each such cluster `c ≠ cl(r)` the agent starts outside `c`, so it must at some point traverse an edge that *enters* `c` from another cluster, costing at least `min_entry[c]`. A traversal enters exactly one cluster, so these first entries are distinct traversals for distinct clusters, and their costs add up: `T*(s) ≥ X(s)`. Also `T*(s) ≥ M(s)` from 3.1. Therefore `T*(s) ≥ max(M, X)` and `h_cl(s) ≤ T*(s) + S(s) = h*(s)`. ∎

**Consistent.** `max` of consistent bounds is consistent, so it suffices to look at `g = max(M, X)` together with the service term, which behaves exactly as in 3.1. We already showed `M(s) ≤ c + M(s')` for moves and `M(s) = M(s')` for completes. For `X`:

* *move within a cluster*: `cl(r)` and `R` unchanged ⇒ `X(s) = X(s') ≤ c + X(s')`.
* *move `r → n` crossing from `a = cl(r)` to `b = cl(n)`*, `a ≠ b`. The two sums differ only in the terms for `a` and `b`: `X(s) − X(s') = [b ∈ C_rem]·min_entry[b] − [a ∈ C_rem]·min_entry[a] ≤ min_entry[b] ≤ w(r,n) = c`, because `(r,n)` is an edge entering `b` from another cluster.
* *complete `r`*: only `R` changes, and only by removing `r`, which belongs to the *current* cluster `cl(r)`; that cluster is excluded from the sum anyway, so `X(s) = X(s')`.

So `g(s) ≤ c + g(s')` for moves and `g(s) = g(s')` for completes; with `h = g + S` the same algebra as in 3.1 gives `h_cl(s) ≤ c + h_cl(s')`. `h_cl(goal) = 0`. ∎

**Dominance.** `h_cl ≥ h_far` everywhere (`max(M, X) ≥ M`). With two consistent heuristics, A\* using the larger one expands no more nodes than with the smaller one (up to tie-breaking), because any node with `f_far(n) < C*` expanded under `h_cl` is also expanded under `h_far`.

### 3.3 Empirical check

`tests/test_astar.py::test_heuristics_are_consistent_and_zero_at_goal` enumerates **every** reachable transition of the campus problem (22 528 states) and asserts `h(s) ≤ c + h(s')`, `h(goal) = 0` and `h_cl ≥ h_far`. This supports the proofs above.

## 4. Experimental results (campus map, start = `gate`, all 11 steps)

Optimal cost = **135** (travel 71 + service 64), identical for all three searches.

| Search | Nodes expanded | Nodes generated |
|---|---|---|
| UCS (`h = 0`) | 21 861 | 25 764 |
| A\* + farthest-remaining-room | 1 019 | 2 025 |
| A\* + cluster-aware | **977** | **1 791** |

Textbook check (Romania map, Arad → Bucharest): optimal cost 418 via Sibiu → Rimnicu Vilcea → Pitesti; UCS expands 13 nodes, A\* expands 5.

A\* expands about 21× fewer nodes than UCS on the campus problem while returning the same optimal plan. The cluster-aware heuristic is slightly better than the simple one because the crossing bound is the larger term in many states (for example 94 vs 89 at `lhtc_gate` with nothing completed).

## 5. Limitations

* Waiting time at counters is random and is not modelled in the deterministic search cost. The plan is optimal for travel + service only; the stochastic simulation may add waiting time on top.
* Optimality relies on the closed set + consistent `h`. A custom heuristic that is only admissible could return a sub-optimal plan with this implementation.
* The state space grows as `|V|·2^|G|`; it is fine for 11 steps (22 528 states) but UCS does not scale to many more steps.