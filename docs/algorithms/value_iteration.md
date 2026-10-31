# Value Iteration with Congestion States

Issue: #4. Code: `algorithms/value_iteration.py`. Evaluation: `experiments/eval_value_iteration.py`. Tests: `tests/value_iteration_test.py`.

## Model

A state is `(room, completed steps, congestion level)`.

- `room`: one of the 11 campus nodes.
- `completed steps`: any subset of the 11 steps in `steps.json`, stored as a bitmask.
- `congestion level`: low, medium or high, describing the cluster (academic, hostel or neutral) that the current room belongs to.

Actions are the ones `CampusProblem.actions` allows: move to a neighbouring room, or complete the step in the current room. Reward is the negative of time spent. A state with every step completed is terminal with value 0.

### State space size

11 rooms x 2^11 completed sets x 3 congestion levels = 67,584 states. The 33 states with all steps completed are terminal.

### Costs

- Move: the edge weight, deterministic, as in the simulator.
- Complete: expected queue wait plus `mean_service_time`. The service term matches what `run_simulation` adds to `service_time`. The wait term is `load x mean service ticks of the room`, where `load` is the representative queue length per counter for the current congestion level.

`RoomCounter` samples service time as `max(1, round(Gamma(2, mean/4)))`. The rounding and the floor of one tick raise the real mean above `mean/2` (for example 5.02 ticks for a configured mean of 10, 1.25 for a configured mean of 2). The solver computes this discretised mean exactly from the gamma CDF and uses it for both arrival rates and wait cost.

### Congestion dynamics

Transition probabilities are estimated from the simulator's own queue model rather than invented:

1. For each cluster, build one `RoomCounter` per room.
2. Run 20,000 ticks with a fixed seed. Each room receives Poisson arrivals at 0.8 of its capacity.
3. Cluster load per tick is the mean of `queue_length / num_counters` over its rooms. It is binned with edges 0.5 and 1.5.
4. Count level-to-level changes to get a per-tick matrix `P_c`.

A move of `w` ticks inside a cluster uses `P_c^w`. A completion lasting `d` ticks uses `P_c^d`. A move into a different cluster draws the new level from the destination cluster's long-run occupancy.

| cluster | occupancy (low, med, high) | mean load per level |
|---|---|---|
| academic | 0.315, 0.590, 0.095 | 0.28, 0.88, 1.84 |
| hostel | 0.274, 0.432, 0.293 | 0.20, 0.88, 2.83 |
| neutral | 0.419, 0.170, 0.411 | 0.00, 1.00, 4.36 |

## Bellman backup

With `gamma = 0.99` and `T_c(b' | b, k)` the k-tick transition of cluster `c`:

```
V(r, S, b) = max over actions a of Q(r, S, b, a)

Q(r, S, b, move r') = -w(r, r') + gamma * sum_b' T(b' | b, w) * V(r', S, b')

Q(r, S, b, complete) = -(wait(r, b) + service(r)) + gamma * sum_b' T(b' | b, d) * V(r, S + {r}, b')

V(r, all steps, b) = 0
```

The backup is vectorised with numpy: one matrix product per directed edge (36) and one per room (11) per sweep.

## Convergence

The solver stops when the sup-norm change between sweeps is below `epsilon * (1 - gamma) / (2 * gamma)`. This guarantees the greedy policy is epsilon-optimal: its value is within `epsilon` of the optimum.

With `epsilon = 1e-4` the threshold is 5.05e-7. The default run converges in 176 sweeps with a final change of 4.88e-7, about 3 seconds including the congestion estimation. Re-running with the same seed gives identical values and policy.

## Policy evaluation

Model value at the start state (expected negative time to finish all steps):

| start congestion | value |
|---|---|
| low | -144.43 |
| medium | -145.68 |
| high | -149.87 |

More congestion never improves the value, as expected.

The policy completes steps in this order: gate, L-103, L-206, L-106, L-105, L-104, lhtc_gate, panini_gate, shop, office, room. It needs 24 actions.

Simulator results over seeds 0 to 19, against a baseline that always walks to the nearest unfinished room:

| agent | travel | wait | service | total |
|---|---|---|---|---|
| ValueIterationAgent | 71.0 | 11.0 | 64.0 | 146.0 |
| NearestRoomAgent | 78.0 | 11.0 | 64.0 | 153.0 |

The model's -144.43 is close to the simulator's 146.0.

## Limitations

- `CampusSimulator.run_simulation` charges exactly one tick of wait per step and the configured mean for service, so results do not vary with the seed and the queue model is not exercised. The comparison above reflects routing only.
- The simulator passes `(room, completed)` without a congestion level. In that case `select_action` assumes the most likely level of the room's cluster. To use live congestion, pass a third element `level` in the state.
- Holding the level fixed at high while rolling the policy forward can make the agent wait by moving back and forth, because in the model the level would eventually change. This only happens when the level is pinned externally.
