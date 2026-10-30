# Environment Characteristics & Learning Specifications

## Environment Classification

- **Observability:** Partially Observable (queues are hidden or partially visible based on observation settings).
- **State Space:** Stochastic (service times and wait times vary based on background crowd dynamics).
- **Time Horizon:** Sequential (current routing choices heavily impact future queue delays).
- **Dynamics:** Dynamic (other background students move and consume counter capacity concurrently).
- **Environment Type:** Discrete (10 mins simulation ticks) and Multi-agent.

## Mitchell's (T, E, P) Framework Statements

### For Q-Learning (Reinforcement Learning Agent)

- **Task ($T$):** Navigating the campus reporting checklist and sequencing room visits to minimize total completion time under stochastic queue delays.
- **Performance Experience ($P$):** Mean and maximum completion time across 50 fixed-seed simulation runs.
- **Training Experience ($E$):** Direct online interaction episodes with the campus simulator, observing state transitions $(s, a, r, s')$ under varying epsilon-greedy exploration policies.

### For Value Iteration (MDP Planner)

- **Task ($T$):** Computing an optimal policy mapping states (location, completed set, cluster congestion bins) to the next reporting room.
- **Performance Experience ($P$):** Plan cost and optimality gap compared against deterministic benchmarks.
- **Training Experience ($E$):** Offline model-based computation utilizing known state transition probabilities derived from the simulator's service-time distribution and cluster congestion models.
