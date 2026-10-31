import numpy as np

from agents.base_agent import BaseAgent
from core.counters import RoomCounter
from core.seeding import get_rng

LEVELS = ("low", "medium", "high")


class ValueIterationAgent(BaseAgent):
    """Model-based agent that solves the campus MDP with value iteration.

    A state is (room, completed steps, congestion level of the current room's
    cluster). Actions are moves to a neighbouring room or completing the step
    in the current room. The reward is the negative time spent. Congestion
    dynamics are estimated by simulating RoomCounter queues with a fixed seed.
    """

    def __init__(self, problem, gamma=0.99, epsilon=1e-4, max_iterations=2000,
                 seed=42, utilisation=0.8, load_edges=(0.5, 1.5), sim_ticks=20000):
        super().__init__(problem, name="ValueIterationAgent")
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.seed = seed
        self.utilisation = utilisation
        self.load_edges = np.array(load_edges, dtype=float)
        self.sim_ticks = sim_ticks

        self.nodes = sorted(problem.graph.nodes)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.rooms = sorted(problem.steps_config)
        self.room_bit = {room: i for i, room in enumerate(self.rooms)}
        self.num_masks = 1 << len(self.rooms)
        self.full_mask = self.num_masks - 1
        self.complete_code = len(self.nodes)
        self.cluster_of = {node: problem.graph.nodes[node]["cluster"] for node in self.nodes}

        self.tick_matrix = {}
        self.occupancy = {}
        self.level_load = {}
        for cluster in sorted(set(self.cluster_of.values())):
            self._estimate_cluster(cluster)
        self.default_level = {c: int(np.argmax(p)) for c, p in self.occupancy.items()}

        self._build_actions()
        self.solve()

    @property
    def state_space_size(self):
        return int(np.prod(self.values.shape))

    def _mean_service(self, room):
        scale = float(self.problem.steps_config[room].get("mean_service_time", 3.0)) / 4.0
        ticks = np.arange(1, 400)
        upper = ticks + 0.5
        cdf = 1 - np.exp(-upper / scale) * (1 + upper / scale)
        probs = np.diff(np.concatenate(([0.0], cdf)))
        return float((ticks * probs).sum())

    def _estimate_cluster(self, cluster):
        rooms = [r for r in self.rooms if self.cluster_of.get(r) == cluster]
        fallback = np.concatenate(([0.0], self.load_edges))
        if not rooms:
            self.tick_matrix[cluster] = np.eye(3)
            self.occupancy[cluster] = np.full(3, 1 / 3)
            self.level_load[cluster] = fallback
            return

        rng = get_rng(self.seed)
        counters = [RoomCounter(r, self.problem.steps_config) for r in rooms]
        rates = np.array([
            self.utilisation * c.num_counters / self._mean_service(c.room_id) for c in counters
        ])

        moves = np.zeros((3, 3))
        visits = np.zeros(3)
        load_sum = np.zeros(3)
        previous = None
        for _ in range(self.sim_ticks):
            for counter, arrivals in zip(counters, rng.poisson(rates)):
                for _ in range(arrivals):
                    counter.add_to_queue("visitor")
                counter.tick(rng)
            load = np.mean([c.queue_length() / c.num_counters for c in counters])
            level = int(np.searchsorted(self.load_edges, load, side="right"))
            visits[level] += 1
            load_sum[level] += load
            if previous is not None:
                moves[previous, level] += 1
            previous = level

        totals = moves.sum(axis=1, keepdims=True)
        self.tick_matrix[cluster] = np.where(totals > 0, moves / np.maximum(totals, 1), np.eye(3))
        self.occupancy[cluster] = visits / visits.sum()
        self.level_load[cluster] = np.where(visits > 0, load_sum / np.maximum(visits, 1), fallback)

    def _level_after(self, src_cluster, dst_cluster, ticks):
        if src_cluster == dst_cluster:
            return np.linalg.matrix_power(self.tick_matrix[src_cluster], ticks)
        return np.tile(self.occupancy[dst_cluster], (3, 1))

    def _build_actions(self):
        self.moves = []
        for src in self.nodes:
            for dst, weight in self.problem.graph.get_neighbors(src).items():
                trans = self._level_after(self.cluster_of[src], self.cluster_of[dst], int(weight))
                self.moves.append((self.node_index[src], self.node_index[dst], float(weight), trans))

        all_masks = np.arange(self.num_masks)
        self.completions = []
        for room in self.rooms:
            if room not in self.node_index:
                continue
            cluster = self.cluster_of[room]
            service = float(self.problem.steps_config[room].get("mean_service_time", 3.0))
            wait = self.level_load[cluster] * self._mean_service(room)
            cost = wait + service
            trans = np.array([
                np.linalg.matrix_power(self.tick_matrix[cluster], max(1, int(round(d))))[b]
                for b, d in enumerate(cost)
            ])
            bit = self.room_bit[room]
            masks = all_masks[(all_masks >> bit) & 1 == 0]
            self.completions.append((self.node_index[room], 1 << bit, masks, cost, trans))

    def _backup(self, values):
        best = np.full_like(values, -np.inf)
        choice = np.zeros(values.shape, dtype=int)

        for src, dst, weight, trans in self.moves:
            q = -weight + self.gamma * values[dst] @ trans.T
            better = q > best[src]
            best[src] = np.where(better, q, best[src])
            choice[src] = np.where(better, dst, choice[src])

        for node, bit, masks, cost, trans in self.completions:
            q = -cost + self.gamma * values[node][masks | bit] @ trans.T
            better = q > best[node, masks]
            best[node, masks] = np.where(better, q, best[node, masks])
            choice[node, masks] = np.where(better, self.complete_code, choice[node, masks])

        best[:, self.full_mask, :] = 0.0
        return best, choice

    def solve(self):
        """Run Bellman backups until the sup-norm change is within epsilon-optimality."""
        if self.gamma < 1:
            threshold = self.epsilon * (1 - self.gamma) / (2 * self.gamma)
        else:
            threshold = self.epsilon

        values = np.zeros((len(self.nodes), self.num_masks, len(LEVELS)))
        residual = np.inf
        iterations = 0
        policy = None
        while iterations < self.max_iterations and residual >= threshold:
            new_values, policy = self._backup(values)
            residual = float(np.abs(new_values - values).max())
            values = new_values
            iterations += 1

        self.values = values
        self.policy = policy
        self.iterations = iterations
        self.residual = residual
        self.converged = residual < threshold
        return self

    def _mask(self, completed):
        return sum(1 << self.room_bit[r] for r in completed if r in self.room_bit)

    def value(self, state):
        """Expected negative time-to-go from a state, using the default congestion level."""
        room, completed = state[0], state[1]
        level = state[2] if len(state) > 2 else self.default_level[self.cluster_of[room]]
        return float(self.values[self.node_index[room], self._mask(completed), level])

    def select_action(self, state, metrics=None):
        """Return the optimal action for a (room, completed[, congestion level]) state."""
        room, completed = state[0], state[1]
        if self.problem.goal_test((room, completed)):
            return None
        level = state[2] if len(state) > 2 else self.default_level[self.cluster_of[room]]
        choice = self.policy[self.node_index[room], self._mask(completed), level]
        if choice == self.complete_code:
            return ("complete", room)
        return ("move", self.nodes[choice])
