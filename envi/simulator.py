import json
from envi.campus_graph import CampusGraph
from core.counters import RoomCounter
from core.metrics import MetricsTracker
from core.seeding import get_rng
from core.problem import CampusProblem

from algorithms.greedy_nearest import GreedyNearestAgent
from algorithms.bfs import BFSAgent
from algorithms.dfs import DFSAgent
from algorithms.astar import AStarAgent

class CampusSimulator:
    def __init__(self, campus_map_path: str = "data/campus_map.json", steps_config_path: str = "data/steps.json", seed: int = 42, profile=None, rule_pack_version: str = "v1"):
        self.graph = CampusGraph(campus_map_path)
        
        with open(steps_config_path, 'r') as f:
            self.steps_config = json.load(f)
            
        self.rng = get_rng(seed)
        self.bg_rng = get_rng(seed + 8888)

        self.problem = CampusProblem(self.graph, steps_config=self.steps_config,
                                     profile=profile, rule_pack_version=rule_pack_version)
        
        self.counters = {
            room_id: RoomCounter(room_id, self.steps_config)
            for room_id in self.steps_config.keys()
        }

        self.room_ids = list(self.steps_config.keys())
        agent_classes = [GreedyNearestAgent, DFSAgent, BFSAgent]

        self.background_agents = []
        for i in range(20):
            start_room = self.bg_rng.choice(self.room_ids)
            target_room = self.bg_rng.choice([r for r in self.room_ids if r != start_room])
            algo_class = self.bg_rng.choice(agent_classes)
            bg_agent_instance = algo_class(self.problem, name=f"student_{i+1}")

            self.background_agents.append({
                "id": f"student_{i+1}",
                "agent": bg_agent_instance,
                "current_room": start_room,
                "completed_steps": frozenset([target_room]),
                "travel_ticks_remaining": 0,
                "current_destination": None
            })
    
    def _update_background_agents(self):
        """Advances background student agents by 1 tick to create realistic congestion."""
        for bg in self.background_agents:
            if bg["travel_ticks_remaining"] > 0:
                bg["travel_ticks_remaining"] -= 1
                if bg["travel_ticks_remaining"] == 0:
                    bg["current_room"] = bg["current_destination"]
                    bg["current_destination"] = None
                continue
            
            if not bg.get("path"):
                start_room = bg["current_room"]
                target_room = self.bg_rng.choice([r for r in self.room_ids if r != start_room])
                bg["path"] = self._compute_shortest_path(start_room, target_room)
                if not bg["path"]:
                    continue

            target = bg["path"].pop(0)
            neighbors = self.graph.get_neighbors(bg["current_room"])
            if target in neighbors:
                weight = neighbors[target]
                bg["travel_ticks_remaining"] = max(1, int(weight) - 1)
                bg["current_destination"] = target

                if target in self.counters and self.bg_rng.random() < 0.3:
                    self.counters[target].add_to_queue(bg["id"])

    def run_simulation(self, agent, max_steps: int = 500, congestion: bool = True, start_room: str = None, completed_steps: frozenset = None) -> dict:
        metrics = MetricsTracker()
        if start_room is not None or completed_steps is not None:
            init_room = start_room if start_room is not None else self.problem.start_room
            init_completed = completed_steps if completed_steps is not None else frozenset()
            state = (init_room, frozenset(init_completed))
        else:
            state = self.problem.initial_state()
        trace = []
        background_steps = []
        current_room, completed_steps = state
        step_count = 0

        travel_time = 0.0
        wait_time = 0.0
        service_time = 0.0

        while step_count < max_steps:
            step_count += 1
            
            if congestion:
                for counter in self.counters.values():
                    counter.tick(self.rng)            
                self._update_background_agents()

            try:
                action = agent.select_action((current_room, completed_steps), metrics=None)
            except TypeError:
                try:
                    action = agent.select_action((current_room, completed_steps))
                except Exception:
                    action = None

            if not action:
                legal_actions = self.problem.actions((current_room, completed_steps))
                if legal_actions:
                    action = legal_actions[0]
                else:
                    break  

            action_type, target = action
            if action_type == "move":
                neighbors = self.graph.get_neighbors(current_room)
                weight = neighbors.get(target, 1.0)
                travel_time += float(weight)
                metrics.add_travel(weight)
                current_room = target

            elif action_type == "complete":
                mean_service = float(self.steps_config.get(target, {}).get("mean_service_time", 3.0))

                if congestion and target in self.counters:
                    counter = self.counters[target]
                    counter.add_to_queue(getattr(agent, "name", "agent"))
                    
                    queue_list = getattr(counter, 'queue', [])
                    queue_len = len(queue_list) if isinstance(queue_list, list) else 1
                    
                    # Calculate dynamic node crowding from background workers present at target room
                    room_crowd = sum(1 for bg in self.background_agents if bg["current_room"] == target)
                    effective_crowd = max(queue_len, 1 + room_crowd)

                    step_wait = max(1, int(effective_crowd * (mean_service / 2.0)))
                    
                    step_count += step_wait
                    wait_time += float(step_wait)
                    metrics.add_wait(step_wait)

                completed_steps = frozenset(list(completed_steps) + [target])
                service_duration = mean_service
                service_time += service_duration
                metrics.add_service(service_duration)
            
            trace.append({
                "step": step_count,
                "current_room": current_room,
                "completed_steps": list(completed_steps),
                "action_taken": [action_type, target]
            })

            if congestion:
                background_steps.append([
                    {
                        "id": bg["id"],
                        "current_room": bg["current_room"]
                    }
                    for bg in self.background_agents
                ])

            if self.problem.goal_test((current_room, completed_steps)):
                break

        total_cost = travel_time + wait_time + service_time
        return {
            "agent": getattr(agent, "name", "Agent"),
            "total_steps": step_count,
            "metrics": {
                "travel_time": travel_time,
                "wait_time": wait_time,
                "service_time": service_time,
                "total_cost": total_cost
            },
            "trace": trace,
            "background_steps": background_steps if congestion else []
        }
    
    def _compute_shortest_path(self, start: str, goal: str) -> list:
        """Lightweight BFS for background agents to compute paths instantaneously."""
        from collections import deque
        if start == goal:
            return []
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                return path[1:]
            for neighbor in self.graph.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []