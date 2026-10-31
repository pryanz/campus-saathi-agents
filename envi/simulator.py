import json
from envi.campus_graph import CampusGraph
from core.counters import RoomCounter
from core.metrics import MetricsTracker
from core.seeding import get_rng

class CampusSimulator:
    def __init__(self, campus_map_path: str = "data/campus_map.json", steps_config_path: str = "data/steps.json", seed: int = 42):
        self.graph = CampusGraph(campus_map_path)
        
        with open(steps_config_path, 'r') as f:
            self.steps_config = json.load(f)
            
        self.rng = get_rng(seed)
        
        self.counters = {
            room_id: RoomCounter(room_id, self.steps_config, num_counters=2)
            for room_id in self.steps_config.keys()
        }

    def run_simulation(self, agent, max_steps: int = 500) -> dict:
        metrics = MetricsTracker()
        state = ("gate", frozenset())  # (current_room, completed_steps)
        
        #Tracking travel progress: if an agent is traveling along an edge, store remaining travel time
        travel_ticks_remaining = 0
        current_destination = None

        time_step = 0
        while time_step < max_steps and not agent.problem.goal_test(state):
            time_step += 1
            
            for counter in self.counters.values():
                counter.tick(self.rng)

            if travel_ticks_remaining > 0:
                travel_ticks_remaining -= 1
                metrics.add_travel(1.0)
                if travel_ticks_remaining == 0:
                    current_room, completed = state
                    state = (current_destination, completed)
                continue

            action = agent.select_action(state, metrics)
            if not action:
                break  

            action_type, target = action
            current_room, completed_steps = state

            if action_type == "move":
                if target == current_room:
                    continue
                # Get edge weight (travel time) from graph
                neighbors = self.graph.get_neighbors(current_room)
                if target in neighbors:
                    travel_ticks_remaining = neighbors[target] - 1  # 1 tick passes immediately
                    current_destination = target
                    metrics.add_travel(1.0)
                    if travel_ticks_remaining <= 0:
                        state = (target, completed_steps)

            elif action_type == "complete":
                # Check if room has a counter and join queue
                if target in self.counters:
                    counter = self.counters[target]
                    counter.add_to_queue(agent.name)
                    
                    wait_time = 0
                    while target not in completed_steps:
                        time_step += 1
                        wait_time += 1
                        counter.tick(self.rng)
                        break
                    
                    state = (current_room, completed_steps | {target})
                    metrics.add_wait(wait_time)
                    service_duration = self.steps_config[target].get("mean_service_time", 3.0)
                    metrics.add_service(service_duration)

        return metrics.summary()