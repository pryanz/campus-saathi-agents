import numpy as np

class RoomCounter:
    def __init__(self, room_id: str, steps_config: dict, num_counters: int = 4):
        self.room_id = room_id
        self.num_counters = num_counters

        step_info = steps_config.get(room_id, {})
        self.mean_service_time = step_info.get("mean_service_time", 3.0)
        
        self.queue = []
        
        self.busy_counters = [0] * num_counters
    
    def queue_length(self) -> int:
        #this will return number of agent currently in queue
        return len(self.queue)

    def add_to_queue(self, agent_id: str):
        #adds agent to the back of queue
        self.queue.append(agent_id)

    def tick(self, rng: np.random.Generator):
        self.busy_counters = [max(0, t - 1) for t in self.busy_counters]

        for i in range(self.num_counters):
            if self.busy_counters[i] == 0 and self.queue:
                _agent_id = self.queue.pop(0)
                
                scale = self.mean_service_time / 4.0
                sampled_time = max(1, int(round(rng.gamma(shape=2.0, scale=scale))))
                
                self.busy_counters[i] = sampled_time