class MetricsTracker:
    def __init__(self):
        self.travel_time = 0.0
        self.wait_time = 0.0
        self.service_time = 0.0
        self.nodes_expanded = 0

    def add_travel(self, duration: float):
        self.travel_time += duration

    def add_wait(self, duration: float):
        self.wait_time += duration

    def add_service(self, duration: float):
        self.service_time += duration

    def increment_node_expansion(self, count: int = 1):
        self.nodes_expanded += count

    @property
    def total_cost(self) -> float:
        
        return self.travel_time + self.wait_time + self.service_time

    def summary(self) -> dict:
        return {
            "travel_time": self.travel_time,
            "wait_time": self.wait_time,
            "service_time": self.service_time,
            "total_cost": self.total_cost,
            "nodes_expanded": self.nodes_expanded
        }