import json
import heapq

class CampusGraph:
    def __init__(self, json_path: str = "data/campus_map.json"):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.nodes = data["nodes"]
        self.adj_list = {node: {} for node in self.nodes}
        
        for edge in data["edges"]:
            u, v, weight = edge["from"], edge["to"], edge["weight"]
            self.adj_list[u][v] = weight
            self.adj_list[v][u] = weight

    def get_neighbors(self, node: str) -> dict:
        return self.adj_list.get(node, {})

    def shortest_path_weight(self, start: str, target: str) -> int:
        if start == target:
            return 0
        
        pq = [(0, start)]
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node == target:
                return current_dist
            
            if current_dist > distances[current_node]:
                continue
                
            for neighbor, weight in self.adj_list[current_node].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
                    
        return float('inf')