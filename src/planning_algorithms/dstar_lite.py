import heapq
import math
from typing import Dict, Tuple, List, Any
import networkx as nx


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PriorityQueue:
    def __init__(self):
        self._queue: List[Tuple[Tuple[float, float], Any]] = []

    def push(self, item: Any, key: Tuple[float, float]):
        heapq.heappush(self._queue, (key, item))

    def pop(self) -> Any:
        return heapq.heappop(self._queue)[1]

    def top_key(self) -> Tuple[float, float]:
        if not self._queue:
            return (float('inf'), float('inf'))
        return self._queue[0][0]

    def remove(self, item: Any):
        for i, (_, it) in enumerate(self._queue):
            if it == item:
                del self._queue[i]
                heapq.heapify(self._queue)
                break

    def __contains__(self, item: Any) -> bool:
        return any(it == item for _, it in self._queue)


class DStarLite:
    def __init__(self, graph: nx.Graph, goal: int, heuristic_fn=heuristic):
        self.graph = graph
        self.goal = goal
        self.h = heuristic_fn
        self.g: Dict[int, float] = {n: float('inf') for n in graph.nodes}
        self.rhs: Dict[int, float] = {n: float('inf') for n in graph.nodes}
        self.U = PriorityQueue()
        self.k_m = 0.0
        self.rhs[goal] = 0.0
        self.U.push(goal, self.calculate_key(goal))
        self.start = goal

    def calculate_key(self, node: int) -> Tuple[float, float]:
        g_rhs = min(self.g[node], self.rhs[node])
        return (g_rhs + self.h(self.start_pos(node), self.start_pos(self.start)) + self.k_m, g_rhs)

    def start_pos(self, node: int) -> Tuple[int, int]:
        pos = self.graph.nodes[node]['pos']
        return (int(pos[0]), int(pos[1]))

    def update_vertex(self, u: int):
        if u != self.goal:
            min_cost = float('inf')
            for v in self.graph.neighbors(u):
                cost = self.graph[u][v]['weight'] + self.g[v]
                if cost < min_cost:
                    min_cost = cost
            self.rhs[u] = min_cost
        if u in self.U:
            self.U.remove(u)
        if self.g[u] != self.rhs[u]:
            self.U.push(u, self.calculate_key(u))

    def compute_shortest_path(self):
        while self.U.top_key() < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]:
            k_old = self.U.top_key()
            u = self.U.pop()
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.graph.predecessors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.graph.predecessors(u):
                    self.update_vertex(s)

    def replan(self, start: int) -> List[int]:
        if start not in self.graph:
            raise ValueError('Start node not in graph')
        self.start = start
        self.compute_shortest_path()
        path = []
        current = start
        if self.g[current] == float('inf'):
            return path
        while current != self.goal:
            path.append(current)
            min_cost = float('inf')
            next_node = None
            for v in self.graph.neighbors(current):
                cost = self.graph[current][v]['weight'] + self.g[v]
                if cost < min_cost:
                    min_cost = cost
                    next_node = v
            if next_node is None:
                return []
            current = next_node
        path.append(self.goal)
        return path
