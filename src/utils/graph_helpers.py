from typing import Tuple
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def filter_graph(graph: nx.Graph, dist: np.ndarray, clearance: float, step: float) -> nx.Graph:
    G = nx.Graph()
    for n, data in graph.nodes(data=True):
        x, y = data['pos']
        if dist[int(y), int(x)] >= clearance:
            G.add_node(n, pos=(x, y))
    tree = None
    if G.number_of_nodes() == 0:
        return G
    for u, v, w in graph.edges(data='weight'):
        if u in G and v in G and w <= step:
            x1, y1 = graph.nodes[u]['pos']
            x2, y2 = graph.nodes[v]['pos']
            line = list(zip(*bresenham_line(int(x1), int(y1), int(x2), int(y2))))
            if all(dist[int(y), int(x)] >= clearance for x, y in zip(*line)):
                G.add_edge(u, v, weight=w)
    return G


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def snap_point(graph: nx.Graph, point: Tuple[int, int]) -> int:
    nodes = [graph.nodes[n]['pos'] for n in graph.nodes]
    tree = cKDTree(nodes)
    dist, idx = tree.query(point)
    return list(graph.nodes)[idx]
