from Graph import *
from Path import *


def dijkstra(graph, source):
    """
    Dijkstra's algorithm starting at "source"
    :return: a list of Path objects where list[i] is the shortest path from source to vertex i
    """
    q = set()  # set of unvisited nodes

    dist = dict()  # dist[i] is distance from source to vertex i
    prev = dict()  # prev[i] is the previous node in the optimal path from source to i
    # foreach vertex in the graph
    for vertex in range(0, graph.num_vertices):
        dist[vertex] = math.inf
        prev[vertex] = None
        q.add(vertex)

    dist[source] = 0  # duh

    while len(q) > 0:
        min_vertex = None
        for vertex in q:
            if min_vertex is None:
                min_vertex = vertex
            elif dist[vertex] < dist[min_vertex]:
                min_vertex = vertex

        if min_vertex is None:
            break

        q.remove(min_vertex)

        for neighbor in graph.get_neighbors(min_vertex):
            alternate = dist[min_vertex] + graph.get_weight(min_vertex, neighbor)
            if alternate < dist[neighbor]:
                dist[neighbor] = alternate
                prev[neighbor] = min_vertex

    # at this point, we will have discovered the shortest paths from source to every other node
    # now we'll walk backwards along each path to construct the Path objects
    shortest_paths = dict()
    for vertex in range(0, graph.num_vertices):
        path_to_vertex = []
        node_along_path = vertex
        while prev[node_along_path] is not None:
            path_to_vertex.insert(0, prev[node_along_path])
            node_along_path = prev[node_along_path]

        path = Path(graph)
        for node in path_to_vertex:
            path.extend(node)
        path.extend(vertex)
        shortest_paths[vertex] = path

    return shortest_paths


def bfs(graph, start):
    """
    Breadth-first search over the specified graph starting at vertex with index "start"
    """
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            neighbors = graph.get_neighbors(vertex)
            queue.extend(neighbors - visited)
    return visited
