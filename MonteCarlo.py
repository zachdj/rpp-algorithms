"""
    Implementation of the Monte Carlo approach for finding approximate solutions to the Rural Postman Problem
    proposed by Cordoba et. al. in this paper:
    http://www.sciencedirect.com/science/article/pii/S0305054898000288

    author: Zach Jones
    date: 4/29/17
"""

from GraphUtils import *
from random import *
import time

verbose = False  # enable/disable verbose mode
g = Graph(filename="test-graphs/sparse/sparse1")

# configuration variables
max_it = 25
iteration_counter = 0
alpha = 1

# shortest tour found
shortest_tour = None
shortest_tour_path = None
shortest_tour_cost = math.inf

start_time = time.monotonic_ns()

while iteration_counter < max_it:
    iteration_counter += 1
    # tour[i][j] holds the count of how many times edge (i, j) was traversed
    tour = [[0 for col in range(g.num_vertices)] for row in range(g.num_vertices)]
    tour_path = Path(g)
    edge_required = [[False for col in range(g.num_vertices)] for row in range(g.num_vertices)]
    for edge in g.get_required():
        edge_required[edge[0]][edge[1]] = True
        edge_required[edge[1]][edge[0]] = True

    cost = 0
    num_required = len(g.get_required())
    # probabilities[i][j] holds probability of traversing edge (i, j)
    probabilities = [[0 for col in range(g.num_vertices)] for row in range(g.num_vertices)]
    for v1 in range(0, g.num_vertices):
        for v2 in range(v1, g.num_vertices):
            weight = g.get_weight(v1, v2)
            if weight > 0:
                probabilities[v1][v2] = weight ** -alpha
            else:
                probabilities[v1][v2] = 0

    # randomly place the "vehicle" on a vertex
    current_pos = randint(0, g.num_vertices - 1)
    starting_pos = current_pos
    tour_path.extend(starting_pos)
    if verbose:
        print("Starting at %s" % current_pos)

    while num_required > 0:
        neighbors = g.get_neighbors(current_pos)
        type_1_edges = g.get_type_1_edges(current_pos, edge_required)
        type_2_edges = g.get_type_2_edges(current_pos, edge_required)

        selected_edge = None

        if len(type_1_edges) > 0:
            # randomly pick a type 1 edge to traverse - all T1 edges have equal probability to be selected
            if verbose:
                print("Choosing type 1 edge")

            selection = randint(0, len(type_1_edges) - 1)
            selected_edge = type_1_edges[selection]
        elif len(type_2_edges) > 0:
            # pick a type 2 vertex based on their probabilities
            if verbose:
                print("Choosing type 2 edge")

            prob_sum = 0
            for edge in type_2_edges:
                prob_sum += probabilities[edge[0]][edge[1]]
            selection = random() * prob_sum
            prob_sum = 0
            for edge in type_2_edges:
                prob_sum += probabilities[edge[0]][edge[1]]
                if selection <= prob_sum:
                    selected_edge = edge
                    break
        else:
            # randomly select a neighboring edge based on their probabilities
            if verbose:
                print("Choosing another edge")
            prob_sum = 0
            for neighbor in neighbors:
                prob_sum += probabilities[current_pos][neighbor]
            selection = random() * prob_sum
            prob_sum = 0
            for neighbor in neighbors:
                prob_sum += probabilities[current_pos][neighbor]
                if selection <= prob_sum:
                    selected_edge = (current_pos, neighbor)
                    break

        # we have selected an edge to traverse
        if verbose:
            print("Traversing edge", selected_edge)

        if edge_required[selected_edge[0]][selected_edge[1]]:
            edge_required[selected_edge[0]][selected_edge[1]] = False
            edge_required[selected_edge[1]][selected_edge[0]] = False
            num_required -= 1

        # traverse the edge
        tour[selected_edge[0]][selected_edge[1]] += 1
        tour[selected_edge[1]][selected_edge[0]] += 1
        tour_path.extend(selected_edge[1])
        cost += g.get_weight(selected_edge[0], selected_edge[1])
        current_pos = selected_edge[1]
    # end while

    # we've now traversed all required edges
    # complete the tour by adding a copy of each edge on the shortest path from current_pos to the starting_pos
    shortest_path_to_beginning = dijkstra(g, current_pos)[starting_pos]
    for edge in shortest_path_to_beginning.get_edges():
        tour[edge[0]][edge[1]] += 1
        tour[edge[1]][edge[0]] += 1
        tour_path.extend(edge[1])
        cost += g.get_weight(edge[0], edge[1])

    # SR1
    if verbose:
        print("Cost before SR1: %s" % cost)

    for vertex1 in range(0, len(tour)):
        for vertex2 in range(0, len(tour)):
            while tour[vertex1][vertex2] > 3:
                tour[vertex1][vertex2] -= 2
                tour[vertex2][vertex1] -= 2
                cost -= g.get_weight(vertex1, vertex2)
    if verbose:
        print("Cost before SR1: %s" % cost)

    # SR2
    # create graph object for this tour used to find connected components - this is not an accurate representation of the tour as a subgraph!
    if verbose:
        print("Cost before SR2: %s" % cost)

    tour_graph = Graph(adjacency_matrix=tour, required_matrix=[[]], consider_zero_disconnected=True)
    for vertex1 in range(0, len(tour)):
        for vertex2 in range(0, len(tour)):
            if not g.is_required(vertex1, vertex2) and tour[vertex1][vertex2] == 2:
                reachable_before = bfs(tour_graph, vertex1)
                tour_graph.representation[vertex1][vertex2] = None
                reachable_after = bfs(tour_graph, vertex1)
                if len(reachable_after) == len(reachable_before):  # graph is still connected
                    tour[vertex1][vertex2] = 0
                    tour[vertex2][vertex1] = 0
                    cost -= 2 * g.get_weight(vertex1, vertex2)
                else:
                    tour_graph.representation[vertex1][vertex2] = 2

    if verbose:
        print("Cost after SR2: %s" % cost)

    # SR3
    if verbose:
        print("Cost before SR3: %s" % cost)

    double_edge = None
    for vertex1 in range(0, len(tour)):
        for vertex2 in range(0, len(tour)):
            if tour[vertex1][vertex2] == 2:
                double_edge = (vertex1, vertex2)
                break
    if double_edge is not None:
        double_path = Path(tour_graph, double_edge[0])
        double_path.extend(double_edge[1])
        expansion_found = True
        while expansion_found:
            end_vertex = double_path.vertices[-1]
            neighbors = tour_graph.get_neighbors(end_vertex)
            expansion_found = False
            for neighbor in neighbors:
                if neighbor not in double_path.vertices and tour[end_vertex][neighbor] == 2:
                    expansion_found = True
                    double_path.extend(neighbor)
                    break
        # remove one copy of each edge in double_path and add an edge along the shortest path
        start_vertex = double_path.vertices[0]
        end_vertex = double_path.vertices[-1]
        for edge in double_path.get_edges():
            tour[edge[0]][edge[1]] -= 1
            tour[edge[1]][edge[0]] -= 1
            cost -= g.get_weight(edge[0], edge[1])
        for edge in dijkstra(g, start_vertex)[end_vertex].get_edges():
            tour[edge[0]][edge[1]] += 1
            tour[edge[1]][edge[0]] += 1
            cost += g.get_weight(edge[0], edge[1])

    if verbose:
        print("Cost after SR3: %s" % cost)

    if verbose:
        print("Found tour with cost %s" % cost)

    if cost < shortest_tour_cost:
        shortest_tour = tour
        shortest_tour_path = tour_path
        shortest_tour_cost = cost

# end outer while


time_elapsed = time.monotonic_ns() - start_time

print("Done searching!\nBest tour found has cost %s" % shortest_tour_cost)
print("Path before simplification routines:\n %s" % shortest_tour_path)

print("Time elapsed: %s ms" % (time_elapsed * 1000))
