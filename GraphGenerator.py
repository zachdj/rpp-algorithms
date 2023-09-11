"""
    Contains functions to randomly generate graphs
"""

from random import *

from Graph import Graph


def to_file(graph, filename="graph"):
    """
    Writes the given Graph object to a file with the given name/path
    """
    graph_string = ""
    required_string = ""
    for row in range(len(graph.representation)):
        for col in range(len(graph.representation)):
            graph_string += str(graph.representation[row][col])
            required_string += str(graph.required[row][col])
            if col < len(graph.representation) - 1:
                graph_string += ","
                required_string += ","
        graph_string += "\n"
        required_string += "\n"

    with open(filename + ".csv", "w") as text_file:
        print(graph_string, file=text_file)
    with open(filename + "_required.csv", "w") as text_file:
        print(required_string, file=text_file)


def add_edge(graph, weight, i, j):
    """
    Adds an edge between vertex i and vertex j to the specified graph with the given weight
    """
    graph[i][j] = weight
    # graph[j][i] = weight


def generate(num_vertices, num_edges, num_required, min_weight=1, max_weight=30):
    """
    Generates a graph with the given number of vertices, edges, and required edges
    """
    # initially disconnected collection of vertices
    representation = [[-1 for i in range(num_vertices)] for j in range(num_vertices)]
    vertices = set(range(num_vertices))

    edges = set()
    # keeps track of how many edges are still required to add
    edges_to_add = num_edges
    # keeps track of how many edges are still required to add
    required_edges_to_add = num_required

    # Create two partitions, aux_s and aux_t. Initially store all vertices in aux_s.
    aux_s = set(range(num_vertices))
    aux_t = set()

    # Pick a random node, and mark it as visited and the current node.
    current_node = sample(list(aux_s), 1).pop()
    aux_s.remove(current_node)
    aux_t.add(current_node)

    # Create a random connected graph.
    while aux_s:
        # Randomly pick the next node to visit
        neighbor_node = sample(list(vertices), 1).pop()
        # If the new node hasn't been visited, add the edge from current to new.
        if neighbor_node not in aux_t:
            edge = (current_node, neighbor_node)
            # print("Adding edge between %s and %s" % (current_node+1, neighbor_node+1))
            weight = randint(min_weight, max_weight)
            add_edge(representation, weight, current_node, neighbor_node)
            edges.add(edge)
            edges_to_add -= 1
            aux_s.remove(neighbor_node)
            aux_t.add(neighbor_node)
        # Set the new node as the current node.
        current_node = neighbor_node

    # add random edges until we have the specified amount:
    while edges_to_add > 0:
        # choose a random vertex
        start = sample(list(vertices), 1).pop()
        neighbors = map(lambda x: x[1] if x[0] == start else x[0],
                        filter(lambda x: x[0] == start or x[1] == start, edges))
        # choose a vertex not in neighbors
        candidates = (vertices - set(neighbors)) - {start}
        if candidates:
            end = sample(list(candidates), 1).pop()
            edge = (start, end)
            # print("Adding additional edge between %s and %s" % (start+1, end+1))
            weight = randint(min_weight, max_weight)
            add_edge(representation, weight, start, end)
            edges.add(edge)
            edges_to_add -= 1

    # mark random edges as required
    # representation of required edges
    required = [[0 for i in range(num_vertices)] for j in range(num_vertices)]
    required_edges = sample(list(edges), num_required)
    for edge in required_edges:
        add_edge(required, 1, edge[0], edge[1])
        required_edges_to_add -= 1

    graph = Graph(adjacency_matrix=representation, required_matrix=required, consider_zero_disconnected=True)
    return graph


# generate some graphs and write them to a file

# sparse graphs - 15% of all possible edges
sg1 = generate(20, 57, 10)
to_file(sg1, "test-graphs/sparse1")
sg2 = generate(40, 234, 40)
to_file(sg2, "test-graphs/sparse2")
sg3 = generate(60, 531, 100)
to_file(sg3, "test-graphs/sparse3")
sg4 = generate(80, 948, 190)
to_file(sg4, "test-graphs/sparse4")
sg5 = generate(100, 1485, 295)
to_file(sg5, "test-graphs/sparse5")
print("Done with sparse graphs")

# moderate graphs - 25% of all possible edges
mg1 = generate(20, 95, 19)
to_file(mg1, "test-graphs/moderate1")
mg2 = generate(40, 290, 58)
to_file(mg2, "test-graphs/moderate2")
mg3 = generate(60, 885, 177)
to_file(mg3, "test-graphs/moderate3")
mg4 = generate(80, 1580, 316)
to_file(mg4, "test-graphs/moderate4")
mg5 = generate(100, 2475, 495)
to_file(mg5, "test-graphs/moderate5")
print("Done with moderate graphs")

# dense graphs - 50% of all possible edges
dg1 = generate(20, 190, 38)
to_file(dg1, "test-graphs/dense1")
print("Done with dense1")
dg2 = generate(40, 780, 156)
to_file(dg2, "test-graphs/dense2")
print("Done with dense2")
dg3 = generate(60, 1770, 354)
to_file(dg3, "test-graphs/dense3")
print("Done with dense3")
dg4 = generate(80, 3160, 632)
to_file(dg4, "test-graphs/dense4")
print("Done with dense4")
dg5 = generate(100, 4950, 990)
to_file(dg5, "test-graphs/dense5")
print("Done with dense5")
