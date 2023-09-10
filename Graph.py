import csv
import math


#  The Graph class represents a graph and exposes methods for manipulating graphs
class Graph(object):

    # constructor can be called in two ways
    # either specify the filename, or specify the adjacency matrix and matrix describing required edges
    # the consider_zero_disconnected parameter tells whether this graph should consider edges with a weight of zero
    # as non-existent.  Default value of False allows for edges with weight zero
    def __init__(self, filename=None, adjacency_matrix=None, required_matrix=None, consider_zero_disconnected=False):
        self.representation = []
        self.required = []
        self.consider_zero_disconnected = consider_zero_disconnected
        if filename is not None:
            with open(filename + ".csv", 'rt') as file:
                reader = csv.reader(file, delimiter=',')
                row = next(reader, False)
                self.num_vertices = len(row)
                while row:
                    row_contents = []
                    for item in row:
                        if len(item.strip()) > 0:
                            row_contents.append(int(item.strip()))
                        else:
                            row_contents.append(None)

                    self.representation.append(row_contents)
                    row = next(reader, False)

            with open(filename + "_required.csv", 'rt') as file:
                reader = csv.reader(file, delimiter=',')
                row = next(reader, False)
                while row:
                    row_contents = []
                    for item in row:
                        row_contents.append(int(item.strip()))

                    self.required.append(row_contents)
                    row = next(reader, False)

        elif adjacency_matrix is not None and required_matrix is not None:
            self.representation = adjacency_matrix
            self.required = required_matrix
            self.num_vertices = len(self.representation)
        else:
            print("WARNING: constructing empty graph")
            self.num_vertices = 0

        # count edges
        self.num_edges = 0
        for i in range(0, self.num_vertices):
            for j in range(0, self.num_vertices):
                if i != j and self.is_connected(i, j):
                    self.num_edges += 1

    # tests if two vertices are connected
    def is_connected(self, i, j):
        if i < self.num_vertices and j < self.num_vertices:
            if self.consider_zero_disconnected:
                return self.representation[i][j] is not None and self.representation[i][j] > 0
            else:
                return self.representation[i][j] is not None and self.representation[i][j] >= 0
        else:
            return False

    # determines if the edge between vertex i and vertex j is required
    def is_required(self, i, j):
        if i < self.num_vertices and j < self.num_vertices:
            return self.required[i][j] == 1
        else:
            return False

    def set_required(self, i, j, new_val):
        """
        Allows an edge to be mark as required or not required
        """
        if new_val:
            self.required[i][j] = 1
            # self.required[j][i] = 1
        else:
            self.required[i][j] = 0
            # self.required[j][i] = 0

    def get_weight(self, i, j):
        """
        Gets the weight of the edge between two vertices - returns infinity if they are not connected
        """
        if self.is_connected(i, j):
            return self.representation[i][j]
        else:
            return math.inf

    def get_required(self):
        """
        returns a list of all required edges in the graph
        """
        required = []
        for i in range(0, self.num_vertices):
            for j in range(0, self.num_vertices):
                if self.is_required(i, j):
                    required.append((i, j))

        return required

    # gets vertices adjacent to specified vertex
    def get_neighbors(self, vertex):
        neighbors = set()
        for other in range(0, self.num_vertices):
            if other != vertex and self.is_connected(vertex, other):
                neighbors.add(other)

        return neighbors

    # gets required edges that are incident to the specified vertex
    # required[i][j] should be true is edge (i, j) is required
    def get_type_1_edges(self, vertex, required):
        edges = []
        neighbors = self.get_neighbors(vertex)
        for neighbor in neighbors:
            if required[vertex][neighbor]:
                edges.append((vertex, neighbor))

        return edges

    # gets edges incident to "vertex" which are also incident to a required vertex
    # required[i][j] should be true is edge (i, j) is required
    def get_type_2_edges(self, vertex, required):
        edges = []
        neighbors = self.get_neighbors(vertex)

        for other in neighbors:
            others_neighbors = self.get_neighbors(other)
            for neighbor in others_neighbors:
                if required[other][neighbor]:
                    edges.append((vertex, other))
                    break

        return edges
