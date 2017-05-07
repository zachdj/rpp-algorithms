class Path(object):
    def __init__(self, graph, vertex=None):
        self.graph = graph
        self.cost = 0
        self.vertices = []
        if vertex is not None:
            self.vertices.append(vertex)

    def extend(self, vertex):
        if len(self.vertices) > 0:
            if self.graph.is_connected(self.vertices[-1], vertex):
                self.cost += self.graph.get_weight(self.vertices[-1], vertex)
                self.vertices.append(vertex)
            else:
                raise Exception("Tried to add non-existent edge (%s, %s) to path" % (self.vertices[-1], vertex))
        else:
            # first vertex
            self.vertices.append(vertex)

    # returns a list of edges (i, j) that describes this path
    def get_edges(self):
        edges = []
        for idx in range(0, len(self.vertices) - 1):
            v1 = self.vertices[idx]
            v2 = self.vertices[idx+1]
            edges.append((v1, v2))

        return edges

    # returns a string representation of this path
    def __str__(self):
        string = ""
        for idx in range(0, len(self.vertices)):
            vertex = self.vertices[idx]
            if idx == len(self.vertices) - 1:
                string += "%s" % vertex
            else:
                string += "%s -> " % vertex

        return string
