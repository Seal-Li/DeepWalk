from random import randint
from collections import defaultdict


class Graph(object):
    def __init__(self):
        self.graph = defaultdict(set)
        self.nodes = []
        self.encoder = {}
        self.decoder = {}

    def construct_graph(self, srcs, dsts, direction=False):
        srcs, dsts = self.node_encoder(srcs, dsts)
        nodes = set()
        if direction:
            for src, dst in zip(srcs, dsts):
                self.graph[src].add(dst)
                nodes.update({src, dst})
        else:
            for src, dst in zip(srcs, dsts):
                self.graph[src].add(dst)
                self.graph[dst].add(src)
                nodes.update({src, dst})
        self.nodes = list(nodes)
        return self.graph

    def random_graph(self, numNodes=1000, numEdges=10000, direction=False):
        srcs = [randint(1, numNodes) for _ in range(numEdges)]
        dsts = [randint(1, numNodes) for _ in range(numEdges)]
        srcs, dsts = self.node_encoder(srcs, dsts)
        nodes = set()
        if direction:
            for src, dst in zip(srcs, dsts):
                self.graph[src].add(dst)
                nodes.update({src, dst})
        else:
            for src, dst in zip(srcs, dsts):
                self.graph[src].add(dst)
                self.graph[dst].add(src)
                nodes.update({src, dst})
        self.nodes = list(nodes)
        return self.graph

    def node_encoder(self, srcs, dsts):
        nodes = srcs + dsts
        node_count = {}
        for node in nodes:
            if node in node_count:
                node_count[node] += 1
            else:
                node_count[node] = 0
        count_sort = sorted(node_count.items(), key=lambda x:x[1], reverse=True)

        for index, element in enumerate(count_sort):
            # start from 1, not 0
            node_id = index + 1
            self.encoder[element[0]] = node_id
            self.decoder[node_id] = element[0]

        srcs = [self.encoder[src] for src in srcs]
        dsts = [self.encoder[dst] for dst in dsts]
        return srcs, dsts

    def num_nodes(self):
        return len(self.encoder)

    def num_edges(self):
        return sum(len(value) for _, value in self.graph.items())

    def degrees(self):
        return [len(self.graph[i]) for i in range(1, len(self.graph.items()) + 1)]
    
    def node_type(self, nodeTypeDict):
        return {self.encoder[key]: value for key, value in nodeTypeDict.items()}


if __name__ == "__main__":
    Graph = Graph()
    # graph = Graph.random_graph(numNodes=1000, numEdges=10000)
    srcs = [randint(1, 100) for _ in range(500)]
    dsts = [randint(1, 100) for _ in range(500)]
    graph = Graph.construct_graph(srcs, dsts)
    print(graph)