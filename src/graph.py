from random import randint
from collections import defaultdict


class Graph(object):
    """Create Graph base class
    """
    def __init__(self):
        self.graph = defaultdict(set)
        self.nodes = []
        self.encoder = {}
        self.decoder = {}


    def construct_graph(self, srcs, dsts, direction=False):
        """construct a graph with your dataset,
        if set direction is True, you will get a directed graph
        """
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
        """ get a graph example, default is undirected graph 
        """
        srcs = [str(randint(1, numNodes)) for _ in range(numEdges)]
        dsts = [str(randint(1, numNodes)) for _ in range(numEdges)]
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
        """set graph node with new id, start from 1, not 0 
        """
        nodes = srcs + dsts
        node_count = {}
        for node in nodes:
            if node in node_count:
                node_count[node] += 1
            else:
                node_count[node] = 0

        count_sort = sorted(
            node_count.items(),
            key=lambda x:x[1],
            reverse=True
        )

        for index, element in enumerate(count_sort):
            node_id = index + 1
            self.encoder[element[0]] = node_id
            self.decoder[node_id] = element[0]

        srcs = [self.encoder[src] for src in srcs]
        dsts = [self.encoder[dst] for dst in dsts]
        return srcs, dsts


    def num_nodes(self):
        """return nodes number of graph
        """
        return len(self.nodes)


    def num_edges(self):
        """return edges number of graph
        """
        return sum(len(value) for _, value in self.graph.items())


    def degrees(self):
        """return out degrees of graph nodes
        """
        return [len(self.graph[i]) for i in range(1, len(self.graph.items()) + 1)]


    def node_type(self, nodeTypeDict):
        """record nodes type for heterogeneous graph,
        the function will transform node id from original id to encoder id
        """
        return {self.encoder[key]: value for key, value in nodeTypeDict.items()}


    def encoder_new_edges(self, srcs, dsts):
        encoder_srcs, encoder_dsts = [], []
        for src, dst in zip(srcs, dsts):
            encoder_srcs.append(self.encoder[src])
            encoder_dsts.append(self.encoder[dst])
        return encoder_srcs, encoder_dsts