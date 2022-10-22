import random

class Walk(object):
    def __init__(self, nodes, graph, numWalks, walkLength):
        self.nodes = nodes * numWalks
        self.graph = graph
        self.walks = numWalks
        self.length = walkLength

    def walk(self):
        traces = []
        for node in self.nodes:
            trace = [node]
            while len(trace) <= self.length:
                candidates = self.graph[int(node)]
                # print(candidates)
                if candidates:
                    node = random.choice(list(candidates))
                    trace.append(str(node))
                else:
                    trace.append("0")
            traces.append(trace)
        return traces
