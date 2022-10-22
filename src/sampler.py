import torch
from walk import Walk

class Sampler(object):
    """ defined an base Sampler
    """
    def __init__(self, g, numWalks, walkLength, windowSize, degrees, numNegative):
        self.g = g
        self.numWalks = numWalks
        self.walkLength = walkLength
        self.windowSize = windowSize
        self.numNegative = numNegative
        self.sampleWeights = [max(degrees) - degree for degree in degrees]
    
    def sample(self, nodes):
        """get training samples
        """
        positive_pairs = self.positive_sample(nodes)
        negative_pairs = self.negative_sample(positive_pairs)
        srcs, dsts, labels = [], [], []
        for pair in positive_pairs + negative_pairs:
            src, dst, label = pair
            srcs.append(int(src))
            dsts.append(int(dst))
            labels.append(int(label))
        return torch.tensor(srcs), torch.tensor(dsts), torch.tensor(labels)
    
    def positive_sample(self, nodes):
        """ generate positive pairs
        """
        traces = Walk(
            nodes,
            self.g.graph,
            self.numWalks,
            self.walkLength
        ).walk()
        
        positive_pairs = []
        for trace in traces:
            for i in range(len(trace)):
                center = trace[i]
                left = max(0, i - self.windowSize)
                right = min(len(trace), i + self.windowSize + 1)
                positive_pairs.extend(
                    [[center, x, 1]
                    for x in trace[left:i]]
                )
                positive_pairs.extend(
                    [[center, x, 1]
                    for x in trace[i+1:right]]
                )
        return positive_pairs
    
    def negative_sample(self, positive_pairs):
        """generate negative pairs
        """
        negative_srcs = [
            positive_pair[0] 
            for positive_pair in positive_pairs
        ] * self.numNegative
        negative_dsts = torch.multinomial(
            torch.tensor(self.sampleWeights).float(),
            int(len(positive_pairs) * self.numNegative),
            replacement=True
        ).tolist()
        negative_pairs = [
            [src, str(dst), 0]
            for src, dst in zip(negative_srcs, negative_dsts)
            ]
        return negative_pairs