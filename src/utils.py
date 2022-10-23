import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="C:/Users/lihai/Desktop/paper and code/DeepWalk/DeepWalk")
    parser.add_argument("--numWalks", type=int, default=5)
    parser.add_argument("--walkLength", type=int, default=30)
    parser.add_argument("--windowSize", type=int, default=2)
    parser.add_argument("--numNegative", type=int, default=1)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument("--shrinkStep", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--logStep", type=int, default=100)
    return parser.parse_args()

def load_train_edges(path):
    srcs, dsts = [], []
    with open(f"{path}/data/train_edges.txt", "r") as f:
        for line in f:
            edge = line.strip("\n").split(" ")
            srcs.append(edge[0])
            dsts.append(edge[1])
    return [srcs, dsts]

def load_test_edges(path):
    srcs, dsts, labels = [], [], []
    with open(f"{path}/data/test_edges.txt", "r") as f:
        for line in f:
            edge = line.strip("\n").split(" ")
            srcs.append(edge[0])
            dsts.append(edge[1])
            labels.append(int(edge[2]))
    return [srcs, dsts, labels]

def load_nodes_type(path):
    node_type = {}
    with open(f"{path}/data/node_type.txt", "r") as f:
        for line in f:
            nodeType = line.strip("\n").split(" ")
            node_type[nodeType[0]] = nodeType[1]
    return node_type