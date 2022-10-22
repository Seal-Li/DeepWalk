import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numWalks", type=int, default=10)
    parser.add_argument("--walkLength", type=int, default=50)
    parser.add_argument("--windowSize", type=int, default=2)
    parser.add_argument("--numNegative", type=int, default=1)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--shrinkage", type=float, default=0.8)
    parser.add_argument("--shrinkStep", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--logStep", type=int, default=100)
    return parser.parse_args()
