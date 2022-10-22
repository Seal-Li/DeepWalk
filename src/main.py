from pyro import sample
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
from graph import Graph
from model import DeepWalk
from sampler import Sampler
from utils import arg_parser

def train(args, g):
    sampler = Sampler(
        g, 
        args.numWalks, 
        args.walkLength, 
        args.windowSize, 
        g.degrees(), 
        args.numNegative
    )

    dataloader = DataLoader(
        [str(node_id) for node_id in range(g.num_nodes())],
        batch_size=args.batchSize,
        shuffle=True,
        collate_fn=lambda nodes: sampler.sample(nodes)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepWalk(args.dim, g.num_nodes())
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.shrinkage)

    for epoch in range(args.epoch):
        epoch_total_loss = 0
        for step, (srcs, dsts, labels) in enumerate(dataloader):
            srcs, dsts, labels = srcs.to(device), dsts.to(device), labels.to(device)
            srcs_embedding, dsts_embedding = model(srcs, dsts)
            loss = model.loss(srcs_embedding, dsts_embedding, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_total_loss += loss.item()
            if step % args.logStep == 0:
                print('Epoch {:05d} | Step {:05d} | Step Loss {:.4f} | Epoch Avg Loss: {:.4f}'\
                    .format(epoch, step, loss.item(), epoch_total_loss / (step + 1)))
        if (epoch + 1) % args.shrinkStep == 0:
            scheduler.step()
    return model

def evaluate(testEdges, model):
    
    return None


if __name__ == "__main__":
    args = arg_parser()
    g = Graph()
    graph = g.random_graph(2000, 20000, direction=True)
    model = train(args, g)
    model.save_embeddings()
    torch.save(model, "./model.pt")