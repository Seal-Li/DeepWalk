import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
from graph import Graph
from model import DeepWalk
from sampler import Sampler
import utils
import torch.nn.functional as F
from tqdm import tqdm

def train(args, g, testEdges):
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
        evaluate(model, testEdges, g)
        if (epoch + 1) % args.shrinkStep == 0:
            scheduler.step()
    return model

def evaluate(model, testEdges, g):
    labels = testEdges[-1]
    srcs, dsts = testEdges[0], testEdges[1]
    srcs_embedding = model.embedding(torch.tensor(srcs))
    dsts_embedding = model.embedding(torch.tensor(dsts))
    # print(srcs_embedding)
    preds = torch.sigmoid(torch.sum(srcs_embedding * dsts_embedding, dim=1))\
        .cpu().detach().numpy().tolist()
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
    print("Evaluate link prediction AUC: {:.4f}".format(metrics.auc(fpr, tpr)))
    return None

if __name__ == "__main__":
    args = utils.arg_parser()
    g = Graph()
    trainEdges = utils.load_train_edges(args.path)
    testEdges = utils.load_test_edges(args.path)
    graph = g.construct_graph(trainEdges[0], trainEdges[1])
    testEdges[0], testEdges[1] = g.encoder_new_edges(testEdges[0], testEdges[1])
    model = train(args, g, testEdges)
    model.save_embeddings()
    torch.save(model, "./model.pt")