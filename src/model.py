import torch

class DeepWalk(torch.nn.Module):
    def __init__(self, dims, numNodes):
        super(DeepWalk, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dims = dims
        self.numNodes = numNodes
        self.embedding = torch.nn.Embedding(numNodes+1, dims).to(self.device)

    def forward(self, srcs, dsts):
        srcs_embedding = self.embedding(srcs.int())
        dsts_embedding = self.embedding(dsts.int())
        return srcs_embedding, dsts_embedding
    
    def loss(self, srcs_embedding, dsts_embedding, labels):
        prob = torch.sigmoid(torch.sum(srcs_embedding * dsts_embedding, axis=1))
        prob = torch.clamp(prob, min=1e-7, max=1 - 1e-7)
        v = torch.mean(- (labels * torch.log(prob) + (1 - labels) * torch.log(1 - prob)))
        return v
    
    def save_embeddings(self):
        node_ids = torch.arange(self.numNodes + 1)
        with open("./embeddings.txt", "w+") as f:
            for node_id in node_ids:
                node_vector = self.embedding(node_id).to("cpu").tolist()
                vector = " ".join(list(map(lambda vec: str(vec), node_vector)))
                f.write(f"{node_id} {vector}\n")
        return None
            
    