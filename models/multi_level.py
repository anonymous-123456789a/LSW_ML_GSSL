import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import time
import numpy as np
import random
from tqdm import tqdm
from collections import deque
from models.intrinsic_dimension import computeLID, computeID
from torch_cluster import random_walk
from models.utils import *
from models.first_eval_protocol_func import *

import sys
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def generate_mask_tensor(s):
    # Generate a random tensor with zeros and ones
    tensor = torch.zeros(s, s)

    # Generate random indices for each column where the value will be one
    indices = torch.randint(0, s - 1, (s,))

    # Adjust the indices to avoid diagonal positions
    for i in range(s):
        if indices[i] >= i:
            indices[i] += 1

    # Set the value to one at the random indices
    tensor[torch.arange(s), indices] = 1

    # Make sure the diagonal positions contain zero
    tensor.fill_diagonal_(0)

    return tensor

def replace_ones_with_zeros(matrix):
    # Iterate over each row in the matrix
    for row in matrix:
        # Find the indices where elements are one
        one_indices = (row == 1).nonzero(as_tuple=True)[0]

        # Randomly select K-2 indices from the ones to replace with zeros
        # Ensure the selected indices do not include the zero's index
        indices_to_zero = one_indices[torch.randperm(len(one_indices))[:len(one_indices) - 1]]

        # Set the selected indices to zero
        row[indices_to_zero] = 0

    return matrix

def find_k_hop_neighborhood(row, col, k):
    # Construct the adjacency list
    n = max(row.max(), col.max()) + 1  # Assuming nodes are labeled from 0
    nodes = np.arange(0, n.item())
    adjacency_list = [[] for _ in range(n.item())]  # Convert to int for indexing
    for r, c in zip(row.tolist(), col.tolist()):
        adjacency_list[r].append(c)
        adjacency_list[c].append(r)  # Graph is undirected

    # Function to find the k-hop neighborhood for a given node
    l = []
    for node in nodes:
        visited = {node}
        queue = deque([(node, 0)])
        while queue:
            current_node, depth = queue.popleft()
            if depth < k:
                for neighbor in adjacency_list[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        l.append(visited)
    return l


def find_external_node_for_each_node(row, col, k_hop_neighbors):
    # Construct the adjacency list
    n = max(row.max(), col.max()) + 1  # Assuming nodes are labeled from 0
    adjacency_list = [[] for _ in range(n.item())]  # Convert to int for indexing
    for r, c in zip(row.tolist(), col.tolist()):
        adjacency_list[r].append(c)
        adjacency_list[c].append(r)  # Graph is undirected

    # Find a non-neighbor node for each node by random selection
    def find_non_neighbor(node):
        neighbors = k_hop_neighbors[node]
        found = False
        num_nodes = len(adjacency_list)
        nodes_list = list(range(num_nodes))
        while found == False:
            candidate = random.choice(nodes_list)
            if candidate not in neighbors:
                non_neighbor = candidate
                found = True
        return non_neighbor

    # Iterate over all nodes to find a non-neighbor node
    non_neighbors = [find_non_neighbor(node) for node in range(len(adjacency_list))]

    return non_neighbors

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, edges,
                 tau: float = 0.5, m: float = 1.0, gamma: float = 1.5, batch: int = 0, num_clust: int = 7, walk_length: int = 2):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.edges = edges
        self.m: float = m
        self.gamma: float = gamma
        self.batch = batch
        self.walk_length = walk_length
        self.cluster_number = num_clust
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.epch = 0
        self.c1, self.p1, self.c2, self.p2, self.c3 = 0, 0, 0, 0, 0
        row, col = self.edges
        self.k_hop_neighbors = find_k_hop_neighborhood(row, col, self.walk_length)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return torch.mm(z1, z2.t())


    def clustering(self, embeddings, num_clust):
        embeddings = embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clust).fit(embeddings)
        centroids = kmeans.cluster_centers_
        preds = kmeans.predict(embeddings)
        preds = np.eye(self.cluster_number)[preds]
        preds = torch.tensor(preds, dtype=torch.float)
        centroids = torch.tensor(centroids, dtype=torch.float)
        return centroids, preds

    def readout(self, z: torch.Tensor) -> torch.Tensor:
        s = z.mean(dim=0)
        return torch.unsqueeze(s, dim=1)
    def random_walks_mean(self, z: torch.Tensor) -> torch.Tensor:
        # Create self-loop edges for all nodes (node_i -> node_i)
        self_loops = torch.stack([torch.arange(len(z),device=z.device), torch.arange(len(z),device=z.device)], dim=0)

        # Concatenate the existing edges with the self-loops
        self.edges = torch.cat([self.edges, self_loops], dim=1)
        row, col = self.edges
        non_neighbors = find_external_node_for_each_node(row, col, self.k_hop_neighbors)

        #start = torch.unique(row)
        start = torch.unique(torch.cat([row,col]))

        walk = random_walk(row=row, col=col, start=start, walk_length=self.walk_length, p=1.0, q=1.0)
        walk = walk[:, 1:]

        # znp = z.detach().numpy()
        znp = z.cpu().detach().numpy()
        pr = torch.Tensor([znp[i].mean(0) for i in walk.cpu().detach().numpy()])
        return pr, non_neighbors

    def proximity(self, z: torch.Tensor) -> torch.Tensor:
        return self.random_walks_mean(z)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:

        ################################Node level######################################
        device = z1.device

        mask_tensor = generate_mask_tensor(z1.shape[0])
        mask_tensor = mask_tensor.to(device)
        Sp_1 = torch.mm(z1, z2.t()).diag()
        Sn_1 = torch.mul(torch.mm(z1, z2.t()), mask_tensor).sum(1)
        ap_1 = torch.clamp_min(-Sp_1.detach() + 1 + self.m, min=0.)
        an_1 = torch.clamp_min(Sn_1.detach() + self.m, min=0.)


        """
        #################################Proximity level################################
        prox, non_negative = self.proximity(z2)
        n_nodes = z1.shape[0]
        mask_tensor = torch.zeros((n_nodes, n_nodes))
        mask_tensor[torch.arange(n_nodes), non_negative] = 1.0
        mask_tensor = mask_tensor.to(device)
        prox = prox.to(device)
        Sp_4 = torch.mm(z1, prox.t()).diag()
        Sn_4 = torch.mul(torch.mm(z1, prox.t()), mask_tensor).sum(1)
        ap_4 = torch.clamp_min(-Sp_4.detach() + 1 + self.m, min=0.)
        an_4 = torch.clamp_min(Sn_4.detach() + self.m, min=0.)

        #################################Cluster level##################################

        """
        """
        c1, p1 = self.clustering(z1, self.cluster_number)
        c2, p2 = self.clustering(z2, self.cluster_number)
        c3, _ = self.clustering(z3, self.cluster_number)
        self.c1, self.p1, self.c2, self.p2, self.c3 = c1.to(device), p1.to(device), c2.to(device), p2.to(device), c3.to(device)
        Sp_2 = (torch.mul(torch.mm(z1, self.c2.T), self.p1).sum(1))
        Sn_2 = (torch.mul(torch.mm(z1, self.c2.T), replace_ones_with_zeros(1-self.p1)).sum(1))
        ap_2 = torch.clamp_min(-Sp_2.detach() + 1 + self.m, min=0.)
        an_2 = torch.clamp_min(Sn_2.detach() + self.m, min=0.)
        """
        ##################################### graph level ###########################################

        s2 = self.readout(z2)
        s3 = self.readout(z3)
        Sp_3 = torch.mm(z1, s2)
        Sn_3 = torch.mm(z1, s3)
        ap_3 = torch.clamp_min(-Sp_3.detach() + 1 + self.m, min=0.)
        an_3 = torch.clamp_min(Sn_3.detach() + self.m, min=0.)




        delta_p = 1 - self.m
        delta_n = self.m
        ll = torch.log(1 + torch.exp(self.gamma * (an_1 * (Sn_1 - delta_n) - ap_1 * (Sp_1 - delta_p) + an_3 * (Sn_3 - delta_n) - ap_3 * (Sp_3 - delta_p))))


        return ll

    def semi_loss_batch(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        batch_size = self.batch
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1

        f = lambda x1, x2: torch.exp(torch.mm(x1, x2) / self.tau)

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]

            intra_view = f(z1[mask], z1.t())
            inter_view = f(z1[mask], z2.t())
            neg_aug = f(z1[mask], z3.t())

            losses.append(-torch.log(
                inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
                / (intra_view.sum(1) + inter_view.sum(1) + neg_aug.sum(1)
                   - intra_view[:, i * batch_size:(i + 1) * batch_size].diag()
                   - inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
                   - neg_aug[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             z3: torch.Tensor, mean: bool = True):
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        z3 = self.projection(z3)

        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        z3 = F.normalize(z3)

        if (self.batch == 0):
            l1 = self.semi_loss(z1, z2, z3)
            l2 = self.semi_loss(z2, z1, z3)
        else:
            l1 = self.semi_loss_batch(z1, z2, z3)
            l2 = self.semi_loss_batch(z2, z1, z3)

        ret = 0.5*(l1+l2)
        ret = ret.mean() if mean else ret.sum()

        return ret


def train_and_extract_ID_LID_multi_level(
        data,
        save=False,
        model_path='pretrained_models/cora/full_model/node.pth',
        weight_path='pretrained_models/cora/weights/node.pth',
        pretrained_weights=None,
        pretrained_model=None,
        m=0.1,
        gamma=1.5,
        pretrained=False,
        plot=True,
        title=None,
        learning_rate=0.0005,
        num_hidden=128,
        num_proj_hidden=128,
        activation=nn.PReLU(),
        base_model=GCNConv,
        num_layers=2,
        drop_edge_rate_1=0.2,
        drop_edge_rate_2=0.4,
        drop_feature_rate_1=0.3,
        drop_feature_rate_2=0.4,
        drop_scheme='uniform',
        tau=0.4,
        num_epochs=200,
        patience=200,
        num_epochs_eval=500,
        num_classes=7,
        weight_decay=0.00001,
        batch=0,
        rd_seed=129,
        device=torch.device("cpu"),
):
    torch.manual_seed(rd_seed)
    random.seed(rd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = data.to(device)

    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, data.edge_index, tau=tau, batch=batch, num_clust=num_classes, m=m, gamma=gamma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if pretrained:
        model.load_state_dict(torch.load(pretrained_weights))
        model.to(device)

    dict = {'accuracy': [],
            'kmeans': [],
            'nmi': [],
            'ari': []}

    best_accuracy = 0
    cpt_wait = 0
    with tqdm(total=num_epochs, desc='(T)') as pbar:
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            model.epch = epoch
            edge_index_1 = drop_edge(data, device, drop_scheme, drop_edge_rate_1)
            edge_index_2 = drop_edge(data, device, drop_scheme, drop_edge_rate_2)

            x_1 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_1)
            x_2 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_2)
            x_3 = shuffle(data.x)

            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)
            z3 = model(x_3, data.edge_index)
            loss = model.loss(z1, z2, z3)
            loss.backward()
            optimizer.step()


            if (epoch > 0):
                model.eval()
                with torch.no_grad():
                    accuracy = classifier(model, data, LogReg(num_hidden, num_classes), device,
                                          n_epochs=num_epochs_eval)
                    dict['accuracy'].append(accuracy)
                    res_clust = clustering_evaluation(model, data, num_classes)
                    dict['kmeans'].append(res_clust[0])
                    dict['nmi'].append(res_clust[1])
                    dict['ari'].append(res_clust[2])
                    pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy, "nmi": res_clust[1]})

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        if save:
                            torch.save(model, model_path)
                            torch.save(model.state_dict(), weight_path)

                    else:
                        cpt_wait += 1
                        if cpt_wait > patience:
                            break


            pbar.update()
    print("=== Final ===")
    model.load_state_dict(torch.load(weight_path))
    return model, dict
