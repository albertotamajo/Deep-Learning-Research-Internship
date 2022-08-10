from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch_geometric.nn as gnn
import networkx as netx
from torch_geometric.utils import sort_edge_index, add_self_loops
from torch_scatter import scatter_max


class XConvBatch(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dim: int, kernel_size: int,
                 hidden_channels: Optional[int] = None, dilation: int = 1, bias: bool = True, num_workers: int = 1):
        super().__init__()
        self.xconv = gnn.XConv(in_channels, out_channels, dim, kernel_size,
                               hidden_channels, dilation, bias, num_workers)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # x size = (N,I,F) where N=batch number, I=members, F=member feature dimensionality
        # x could be None
        # pos size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        N, I, D = pos.size()
        output = None
        if x is None:
            output = self.xconv(x, pos.view(N*I, -1)).view(N, I, -1)
        else:
            output = self.xconv(x.view(N*I, -1), pos.view(N*I, -1)).view(N, I, -1)
        # output size = (N,I,F') where N=batch number, I=members, F'= out_channels
        return output


class ShrinkingLayer(nn.Module):
    def __init__(self, mlp: nn.Module, learning_rate: int, k: int, kmeansInit, n_init, sigma: nn.Module, F: nn.Module, W: nn.Module,
                 M: nn.Module, B: nn.Module, C, P, mlp1: nn.Module, mlp2: nn.Module):
        """
        A shrinking layer is a stacked sequence of modules:
            -Self-correlation
            -K-Means Convolution
            -Local adaptive Feature Aggregation
            -Graph Max Pool

        :param mlp: mlp: R^C -> R^C
        :param learning_rate: learning rate for the self-correlation module
        :param k: number of clusters for each point cloud
        :param kmeansInit: initializer for the kmeans algorithm
        :param n_init: number of restarts for the kmeans algorithm
        :param sigma: sigma: R^(C+P) -> R^(C+P)
        :param F: F: R^C -> R^(C x (C+P))
        :param W: W: R^C -> R^(C x (C+P))
        :param M: M: R^(C+P) -> R
        :param B: B: R^(C+P) -> R^(C+P)
        :param C: dimensionality of each point
        :param P: augmentation
        :param mlp1: R^(C+P) -> R^(C+P)
        :param mlp2: R^(C+P) -> R^(C+P)
        """
        super().__init__()
        self.selfCorr = SelfCorrelation(mlp, learning_rate)
        self.kmeansConv = KMeansConv(k, kmeansInit, n_init, sigma, F, W, M, B, C, P)
        self.localAdaptFeaAggre = LocalAdaptiveFeatureAggregation(mlp1, mlp2)
        self.graphMaxPool = GraphMaxPool(k)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = self.selfCorr(feature_matrix_batch)
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch, conv_feature_matrix_batch, cluster_index = self.kmeansConv(feature_matrix_batch)
        feature_matrix_batch = self.localAdaptFeaAggre(feature_matrix_batch, conv_feature_matrix_batch)
        output = self.graphMaxPool(feature_matrix_batch, cluster_index)
        # output size = (N,K,D) where N=batch number, K=members, D=member dimensionality
        return output


class SelfCorrelation(nn.Module):
    """
    This module tries to learn the correlation of each point with respect to itself
    """
    def __init__(self, mlp: nn.Module, learning_rate: int = 1.0):
        super().__init__()
        self.mlp = mlp
        self.learning_rate = torch.tensor(learning_rate, requires_grad=True)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        N, I, D = feature_matrix_batch.size()
        feature_matrix_batch = feature_matrix_batch.view(-1, D)
        # feature_matrix_batch size = (L,D) where L=N*I, D=member dimensionality
        Weight = self.mlp(feature_matrix_batch)
        # Weight size = (L,D) where L=N*I, D=member dimensionality
        output = (self.learning_rate * feature_matrix_batch * Weight) + feature_matrix_batch
        # output size = (L,D) where L=N*I, D=member dimensionality
        output = output.view(N, I, D)
        # output size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        # output size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        return output


class KMeansConv(nn.Module):
    """
    This module applies the k-means algorithm to obtain k clusters.
    A customised convolution operation is applied to each cluster separately.
    """
    def __init__(self, k: int, kmeansInit, n_init: int, sigma: nn.Module, F: nn.Module, W: nn.Module, M: nn.Module, B: nn.Module, C: int, P: int):
        super().__init__()
        self.k = k
        self.kmeansInit = kmeansInit
        self.n_init = n_init
        #self.kmeans = KMeans(n_clusters=k, init=kmeansInit, n_init=n_init)
        self.conv = Conv(sigma, F, W, M, B, C, P)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        N, I, D = feature_matrix_batch.size()
        clusters = []
        for i, feature_matrix in enumerate(feature_matrix_batch):
            kmeans = KMeans(n_clusters=self.k, init=self.kmeansInit, n_init=self.n_init)
            feature_matrix_numpy = torch.clone(feature_matrix).detach().cpu().numpy()
            #print(feature_matrix.size())
            #print(torch.unique(feature_matrix, dim = 0).size())
            #print(feature_matrix.size())
            #print(feature_matrix_numpy.shape)
            kmeans = kmeans.fit(feature_matrix_numpy)#feature_matrix.clone().detach().cpu().numpy())
            labels = np.apply_along_axis(lambda x: x + (i*self.k), axis=0, arr=kmeans.labels_)
            clusters.extend(labels)
        clusters = np.asarray(clusters)
        list1 = []
        list2 = []
        for i in range(self.k*N):
            indices = np.argwhere(clusters == i).flatten().tolist()
            if len(indices) != 1:
                edges = [e for e in netx.complete_graph(indices).edges]
                inverse_edges = list(map(lambda x: (x[1], x[0]), edges))
                edges.extend(inverse_edges)
                unzip = list(zip(*edges))
                list1.extend(unzip[0])
                list2.extend(unzip[1])
            else:
                list1.append(indices[0])
                list2.append(indices[0])

        edge_index = torch.tensor([list1, list2], dtype=torch.long, device=getDevice(feature_matrix_batch))
        edge_index = sort_edge_index(add_self_loops(edge_index)[0])
        conv_feature_matrix_batch = self.conv(feature_matrix_batch.view(-1, D), edge_index).view(N, I, -1)
        # conv_feature_matrix_batch size = (N,I,L) where N=batch number, I=members, L=C+P
        return feature_matrix_batch, conv_feature_matrix_batch, torch.tensor(clusters, dtype=torch.long, device=getDevice(feature_matrix_batch))


# class KMeansConv(nn.Module):
#     def __init__(self, k: int, kmeansInit, sigma: nn.Module, F: nn.Module, W: nn.Module, M: nn.Module, out_features: int, dtype=torch.float32):
#         super().__init__()
#         self.k = k
#         self.kmeans = KMeans(n_clusters=k, init=kmeansInit, n_init=1)
#         self.conv = Conv(sigma, F, W, M, out_features)
#         self.dtype = dtype
#
#     def forward(self, feature_matrix_batch):
#         # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
#         N, I, D = feature_matrix_batch.size()
#         labels = torch.empty(N, I, requires_grad=False, device=getDevice(feature_matrix_batch))
#         # labels size = (N,I) where N=batch number, I=members
#         highest_num_members = 0
#         for i, feature_matrix in enumerate(feature_matrix_batch):
#             kmeans = self.kmeans.fit(feature_matrix.clone().detach().cpu().numpy())
#             highest_members = np.amax(np.bincount(kmeans.labels_))
#             if highest_members > highest_num_members:
#                 highest_num_members = highest_members
#             labels[i] = torch.tensor(kmeans.labels_)
#         clusters_batch = torch.tensor(0, dtype=self.dtype, device=getDevice(feature_matrix_batch)).repeat(N, self.k, highest_num_members, D)
#         for i in range(N):
#             for j in range(self.k):
#                 cluster = feature_matrix_batch[i, (labels[i] == j).nonzero()].squeeze()  # nodes of the jth cluster in the ith batch
#                 # cluster size = (M,D) where M=members, D=member dimensionality
#                 size = cluster.size()[0] if cluster.dim() > 1 else 1
#                 clusters_batch[i, j, :size] = cluster
#
#         conv_clusters_batch = self.conv(clusters_batch)
#         return clusters_batch, conv_clusters_batch


def getDevice(t: torch.Tensor):
    if t.is_cuda:
        return f"cuda:{t.get_device()}"
    else:
        return "cpu"


class KMeansInitMostDistantFromMean:
    def __call__(self, *args, **kwargs):
        X, k = args
        mean = np.mean(X, axis=0)
        arg_sorted = np.argsort(np.apply_along_axis(lambda y: euclidean(mean, y), 1, X))
        output = X[np.flip(arg_sorted)[:k]]
        return output


class KMeansInit:
    def __call__(self, *args, **kwargs):
        X, k = args
        current_centroids = np.expand_dims(np.mean(X, axis=0), 0)
        for i in range(k - 1):
            X, current_centroids = self.next_centroid(X, current_centroids)

        return current_centroids

    def next_centroid(self, X, curr_centroids):
        highest_dist = 0.0
        next_centroid = None
        next_centroid_index = None
        for i, x in enumerate(X):
            max_dist = np.amax(np.apply_along_axis(lambda y: euclidean(x, y), 1, curr_centroids))
            if max_dist > highest_dist:
                next_centroid = x
                highest_dist = max_dist
                next_centroid_index = i

        return np.delete(X, next_centroid_index, 0), np.append(curr_centroids, np.expand_dims(next_centroid, 0), 0)


# class Conv(nn.Module):
#     def __init__(self, sigma: nn.Module, F: nn.Module, W: nn.Module, M: nn.Module, out_features):
#         super().__init__()
#         self.sigma = sigma
#         self.F = F
#         self.W = W
#         self.M = M
#         self.bias = torch.tensor([1.0,2.0,3.0,1.0,2.0,3.0], requires_grad=True, device="cuda")
#
#     def forward(self, clusters_batch):
#         torch.set_printoptions(threshold=10000)
#         # clusters_batch size = (N,K,I,D) where N=batch number, K=cluster number, I=cluster members, D=member dimensionality
#         N, K, I, D = clusters_batch.size()
#         clusters_batch_expanded = clusters_batch.unsqueeze(2).expand(-1, -1, I, -1, -1)
#         # clusters_batch_expanded size = (N,K,I,I,D) where where N=batch number, K=cluster number, I=cluster members, D=member dimensionality
#         single_vectors_batch = torch.repeat_interleave(clusters_batch, I, dim=2).view(N, K, I, I, D)
#         # single_vectors_batch size = (N,K,I,I,D) where N=batch number, K=cluster number, I=cluster members, D=member dimensionality
#         clusters_batch_subtracted = clusters_batch_expanded - single_vectors_batch
#         # clusters_batch_subtracted size = (N,K,I,I,D) where N=batch number, K=cluster number, I=cluster members, D=member dimensionality
#         clusters_batch_subtracted = clusters_batch_subtracted.view(N*K*I*I, D)
#         # clusters_batch_subtracted size = (L,D) where L=N*K*I*I, D=member dimensionality
#         A = torch.sum(self.F(clusters_batch_subtracted).view(N, K, I, I, -1), dim=3)
#         # A size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality  - D'=out_feature
#         Weight = self.M(A.view(N*K*I, -1)).view(N, K, I, -1)
#         # Weight size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality  - D'=out_feature
#         A = A * Weight
#         # A size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality  - D'=out_feature
#         B = self.W(clusters_batch.view(N*K*I,D)).view(N, K, I, -1)
#         # B size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality  - D'=out_feature
#         bias = self.bias.expand(I, -1)
#         # bias size = (I,D') where I=cluster members, D'=member dimensionality  - D'=out_feature
#         output = A + B + bias
#         # output size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality  - D'=out_feature
#         output = self.sigma(output.view(N*K*I, -1)).view(N, K, I, -1)
#         # output size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality  - D'=out_feature
#         print(output)
#         return output


class Conv(gnn.MessagePassing):
    def __init__(self, sigma: nn.Module, F: nn.Module, W: nn.Module, M: nn.Module, B: nn.Module, C: int, P: int):
        """
        Customised convolution operation

        :param sigma: sigma: R^(C+P) -> R^(C+P)
        :param F: F: R^C -> R^(C x (C+P)) where C is the dimensionality of the points and P>=0 is an hyperparameter of the model
        :param W: W: R^C -> R^(C x (C+P)) where C is the the dimensionality of the points and P>=0 is an hyperparameter of the model
        :param M: M: R^(C+P) -> R where C is the dimensionality of the points and P>=0 is an hyperparameter of the model
        :param B: B: R^(C+P) -> R^(C+P) where C is the dimensionality of the points and P>=0 is an hyperparameter of the model
        """
        super().__init__(aggr="mean")
        self.sigma = sigma
        self.F = F
        self.W = W
        self.M = M
        self.C = C
        self.P = P
        self.B = B

    def forward(self, feature_matrix, edge_index):
        # feature_matrix size = (N,C) where N=number points, C=points dimensionality
        # edge_index size = (2,E) where N=number points, E=edges
        return self.propagate(edge_index, feature_matrix=feature_matrix)

    def message(self, feature_matrix_i, feature_matrix_j):
        # feature_matrix_i size = (E, C) where E=edges, C=point dimensionality
        # feature_matrix_j size = (E, C) where E=edges, C=point dimensionality
        message = self.F(feature_matrix_j - feature_matrix_i)
        # message size = (E, M) where E=edges and M=C x (C+P)
        message = message.view(-1, self.C + self.P, self.C)
        # message size = (E, M, L) where E=edges, M=C+P, L=C
        feature_matrix_i_ = feature_matrix_i.unsqueeze(2)
        # feature_matrix_i_ size = (E,C,1) where E=edges, C=point dimensionality
        output = torch.bmm(message, feature_matrix_i_).squeeze()
        # output size = (E,M) where E=edges, M=C+P
        return output

    def update(self, aggr_out, feature_matrix):
        # aggr_out size = (N,L) where N=number points and L=C+P
        # feature_matrix size = (N,C) where N=number points, C=points dimensionality
        Weight = self.M(aggr_out)
        # Weight size = (N,1) where N=number points
        aggr_out = aggr_out * Weight
        # aggr_out size = (N,L) where N=number points and L=C+P
        transform = self.W(feature_matrix)
        # transform size = (N, M) where N=number points, M=(C x (C+P))
        transform = transform.view(-1, self.C + self.P, self.C)
        # transform size = (N, M, L) where N=number points, M=C+P, L=C
        feature_matrix = feature_matrix.unsqueeze(2)
        # feature_matrix size = (N,C,1) where N=number points, C=points dimensionality
        transformation = torch.bmm(transform, feature_matrix).squeeze()
        # transformation size = (N,L) where N=number points, L=C+P
        aggr_out = aggr_out + transformation
        # aggr_out size = (N,L) where N=number points, L=C+P
        adder = self.B(aggr_out)
        # adder size = (N,L) where N=number points, L=C+P
        output = aggr_out + adder
        # output size = (N,L) where N=number points, L=C+P
        output = self.sigma(output)
        # output size = (N,L) where N=number points, L=C+P
        return output


class LocalAdaptiveFeatureAggregation(nn.Module):
    def __init__(self, mlp1: nn.Module, mlp2: nn.Module):
        """
        This module tries to learn the correlation between each clusters and its convoluted counterparts

        :param mlp1: R^(C+P) -> R^(C+P)
        :param mlp2: R^(C+P) -> R^(C+P)
        """
        super().__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.softmax = nn.Softmax(0)

    def forward(self, feature_matrix_batch: torch.Tensor, conv_feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=cluster members, D=member dimensionality
        # conv_feature_matrix_batch size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        N, I, D = feature_matrix_batch.size()
        N_, I_, D_ = conv_feature_matrix_batch.size()
        augmentation = D_ - D
        if augmentation > 0:  # the dimensionality of each point needs to be increased
            feature_matrix_batch = F.pad(feature_matrix_batch, (0, augmentation))
            # feature_matrix_batch size = (N,I,D') where N=batch number, I=members, D'=C+P
        elif augmentation < 0:  # the dimensionality of each point needs to be decreased
            feature_matrix_batch = torch.nn.MaxPool1d((-1*augmentation)+1, stride=1)
            # feature_matrix_batch size = (N,I,D') where N=batch number, I=members, D'=C+P
        S1 = torch.mean(feature_matrix_batch, 1)
        # S1 size = (N,D') where N=batch number, D'=C+P
        S2 = torch.mean(conv_feature_matrix_batch, 1)
        # S2 size = (N,D') where N=batch number, D'=C+P
        Z1 = self.mlp1(S1)
        # Z1 size = (N,D') where N=batch number, D'=C+P
        Z2 = self.mlp2(S2)
        # Z2 size = (N,D') where N=batch number, D'=C+P
        M = self.softmax(torch.stack((Z1, Z2), 0))
        # torch.stack((Z1, Z2), 0)) size = (2,N,D') where N=batch number, D'=C+P
        # M size = (2,N,D') where N=batch number, D'=C+P
        M1 = M[0]
        # M1 size = (N,D') where N=batch number, D'=C+P
        M2 = M[1]
        # M2 size = (N,D') where N=batch number, D'=C+P
        M1 = M1.unsqueeze(1).expand(-1, I, -1)
        M2 = M2.unsqueeze(1).expand(-1, I, -1)
        # M1 size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        # M2 size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        output = (M1 * feature_matrix_batch) + (M2 * conv_feature_matrix_batch)
        # output size = (N,I,D') where N=batch number, I=cluster members, D'=C+P
        return output


# class LocalAdaptiveFeatureAggregation(nn.Module):
#     def __init__(self, mlp1: nn.Module, mlp2: nn.Module, weight=None, device="cpu"):
#         super().__init__()
#         self.mlp1 = mlp1
#         self.mlp2 = mlp2
#         self.softmax = nn.Softmax(1)
#         self.weight = weight if weight is not None else torch.randn(1, requires_grad=True, device=device)
#
#     def forward(self, clusters_batch, conv_clusters_batch):
#         # clusters_batch size = (N,K,I,D) where N=batch number, K=cluster number, I=cluster members, D=member dimensionality
#         # conv_clusters_batch size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality
#         N,K,I,D = clusters_batch.size()
#         N_,K_,I_,D_ = conv_clusters_batch.size()
#         augmentation = D_ - D
#         if augmentation > 0:
#             clusters_batch = F.pad(clusters_batch, (0, augmentation))
#             # clusters_batch size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality
#
#         # the mean of the elements of each cluster is replaced by this operation which uses a learnable parameter instead of the conventional 1/N
#         S1 = torch.sum(clusters_batch, 2) * self.weight
#         S2 = torch.sum(conv_clusters_batch, 2) * self.weight
#         # S1 size = (N,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         # S2 size = (N,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         Z1 = self.mlp1(S1.view(N*K, -1)).view(N, K, -1)
#         Z2 = self.mlp2(S2.view(N*K, -1)).view(N, K, -1)
#         # Z1 size = (N,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         # Z2 size = (N,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         M = self.softmax(torch.stack((Z1, Z2), 1))
#         # torch.stack((Z1, Z2), 1)) size = (N,2,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         # M size = (N,2,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         M1 = M[:, 0]
#         M2 = M[:, 1]
#         # M1 size = (N,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         # M2 size = (N,K,D') where N=batch number, K=cluster number, D'=member dimensionality
#         M1 = M1.unsqueeze(2).expand(-1, -1, I, -1)
#         M2 = M2.unsqueeze(2).expand(-1, -1, I, -1)
#         # M1 size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality
#         # M2 size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality
#         output = (M1 * clusters_batch) + (M2 * conv_clusters_batch)
#         # output size = (N,K,I,D') where N=batch number, K=cluster number, I=cluster members, D'=member dimensionality
#         return output


class GraphMaxPool(nn.Module):
    def __init__(self, k: int):
        """
        This module shrinks every clusters into a single node applying the MAX operation
        :param k: number of clusters for each batch
        """
        super().__init__()
        self.k = k

    def forward(self, feature_matrix_batch: torch.Tensor, cluster_index: torch.Tensor):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        # cluster_index size = (M) where M=N*I
        N, I, D = feature_matrix_batch.size()
        feature_matrix_batch = feature_matrix_batch.view(-1, D)
        # feature_matrix_batch size = (M,D) where M=N*I, D=member dimensionality
        output = scatter_max(feature_matrix_batch, cluster_index, dim=0)[0]
        # output size = (L,D) where L=k*N, D=member dimensionality
        output = output.view(N, self.k, -1)
        #output size = (N,K,D) where N=batch number, K=clusters, D=member dimensionality
        return output


class GraphConvPool3DPnet(nn.Module):
    """
    GraphConvPool3DPNet is a 3D point clouds artificial neural network classifier.
    It consists of two main sections:

        - Shrinking layers => a stacked sequence of layers (each layer is called Shrinking Layer).
                              Given a shrinking layer N, the shrinking layer N+1 receives the output of N.

        - Classifier => => a classic MLP which receives as input the output of the Shrinking Layers section
                           and outputs the probability distribution for categorisation purposes
    """
    def __init__(self, xConvLayers: [XConvBatch], shrinkingLayers: [ShrinkingLayer], mlp: nn.Module):
        super().__init__()
        self.xConvLayers = xConvLayers
        self.neuralNet = nn.Sequential(*shrinkingLayers, mlp)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # x should be None when starting with a point cloud with no features apart from the euclidean coordinates
        # pos size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        output = x
        for xconv in self.xConvLayers:
            output = xconv(output, pos)

        feature_matrix_batch = torch.cat((pos, output), 2) if output is not None else pos
        return self.neuralNet(feature_matrix_batch)