import copy

from graphConvPool3DPnet import SelfCorrelation, KMeansConv, LocalAdaptiveFeatureAggregation, GraphMaxPool
from threading import Thread
import torch
import torch.nn as nn


class ShrinkingLayerStack(nn.Module):
    def __init__(self, input_stack: int, stack_fork: int, mlp: nn.Module, learning_rate: int, k: int, kmeansInit, n_init, sigma: nn.Module, F: nn.Module, W: nn.Module,
                 M: nn.Module, B: nn.Module, C, P, mlp1: nn.Module, mlp2: nn.Module):
        """
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
        self.stack_fork = stack_fork
        stack_size = input_stack * stack_fork
        self.selfCorrStack = SelfCorrelationStack(stack_size, mlp, learning_rate)
        self.kmeansConvStack = KMeansConvStack(stack_size, k, kmeansInit, n_init, sigma, F, W, M, B, C, P)
        self.localAdaptFeaAggreStack = LocalAdaptiveFeatureAggregationStack(stack_size, mlp1, mlp2)
        self.graphMaxPoolStack = GraphMaxPoolStack(stack_size, k)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (S,N,I,D) where S= input stack, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = torch.repeat_interleave(feature_matrix_batch, self.stack_fork, dim=0)
        # feature_matrix_batch size = (S',N,I,D) where S'=stack_size, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = self.selfCorrStack(feature_matrix_batch)
        # feature_matrix_batch size = (S',N,I,D) where S'=stack_size, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch_, conv_feature_matrix_batch, cluster_index = self.kmeansConvStack(feature_matrix_batch)
        feature_matrix_batch = self.localAdaptFeaAggreStack(feature_matrix_batch, conv_feature_matrix_batch)
        output = self.graphMaxPoolStack(feature_matrix_batch, cluster_index)
        # output size = (S',N,K,D) where S'=stack_size, N=batch number, K=members, D=member dimensionality
        return output


class SelfCorrelationStack(nn.Module):
    def __init__(self, stack_size: int, mlp: nn.Module, learning_rate: int = 1.0):
        super().__init__()
        self.selfCorrelationStack = nn.ModuleList([SelfCorrelation(copy.deepcopy(mlp), learning_rate) for i in range(stack_size)])

    def forward(self, feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S=stack_size, N=batch number, I=members, D=member dimensionality
        output = selfCorrThreader(self.selfCorrelationStack, feature_matrix_batch)
        # output size = (S,N,I,D) where where S=stack_size, N=batch number, I=members, D=member dimensionality
        return output


class KMeansConvStack(nn.Module):
    def __init__(self, stack_size: int, k: int, kmeansInit, n_init: int, sigma: nn.Module, F: nn.Module, W: nn.Module,
                 M: nn.Module, B: nn.Module, C: int, P: int):
        super().__init__()
        self.kmeansConvStack = nn.ModuleList([
            KMeansConv(k, kmeansInit, n_init, copy.deepcopy(sigma), copy.deepcopy(F), copy.deepcopy(W),
                       copy.deepcopy(M), copy.deepcopy(B), C, P) for i in range(stack_size)])

    def forward(self, feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S=stack size, N=batch number, I=members, D=member dimensionality
        feature_matrix_batch, conv_feature_matrix_batch, cluster_index = kmeansConvThreader(self.kmeansConvStack,
                                                                                            feature_matrix_batch)
        # feature_matrix_batch size = (S,N,I,D) where where S=stack_size, N=batch number, I=members, D=member dimensionality
        # conv_feature_matrix_batch size = (S,N,I,D) where where S=stack_size, N=batch number, I=members, D=member dimensionality
        # cluster_index size = (S,M) where S=stack_size, M=N*I
        return feature_matrix_batch, conv_feature_matrix_batch, cluster_index


class LocalAdaptiveFeatureAggregationStack(nn.Module):
    def __init__(self, stack_size: int, mlp1: nn.Module, mlp2: nn.Module):
        super().__init__()
        self.localAdaptFeatAggreStack = nn.ModuleList([LocalAdaptiveFeatureAggregation(copy.deepcopy(mlp1), copy.deepcopy(mlp2)) for i
                                         in range(stack_size)])

    def forward(self, feature_matrix_batch: torch.Tensor, conv_feature_matrix_batch: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S = stack size, N=batch number, I=cluster members, D=member dimensionality
        # conv_feature_matrix_batch size = (S,N,I,D') where S= stack size, N=batch number, I=cluster members, D'=C+P
        output = threader(self.localAdaptFeatAggreStack, feature_matrix_batch, conv_feature_matrix_batch)
        # output size = (S,N,I,D') where S= stack size, N=batch number, I=cluster members, D'=C+P
        return output


class GraphMaxPoolStack(nn.Module):
    def __init__(self, stack_size: int, k: int):
        super().__init__()
        self.graphMaxPoolStack = nn.ModuleList([GraphMaxPool(k) for i in range(stack_size)])

    def forward(self, feature_matrix_batch: torch.Tensor, cluster_index: torch.Tensor):
        # feature_matrix_batch size = (S,N,I,D) where S=stack size, N=batch number, I=members, D=member dimensionality
        # cluster_index size = (S,M) where S=stack size, M=N*I
        output = threader(self.graphMaxPoolStack, feature_matrix_batch, cluster_index)
        # output size = (S,N,K,D) where S=stack size, N=batch number, K=clusters, D=member dimensionality
        return output


def selfCorrThreader(modules, input_tensor):
    list_append = []
    threads = []
    for i, t in enumerate(input_tensor):
        threads.append(Thread(target=selfCorrAppender, args=(modules[i], t, list_append, i)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    list_append.sort()
    list_append = list(map(lambda x: x[1], list_append))
    return torch.stack(list_append)


def selfCorrAppender(module, tensor, list_append, index):
    list_append.append((index, module(tensor)))


def kmeansConvThreader(modules, input_tensor):
    list1_append = []
    list2_append = []
    list3_append = []
    threads = []
    for i, t in enumerate(input_tensor):
        threads.append(
            Thread(target=kmeansAppender, args=(modules[i], t, list1_append, list2_append, list3_append, i)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    list1_append.sort()
    list2_append.sort()
    list3_append.sort()
    list1_append = list(map(lambda x: x[1], list1_append))
    list2_append = list(map(lambda x: x[1], list2_append))
    list3_append = list(map(lambda x: x[1], list3_append))
    return torch.stack(list1_append), torch.stack(list2_append), torch.stack(list3_append)


def kmeansAppender(module, input, list1_append, list2_append, list3_append, index):
    x, y, z = module(input)
    list1_append.append((index, x))
    list2_append.append((index, y))
    list3_append.append((index, z))


def threader(modules, input_tensor1, input_tensor2):
    list_append = []
    threads = []
    for i, t in enumerate(input_tensor1):
        threads.append(Thread(target=threaderAppender, args=(modules[i], t, input_tensor2[i], list_append, i)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    list_append.sort()
    list_append = list(map(lambda x: x[1], list_append))
    return torch.stack(list_append)


def threaderAppender(module, t1, t2, list_append, index):
    list_append.append((index, module(t1, t2)))


class GraphConvPool3DPnetStack(nn.Module):
    def __init__(self, shrinkingLayersStack: [ShrinkingLayerStack], mlp: nn.Module):
        super().__init__()
        self.neuralNet = nn.Sequential(*shrinkingLayersStack)
        self.mlp = mlp

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # x should be None when starting with a point cloud with no features apart from the euclidean coordinates
        # pos size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = pos.unsqueeze(0)
        # feature_matrix_batch size = (1,N,I,D) where N=batch number, I=members, D=member dimensionality
        output = self.neuralNet(feature_matrix_batch)
        # output size = (S,N,D) where S= stack size, N=batch number, D'=member dimensionality
        output = torch.mean(output, dim=0)
        # output size = (N,D) where N=batch number, D'=member dimensionality
        return self.mlp(output)