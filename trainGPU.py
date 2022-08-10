# Built-in modules
from stackGraphConvPool3DPnet import GraphConvPool3DPnetStack, ShrinkingLayerStack
from graphConvPool3DPnet import getDevice

# Third-party modules
import torch.optim
from torch_geometric.datasets import ModelNet
import torch_geometric.data as data
from torch_geometric.transforms import SamplePoints
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import os
import statistics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

torch.set_printoptions(threshold=10_000)
torch.autograd.set_detect_anomaly(True)

# Hyper-parameters
sample_points = 800
batch_size = 16
dimensionality = 3
learning_rate = 0.01
epochs = 500
out_points_decay = 5


class MLP2Layers(nn.Module):
    def __init__(self, in_feature1, out_feature1, out_feature2, out_feature3, out_feature4, out_feature5, out_feature6):
        super().__init__()
        lin1 = nn.Linear(in_feature1, out_feature1)
        lin2 = nn.Linear(out_feature1, out_feature2)
        lin3 = nn.Linear(out_feature2, out_feature3)
        lin4 = nn.Linear(out_feature3, out_feature4)
        lin5 = nn.Linear(out_feature4, out_feature5)
        lin6 = nn.Linear(out_feature5, out_feature6)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.xavier_uniform_(lin3.weight)
        nn.init.xavier_uniform_(lin4.weight)
        nn.init.xavier_uniform_(lin5.weight)
        nn.init.xavier_uniform_(lin6.weight)
        self.neuralNet = nn.Sequential(
            lin1,
            nn.ReLU(),
            lin2,
            nn.ReLU(),
            lin3,
            nn.Tanh(),
            lin4,
            nn.ReLU(),
            lin5,
            nn.ReLU(),
            lin6,
            nn.ReLU()
        )

    def forward(self, X):
        return self.neuralNet(X)



# ----------------------------------------------------------BEGINNING MODEL--------------------------------------------#
# Shrinking Layer 1
input_feature = 3
out_points = sample_points // out_points_decay
input_stack = 1
stack_fork = 2
n_init = 1
C = input_feature
P = 3
out_feature = C + P
mlp = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer1 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                      M, B, C, P, mlp1, mlp2)

# Shrinking Layer 2
out_points = out_points // out_points_decay
n_init = 1
input_stack = input_stack * stack_fork
stack_fork = 3
C = input_feature
P = 3
out_feature = C + P
mlp = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer2 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                      M, B, C, P, mlp1, mlp2)

# Shrinking Layer 3
out_points = 1
n_init = 1
input_stack = input_stack * stack_fork
stack_fork = 2
C = input_feature
P = 3
out_feature = C + P
mlp = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2Layers(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer3 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                      M, B, C, P, mlp1, mlp2)

# Shrinking layers
shrinkingLayers = [shrinkingLayer1, shrinkingLayer2, shrinkingLayer3]


# MLP classifier
class MLPClassifer(nn.Module):
    def __init__(self, in_feature: int):
        super().__init__()
        lin1 = nn.Linear(in_feature, in_feature * 2)
        lin2 = nn.Linear(in_feature * 2, in_feature * 2 + 10)
        lin3 = nn.Linear(in_feature * 2 + 10, in_feature * 2 + 20)
        lin4 = nn.Linear(in_feature * 2 + 20, in_feature * 2 + 10)
        lin5 = nn.Linear(in_feature * 2 + 10, 10)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.xavier_uniform_(lin3.weight)
        nn.init.xavier_uniform_(lin4.weight)
        nn.init.xavier_uniform_(lin5.weight)
        self.main = nn.Sequential(
            lin1,
            nn.ReLU(),
            lin2,
            nn.ReLU(),
            lin3,
            nn.ReLU(),
            lin4,
            nn.ReLU(),
            lin5
        )

    def forward(self, feature_matrix_batch):
        output = self.main(feature_matrix_batch.squeeze())
        return output


mlpClassifier = MLPClassifer(input_feature)


# ---------------------------------------------FUNCTIONS-------------------------------------------------------------- #

def save_checkpoint(model, optimizer, scheduler, epoch, epoch_losses, training_accuracies, test_losses, test_accuracies, learning_rates, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'epoch_losses': epoch_losses,
        'training_accuracies': training_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'learning_rates': learning_rates
    }, save_path)


def load_checkpoint(model, optimizer, scheduler, load_path):
    try:
        checkpoint = torch.load(load_path)
        print("Progress file in the folder")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state diction read")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state dictionary read")
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Scheduler state dictionary read")
        epoch = checkpoint['epoch']
        print("Epoch read")
        print(epoch + 1)
        return epoch + 1
    except:
        print("Progress file not in the folder")
        return 0


def printLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']}")


# Training loop
def training_loop(gpu, training_dataloader, model, loss_fn, optimizer):
    losses = []
    correct = 0
    for batch_n, batch in enumerate(training_dataloader):
        batch_size = int(batch.batch.size()[0] / sample_points)
        X = batch.pos.cuda(non_blocking=True).view(batch_size, sample_points, -1) + torch.normal(
            torch.zeros(batch_size, sample_points, dimensionality), torch.full((batch_size, sample_points,
                                                                               dimensionality), fill_value=0.1)).cuda(gpu)
        y = batch.y.cuda(non_blocking=True).flatten()
        # Compute predictions
        pred = model(None, X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_n % 25 == 0:
            torch.distributed.reduce(loss, 0)
            if gpu == 0:
                loss, current = loss.item(), batch_n * len(X)
                print(f"loss: {loss:>7f}")

    return torch.tensor(losses, device=f"cuda:{gpu}"), torch.tensor(correct, device=f"cuda:{gpu}")


# Test loop
def test_loop(gpu, test_dataloader, model, loss_fn):
    test_losses = []
    correct = 0

    with torch.no_grad():
        for batch_n, batch in enumerate(test_dataloader):
            batch_size = int(batch.batch.size()[0] / sample_points)
            X = batch.pos.cuda(non_blocking=True).view(batch_size, sample_points, -1)
            y = batch.y.cuda(non_blocking=True).flatten()
            pred = model(None, X)
            test_losses.append(loss_fn(pred, y).item())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss = statistics.mean(test_losses)
    print(f"{gpu}: test accuracy {correct}")
    return torch.tensor(correct, device=f"cuda:{gpu}"), torch.tensor(test_loss, device=f"cuda:{gpu}")


def createdir(path):
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created")
    except FileExistsError:
        print(f"Directory '{path}' already exists")


def next_training_number(path):
    listdir = os.listdir(path)
    if listdir == []:
        return 1
    else:
        list_number = map(lambda x: int(x.replace("train", "")), filter(lambda x: x.startswith("train"), listdir))
        return max(list_number) + 1 if list_number is not [] else 1


def train_optimisation(gpu, gpus, training_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, dir_path):
    epoch_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    learning_rates = []
    for i in range(epochs):
        if gpu == 0:
            print(f"Epoch {i + 1}\n-------------------------------")

        losses, training_accuracy = training_loop(gpu, training_dataloader, model, loss_fn, optimizer)
        average_loss = torch.mean(losses)
        torch.distributed.reduce(average_loss, 0, torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(training_accuracy, 0, torch.distributed.ReduceOp.SUM)
        test_accuracy, test_loss = test_loop(gpu, test_dataloader, model, loss_fn)
        torch.distributed.reduce(test_accuracy, 0, torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(test_loss, 0, torch.distributed.ReduceOp.SUM)
        if gpu == 0:  # the following operations are performed only by the process running in the first gpu
            average_loss = average_loss / torch.tensor(gpus, dtype=torch.float)  # average loss among all gpus
            print(test_accuracy)
            test_accuracy = test_accuracy / torch.tensor(len(test_dataloader.dataset),
                                                         dtype=torch.float) * torch.tensor(100.0)
            training_accuracy = training_accuracy / torch.tensor(len(training_dataloader.dataset),
                                                                 dtype=torch.float) * torch.tensor(100.0)
            print(test_accuracy)
            test_loss = test_loss / torch.tensor(gpus, dtype=torch.float)
            epoch_losses.append(average_loss.item())
            training_accuracies.append(training_accuracy.item())
            test_losses.append(test_loss.item())
            test_accuracies.append(test_accuracy.item())
            learning_rates.append((optimizer.param_groups[0])["lr"])
            print(f"Training average loss: {average_loss.item()}")
            print(f"Training accuracy: {training_accuracy.item()}")
            print(f"Test average loss: {test_loss.item()}")
            print(f"Test average accuracy: {test_accuracy.item()}%")
            printLearningRate(optimizer)
            #scheduler.step(test_loss.item())
            if average_loss.item() <= 0.35:
                for param_group in optimizer.param_groups:
                    print("Learning rate changed to 0.001")
                    param_group['lr'] = 0.001
            save_checkpoint(model, optimizer, scheduler, i, epoch_losses, training_accuracies, test_losses, test_accuracies, learning_rates,
                            os.path.join(dir_path, f"epoch{i}.pth"))

# ---------------------------------------------FUNCTIONS-------------------------------------------------------------- #


# ----------------------------------------------MULTI-GPU MODEL------------------------------------------------------- #

def train(gpu, gpus, world_size):
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl') #TODO
    dir_path = None
    if gpu == 0:
        dir_path = "stackgraphConvPool3DPnet"
        createdir(dir_path)
        training_number = next_training_number(dir_path)
        dir_path = os.path.join(dir_path, f"train{training_number}")
        createdir(dir_path)

    model = GraphConvPool3DPnetStack(shrinkingLayers, mlpClassifier)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    loss_fn = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.01, patience=10,
    #                                                       threshold_mode='abs', verbose=True)

    model = DDP(model, device_ids=[gpu])

    training_data = ModelNet("ModelNet10_train_data", transform=lambda x: SamplePoints(sample_points)(x))
    training_sampler = DistributedSampler(training_data, num_replicas=world_size)
    training_dataloader = data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=False, num_workers=0,
                                          pin_memory=True, sampler=training_sampler)

    test_data = ModelNet("Modelnet10_test_data", train=False, transform=lambda x: SamplePoints(sample_points)(x))
    test_sampler = DistributedSampler(test_data, num_replicas=world_size)
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True, sampler=test_sampler)

    train_optimisation(gpu, gpus, training_dataloader, test_dataloader, model, loss_fn, optimizer, None, dir_path)


if __name__ == '__main__':
    gpus = torch.cuda.device_count()
    nodes = 1
    world_size = nodes * gpus
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    train(args.local_rank, 12, 12) # TODO

