# Built-in modules
from stackGraphConvPool3DPnet import GraphConvPool3DPnetStack, ShrinkingLayerStack

# Third-party modules
import torch.optim
from torch_geometric.datasets import ModelNet
import torch_geometric.data as data
from torch_geometric.transforms import SamplePoints
import torch.nn as nn
import os
import statistics

torch.set_printoptions(threshold=10_000)
torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Hyper-parameters
sample_points = 800
batch_size = 16
learning_rate = 0.001
epochs = 500
out_points_decay = 5


class MLP2Layers(nn.Module):
    def __init__(self, in_feature1, out_feature1, out_feature2, out_feature3):
        super().__init__()
        self.neuralNet = nn.Sequential(
            nn.Linear(in_feature1, out_feature1),
            nn.ReLU(),
            nn.Linear(out_feature1, out_feature2),
            nn.ReLU(),
            nn.Linear(out_feature2, out_feature3),
            nn.Tanh()
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
mlp = MLP2Layers(C, C + 5, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 5, C + 10, C*(C+P))
W = MLP2Layers(C, C + 5, C + 10, C*(C+P))
M = MLP2Layers(C+P,  C + P + 5, C + P + 10, 1)
B = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
mlp1 = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
mlp2 = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
input_feature = out_feature
shrinkingLayer1 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2)

# Shrinking Layer 2
out_points = out_points // out_points_decay
n_init = 1
input_stack = input_stack * stack_fork
stack_fork = 2
C = input_feature
P = 3
out_feature = C + P
mlp = MLP2Layers(C, C + 5, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 5, C + 10, C*(C+P))
W = MLP2Layers(C, C + 5, C + 10, C*(C+P))
M = MLP2Layers(C+P,  C + P + 5, C + P + 10, 1)
B = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
mlp1 = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
mlp2 = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
input_feature = out_feature
shrinkingLayer2 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2)


# Shrinking Layer 3
out_points = 1
n_init = 1
input_stack = input_stack * stack_fork
stack_fork = 2
C = input_feature
P = 3
out_feature = C + P
mlp = MLP2Layers(C, C + 5, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 5, C + 10, C*(C+P))
W = MLP2Layers(C, C + 5, C + 10, C*(C+P))
M = MLP2Layers(C+P,  C + P + 5, C + P + 10, 1)
B = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
mlp1 = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
mlp2 = MLP2Layers(C+P,  C + P + 5, C + P + 10, C+P)
input_feature = out_feature
shrinkingLayer3 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2)


# Shrinking layers
shrinkingLayers = [shrinkingLayer1, shrinkingLayer2, shrinkingLayer3]


# MLP classifier
class MLPClassifer(nn.Module):
    def __init__(self, in_feature: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, in_feature * 2),
            nn.ReLU(),
            nn.Linear(in_feature * 2, in_feature * 2 + 20),
            nn.ReLU(),
            nn.Linear(in_feature * 2 + 20, 10)
        )

    def forward(self, feature_matrix_batch):
        output = self.main(feature_matrix_batch.squeeze())
        return output


mlpClassifier = MLPClassifer(input_feature)

model = GraphConvPool3DPnetStack(shrinkingLayers, mlpClassifier)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# -------------------------------------------------------END MODEL-----------------------------------------------------#

# Dataset and Dataloaders
training_data = ModelNet("ModelNet10_train_data", transform=lambda x: SamplePoints(sample_points)(x))
test_data = ModelNet("Modelnet10_test_data", train=False, transform=lambda x: SamplePoints(sample_points)(x))
training_dataloader = data.DataLoader(training_data, batch_size, True)
test_dataloader = data.DataLoader(test_data, batch_size, True)

# Loss function, optimizer, scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.1, patience=10, threshold_mode='abs',
                                                       verbose=True)


def save_checkpoint(model, optimizer, scheduler, epoch, epoch_losses, test_losses, test_accuracies, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'epoch_losses': epoch_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
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


# Training loop
def training_loop():
    losses = []
    for batch_n, batch in enumerate(training_dataloader):
        if batch_n == len(training_dataloader) - 1:
            break
        size = len(training_dataloader.dataset)
        X = batch.pos.to(device).view(batch_size, sample_points, -1)
        y = batch.y.to(device).flatten()
        # Compute predictions
        pred = model(None, X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_n % 25 == 0:
            loss, current = loss.item(), batch_n * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return losses


# Test loop
def test_loop():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch_n, batch in enumerate(test_dataloader):
            if batch_n == len(test_dataloader) - 1:
                break
            X = batch.pos.to(device).view(batch_size, sample_points, -1)
            y = batch.y.to(device).flatten()
            pred = model(None, X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct, test_loss


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


dir_path = "stackgraphConvPool3DPnet"
createdir(dir_path)
training_number = next_training_number(dir_path)
dir_path = os.path.join(dir_path, f"train{training_number}")
createdir(dir_path)

epoch_losses = []
test_losses = []
test_accuracies = []
# Optimization loop
for i in range(epochs):
    print(f"Epoch {i + 1}\n-------------------------------")
    losses = training_loop()
    average_loss = statistics.mean(losses)
    print(f"Average loss: {average_loss}")
    epoch_losses.append(average_loss)
    test_accuracy, test_loss = test_loop()
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    scheduler.step(test_loss)
    save_checkpoint(model, optimizer, scheduler, i, epoch_losses, test_losses, test_accuracies,
                    os.path.join(dir_path, f"epoch{i}.pth"))

# Save the model
torch.save(model, os.path.join(dir_path, "stackgraphConvPool3DPnet.pth"))