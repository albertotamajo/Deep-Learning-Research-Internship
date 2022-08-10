# Built-in modules
from point2node import Point2Node

# Third-party modules
import torch.optim
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Hyper-parameters
sample_points = 1024
input_feature = 0
dim = 3
k = 10
d = 1
r = 5
out_xconv1 = 5
out_xconv2 = 7
batch_size = 16
learning_rate = 0.001
epochs = 200

# Model
model = Point2Node(sample_points, input_feature, dim, k, d, r, out_xconv1, out_xconv2)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# model progress
load_path = "progress.pth"

# Dataset and Dataloaders
training_data = ModelNet("ModelNet10_train_data", transform=lambda x: (
    SamplePoints(sample_points)(x).pos, x.y.to(torch.long)))  # torch.zeros(10, dtype=torch.float).scatter_(0, x.y, 1))

test_data = ModelNet("Modelnet10_test_data", train=False, transform=lambda x: (
    SamplePoints(sample_points)(x).pos, x.y.to(torch.long)))  # torch.zeros(10, dtype=torch.float).scatter_(0, x.y, 1))

training_dataloader = DataLoader(training_data, len(training_data), True, pin_memory=True)
for x in training_dataloader:
    training_data = TensorDataset(x[0].to(device), x[1].to(device))

training_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

# Loss function, optimizer, scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.1, patience=5, threshold_mode='abs',
                                                       verbose=True)


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, save_path)


def load_checkpoint(model, optimizer, scheduler, load_path):
    try:
        checkpoint = torch.load(load_path)
        print("Progress file in the folder")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state diction read")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state dictionary read")
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # print("Scheduler state dictionary read")
        epoch = checkpoint['epoch']
        print("Epoch read")
        print(epoch + 1)
        return epoch + 1
    except:
        print("Progress file not in the folder")
        return 0


# Training loop
def training_loop():
    for batch, (X, y) in enumerate(training_dataloader):
        if batch == len(training_dataloader) - 1:
            break
        size = len(training_dataloader.dataset)
        # Compute predictions
        pred = model(None, X)  # TODO
        loss = loss_fn(pred, y.flatten())  # TODO

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Test loop
def test_loop():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(None, X.to(device))
            test_loss += loss_fn(pred, y.to(device).flatten()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss


# Optimization loop
for i in range(load_checkpoint(model, optimizer, scheduler, load_path), epochs):
    print(f"Epoch {i + 1}\n-------------------------------")
    training_loop()
    scheduler.step()
    save_checkpoint(model, optimizer, "progress.pth", i)
    if i == (epochs - 1):
        test_loop()

# Save the model
torch.save(model, "point2node.pth")
