import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def sketch(folder, max_loss):
    list_epoch_files = map(lambda x: os.path.join(folder, x), filter(lambda x: x.startswith("epoch"), os.listdir(folder)))
    highest_epo = highest_epoch(folder)
    highest_epo_file = os.path.join(folder, f"epoch{highest_epo}.pth")
    highest_epo_dict = torch.load(highest_epo_file)
    epochs_range = np.arange(highest_epo + 1)
    fig, axis = plt.subplots(3, 1)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axis[0].set_title("Training losses vs Test losses over epochs")
    axis[0].plot(epochs_range, highest_epo_dict["epoch_losses"], label="Training losses")
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Loss")
    axis[0].plot(epochs_range, highest_epo_dict["test_losses"], label="Test losses")
    axis[0].set_ylim([0, max_loss])
    axis[0].legend(loc="upper right")
    axis[0].set_title("Learning rate over epochs")
    try:
        axis[1].set_title("Learning rate over epochs")
        axis[1].plot(epochs_range, highest_epo_dict["learning_rates"])
        axis[1].set_xlabel("Epochs")
        axis[1].set_ylabel("Learning rate")
    except:
        pass
    axis[2].set_title("Training accuracies vs Test accuracies over epochs")
    axis[2].set_xlabel("Epochs")
    axis[2].set_ylabel("Accuracy")
    axis[2].plot(epochs_range, highest_epo_dict["test_accuracies"], label="Test accuracies")
    try:
        axis[2].plot(epochs_range, highest_epo_dict["training_accuracies"], label="Training accuracies")
    except:
        pass
    axis[2].legend(loc="upper right")
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()



def highest_epoch(folder):
    listdir = os.listdir(folder)
    list_number = map(lambda x: int(x.replace(".pth", "")), map(lambda x: x.replace("epoch", ""), filter(lambda x: x.startswith("epoch"), listdir)))
    return max(list_number)


def learning_rates(folder):
    listdir = os.listdir(folder)
    list_files = filter(lambda x: x.startswith("epoch"), listdir)
    sort = np.argsort(np.asarray(map(lambda x: int(x.replace(".pth", "")), map(lambda x: x.replace("epoch", ""), list_files))))
    print(sort)
    list_files = list(map(lambda x: os.path.join(folder, x), list_files))
    lrates = []
    for i in sort:
        dict = torch.load(list_files[i])["optimizer_state_dict"]["param_groups"]
        lrates.append(dict[0]["lr"])
    return np.asarray(lrates)

def plot(file):
    dict = torch.load(file)
    losses = dict['train_d_losses']
    dict['train_d_losses']
    -3.4371751098660752e-06

torch.load("modelNet10Benchmarks")
