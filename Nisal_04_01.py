# Nisal, Prajakta
# 1002_174_111
# 2024_10_22
# Assignment_04_01

import random
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred, n_classes=10):
    #
    # Compute the confusion matrix for a set of predictions
    #
    con_mat = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        con_mat[true_label, predicted_label] += 1
    return con_mat


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def train_cnn_torch(train_dl, test_dl, lr=0.01, epochs=1, test_mode=False):
    # Define the CNN architecture
    # Your neural network should have this exact architecture (it's okay to hardcode)

    model = CustomModel()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    train_loss_history = []
    train_accuracy_history = []

    loss_history = []
    accuracy_history = []

    con_mat = None

    y_true = []
    y_pred = []

    # print(f"epochs::{epochs}")
    for epoch in range(epochs):

        # training phase
        model.train()

        running_loss = 0
        correct_predictions = 0
        total_samples = 0

        for index, (inputs, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = loss_function(y_hat, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, y_index = torch.max(y_hat, 1)
            correct_predictions += (y_index == labels).sum().item()
            total_samples += labels.size(0)

            if test_mode:
                # print(f"{index} : hello")
                break
        d = 1 if test_mode else len(train_dl)
        train_loss_history.append(running_loss / d)
        train_accuracy_history.append(correct_predictions / total_samples)

        # evaluatin phase
        model.eval()

        running = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for index, (inputs, labels) in enumerate(test_dl):

                y_hat = model(inputs)
                loss = loss_function(y_hat, labels)

                running += loss.item()
                _, y_index = torch.max(y_hat, 1)
                correct += (y_index == labels).sum().item()
                total += labels.size(0)

                y_true.extend(labels.numpy())
                y_pred.extend(y_index.numpy())

                if test_mode:
                    # print(f"{index}::hellooo")
                    break
            d = 1 if test_mode else len(test_dl)
            # print(f"{correct_predictions}:{total}:{y_index}")
            loss_history.append(running / d)
            accuracy_history.append(correct / total)

    # saving
    torch.save(model.state_dict(), "cnn.pt")

    # default values for no training
    if epochs == 0:
        return model, np.array([]), None, np.array([])

    con_mat = confusion_matrix(np.array(y_true), np.array(y_pred), n_classes=10)
    if con_mat is not None and con_mat.size > 0:
        plt.matshow(con_mat, cmap="viridis")
        plt.colorbar()
        plt.savefig("confusion_matrix.png")
        plt.close()

    # print(f"ACCURACY_HISTORY: {accuracy_history}")
    return model, np.array(loss_history), con_mat, np.array(accuracy_history)


# if we increase the number of epochs to 100, it gives an accuracy of around 80%
# if we change activation function to Tanh(), the model gives a 92% accuracy for 100 epochs which is a significant improvement.
