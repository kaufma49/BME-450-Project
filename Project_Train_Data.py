# ECG Arrhythmia Classification using Neural Networks
# MIT-BIH Dataset (Signal-Based)
# BME 450

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix


# DATASET CLASS

class ECGDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)

        self.X = data.iloc[:, :-1].values.astype(np.float32)
        self.y = data.iloc[:, -1].values.astype(np.int64)

        # normalize each heartbeat
        self.X = (self.X - self.X.mean(axis=1, keepdims=True)) / \
                 (self.X.std(axis=1, keepdims=True) + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        signal = torch.tensor(self.X[idx]).unsqueeze(0)  # (1,187)
        label = torch.tensor(self.y[idx])
        return signal, label


# LOAD DATA

train_data = ECGDataset("mitbih_train.csv")
test_data = ECGDataset("mitbih_test.csv")

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))


# VISUALIZE SAMPLE ECG

sample_signal, sample_label = train_data[0]

plt.plot(sample_signal.squeeze().numpy())
plt.title(f"Sample ECG Beat - Class {sample_label}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# MODEL DEFINITION (1D CNN)

class ECGNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(32 * 46, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# TRAIN LOOP

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = (batch + 1) * len(X)
        print(f"loss: {loss.item():>7f} [{current}/{size}]")


# TEST LOOP

def test_loop(dataloader, model, loss_fn):

    model.eval()

    total_loss = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            total_loss += loss_fn(pred, y).item()
            predicted = pred.argmax(1)

            correct += (predicted == y).sum().item()

            all_preds.extend(predicted.numpy())
            all_labels.extend(y.numpy())

    accuracy = correct / len(dataloader.dataset)

    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print("Confusion Matrix:\n", cm)


# TRAINING SETUP

model = ECGNet(num_classes=5)

batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# TRAINING

epochs = 15

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n----------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)

print("Done!")


# FINAL PREDICTION

with torch.no_grad():
    signal, label = test_data[0]
    output = model(signal.unsqueeze(0))

print("Raw output:", output)
print("Predicted class:", torch.argmax(output).item())
print("True class:", label.item())