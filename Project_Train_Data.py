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



# DATASET

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, label_override=None):

        data = pd.read_csv(csv_file, header=None)

        self.X = data.iloc[:, :-1].values.astype("float32")

        if label_override is None:
            self.y = data.iloc[:, -1].values.astype("int64")
        else:
            self.y = np.full(len(self.X), label_override)

        # per-beat normalization
        self.X = (self.X - self.X.mean(axis=1, keepdims=True)) / \
                 (self.X.std(axis=1, keepdims=True) + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        signal = torch.tensor(self.X[idx]).unsqueeze(0)
        label = torch.tensor(self.y[idx])
        return signal, label



# LOAD DATA

train_data = ECGDataset(r"C:/Users/Maggie/OneDrive/Documents/BME 450/Final Project/mitbih_train.csv")
test_data  = ECGDataset(r"C:/Users/Maggie/OneDrive/Documents/BME 450/Final Project/mitbih_test.csv")

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))


# SAMPLE VISUALIZATION

sample_signal, sample_label = train_data[0]

plt.plot(sample_signal.squeeze().numpy())
plt.title(f"Sample ECG Beat - Class {sample_label}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# MODEL

class ECGNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

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

        print(f"loss: {loss.item():.6f}")


# TEST LOOP
def test_loop(dataloader, model, loss_fn):

    model.eval()

    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            predicted = pred.argmax(1)

            correct += (predicted == y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = correct / len(dataloader.dataset)

    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print("Confusion Matrix:\n", cm)

    
    # PLOT CONFUSION MATRIX
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.show()


# SETUP
model = ECGNet(num_classes=5)

batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# TRAINING
epochs = 15
train_losses = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n----------------------")

    model.train()
    epoch_loss = 0

    for X, y in train_loader:
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print("Epoch loss:", avg_loss)

    test_loop(test_loader, model, loss_fn)

print("Done!")


# TRAINING CURVE
plt.plot(train_losses)
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# FINAL PREDICTION
with torch.no_grad():
    signal, label = test_data[0]
    output = model(signal.unsqueeze(0))

print("Raw output:", output)
print("Predicted class:", torch.argmax(output).item())
print("True class:", label.item())
       
