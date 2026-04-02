# PyTorch train script
# ECG Arrhythmia Classification (BME 450)

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np



# DATASET LOADING


train_path = r"./data/train"
test_path = r"./data/test"

transform = transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor()])

training_data = ImageFolder(root=train_path, transform=transform)
test_data = ImageFolder(root=test_path, transform=transform)

categories = training_data.classes
print("Arrhythmia Classes:", categories)




# PRINT SAMPLE DATA


sample_num = 0

print('Image size:', training_data[sample_num][0].shape)
print('Label:', training_data[sample_num][1])

ima = training_data[sample_num][0]

print('min,max,mean,std:',ima.min().item(),ima.max().item(),ima.mean().item(),ima.std().item())

iman = ima.permute(1, 2, 0)
plt.imshow(iman)
plt.title(categories[training_data[sample_num][1]])
plt.show()


# MODEL 1 — FULLY CONNECTED NN

class FCNet(nn.Module):
    def __init__(self, num_classes):
        super(FCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(128*128*3, 128)
        self.dropout = nn.Dropout(0.3)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        return self.l3(x)


# MODEL 2 — CNN 


class CNNNet(nn.Module):
    def __init__(self, num_classes):
        super(CNNNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
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



# TEST LOOP + METRICS

def test_loop(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            predicted = pred.argmax(1)

            correct += (predicted == y).type(torch.float).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    accuracy = correct / size

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print("Confusion Matrix:\n", cm)


# TRAINING SETUP

num_classes = len(categories)

model = CNNNet(num_classes)   # change to FCNet(...) to compare models

batch_size = 16

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# TRAINING

epochs = 15

for t in range(epochs):
    print(f"\nEpoch {t+1}\n---------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Training Complete!")


# FINAL PREDICTION

sample_img, true_label = test_data[0]

with torch.no_grad():
    output = model(sample_img.unsqueeze(0))

print("Raw output:", output)
print("Predicted:", categories[torch.argmax(output).item()])
print("True:", categories[true_label])