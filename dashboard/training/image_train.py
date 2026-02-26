# @title Training Loop
import torch.optim as optim
import torch
import time
import torch.nn as nn
from data.image_data_loader import get_pneumonia_loaders
from models.image_models.simple_cnn import SmallCNN,BetterCNN
from models.image_models.resnet import ResNet8
train_loader, test_loader= get_pneumonia_loaders()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_epoch(model):
     
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy


def test_epoch(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train_loop(model,num_epochs=15):
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model)
        test_acc = test_epoch(model)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f} seconds")
        print("--------------------")