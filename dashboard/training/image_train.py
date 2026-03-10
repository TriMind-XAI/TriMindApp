# @title Training Loop
import torch.optim as optim
import torch
import streamlit as st
import time
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses = []
train_accuracies = []
test_accuracies = []

def train_epoch(model,train_loader,criterion,optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

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


def test_epoch(model,test_loader):
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

def train_loop(model,train_loader,test_loader,num_epochs=15):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc = train_epoch(model,train_loader,criterion,optimizer)
        test_acc = test_epoch(model,test_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Test Acc: {test_acc:.2f}%")
        print("--------------------")
    return train_accuracies, test_accuracies