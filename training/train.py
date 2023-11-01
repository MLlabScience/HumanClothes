# training/train.py
import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, trainloader, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(epoch, running_loss / len(trainloader))
