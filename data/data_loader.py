# data/data_loader.py
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_data(PATH):
    pass

def data_transforms():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform
