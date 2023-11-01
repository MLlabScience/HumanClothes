# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.data_loader import load_data, data_transforms
from models.model import MyModel, create_optimizer
from training.train import train_model
from utils.image_shower import image_shower

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

