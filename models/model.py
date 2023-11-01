# models/model.py
import torch.nn as nn
import torchvision

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        pass

def create_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer
