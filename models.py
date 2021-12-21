""" Holds model classes and a getter function to access them """

from torch import nn
from argparse import Namespace
import torch.nn.functional as F
import torchvision
import math
import numpy as np
import torch
from typing import Optional, Union


class ResNet18(nn.Module):
    """Wrapper class for the ResNet18 model (imported from Torch).

    Attributes:
        model: The Torch model.
        dim: Dimension of the last embedding layer

    """

    def __init__(self, num_classes: int = 10, one_channel: bool = False, pretrained: bool = False):
        """Instantiate object of class ResNet18.

        Args:
            num_classes: Number of classes (for applying model to classification task).
            one_channel: Whether or not input data has one colour channel (for MNIST).
            pretrained: Whether or not to get pretrained model from Torch.

        """
        super().__init__()
        if pretrained:
            self.model = torchvision.models.resnet18(pretrained=True)
        else:
            self.model = torchvision.models.resnet18(num_classes=num_classes)
        self.dim = 512
        if one_channel:
            # Set number of input channels to 1 (since MNIST images are greyscale)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.model(x)

    def get_dim(self):
        return self.dim


class MLP(nn.Module):
    ''' A basic 3 layer MLP '''

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet300(nn.Module):
    ''' A common network used for pruning analysis '''

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super(LeNet300, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleConv(nn.Module):
    ''' A simple convolutional neural net '''

    def __init__(self, num_channels: int, num_classes: int):
        super(SimpleConv, self).__init__()

        fc1_in = 784 if (num_channels == 1) else 1024
        self.conv1 = nn.Conv2d(num_channels, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(4)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(fc1_in, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x




################################################################################################################################################

def get_model(model_type: str, args: Namespace):
    ''' Getter function to return the appropraite version of a model '''

    if args.dataset == 'cifar10':
        input_dim = 32*32*3
        num_channels = 3
        num_classes = 10
    elif (args.dataset == 'mnist') or (args.dataset == 'fashion_mnist'):
        input_dim = 28*28
        num_channels = 1
        num_classes = 10
    elif args.dataset == 'imagenet':
        input_dim = 224*224
        num_channels = 3
        num_classes = 1000
    else:
        raise ValueError(f'Model hyper-parameters unavailable for dataset {args.dataset}')

    if model_type == 'mlp':
        model = MLP(input_dim, num_classes)    
    elif model_type == 'lenet300':
        model = LeNet300(input_dim, num_classes)
    elif model_type == 'simple_conv':
        model = SimpleConv(num_channels, num_classes)
    elif model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes, one_channel=(num_channels == 1), pretrained=args.pretrained)
    else:
        return ValueError(f'Model {args.model} is unavailable')

    if (args.load_path is not None):
        model.load_state_dict(torch.load(args.load_path))

    return model
