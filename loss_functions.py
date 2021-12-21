from torch._C import Value
import torch.nn as nn
import torch


def get_loss_function(name:str):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'la_roux':
        return la_roux_loss
    else:
        raise ValueError(f'Loss function {name} not available')


def la_roux_loss(x : torch.tensor, target : torch.tensor):
    pass