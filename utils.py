''' A place to put helpful functions that don't fit anywhere else. Mostly for testing and development. '''
from typing import Optional
import numpy as np
from torch import nn
import torch


def get_mse(A):
    ''' Get mean square error '''
    m = A.shape[0]
    n = A.shape[1]
    return np.linalg.norm(A, ord='fro')**2 / (m*n)


def get_rel_MSE(B, A):
    ''' Define relative MSE'''
    top = np.linalg.norm(A - B, ord='fro')**2
    bottom = np.linalg.norm(A, ord='fro')**2
    return top/bottom


def is_close(x, y, decimals: int = 2):
    ''' Check if two tensors are close. Replaces equality for many floating point ops '''
    x = torch.round(x * 10**decimals) / (10**decimals)
    y = torch.round(y * 10**decimals) / (10**decimals)

    return all(x == y)


def get_psnr(B, A):
    ''' Define peak signal to noise ration'''
    m = A.shape[0]
    n = A.shape[1]
    maxa = np.max(np.abs(A))
    mse = (np.linalg.norm(A-B)**2) / (n*m)
    return 10 * np.log10(maxa**2 / mse)


def generate_A(n, r):
    ''' Generate random full-rank symetric matrix with r dominant singular values '''
    G = np.random.randn(n, n) / np.sqrt(n)
    A = (G + G.transpose())/2
    U = np.zeros([n, n])
    for i in range(r):
        u = np.random.randn(n, 1)/np.sqrt(n)
        U += u @ u.transpose()

    A += 20 * U
    return A


def generate_Amn(m, n, r):
    ''' Generate random full-rank matrix with r dominant singular values '''
    G = np.random.randn(m, n) / np.sqrt(n)

    U = np.zeros([m, n])
    for i in range(r):
        u = np.random.randn(n, 1)/np.sqrt(n)
        v = np.random.randn(m, 1)/np.sqrt(m)
        U += v @ u.transpose()

    G += 20 * U
    return G


def num_parameters(model: nn.Module):
    ''' Get the number of trainable parameters in a nn.Module '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layers(model: nn.Module, layers: list = []):
    ''' Return every layer in a nn.Module '''
    children = list(model.children())
    if len(children) == 0:
        layers.append(model)
    else:
        for child in children:
            get_layers(child, layers)

    return layers


def get_convolutional2d_names(model: nn.Module):
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            names.append(name)
    return names


def get_linear_names(model: nn.Module):
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            names.append(name)
    return names


def reduction_factor(n: int, m: int, r: Optional[int] = None, bits: int = 32):
    ''' Get reduction factor of performing q-SVD [ie size(A) = f size(USV)]. Assume 32bit floats by default. Assume no sparsity '''

    if r is None:
        r = min(m, n)

    Acost = n*m*bits
    USVcost = r*(n + m + bits)

    return Acost / USVcost
