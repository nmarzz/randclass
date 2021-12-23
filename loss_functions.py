from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import torch


def get_loss_function(name: str):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'la_roux':
        return la_roux_loss
    else:
        raise ValueError(f'Loss function {name} not available')


def la_roux_loss(x: torch.tensor, target: torch.tensor, model: nn.Module, old_parameters: dict):
    ''' The loss function given by -sum p(y|x , old_params) log(p(y|x, variable_params) '''

    output = model(x)
    with torch.no_grad():
        model.eval()
        params = model.state_dict()        
        model.load_state_dict(old_parameters)
        pold = F.softmax(model(x),dim = 1)
        model.load_state_dict(params)
        model.train()
    
    
    logp = torch.log(F.softmax(output,dim = 1))   
    q = F.one_hot(target,num_classes = logp.shape[1])

    # loss = -torch.sum(q * logp) / q.shape[0]
    loss = -torch.sum(q * logp * pold) / q.shape[0]
    
    
    
    # This recreates cross entropy
    # output = model(x)
    # logp = torch.log(F.softmax(output,dim = 1))   
    # q = F.one_hot(target,num_classes = logp.shape[1])

    # loss = -torch.sum(q * logp) / q.shape[0]


    return loss, output
