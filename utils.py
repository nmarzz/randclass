''' A place to put helpful functions that don't fit anywhere else. Mostly for testing and development. '''
from torch import nn


def freeze_embedder(model: nn.Module):
    ''' Freeze all layers except for the final linear classification layer    
        Occurs in place        
     '''

    # Get layers to freeze
    try:
        features = list(model.model.children())[:-1]
    except AttributeError:
        features = list(model.children())[:-1]

    # Freeze parameters in layer
    for l in features:
        for param in l.parameters():
            param.requires_grad = False
