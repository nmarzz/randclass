''' Utility to load and evaluate a model on a particular dataset.

Can handle:
    - decomposed models (with the --decomposed flag)
    - vanilla saved models
'''
import argparse
import torch
from torch import nn
import numpy as np
from qsvd.loaders import get_loader
from qsvd.models import get_model
from qsvd.training import predict
from qsvd.psvd import load_decomposed_model


def get_args(parser):
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], metavar='D', help='Choice of train, val, test')
    parser.add_argument('--decomposed', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-hidden', type=int)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--load-path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=1331, metavar='S')

    args = parser.parse_args()

    return args


def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Evaluate a model')
    args = get_args(parser)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    train_loader, val_loader = get_loader(args.dataset, args.batch_size, distributed=False)
    if args.split == 'train':
        loader = train_loader
    else:
        loader = val_loader

    if args.decomposed:
        model, _, _, _, _ = load_decomposed_model(load_path=args.load_path, args=args)
    else:
        model = get_model(args.model, args)

    # print('FC1')
    # print(torch.sum(model.fc1.V == 0))
    # print(torch.sum(model.fc1.V != 0))
    # print(torch.sum(model.fc1.Ut == 0))
    # print(torch.sum(model.fc1.Ut != 0))

    # print('FC2')
    # print(torch.sum(model.fc2.V == 0))
    # print(torch.sum(model.fc2.V != 0))
    # print(torch.sum(model.fc2.Ut == 0))
    # print(torch.sum(model.fc2.Ut != 0))

    # print('FC3')
    # print(torch.sum(model.fc3.V == 0))
    # print(torch.sum(model.fc3.V != 0))
    # print(torch.sum(model.fc3.Ut == 0))
    # print(torch.sum(model.fc3.Ut != 0))

    model.to(device)

    metrics = predict(model, device, loader, nn.CrossEntropyLoss())

    print('Loss: {}'.format(metrics[0]))
    print('Top-1 Accuracy: {}'.format(metrics[1]))
    print('Top-5 Accuracy: {}'.format(metrics[2]))


if __name__ == '__main__':
    main()
