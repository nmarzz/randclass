''' Main access point to training models in this project. '''

from training import predict
from models import get_model
from loaders import get_loader
from logger import Logger
from utils import freeze_embedder

import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp


def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--model', type=str, default='mlp')    
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--load-path', type=str)    
    parser.add_argument('--pretrained', action='store_true')  
    parser.add_argument('--train-set', action='store_true')    
    parser.add_argument('--freeze-embedder', action='store_true')
    parser.add_argument('--random-labels', action='store_true')
    parser.add_argument('--device', type=str, nargs='+', default=['cpu'],
                        help='Name of CUDA device(s) being used (if any). Otherwise will use CPU. \
                            Can also specify multiple devices (separated by spaces) for multiprocessing.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    return args


def main_worker(idx: int, num_gpus: int, distributed: bool, args: argparse.Namespace):

    device = torch.device(args.device[idx])
    
    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:29501',
                                world_size=num_gpus, rank=idx)

    # Get the data
    batch_size = int(args.batch_size / num_gpus)
    train_loader, val_loader = get_loader(name=args.dataset, batch_size=batch_size, distributed=distributed)

    if args.train_set:
        loader = train_loader
    else:
        loader = val_loader

    # Get model    
    model = get_model(args.model, args)
    if args.freeze_embedder:
        freeze_embedder(model)
        
    model.to(device)   

    loss, acc1, acc5 = predict(model, device, loader, loss_function = nn.CrossEntropyLoss())

    print(loss)
    print(acc1)
    print(acc5)
    


def main():
    parser = argparse.ArgumentParser(description='')
    args = get_args(parser)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    num_gpus = len(args.device)

    # If we are doing distributed computation over multiple GPUs
    if num_gpus > 1:
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, True, args))
    else:
        main_worker(0, 1, False, args)


if __name__ == '__main__':
    main()
