''' Implements a Trainer class along with model specific trainers that are subtypes of Trainer'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from argparse import Namespace
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from logger import Logger
from loss_functions import get_loss_function, la_roux_loss


class Trainer():
    ''' A Trainer base class that handles universal aspects of training.

    Such as:
        - The training loop
        - Logging
        - training instantiation

    Each subclass of Trainer must include a
        - train_epoch function
        - validate function

    '''

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, logger: Logger, idx: int, args: Namespace):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.early_stop = args.early_stop
        self.lr = args.lr
        self.epochs = args.epochs

        self.iters = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_accs5 = []
        self.val_accs5 = []
        self.idx = idx
        self.logger = logger
        self.eval_set = 'Validation Set'
        self.model_path = logger.get_model_path()
        self.log_dir = logger.get_log_dir()
        self.num_devices = len(args.device)
        self.clip = args.clip
        self.old_parameters = None

        self.model.to(self.device)

        self.loss_name = args.loss_function
        self.loss_function = get_loss_function(args.loss_function)
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError('Optimizer not supported')

        self.sched_str = args.scheduler
        if self.sched_str == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, patience=args.plateau_patience, verbose=True)
            self.plateau_factor = 0.1
            self.change_epochs = []
        elif self.sched_str == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
            self.change_epochs = None
        else:
            raise ValueError('Scheduler not available')

    def train(self):
        epochs_until_stop = self.early_stop
        if self.loss_name == 'la_roux':
                self.old_parameters = self.model.state_dict()

        # Get losses at initialization
        val_loss, val_acc, val_acc5 = self.validate()
        train_loss, train_acc, train_acc5 = self.validate(train = True)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.train_accs5.append(train_acc5)
        self.val_accs5.append(val_acc5)

        for epoch in range(1, self.epochs + 1):
            if self.loss_name == 'la_roux':
                self.old_parameters = self.model.state_dict()

            train_loss, train_acc, train_acc5 = self.train_epoch(epoch)
            val_loss, val_acc, val_acc5 = self.validate()

            self.logger.log('\n')
            self.logger.log(f'Training: Average Loss {train_loss}')
            if train_acc is not None:
                self.logger.log(
                    'Training set: Average top-1 accuracy: {:.2f}'.format(train_acc))
                self.logger.log(
                    'Training set: Average top-5 accuracy: {:.2f}'.format(train_acc5))
                self.logger.log(
                    'Validation set: Average loss: {:.6f}'.format(val_loss))
                self.logger.log('\n')

            if val_acc is not None:
                self.logger.log(
                    'Validation set: Top-1 Accuracy: {:.2f}'.format(val_acc))
                self.logger.log(
                    'Validation set: Top-5 Accuracy: {:.2f}'.format(val_acc5))
                self.logger.log('\n')
                self.logger.log('\n')

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_accs5.append(train_acc5)
            self.val_accs5.append(val_acc5)

            # Check if validation accuracy is worsening
            if val_acc <= max(self.val_accs):
                epochs_until_stop -= 1
                if epochs_until_stop == 0:  # Early stopping initiated
                    break
            else:
                epochs_until_stop = self.early_stop
                if self.idx == 0:
                    # Save model when loss improves
                    current_model_path = self.model_path
                    self.logger.log("Saving model...")

                    if self.num_devices > 1:
                        torch.save(self.model.module.state_dict(),
                                   current_model_path)
                    else:
                        torch.save(self.model.state_dict(), current_model_path)
                    self.logger.log("Model saved.\n")

            if self.sched_str == 'plateau':
                self.scheduler.step(val_loss)
                if self.optimizer.param_groups[0]['lr'] == self.plateau_factor * self.lr:
                    self.change_epochs.append(epoch)
                    self.lr = self.plateau_factor * self.lr
                    self.logger.log(
                        "Learning rate decreasing to {}\n".format(self.lr))
            else:
                self.scheduler.step()

        if self.idx == 0:
            self.train_report()
            self.logger.log_results(self.train_losses, self.val_losses,
                                    self.train_accs, self.val_accs, self.train_accs5, self.val_accs5)

    def train_report(self):
        best_epoch = np.argmax(self.val_accs)
        self.logger.log("Training complete.\n")
        self.logger.log("Best Epoch: {}".format(best_epoch + 1))
        self.logger.log("Training Loss: {:.6f}".format(
            self.train_losses[best_epoch]))
        if self.train_accs[best_epoch] is not None:
            self.logger.log("Training Top-1 Accuracy: {:.2f}".format(
                self.train_accs[best_epoch]))
            self.logger.log("Training Top-5 Accuracy: {:.2f}".format(
                self.train_accs5[best_epoch]))
        self.logger.log("{} Loss: {:.6f}".format(
            self.eval_set, self.val_losses[best_epoch]))
        if self.val_accs[best_epoch] is not None:
            self.logger.log("{} Top-1 Accuracy: {:.2f}".format(
                self.eval_set, self.val_accs[best_epoch]))
            self.logger.log("{} Top-5 Accuracy: {:.2f}".format(
                self.eval_set, self.val_accs5[best_epoch]))

    def train_epoch(self, epoch):
        pass

    def validate(self):
        pass


class GeneralTrainer(Trainer):
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, logger: Logger, idx: int, args: Namespace):
        super().__init__(model, train_loader, val_loader, device, logger, idx, args)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = AverageMeter()
        train_top1_acc = AverageMeter()
        train_top5_acc = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            if self.loss_name == 'la_roux':
                loss, output = self.loss_function(
                    data, target, self.model, self.old_parameters)
            else:
                output = self.model(data)
                loss = self.loss_function(output, target)

            top1_acc, top5_acc = compute_accuracy(output, target)
            train_loss.update(loss.item())
            train_top1_acc.update(top1_acc)
            train_top5_acc.update(top5_acc)
            loss.backward()

            # Clip the gradients for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            if batch_idx % 10 == 0:
                logged_loss = train_loss.get_avg()

                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx *
                    len(data), int(
                        len(self.train_loader.dataset) / self.num_devices),
                    100. * batch_idx / len(self.train_loader), logged_loss)
                self.logger.log(log_str)

        return train_loss.get_avg(), train_top1_acc.get_avg(), train_top5_acc.get_avg()

    def validate(self,train :bool = False ):
        la_roux = self.loss_name == 'la_roux'
        if train:
            return predict(self.model, self.device, self.train_loader, self.loss_function, la_roux=la_roux, old_parameters=self.old_parameters)
        else:
            return predict(self.model, self.device, self.val_loader, self.loss_function, la_roux=la_roux, old_parameters=self.old_parameters)


class AverageMeter():
    """Computes and stores the average and current value

    Taken from the Torch examples repository:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg


def compute_accuracy(output, target):
    ''' Helper function that computes accuracy '''
    with torch.no_grad():
        batch_size = target.shape[0]

        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top1_acc = correct[:1].view(-1).float().sum(0,
                                                    keepdim=True) * 100.0 / batch_size
        top5_acc = correct[:5].reshape(-1).float().sum(0,
                                                       keepdim=True) * 100.0 / batch_size

    return top1_acc.item(), top5_acc.item()


def predict(model: nn.Module, device: torch.device,
            loader: torch.utils.data.DataLoader, loss_function: nn.Module,
            precision: str = '32', calculate_confusion: bool = False, la_roux: bool = False, old_parameters: dict = None) -> tuple([float, float]):
    """Evaluate supervised model on data.

    Args:
        model: Model to be evaluated.
        device: Device to evaluate on.
        loader: Data to evaluate on.
        loss_function: Loss function being used.
        precision: precision to evaluate model with

    Returns:
        Model loss and accuracy on the evaluation dataset.

    """
    model.eval()

    total_loss = 0
    acc1 = 0
    acc5 = 0
    confusion = []
    n = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if la_roux:
                loss, output = loss_function(
                    data, target, model, old_parameters)
            else:
                output = model(data)
                loss = loss_function(output, target)
            
            total_loss += loss.item()
            

            cur_acc1, cur_acc5 = compute_accuracy(output, target)
            acc1 += cur_acc1
            acc5 += cur_acc5
    
    total_loss, acc1, acc5 = total_loss / len(loader), acc1 / \
        len(loader), acc5 / len(loader)

    

    if calculate_confusion:
        confusion = np.mean(np.stack(confusion), axis=0)
        return total_loss, acc1, acc5, confusion.round(2)
    else:
        return total_loss, acc1, acc5
