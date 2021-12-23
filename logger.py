''' Put numbers on a screen and in a file '''
import os
from datetime import datetime
from argparse import Namespace
import yaml

config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)


LOGS_DIR = parsed_config['log_dir']
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


class Logger:
    """Training status and results tracker """

    def __init__(self, args: Namespace,
                 save: bool = True, verbose: bool = True):
        """Instantiate logger object.

        Args:
            args: Command line arguments used to run experiment.
            verbose: Whether or not to print logger info to stdout.

        """

        self.verbose = verbose
        if save:
            self.dir = make_log_dir(args)
            os.mkdir(self.dir)
            self.log_path = os.path.join(self.dir, 'log.txt')
            self.model_path = os.path.join(self.dir, 'model.pt')
            self.log_file = open(self.log_path, 'w')
            self.make_header(args)

            self.train_results_dir = os.path.join(
                self.dir, 'train_results.txt')
            self.val_results_dir = os.path.join(self.dir, 'val_results.txt')
        else:
            self.log_file = None
            self.model_path = None
            self.dir = None

    def make_header(self, args: Namespace) -> None:
        """Start the log with a header giving general experiment info"""
        self.log('Experiment Time: {}'.format(datetime.now()))
        for p in vars(args).items():
            self.log(f' {p[0]}: {p[1]}')
        self.log('\n')

    def log(self, string: str) -> None:
        """Write a string to the log"""
        if self.log_file is not None:
            self.log_file.write(string + '\n')
        if self.verbose:
            print(string)

    def log_results(self, train_loss: list, val_loss: list, train_acc: list, val_acc: list, train_acc5: list, val_acc5: list):
        with open(self.train_results_dir,'w+') as train_file:
            train_file.write('LOSS,ACC,ACC5\n')
            for loss, acc, acc5 in zip(train_loss,train_acc,train_acc5):
                train_file.write(f'{loss},{acc},{acc5}\n')

        with open(self.val_results_dir,'w+') as val_file:
            val_file.write('LOSS,ACC,ACC5\n')
            for loss, acc, acc5 in zip(val_loss,val_acc,val_acc5):
                val_file.write(f'{loss},{acc},{acc5}\n')
            

    def get_model_path(self):
        return self.model_path

    def get_log_dir(self):
        return self.dir


def make_log_dir(args) -> str:
    """Create directory to store log, results file, model"""

    dir = os.path.join(
        LOGS_DIR,
        f'{args.model}_{args.dataset}_{args.optimizer}_lr{args.lr}_batch_size{args.batch_size}')

    if os.path.exists(dir):
        exists = True
        i = 1
        while exists:
            new_dir = dir + '_{}'.format(i)
            exists = os.path.exists(new_dir)
            i += 1
        return new_dir
    else:
        return dir
