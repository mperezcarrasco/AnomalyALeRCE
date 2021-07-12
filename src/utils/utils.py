import torch
import json
import os

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)

class EarlyStopping:
    """Early stopping as the convergence criterion.

        Args:
            args (string): hyperparameters for the training.
            patience (int): the model will stop if it not do improve in a patience number of epochs.

        Returns:
            stop (bool): if the model must stop.
            if_best (bool): if the model performance is better than the previous models.
    """
    def __init__(self, args):
        self.best_metric = 9999
        self.counter = 0
        self.patience = args.patience
        self.directory = args.directory

    def count(self, metric, model):
        is_best = bool(metric < self.best_metric)
        self.best_metric = min(metric, self.best_metric)
        if is_best:
            self.counter = 0
            torch.save(model.state_dict(), '{}/trained_parameters.pth'.format(self.directory))
        else:
            self.counter += 1
        if self.counter > self.patience:
            stop = True
        else:
            stop = False
        return stop, is_best

def save_metrics(metrics, root_dir, mode='test'):
    """save all the metrics."""
    mt_dir = os.path.join(root_dir, 'metrics_{}.json'.format(mode))
    with open(mt_dir, 'w') as mt:
        json.dump(metrics, mt)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def print_and_log(metrics, writer, epoch, mode):
    for metric, value in metrics.items():
        print("{}: {:.3f}".format(metric, value))
        writer.add_scalar('{}_{}'.format(metric, mode), value, epoch)
    return metrics