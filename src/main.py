import argparse
import torch
import os

from preprocess import get_data
from models.main import build_network
from test import test
from utils.utils import weights_init_normal, EarlyStopping, AverageMeter, save_metrics, print_and_log

from torch.utils.tensorboard import SummaryWriter
from torch import optim


def train(args, writer, dataloader_train, dataloader_val):
    """Train the unsupervised model."""
    model = build_network(args).to(args.device)
    model.apply(weights_init_normal)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Setting the early stopping.
    es = EarlyStopping(args)
    for epoch in range(args.epochs):
        model.set_metrics()
        print('Epoch: {}/{}'.format(epoch, args.epochs))
        for _, x, _ , _ in dataloader_train:
            model.train()
            x = x.float().to(args.device)

            loss = model.compute_loss(x)
            metrics = model.compute_metrics(x)

            #Computing gradients
            loss.backward()
            optimizer.step()

            #Zero grading for next iteration.
            optimizer.zero_grad()

        print('Training Metrics...')
        print_and_log(metrics, writer, epoch, 'train')
        losses_v, metrics_v = evaluate(args, model, dataloader_val)
        print('Validation Metrics...')
        print_and_log(metrics_v, writer, epoch, 'val')
        print("##########################################")
        stop, is_best = es.count(losses_v.avg, model)

        if is_best:
            save_metrics(losses_v.avg, args.directory, mode='val')
        if stop:
            break

def evaluate(args, model, dataloader):
    """Evalute the unsupervised model."""
    model.eval()
    model.set_metrics()
    losses = AverageMeter()
    with torch.no_grad():
        for _, x, _, _ in dataloader:
            x = x.float().to(args.device)

            loss = model.compute_loss(x)
            metrics = model.compute_metrics(x)

            losses.update(loss.item(), x.size(0))
    return losses, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TRAINING PAREMETERS
    parser.add_argument('--data_pth', default='../data', type=str,
                        help='Dataset directory path.')
    parser.add_argument('--r', default="./experiments", type=str,
                        help='Results path. Weights and metrics will be stored here.')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--model', default='ae', type=str,
                        help='Model architecture.', choices=['ae', 'vae', 'vade'])
    parser.add_argument('--patience', default=100, type=int,
                        help='Patience for early stopping.')
    parser.add_argument("--outlier", type=str, default='none',
                        help="Class to be used as outlier.")
    parser.add_argument("--hierClass", type=str, default='Transient',
                        help="Hierarchical class.")
    parser.add_argument("--fold", type=int, default=0,
                        help="fold used for this experiment.")
    # MODEL HIPERPARAMETERS
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Optimizer learning rate')
    parser.add_argument('--z_dim', default=128, type=int,
                        help='Latent space dimensionality')
    parser.add_argument('--in_dim', default=152, type=int,
                        help='Number of features in the input.')

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.outlier!='none':
        job_name = '{}_{}_{}_lr{}_ld{}_fold{}'.format(args.model, args.hierClass, args.outlier, args.lr, args.z_dim, args.fold)
    else:
        job_name = '{}_{}_lr{}_ld{}_fold{}'.format(args.model, args.hierClass, args.lr, args.z_dim, args.fold)

    args.directory = os.path.join(args.r, job_name)

    # create dir to store the results.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    writer = SummaryWriter(args.directory)

    dataloader_train, dataloader_val, dataloader_test = get_data(args)
    train(args, writer, dataloader_train, dataloader_val)
    #test(args, dataloader_test)