import argparse
import torch
import os

from preprocess import get_data
from models.main import build_network
from utils.utils import weights_init_normal, EarlyStopping, AverageMeter, save_metrics

from torch.utils.tensorboard import SummaryWriter
from torch import optim


def train(args, writer, dataloader_train, dataloader_val):
    """Train the unsupervised model."""
    model = build_network(args).to(args.device)
    model.apply(weights_init_normal)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Setting the early stopping.
    es = EarlyStopping(args)
    losses = AverageMeter()
    for epoch in range(args.epochs):
        print('Epoch: {}/{}'.format(epoch, args.epochs))
        for x, _, _ in dataloader_train:
            model.train()
            x = x.float().to(args.device)

            loss, metric = model.compute_loss(x)
            loss.backward()
            optimizer.zero_grad()
            losses.update(metric, x.size(0))
        
        print('Training Loss: {:.2f}'.format(losses.avg))
        losses_v = evaluate(args, model, dataloader_val)
        stop, is_best = es.count(losses_v, model)

        writer.add_scalar('loss_train', losses.avg, epoch)
        writer.add_scalar('loss_val', losses_v, epoch)
        if is_best:
            save_metrics(losses_v, args.directory, mode='val')
        if stop:
            break

def evaluate(args, model, dataloader):
    """Evalute the unsupervised model."""
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for x, _, _ in dataloader:
            x = x.float().to(args.device)

            _, metric = model.compute_loss(x)
            losses.update(metric, x.size(0))
    print('Test Loss: {:.2f}'.format(losses.avg))
    return losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TRAINING PAREMETERS
    parser.add_argument('--data_pth', default='../data', type=str,
                        help='Dataset directory path.')
    parser.add_argument('--r', default="./experiments", type=str,
                        help='Results path. Weights and metrics will be stored here.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--model', default='ae', type=str,
                        help='Model architecture.', choices=['ae', 'vae', 'vade'])
    parser.add_argument('--patience', default=150, type=int,
                        help='Patience for early stopping.')
    parser.add_argument("--outlier", type=str, default='Bogus',
                        help="Class to be used as outlier.")
    parser.add_argument("--hierClass", type=str, default='Transient',
                        help="Hierarchical class.")
    parser.add_argument("--fold", type=int, default=0,
                        help="fold used for this experiment.")
    # MODEL HIPERPARAMETERS
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Optimizer learning rate')
    parser.add_argument('--z_dim', default=32, type=int,
                        help='Latent space dimensionality')
    parser.add_argument('--in_dim', default=152, type=int,
                        help='Number of features in the input.')

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    job_name = '{}_{}_{}_lr{}_ld{}_fold{}'.format(args.model, args.hierClass, args.outlier, args.lr, args.z_dim, args.fold)
    args.directory = os.path.join(args.r, job_name)

    # create dir to store the results.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    writer = SummaryWriter(args.directory)

    dataloader_train, dataloader_val, dataloader_test = get_data(args)
    train(args, writer, dataloader_train, dataloader_val)

