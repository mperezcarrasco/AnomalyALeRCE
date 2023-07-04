from src.models.Autoencoder import ae
from src.models.VariationalAutoencoder import vae
from src.models.VaDE import vade
from src.models.DeepSVDD import deepsvdd
from src.models.ClasSVDD import classvdd



import torch
import torch.backends.cudnn as cudnn


def build_network(args):
    """Builds the feature extractor and the projection head.

        Args:
            args: Hyperparameters for the network building.

        Returns:
            model (torch.nn.Module): Network architecture.
    """
    # Checking if the network is implemented.
    implemented_networks = ('ae', 'vae', 'deepsvdd', 'classvdd')
    assert args.model in implemented_networks

    model = None

    if args.model == 'ae':
        model = ae(args)

    elif args.model == 'vae':
        model = vae(args)

    elif args.model == 'vade':
        model = vade(args)

    elif args.model == 'deepsvdd':
        model = deepsvdd(args)

    elif args.model == 'classvdd':
        model = classvdd(args)

    #if torch.cuda.is_available():
    #    if torch.cuda.device_count() > 1:
    #        model= torch.nn.DataParallel(model)
    #    cudnn.benchmark = True
    return model