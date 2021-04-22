from models.Autoencoder import ae
from models.VariationalAutoencoder import vae
from models.VaDE import vade

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
    implemented_networks = ('ae', 'vae', 'vade')
    assert args.model in implemented_networks

    model = None

    if args.model == 'ae':
        model = ae(args)

    elif args.model == 'vae':
        model = vae(args)

    elif args.model == 'vade':
        model = vade(args)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        cudnn.benchmark = True
    return model