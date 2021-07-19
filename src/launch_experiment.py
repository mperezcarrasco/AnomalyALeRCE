import sys, os
import argparse
import numpy as np
from subprocess import check_call

PYTHON = sys.executable

def launch_job(model, latent_dim, lr, hierClass, fold, outlier):
    """
    Function to launch the experimets.
    """
    cmd = "{} main.py --model {} --z_dim {} --lr {} \
              --hierClass {} --fold {} --outlier {}".format(PYTHON,
                                               model,
                                               latent_dim,
                                               lr,
                                               hierClass,
                                               fold,
                                               outlier)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='ae',
                        help="Model to be used for the experiments.")
    parser.add_argument("--hierClass", type=str, default='Transient',
                        help="Hierarchical class to be used for the experiments.")
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Optimizer learning rate')
    parser.add_argument('--z_dim', default=32, type=int,
                        help='Latent space dimensionality')
    args = parser.parse_args()
    
    if args.hierClass=='Transient':
        possible_outliers = ['SLSN',
                             'SNII',
                             'SNIa',
                             'SNIbc']
    elif args.hierClass == 'Stochastic':
        possible_outliers = [ 'AGN' ,
                             'Blazar',
                             'CV/Nova',
                             'QSO',
                             'YSO']
    elif args.hierClass == 'Periodic':
        possible_outliers = ['CEP',
                             'DSCT',
                             'E',
                             'RRL',
                             'LPV']
    
    #outlier = 'none'
    #lrs = [0.0005, 0.0001]
    #z_dims = [32, 84, 128]
    
    for outlier in possible_outliers:
        for fold in range(5):
            launch_job(args.model, args.z_dim, args.lr, args.hierClass, fold, outlier)
