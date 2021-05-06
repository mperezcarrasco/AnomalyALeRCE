import sys, os
import argparse
import numpy as np
from subprocess import check_call

PYTHON = sys.executable

def launch_job(model, latent_dim, lr, hierClass, fold):
    """
    Function to launch the experimets.
    """
    cmd = "{} main.py --model {} --latent_dim {} --lr {} \
              --hierClass {} --fold {} --outlier {}".format(PYTHON,
                                                            model,
                                                            latent_dim,
                                                            lr,
                                                            hierClass,
                                                            fold)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Autoencoder',
                        help="Model to be used for the experiments.")
    parser.add_argument("--hierClass", type=str, default='Transient',
                        help="Hierarchical class to be used for the experiments.")
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

    latent_dims = [32, 64, 128]
    lrs = [0.005, 0.001, 0.0005, 0.0001]
    
    for latent_dim in latent_dims:
        for lr in lrs:
            for fold in range(1):
                launch_job(args.model, latent_dim, lr, args.hierClass, fold)
