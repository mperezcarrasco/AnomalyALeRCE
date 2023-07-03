import argparse
import torch
import os

from src.preprocessing.create_dataloaders import get_data, get_data_ML
from src.evaluate import test, test_ML

from torch.utils.tensorboard import SummaryWriter

from src.train import train,train_ML


def launch_job(args):

    if args.outlier!='none':
        job_name = '{}_{}_{}/fold{}'.format(args.model, args.hierClass, args.outlier, args.fold)
    else:
        job_name = '{}_{}/fold{}'.format(args.model, args.hierClass, args.fold)
        

    args.directory = os.path.join(args.r, job_name)
    print('args.directory', args.directory)
    # create dir to store the results.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    if args.model in ['iforest','ocsvm']:
        dataloader_train, test_features, test_labels = get_data_ML(args)
        clf = train_ML(args, dataloader_train)
        test_ML(args, clf, test_features, test_labels)
    else:
        writer = SummaryWriter(args.directory)
        dataloader_train, dataloader_val, dataloader_test = get_data(args)
        train(args, writer, dataloader_train, dataloader_val)
        dataloader_train, _, dataloader_test = get_data(args)

        if args.model in ['deepsvdd', 'classvdd']:
            test(args, dataloader_test, dataloader_train)
        else:
            test(args, dataloader_test)    
    

def launch_all_outliers(args):

    if args.hierClass=='Transient':
        possible_outliers = ['SLSN',
                             'SNII',
                             'SNIa',
                             'SNIbc']
    elif args.hierClass == 'Stochastic':
        possible_outliers = ['QSO',
                             'YSO',
                             'AGN' ,
                             'Blazar',
                             'CV/Nova',
                             ]
    elif args.hierClass == 'Periodic':
        possible_outliers = ['CEP',
                             'DSCT',
                             'E',
                             'RRL',
                             'LPV']

    for outlier in possible_outliers:
        for fold in range(5):
            args.fold=fold
            args.outlier = outlier
            launch_job(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TRAINING PAREMETERS
    parser.add_argument('--data_pth', default='./data/ztf/data', type=str,
                        help='Dataset directory path.')
    parser.add_argument('--feature_list_pt', default='./data/ztf/data_raw/features_BHRF_model.pkl', type=str,
                        help='BHRF features path.')
    parser.add_argument('--r', default="./experiments", type=str,
                        help='Results path. Weights and metrics will be stored here.')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--model', default='ae', type=str,
                        help='Model architecture.', choices=['ae', 'vae', 'deepsvdd', 'classvdd','iforest','ocsvm'])
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
    
   
    parser.add_argument('--all_outliers', default=False, type=bool,
                        help='Calculate the results of all possible outliers.')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('You are using GPU', args.device)

    


    if args.all_outliers:
        print('Calculating all possibles outliers...', args.lr, args.z_dim)
        launch_all_outliers(args)
    else: 
        launch_job(args)



 



    
    
