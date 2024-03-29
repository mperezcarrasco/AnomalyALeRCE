import argparse
import torch
import os

from src.preprocessing.create_dataloaders import get_data
from src.models.main import build_network
from src.utils.plots import plot_histogram, plot_metrics, plot_event
from src.utils.utils import  AverageMeter, save_metrics
import numpy as np
from sklearn.metrics import auc

def compute_metrics(args, scores, labels):
    """
    Computing the Area under the curve ROC and PR.
    """
    in_scores = scores[labels==0]
    out_scores = scores[labels==1]

    auroc, aupr = compute_roc_pr(args, in_scores, out_scores)
    metrics = {'AU ROC': auroc,
               'AU PR': aupr,
               }
    
    if args.model in ['iforest','ocsvm']:
        auroc=1-auroc
        aupr=1-aupr
        metrics = {'AU ROC': auroc,
                'AU PR': aupr,
                }
    return metrics

def compute_roc_pr(args, inliers_scores, outlier_scores):
    auroc_score, fprs, tprs = auroc(inliers_scores, outlier_scores)
    plot_metrics(args, 'AU ROC', auroc_score, fprs, tprs)
    aupr_score, recalls, precisions = aupr(inliers_scores, outlier_scores)
    plot_metrics(args, 'AU PR', aupr_score, recalls, precisions)
    return auroc_score, aupr_score

def auroc(in_scores, out_scores, eps=1e-5):
    scores = np.concatenate((in_scores, out_scores), axis=0)
    start = np.min(scores)
    end = np.max(scores)   
    gap = (end - start)/100000

    tprs = []
    fprs = []
    for delta in np.arange(end, start, -gap):
        tpr = np.sum(np.sum(out_scores >= delta)) / (float(len(out_scores) + eps))
        fpr = np.sum(np.sum(in_scores >= delta)) / (float(len(in_scores) + eps))
        tprs.append(tpr)
        fprs.append(fpr)
    return auc(fprs, tprs), fprs, tprs

def aupr(in_scores, out_scores, eps=1e-5):
    scores = np.concatenate((in_scores, out_scores), axis=0)
    start = np.min(scores)
    end = np.max(scores)   
    gap = (end - start)/100000
    
    precisions = []
    recalls = []
    for delta in np.arange(end, start, -gap):
        tp = np.sum(np.sum(out_scores >= delta)) #/ np.float(len(out_scores))
        fp = np.sum(np.sum(in_scores >= delta)) #/ np.float(len(in_scores))
        if tp + fp == 0: continue
        precision = tp / (tp + fp + eps)
        recall = tp / (float(len(out_scores))+eps)
        precisions.append(precision)
        recalls.append(recall)
    return auc(recalls, precisions), recalls, precisions

def print_metrics(metrics):
    for metric, value in metrics.items():
        print("{}: {:.3f}".format(metric, value))
    print("##########################################")

def test(args, dataloader, dataloader_c=None):
    """Evaluting the anomaly detection model."""
    model = build_network(args).to(args.device)
    ### Loading the trained model...
    state_dict = torch.load('{}/trained_parameters.pth'.format(args.directory))
    model.load_state_dict(state_dict)
    
    scores = []
    out_labels = []
    
    model.eval()
    if args.model in ['deepsvdd', 'classvdd']:
        model.set_c(dataloader_c)
    with torch.no_grad():
        for _, x, _, y_out in dataloader:
            x = x.float().to(args.device)

            score = model.compute_anomaly_score(x)
            scores.append(score.detach().cpu())
            out_labels.append(y_out.cpu())
    
    scores = torch.cat(scores).numpy()
    out_labels = torch.cat(out_labels).numpy()
    
    metrics = compute_metrics(args, scores, out_labels)
    plot_histogram(args, scores[out_labels==0], scores[out_labels==1])
    print_metrics(metrics)
    save_metrics(metrics, args.directory, 'test')
    return metrics 

def test_ML(args, model,test_features, test_labels):

    scores = model.score_samples(test_features)
    
    metrics = compute_metrics(args, scores, test_labels)
    plot_histogram(args, scores[test_labels==0], scores[test_labels==1])
    print_metrics(metrics)
    save_metrics(metrics, args.directory, 'test')
    return metrics 


def evaluation(args, model, dataloader):
    """Evalute the unsupervised model."""
    model.eval()
    model.set_metrics()
    losses = AverageMeter()
    with torch.no_grad():
        for _, x, y_cl, _ in dataloader:
            x = x.float().to(args.device)
            y_cl = y_cl.long().to(args.device)

            if args.model in ['classvdd']:
                loss = model.compute_loss(x, y_cl)
            else:
                loss = model.compute_loss(x)
            metrics = model.compute_metrics(x)

            losses.update(loss.item(), x.size(0))
    return losses, metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TESTINING PAREMETERS
    parser.add_argument('--data_pth', default='../data', type=str,
                        help='Dataset directory path.')
    parser.add_argument('--r', default="./experiments", type=str,
                        help='Results path. Weights and metrics will be stored here.')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size')
    parser.add_argument('--model', default='ae', type=str,
                        help='Model architecture.', choices=['ae', 'vae', 'vade'])
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
        job_name = '{}_{}_{}_fold{}'.format(args.model, args.hierClass, args.outlier, args.fold)
    else:
        job_name = '{}_{}_lr{}_ld{}_fold{}'.format(args.model, args.hierClass, args.lr, args.z_dim, args.fold)

    args.directory = os.path.join(args.r, job_name)

    # create dir to store the results.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    plot_event(args)

    dataloader_train, _, dataloader_test = get_data(args)
    if args.model in ['deepsvdd', 'classvdd']:
        test(args, dataloader_test, dataloader_train)
    else:
        test(args, dataloader_test)