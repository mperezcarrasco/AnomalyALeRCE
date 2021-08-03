import argparse
import torch
import os

from preprocess import get_data
from models.main import build_network
from utils.utils import save_metrics
from utils.plots import plot_histogram

from sklearn.metrics import auc

def compute_metrics(scores, labels):
    """
    Computing the Area under the curve ROC and PR.
    """
    in_scores = scores[labels==0]
    out_scores = scores[labels==1]

    auroc, aupr = compute_roc_pr(in_scores, out_scores)
    metrics = {'AU ROC': auroc,
               'AU PR': aupr,
               }
    return metrics

def compute_roc_pr(inliers_scores, outlier_scores):
    auroc_score = auroc(inliers_scores, outlier_scores)
    aupr_score = aupr(inliers_scores, outlier_scores)
    return auroc_score, aupr_score

def auroc(in_scores, out_scores):
    scores = np.concatenate((in_scores, out_scores), axis=0)
    start = np.min(scores)
    end = np.max(scores)   
    gap = (end - start)/100000

    tprs = []
    fprs = []
    for delta in np.arange(end, start, -gap):
        tpr = np.sum(np.sum(out_scores >= delta)) / np.float(len(out_scores))
        fpr = np.sum(np.sum(in_scores >= delta)) / np.float(len(in_scores))
        tprs.append(tpr)
        fprs.append(fpr)
    return auc(fprs, tprs)

def aupr(in_scores, out_scores):
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
        precision = tp / (tp + fp)
        recall = tp / np.float(len(out_scores))
        precisions.append(precision)
        recalls.append(recall)
    return auc(recalls, precisions)

def print_metrics(metrics):
    for metric, value in metrics.items():
        print("{}: {:.3f}".format(metric, value))
    print("##########################################")

def test(args, dataloader):
    """Evaluting the anomaly detection model."""
    model = build_network(args).to(args.device)
    ### Loading the trained model...
    state_dict = torch.load('{}/trained_parameters.pth'.format(args.directory))
    model.load_state_dict(state_dict)
    
    scores = []
    out_labels = []
    
    model.eval()
    with torch.no_grad():
        for _, x, _, y_out in dataloader:
            x = x.float().to(args.device)

            score = model.compute_anomaly_score(x)
            scores.append(score.detach().cpu())
            out_labels.append(out_label.cpu())
    
    scores = torch.cat(scores).numpy()
    out_labels = torch.cat(out_labels).numpy()
    
    metrics = compute_metrics(scores, out_labels)
    save_metrics(metrics, args.directory, 'test')
    print_metrics(metrics)
    plot_histogram(args, scores[labels==0], scores[labels==1])


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
        job_name = '{}_{}_{}_lr{}_ld{}_fold{}'.format(args.model, args.hierClass, args.outlier, args.lr, args.z_dim, args.fold)
    else:
        job_name = '{}_{}_lr{}_ld{}_fold{}'.format(args.model, args.hierClass, args.lr, args.z_dim, args.fold)

    args.directory = os.path.join(args.r, job_name)

    # create dir to store the results.
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    writer = SummaryWriter(args.directory)

    _, _, dataloader_test = get_data(args)
    test(args, dataloader_test)