import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorboard.backend.event_processing import event_accumulator

import os, glob

def plot_event(args):
    event = glob.glob(os.path.join(args.directory, 'event*'))
    ea = event_accumulator.EventAccumulator(event[0])
    ea.Reload() 
    ea.Tags()
    metrics =  np.unique([metric.split('_')[0] for metric in ea.Tags()['scalars']])
    fig, axs = plt.subplots(nrows=1, ncols=len(metrics), figsize=(15,8))
    for i, metric in enumerate(metrics):
        axs[i].set_title(metric)
        train = pd.DataFrame(ea.Scalars('{}_train'.format(metric))).value.values
        val = pd.DataFrame(ea.Scalars('{}_val'.format(metric))).value.values
        axs[i].plot(np.arange(len(train)), train, c='k', label='train')
        axs[i].plot(np.arange(len(val)), val, c='b', label='val')
        axs[i].set_title('Learning curve {}'.format(metric), fontsize=25)
        axs[i].set_xlabel('Epoch', fontsize=20)
        axs[i].set_ylabel(metric, fontsize=20)
        axs[i].grid(True)
        axs[i].legend(loc='best', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.savefig('{}/learning_curves.png'.format(args.directory))
    
def plot_histogram(args, scores_in, scores_out):
    plt.figure(figsize=(8,4))
    plt.title('Inliers vs Outliers {}'.format(args.model), fontsize=16)
    plt.hist(scores_in, label='Inliers', bins=25, density=True, histtype='step', color='b')
    plt.hist(scores_out, label='Outliers', bins=25, density=True, histtype='step', color='r')
    plt.legend(fontsize=14)
    plt.savefig('{}/histogram.png'.format(args.directory))
    plt.close()
    
def plot_metrics(args, metric_name, score, m1, m2):
    plt.title('{} curve. Score: {:.2f}'.format(metric_name, score), fontsize=20)
    plt.plot(m1, m2)
    if metric_name=='AU ROC':
        m1_name = 'FPR'
        m2_name = 'TPR'
        ident=[0.0, 1.0]
        plt.plot(ident, ident, c='r')
    elif metric_name=='AU PR':
        m1_name = 'Recall'
        m2_name = 'Precision'
        plt.plot([1.0, 0.0], [0.0, 1.0], c='r')
    plt.xlabel(m1_name, fontsize=18)
    plt.ylabel(m2_name, fontsize=18)
    plt.grid()
    plt.savefig('{}/{}.png'.format(args.directory, metric_name))