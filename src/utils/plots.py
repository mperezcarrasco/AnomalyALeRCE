import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorboard.backend.event_processing import event_accumulator


def plot_event(event):
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
    plt.show()

def plot_events(event):
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
    plt.show()