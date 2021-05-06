import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


def normalize(features):
    # Normalizing data using quntile transformer.
    scaler = QuantileTransformer(n_quantiles=1000)
    scaler.fit(features)
    features = scaler.transform(features)
    features[np.isnan(features)] = 0 #NaN to 0.
    return features.astype('float32')

def weighted_sampler(data, class_):
    #Define weighter sampler for balanced batches processing.
    class_sample_count = np.unique(class_, return_counts=True)[1]
    weights = 1. / torch.Tensor(class_sample_count)
    samples_weight = np.array([weights[t] for t in class_])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
              samples_weight.type('torch.DoubleTensor'),len(samples_weight))
    return sampler

def define_outlier_classes(args, data):
    #Define the type1 (wrong classified) and type2 (real unseen outlier) classes.
    outClasses = np.where((data['hierClass']==data['hierPredtmp']) & (data['classALeRCE']!= args.outlier), 0, data['classALeRCE']) #Inlier:0
    outClasses = np.where(data['hierClass']!=data['hierPredtmp'], 1, outClasses) #Type1:1
    outClasses = np.where(data['classALeRCE']==args.outlier, 2, outClasses) #Type2:2
    return outClasses

def map2numerical(labels):
    # Map string labels to numerical labels.
    labels_maped = labels
    for i, class_ in enumerate(np.unique(labels)):
        labels_maped = np.where(labels==class_, i, labels_maped)
    return labels_maped.astype('int8')

class ALeRCE(object):
    """ALeRCE Custom Dataloader.
    Return objects from the object dataset on the fly.
    """
    def __init__(self, args, data, feature_list):
        """Build and return a dataloader.
        
        Args:
            data (array): NxM matrix where N is the number of cells and M the number of genes in each cell.
            ids (list): List of ids associated to each gene.
        """
        #Features to be in [0,1] range.
        self.features = normalize(data[feature_list])
        self.oid = data.oid.values  
        self.classALeRCE = map2numerical(data['classALeRCE'])
        self.classOut = define_outlier_classes(args, data)
    
    def __len__(self):
        """
        Total number of objects in the object dataset.
        """
        return self.features.shape[0]
    
    def __getitem__(self, index):
        """Return an item from the object dataset.
        
        Args:
            index (int): index of the element to be returned.
        """
        return self.oid[index], self.features[index], self.classALeRCE[index], self.classOut[index]

def get_data(args, feature_list_pth='../data_raw/features_RF_model.pkl'):
    """Build and return the train and test dataloaders.
    
    Args:
        args (dict): dict of arguments.

    Return:
        dataloader_train (torch.nn.Dataloader) = dataloader to be used for training.
        dataloader_test (torch.nn.Dataloader) = dataloader to be used for testing.
    """
    feature_list = pd.read_pickle(feature_list_pth)
    train = pd.read_pickle('{}/train_data_filtered.pkl'.format(args.data_pth))
    test = pd.read_pickle('{}/test_data_filtered.pkl'.format(args.data_pth))

    #Select hierarchical class.
    train = train[train.hierClass==args.hierClass]
    train['hierPredtmp'] = train['hierClass']

    if args.outlier!='none':
        test = test[test['hierPred_without_{}'.format(args.outlier)]==args.hierClass]
        test['hierPredtmp'] = test['hierPred_without_{}'.format(args.outlier)]
    else:
        test = test[test['hierPred']==args.hierClass]
        test['hierPredtmp'] = test['hierPred']

    #Remove the outlier from training set and append to the test set.
    test = pd.concat([test, train[train.classALeRCE==args.outlier]], sort=False)
    train = train[train.classALeRCE!=args.outlier]

    #Defining validation set.
    fold_ixs = pd.read_pickle('{}/fold_{}_ixs.pkl'.format(args.data_pth, args.fold))
    val = train[(train.index.isin(fold_ixs)==False)]
    train = train[(train.index.isin(fold_ixs))]

    #generating dataloaders
    trainALerCE = ALeRCE(args, train, feature_list)
    sampler = weighted_sampler(train, trainALerCE.classALeRCE)
    dataloader_train = DataLoader(trainALerCE, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    dataloader_val = DataLoader(ALeRCE(args, val, feature_list), batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(ALeRCE(args, test, feature_list), batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_val, dataloader_test