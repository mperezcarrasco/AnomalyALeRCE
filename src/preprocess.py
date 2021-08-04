import torch
from torch.utils.data import DataLoader

from utils.normalize import HandcraftedFeaturePreprocessor
import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

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
    outClasses = np.where((data['classALeRCE']!= args.outlier), 0, data['classALeRCE']) #Inlier:0
    outClasses = np.where(data['classALeRCE']==args.outlier, 1, outClasses) #Outlier:1
    return outClasses

def map2numerical(labels):
    # Map string labels to numerical labels.
    labels_maped = labels
    for i, class_ in enumerate(np.unique(labels)):
        labels_maped = np.where(labels==class_, i, labels_maped)
    return labels_maped.astype('int8')

def sample_outliers(data, outlier):
    outlier_proportion = data[data.classALeRCE==outlier].shape[0] / data.shape[0]
    
    if outlier_proportion > 0.1:
        #Number of samples to remove.
        n_rm = round(data[data.classALeRCE==outlier].shape[0] - data.shape[0] * 0.1)

        df_subset = data[data.classALeRCE==outlier].sample(n_rm)
        data = data.drop(df_subset.index)
    else:
        #Number of samples to remove.
        n_rm  = round((data.shape[0] * 0.1 - data[data.classALeRCE==outlier].shape[0]) / 0.1)

        df_subset = data[data.classALeRCE!=outlier].sample(n_rm)
        data = data.drop(df_subset.index)
    return data

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
        feature_preprocessor = HandcraftedFeaturePreprocessor()
        self.features = feature_preprocessor.preprocess(data[feature_list]).values
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
    test = test[test.hierClass==args.hierClass]

    #Remove the outlier from training set and append it to the test set.
    test = pd.concat([test, train[train.classALeRCE==args.outlier]], sort=False)
    test = sample_outliers(test, args.outlier)
    train = train[train.classALeRCE!=args.outlier]

    #Validation set.
    fold_ixs = pd.read_pickle('{}/fold_{}_ixs.pkl'.format(args.data_pth, args.fold))
    val = train[(train.index.isin(fold_ixs)==False)]
    train = train[(train.index.isin(fold_ixs))]

    #generating dataloaders
    trainALerCE = ALeRCE(args, train, feature_list)
    sampler = weighted_sampler(train, trainALerCE.classALeRCE)
    dataloader_train = DataLoader(trainALerCE, batch_size=args.batch_size, sampler=sampler, num_workers=16)
    dataloader_val = DataLoader(ALeRCE(args, val, feature_list), batch_size=args.batch_size, shuffle=False, num_workers=16)
    dataloader_test = DataLoader(ALeRCE(args, test, feature_list), batch_size=args.batch_size, shuffle=False, num_workers=16)

    return dataloader_train, dataloader_val, dataloader_test