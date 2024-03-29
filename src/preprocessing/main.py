import argparse


import itertools
import pickle

from sklearn import  model_selection
from preprocessing.ALeRCE_LC import train_bhrf
from preprocessing.data_utils import map_labels,filter_features, flag_data

import pandas as pd





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TRAINING PAREMETERS
    parser.add_argument('--labels_file', default='../data_raw/ZTF_labels_v8.0.1.csv', type=str,
                        help='Labels filename.')
    parser.add_argument('--features_file', default='../data_raw/features_20210719.parquet', type=str,
                        help='Features filename.')
    parser.add_argument('--features_list', default='../data_raw/features_BHRF_model.pkl', type=str,
                        help='Feature list (contains the features to be used for experiments.)')
    args = parser.parse_args()

    features = pd.read_parquet(args.features_file, engine='pyarrow')
    labels = pd.read_csv(args.labels_file)
    feature_list = pd.read_pickle(args.features_list)

    # Map labels to ALeRCE's Taxonomy (incluiding hierarchical) and remove some bad oids.
    labels = map_labels(labels)
    # Filter features accoding Sanches-Saez et al. 
    features = filter_features(features)
    
    features.rename(columns={'PPE':'Period_fit'}, inplace=True)
    merged_data = pd.merge(features, labels[['oid', 'classALeRCE', 'hierClass']], on='oid')

    # Defined train/test splits (80/20)% as defined in Sanchez-Saez et al.
    train, test = model_selection.train_test_split(merged_data, test_size=0.2, stratify=merged_data['classALeRCE'], random_state=42)

    skf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (train_index, val_index) in enumerate(skf.split(train, train.classALeRCE)):
        fold_path = '../data/fold_{}_ixs.pkl'.format(fold)
        with open(fold_path, 'wb') as f:
                pickle.dump(
                    train_index,
                    f,
                    pickle.HIGHEST_PROTOCOL)

    # Saving train data according to Sanchez-Saez filtering criterion.
    train.to_pickle('../data/train_data.pkl')  

    if args.train_late:
        print("Retraining late classifier...")
        model = train_bhrf(args, train, feature_list, save=True)
        unlabeled = features[~features.index.isin(train.oid)] #removing training set from unlabeled data
        x = unlabeled[feature_list].astype('float32')
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(-999)

        probas = model.predict_proba(x)
        unlabeled['Periodic'] = probas[:,0]
        unlabeled['Stochastic'] = probas[:,1]
        unlabeled['Transient'] = probas[:,2]
        unlabeled.to_pickle('../data/unlabeled_dataset_preds.pkl')
        pass

    #Training models treating each class as an outlier class.
    possible_outliers = ['CEP',
                        'DSCT',
                        'E',
                        'RRL',
                        'LPV',
                        'SLSN',
                        'SNII',
                        'SNIa',
                        'SNIbc',
                        'AGN',
                        'Blazar',
                        'CV/Nova',
                        'QSO',
                        'YSO']
    
    test['hierPred'] = model.predict(test[feature_list].fillna(-999))
    temp_test = test[feature_list]
    # Making predictions removing the outlier class from the dataset and training a late classifier.
    for outlier in possible_outliers:
        print('Training bhrf models removing the class {}'.format(outlier))
        temp_train = train[(train['classALeRCE']!=outlier)] #training data without outlier.
        model = train_bhrf(args, temp_train, feature_list)

        temp_test = test[feature_list] # to test using the BHRF trained without the otlier class.
        temp_test = temp_test.fillna(-999)
        hier_pred = model.predict(temp_test)

        test['hierPred_without_{}'.format(outlier)] = hier_pred
        test.to_pickle('../data/test_data.pkl') #Saving test data according to Sanchez-Saez filtering criterion.

    #NEW FILTER! only curves r and g >= 6 + FLAGS.
    train = train[(train['n_det_1']>=6) & (train['n_det_2']>=6)]
    test = test[(test['n_det_1']>=6) & (test['n_det_2']>=6)]

    # Using FLAGS to remove objects as discussed in ALeRCE's meetings.
    train = flag_data(train)
    test = flag_data(test)
    train.to_pickle('../data/train_data_filtered.pkl')
    test.to_pickle('../data/test_data_filtered.pkl')
    


