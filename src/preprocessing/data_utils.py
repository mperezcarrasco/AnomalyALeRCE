
import pandas as pd
import numpy as np

def map_labels(labels):
    """mapping labels to the ALeRCE's taxonomy.
    ref: The Automatic Learning for the Rapid Classification of Events (ALeRCE) Alert Broker (https://arxiv.org/abs/2008.03303)
    """
    def remove_bad_oid():
        bad_oid = ['ZTF18abslpjy','ZTF18acurqaw','ZTF18aboebre','ZTF18acvvsnu','ZTF19aaydpzi','ZTF19aatevrp','ZTF18abtteyy',
                    'ZTF19aatmtne','ZTF18abtfgqr','ZTF18acetlrs','ZTF18abtmgfn','ZTF18acvvppd','ZTF18aczebty','ZTF18acefhxx',
                    'ZTF18acvhggp','ZTF18adbktyj','ZTF18aarcypa','ZTF18accngee','ZTF18acwvcbz','ZTF19aacypbw','ZTF18acenqto',
                    'ZTF19aapfnym','ZTF18acpefgk','ZTF18aavjcpf','ZTF18aceexmi','ZTF18accnmri','ZTF18acdvvgx',
                    'ZTF18accnbgw','ZTF18acemhyb','ZTF19abqrrto','ZTF19aadolpe','ZTF18abxbmqh','ZTF20aacbwbm']
        return labels.drop(labels[labels.oid.isin(bad_oid)].index)

    def map_to_hierarchical():
        hierarchical_labels = {'CEP': 'Periodic',
                            'DSCT': 'Periodic',
                            'E': 'Periodic',
                            'RRL': 'Periodic',
                            'LPV': 'Periodic',
                            'Periodic-Other': 'Periodic',
                            'SLSN': 'Transient',
                            'SNII': 'Transient',
                            'SNIa': 'Transient',
                            'SNIbc': 'Transient',
                            'AGN': 'Stochastic',
                            'Blazar': 'Stochastic',
                            'CV/Nova': 'Stochastic',
                            'QSO': 'Stochastic',
                            'YSO': 'Stochastic'}

        labels['hierClass'] = labels['classALeRCE'].map(hierarchical_labels)
        return labels.sample(frac=1).reset_index(drop=True)

    labels.loc[(labels['classALeRCE'] == 'RSCVn'), 'classALeRCE'] = 'Periodic-Other'
    labels.loc[(labels['classALeRCE'] == 'SNIIn'), 'classALeRCE'] = 'SNII'
    labels.loc[(labels['classALeRCE'] == 'EA'), 'classALeRCE'] = 'E' 
    labels.loc[(labels['classALeRCE'] == 'EB/EW'), 'classALeRCE'] = 'E'
    labels.loc[(labels['classALeRCE'] == 'Ceph'), 'classALeRCE'] = 'CEP'

    labels = labels[(labels['classALeRCE']!='NLAGN')]
    labels = labels[(labels['classALeRCE']!='NLQSO')]
    labels = labels[(labels['classALeRCE']!='ZZ')]
    labels = labels[(labels['classALeRCE']!='TDE')]
    labels = labels[(labels['classALeRCE']!='SNIIb')]

    labels = remove_bad_oid()
    labels = map_to_hierarchical()

    return labels


def filter_features(features):
    """Filtering features according to the ALerCE's light curve classfier.

    ref: Alert Classification for the ALeRCE Broker System: The Light Curve Classifier (https://arxiv.org/abs/2008.03311)
    """
    features = features.astype('float32')
    features = features[(features['n_det_1']>=6) | (features['n_det_2']>=6)]
    return features.replace([np.inf, -np.inf], np.nan)


def flag_data(data):
    """Adding FLAGS"""
    data = data[((data.hierClass=='Transient') & (data.flag_reference_change==False) & (data.flag_diffpos==True)) | 
                ((data.hierClass=='Periodic') & (data.flag_corrected_g==True) & (data.flag_corrected_r==True) & (data.flag_ndubious_g<1) & (data.flag_ndubious_r<1)) | 
                ((data.hierClass=='Stochastic') & (data.flag_corrected_g==True) & (data.flag_corrected_r==True)& (data.flag_ndubious_g<1) & (data.flag_ndubious_r<1)) ]
    return data