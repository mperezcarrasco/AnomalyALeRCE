import pickle
import pandas as pd

from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier

def build_bhrf():
    """Building the BHRF model accoing to Sanchez-Saez et al."""
    model = RandomForestClassifier(n_estimators=500,
                                    max_features='auto',
                                    max_depth=None,
                                    n_jobs=-1,
                                    bootstrap=True,
                                    class_weight='balanced_subsample',
                                    criterion='entropy',
                                    min_samples_split=2,
                                    min_samples_leaf=1)
    return model


def save_model(model):
    with open('../data/hierarchical_level_RF_model.pkl', 'wb') as f:
                pickle.dump(
                    model,
                    f,
                    pickle.HIGHEST_PROTOCOL)


def filter_class(x_train, y_train, y_train_hier, class_):
    """Filter class for the bottom level of the classifier."""
    x_train_bottom = x_train.loc[y_train_hier==class_, :]
    y_train_bottom = y_train.loc[y_train_hier==class_]
    return x_train_bottom, y_train_bottom

def train_bhrf(args, train, feature_list, save=False):
    """Training module of the BHRF"""

    x_train, y_train, y_train_hier = train[feature_list], train.classALeRCE, train.hierClass
    x_train = x_train.fillna(-999)
    #x_test, y_test_alerce, y_test_hier = test[feature_list], test.classALeRCE, test.hierClass

    rf_model_hier = build_bhrf()

    #Training the top level of the classifier.
    rf_model_hier.fit(x_train, y_train_hier)

    if save:
        save_model(rf_model_hier)

    return rf_model_hier