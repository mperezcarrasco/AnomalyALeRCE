import numpy as np
import pandas as pd
from utils.norm_scales import feature_scales, use_log, use_clip_and_log
from sklearn.preprocessing import QuantileTransformer
from abc import ABC, abstractmethod


banned_features = [
    'mean_mag_1',
    'mean_mag_2',
    'min_mag_1',
    'min_mag_2',
    'Mean_1',
    'Mean_2',
    'n_det_1',
    'n_det_2',
    'n_pos_1',
    'n_pos_2',
    'n_neg_1',
    'n_neg_2',
    'first_mag_1',
    'first_mag_2',
    'MHPS_non_zero_1',
    'MHPS_non_zero_2',
    'MHPS_PN_flag_1',
    'MHPS_PN_flag_2',
    'W1', 'W2', 'W3', 'W4',
    'iqr_1',
    'iqr_2',
    'delta_mjd_fid_1',
    'delta_mjd_fid_2',
    'last_mjd_before_fid_1',
    'last_mjd_before_fid_2',
    'g-r_ml',
    'MHAOV_Period_1', 'MHAOV_Period_2'
]


class FeaturePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        pass


class QuantileFeaturePreprocessor(FeaturePreprocessor):
    def __init__(self):
        balanced_feature_sample = self._get_balanced_feature_sample()
        self.transformer = QuantileTransformer()
        self.transformer.fit(balanced_feature_sample.values)
        self.feature_list = balanced_feature_sample.columns

    def preprocess(self, features: pd.DataFrame):
        x = self.transformer.transform(features[self.feature_list].values)
        x = np.nan_to_num(x, nan=-1.0)
        df = features.copy()
        for col_idx, col in enumerate(df.columns):
            df[col] = x[:, col_idx]
        df = df[sorted(df.columns.values)]
        return df

    def _get_balanced_feature_sample(self):
        rf_probabilities = pd.read_pickle('data/rf_probs_for_objects_wo_labels.pkl')
        rf_classification = rf_probabilities.idxmax(axis=1)

        # Ban features
        features = pd.read_pickle('data/features_without_labels.pkl')
        allowed_features = [f for f in features.columns if f not in banned_features]
        features = features[allowed_features]

        # Same order on axis 0
        pseudo_labels = rf_classification.loc[features.index]

        # Same order on classes
        unique_classes = pseudo_labels.unique()

        spm_features = [f for f in features.columns if 'SPM' in f]
        samples_per_class = []
        for class_alerce in unique_classes:
            subset = pseudo_labels == class_alerce
            features_subset = features[subset].sample(n=100, replace=True)
            if class_alerce not in ['SNIa', 'SNIbc', 'SNII', 'SLSN']:
                features_subset[spm_features] = np.NaN
            samples_per_class.append(features_subset)
        balanced_feature_sample = pd.concat(samples_per_class, axis=0)
        return balanced_feature_sample


class HandcraftedFeaturePreprocessor(FeaturePreprocessor):
    def __init__(self):
        self.feature_scales = feature_scales
        self.use_log = use_log
        self.use_clip_and_log = use_clip_and_log
        self.banned_features = banned_features

    def preprocess(self, features):
        allowed_features = self.get_allowed_features(features.columns.values)
        preprocessed_features = features[allowed_features].copy()
        for feature_name in preprocessed_features.columns:
            scale = self.feature_scales[self.get_key_of_feature_scales(feature_name)]
            if self.should_use_log(feature_name):
                if preprocessed_features[feature_name].min() <= 0.0:
                    print(feature_name)
                preprocessed_features[feature_name] = np.log(preprocessed_features[feature_name])
                min_value = np.log(scale[0])
                max_value = np.log(scale[1])
            elif self.should_use_clip_and_log(feature_name):
                preprocessed_features[feature_name] = np.clip(
                    preprocessed_features[feature_name],
                    a_min=scale[0],
                    a_max=None)
                preprocessed_features[feature_name] = np.log(
                    preprocessed_features[feature_name])
                min_value = np.log(scale[0])
                max_value = np.log(scale[1])
            else:
                min_value = scale[0]
                max_value = scale[1]

            preprocessed_features[feature_name] = (
                (preprocessed_features[feature_name] - min_value)
                / (max_value - min_value)) - 0.5
            preprocessed_features[feature_name] = np.tanh(
                preprocessed_features[feature_name])
            preprocessed_features[feature_name] = np.nan_to_num(
                preprocessed_features[feature_name], nan=-1.1)
        preprocessed_features = preprocessed_features[
            sorted(preprocessed_features.columns.values)]
        return preprocessed_features

    def should_use_log(self, feature_name):
        for f in self.use_log:
            if f in feature_name:
                return True
        return False

    def should_use_clip_and_log(self, feature_name):
        for f in self.use_clip_and_log:
            if f in feature_name:
                return True
        return False

    def get_key_of_feature_scales(self, feature_name):
        for key in self.feature_scales.keys():
            if key in feature_name:
                return key
        raise Exception(f'{feature_name} is not in self.feature_scales')

    def get_allowed_features(self, feature_list):
        allowed_features = [
            f for f in feature_list if f not in self.banned_features]
        return allowed_features
