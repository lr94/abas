import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import re
import pickle as pickle
import loss


SOURCE_LOSS_THRESHOLD = 2.0


def get_centroids(data, labels, num_classes):
    centroids = np.zeros((num_classes, data.shape[1]))
    for cid in range(num_classes):
        # Since we are using pseudolabels to compute centroids, some classes might not have instances according to the
        # pseudolabels assigned by the current model. In that case .mean() would return NaN causing KMeans to fail.
        # We set to 0 the missing centroids
        if (labels == cid).any():
            centroids[cid] = data[labels == cid].mean(0)

    return centroids


def get_clustering_performance(feats, plabels, num_classes, source_feats=None, pca_size=64):
    """
    :param feats: N x out numpy vector
    :param plabels:  N numpy vector
    :param num_classes: int
    :param source_feats
    :param pca_size
    :return: silhouette and calinski harabasz scores
    """
    pca = PCA(pca_size)
    n_samples = feats.shape[0]

    if source_feats is not None:
        feats = np.concatenate((feats, source_feats))

    try:
        x = pca.fit_transform(feats)[:n_samples]
        centroids = get_centroids(x, plabels, num_classes)

        clustering = KMeans(n_clusters=num_classes, init=centroids, n_init=1)

        clustering.fit(x)
        clabels = clustering.labels_
        ss, ch, = silhouette_score(x, clabels), calinski_harabasz_score(x, clabels)
    except ValueError:
        ss, ch = float('nan'), float('nan')

    return ss, ch


def compute_time_consistent_pseudolabels(pseudolabels_per_epoch: np.ndarray, num_classes: int) -> np.ndarray:
    epochs = pseudolabels_per_epoch.shape[0]

    # TODO this should be done in a numpy-oompy way
    global_pseudolabels = np.zeros((pseudolabels_per_epoch.shape[1], num_classes), dtype=int)
    for i in range(epochs):
        pseudolabels = pseudolabels_per_epoch[i, :].astype(int)

        for j in range(pseudolabels.shape[0]):
            global_pseudolabels[j, pseudolabels[j]] += 1

    global_pseudolabels = global_pseudolabels.argmax(axis=1)
    return global_pseudolabels


class ModelCriterion:

    """
    This class represents a mode selection method.

    The name can be:
        regression(file_containing_linear_model.pkl)
    or any specific metric (source_accuracy, target_silhouette_score etc)
    """

    def __init__(self, name):
        # Regression?
        m = re.match(r'regression\((.+)\)', name)
        if m is not None:
            # If yes, get the file name
            self.name = 'regression'
            regression_pickle_file = m.group(1)

            # Import regressor classes
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            # Load the model
            with open(regression_pickle_file, 'rb') as fp:
                criterion_dict = pickle.load(fp)
                self.file_name = regression_pickle_file
                self.metric_names = criterion_dict['info']['metric_names']
                self.info = criterion_dict['info']
                self.regressor = criterion_dict['regressor']
                self.scaler = criterion_dict.get('scaler', None)
        else:
            # Simple "monodimensional" metric
            self.name = name
            self.info = None
            self.metric_names = [name]

        # Convert shorthands in full metric names
        for i, n in enumerate(self.metric_names):
            if n == 'ss':
                self.metric_names[i] = 'target_silhouette_score'
            elif n == 'ch':
                self.metric_names[i] = 'target_calinski_harabasz_score'
            elif n == 'pseudolabels':
                self.metric_names[i] = 'target_pseudolabels'

    def __call__(self, data=None, **kwargs) -> np.ndarray:
        # Argument: dataframe or something similar (lists, dicts...)
        # Convert it to dataframe
        if data is None:
            data = pd.DataFrame([kwargs])
        elif isinstance(data, dict) or isinstance(data, list):
            data = pd.DataFrame(data)

        # Convert the dataframe to a numpy matrix containing the selected metrics (one for single-metric, more for
        # regression)
        metrics = data[self.metric_names].to_numpy()

        # If the model selection method uses a single metric (no regression)
        if self.name != 'regression':
            assert len(self.metric_names) == 1
            # if the metric doesn't end with "loss" it must be maximized!
            if not self.metric_names[0].endswith('_loss'):
                metrics *= -1
            # Flatten and return
            return metrics.reshape(-1)
        else:
            # If the model selection method uses regression, just use the model to predict a value
            scaled_metrics = metrics if self.scaler is None else self.scaler.transform(metrics)
            score = self.regressor.predict(scaled_metrics)
            # Infinite loss where source loss is too high
            score[data['source_class_loss'] > SOURCE_LOSS_THRESHOLD] = -1000
            # The model was trained with accuracy, so it's a score and not a loss. Convert into a loss
            return -score

    def get_best(self, all_metrics: pd.DataFrame):
        # Find best model
        losses = self(all_metrics)

        best_index = losses.argmin()

        return best_index, losses[best_index].item()

    def __repr__(self):
        return self.name
