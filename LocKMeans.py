import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


class LocKMeans:
    """
    Implements a modified K-Means algorithm with size-constraints
    on the number of elements of each cluster

    Parameters:
    -----------
    n_clusters: int; number of clusters
    cluster_size: int, np.array; when int gives the maximal cluster size for every cluster
        when array (n_clusters,) gives a maximal cluster size for each cluster
    initial_centers: np.array; if None cluster centers are initialized randomly
        if array (n_clusters, num_features), it will be used as the initial centers for the algorithm
    """

    def __init__(self, n_clusters=8, cluster_size=None, max_iter=300, hide_pbar=False):
        self.n_clusters_ = n_clusters
        if type(cluster_size) == np.ndarray:
            self.cluster_size_ = cluster_size
        elif type(cluster_size) == int:
            self.cluster_size_ = np.repeat(cluster_size, n_clusters)
        else:
            self.cluster_size_ = None
        self.max_iter_ = max_iter
        self.cluster_centers_ = None
        self.hide_pbar_ = hide_pbar

    def fit(self, X, initial_centers=None):
        """
        Parameters:
        -----------
        X: np.array (n_samples, n_features); data matrix
        initial_centers: np.array; if None cluster centers are initialized randomly
        if array (n_clusters, num_features), it will be used as the initial centers for the algorithm
        """
        n_samples, n_features = X.shape
        if initial_centers is None:
            center_indices = np.random.choice(n_samples, self.n_clusters_)
            initial_centers = X[center_indices]
        if self.cluster_size_ is None:
            avg_cluster_size = n_samples // self.n_clusters_ + 1
            self.cluster_size_ = np.repeat(avg_cluster_size, self.n_clusters_)
        self.cluster_centers_ = initial_centers

        for i in tqdm(range(self.max_iter_), disable=self.hide_pbar_):
            self.labels_ = np.repeat(-1, X.shape[0])
            list_points_in_clusters = [[] for _ in range(self.n_clusters_)]
            list_cluster_size = [0 for _ in range(self.n_clusters_)]
            dist_data_centers = cdist(X, self.cluster_centers_)
            sort_index_data_centers = np.argsort(np.min(dist_data_centers, axis=1))
            for idx in sort_index_data_centers:
                cluster_order = np.argsort(dist_data_centers[idx])
                for cluster_idx in cluster_order:
                    if list_cluster_size[cluster_idx] < self.cluster_size_[cluster_idx]:
                        list_points_in_clusters[cluster_idx].append(X[idx])
                        list_cluster_size[cluster_idx] += 1
                        self.labels_[idx] = cluster_idx
                        break
            new_centers = np.zeros_like(self.cluster_centers_)
            for cluster_idx in range(self.n_clusters_):
                new_centers[cluster_idx] = np.mean(
                    np.array(list_points_in_clusters[cluster_idx]), axis=0
                )
            self.cluster_centers_ = new_centers

    def predict(self, X):
        list_points_in_clusters = [[] for _ in range(self.n_clusters_)]
        dist_data_centers = cdist(X, self.cluster_centers_)
        sort_index_data_centers = np.argsort(np.min(dist_data_centers, axis=1))
        labels = np.repeat(-1, X.shape[0])
        for idx in tqdm(sort_index_data_centers):
            cluster_order = np.argsort(dist_data_centers[idx])
            for cluster_idx in cluster_order:
                if (
                    len(list_points_in_clusters[cluster_idx])
                    < self.cluster_size_[cluster_idx]
                ):
                    list_points_in_clusters[cluster_idx].append(X[idx])
                    labels[idx] = cluster_idx
                    break
        return labels