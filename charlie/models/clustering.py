import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.utils.validation import check_array
from typing import Optional, List, Dict, Any, Union

class CharlieClusterer:
    """
    Hybrid clusterer combining linear (PCA) and nonlinear (t-SNE) projections,
    clustering in both spaces, and using consensus or silhouette-based selection
    for final cluster assignments. Also provides interpretable feature-based
    cluster explanations.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form.
    n_pca : int, default=10
        Number of principal components for PCA projection.
    n_tsne : int, default=2
        Number of t-SNE dimensions for nonlinear projection.
    method : {'kmeans', 'agglo'}, default='kmeans'
        Clustering method to use: 'kmeans' or 'agglo' (agglomerative).
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Final cluster assignments.
    X_pca : ndarray of shape (n_samples, n_pca)
        Data projected via PCA.
    X_tsne : ndarray of shape (n_samples, n_tsne)
        Data projected via t-SNE.
    X : ndarray of shape (n_samples, n_features)
        Original input data.

    Methods
    -------
    fit(X)
        Fit the model to data X.
    label_clusters()
        Generate PCA-based descriptive labels for clusters.
    explain_clusters(feature_names=None)
        Return top features for each cluster based on original features.
    evaluate(y_true=None)
        Return silhouette scores; with y_true, also computes Adjusted Rand Index.
    get_labels()
        Return cluster assignments.
    get_cluster_descriptions()
        Return PCA-based cluster labels.

    Examples
    --------
    >>> from sklearn.datasets import load_wine
    >>> data = load_wine()
    >>> X, y = data.data, data.target
    >>> feature_names = data.feature_names
    >>> model = CharlieClusterer(n_clusters=3, n_pca=8, method='kmeans', random_state=0)
    >>> model.fit(X)
    CharlieClusterer(...)
    >>> print("Evaluation:", model.evaluate(y_true=y))
    Evaluation: {'silhouette_pca': ..., 'silhouette_tsne': ..., 'silhouette_orig': ..., 'ARI': ...}
    >>> print("Cluster Labels:", model.label_clusters())
    Cluster Labels: ['PCA-2 mean=...', ...]
    >>> print("Explanations:", model.explain_clusters(feature_names=feature_names))
    Explanations: ['Cluster 0: top features: alcohol=..., color_intensity=..., flavanoids=...', ...]
    >>> labels = model.get_labels()
    >>> descriptions = model.get_cluster_descriptions()
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_pca: int = 10,
        n_tsne: int = 2,
        method: str = "kmeans",
        random_state: int = 42
    ) -> None:
        if not isinstance(n_clusters, int) or n_clusters <= 1:
            raise ValueError("n_clusters must be an integer greater than 1.")
        if not isinstance(n_pca, int) or n_pca < 1:
            raise ValueError("n_pca must be a positive integer.")
        if not isinstance(n_tsne, int) or n_tsne < 1:
            raise ValueError("n_tsne must be a positive integer.")
        if method not in {"kmeans", "agglo"}:
            raise ValueError("method must be one of {'kmeans', 'agglo'}.")
        if not isinstance(random_state, int):
            raise ValueError("random_state must be an integer.")

        self.n_clusters = n_clusters
        self.n_pca = n_pca
        self.n_tsne = n_tsne
        self.method = method
        self.random_state = random_state
        self.fitted_ = False  

    def fit(self, X: np.ndarray) -> "CharlieClusterer":
        """
        Fit the hybrid clustering model.
        """
        X = check_array(X)  
        if X.shape[0] < self.n_clusters:
            raise ValueError("Number of samples must be >= n_clusters.")
        if self.n_pca > X.shape[1]:
            raise ValueError("n_pca cannot exceed n_features of input X.")

        self.X = X
        self.pca = PCA(n_components=self.n_pca, random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(X)
        self.tsne_model = TSNE(n_components=self.n_tsne, random_state=self.random_state)
        self.X_tsne = self.tsne_model.fit_transform(X)

        if self.method == "kmeans":
            labels_pca = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(self.X_pca)
            labels_tsne = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(self.X_tsne)
        else:
            labels_pca = AgglomerativeClustering(n_clusters=self.n_clusters).fit_predict(self.X_pca)
            labels_tsne = AgglomerativeClustering(n_clusters=self.n_clusters).fit_predict(self.X_tsne)

        # Consensus assignment
        self.labels_ = []
        sil_tsne = silhouette_score(self.X_tsne, labels_tsne)
        sil_pca = silhouette_score(self.X_pca, labels_pca)
        for i in range(X.shape[0]):
            if labels_tsne[i] == labels_pca[i]:
                self.labels_.append(labels_tsne[i])
            else:
                self.labels_.append(labels_tsne[i] if sil_tsne > sil_pca else labels_pca[i])
        self.labels_ = np.array(self.labels_)
        self.fitted_ = True
        return self

    def label_clusters(self) -> List[str]:
        """
        Generate PCA-based labels for clusters.
        """
        if not self.fitted_:
            raise RuntimeError("Must fit before calling label_clusters().")
        cluster_labels = []
        for k in np.unique(self.labels_):
            mask = (self.labels_ == k)
            centroid = np.mean(self.X_pca[mask], axis=0)
            top_feat = np.argmax(np.abs(centroid))
            label = f"PCA-{top_feat} mean={centroid[top_feat]:.2f}"
            cluster_labels.append(label)
        self.cluster_labels_ = cluster_labels
        return cluster_labels

    def explain_clusters(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Provide feature-based explanations for each cluster.
        """
        if not self.fitted_:
            raise RuntimeError("Must fit before calling explain_clusters().")
        explanations = []
        global_mean = np.mean(self.X, axis=0)
        for k in np.unique(self.labels_):
            mask = (self.labels_ == k)
            centroid = np.mean(self.X[mask], axis=0)
            diff = np.abs(centroid - global_mean)
            top_idx = np.argsort(diff)[-3:][::-1]
            if feature_names is not None:
                if len(feature_names) != self.X.shape[1]:
                    raise ValueError("feature_names must match number of features.")
                names = [feature_names[i] for i in top_idx]
            else:
                names = [f"f{i}" for i in top_idx]
            vals = [centroid[i] for i in top_idx]
            explanation = f"Cluster {k}: top features: {', '.join([f'{n}={v:.2f}' for n,v in zip(names,vals)])}"
            explanations.append(explanation)
        return explanations

    def evaluate(self, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate clustering using silhouette scores and optionally ARI.
        """
        if not self.fitted_:
            raise RuntimeError("Must fit before calling evaluate().")
        results = {
            "silhouette_pca": silhouette_score(self.X_pca, self.labels_),
            "silhouette_tsne": silhouette_score(self.X_tsne, self.labels_),
            "silhouette_orig": silhouette_score(self.X, self.labels_),
        }
        if y_true is not None:
            y_true = np.asarray(y_true)
            if y_true.shape[0] != self.X.shape[0]:
                raise ValueError("y_true length does not match number of samples.")
            results["ARI"] = adjusted_rand_score(y_true, self.labels_)
        return results

    def get_labels(self) -> np.ndarray:
        """
        Get cluster assignments.
        """
        if not self.fitted_:
            raise RuntimeError("Must fit before calling get_labels().")
        return self.labels_

    def get_cluster_descriptions(self) -> Optional[List[str]]:
        """
        Get PCA-based cluster labels.
        """
        if not self.fitted_:
            raise RuntimeError("Must fit before calling get_cluster_descriptions().")
        return getattr(self, "cluster_labels_", None)
