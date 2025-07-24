import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError

from charlie.models.clustering import CharlieClusterer 

@pytest.fixture
def blob_data():
    X, y = make_blobs(n_samples=60, centers=3, n_features=6, random_state=1)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return X, y, feature_names

def test_invalid_init_params():
    with pytest.raises(ValueError):
        CharlieClusterer(n_clusters=1)
    with pytest.raises(ValueError):
        CharlieClusterer(n_pca=0)
    with pytest.raises(ValueError):
        CharlieClusterer(n_tsne=0)
    with pytest.raises(ValueError):
        CharlieClusterer(method="wrong")
    with pytest.raises(ValueError):
        CharlieClusterer(random_state="foo")

def test_fit_and_attributes(blob_data):
    X, y, _ = blob_data
    model = CharlieClusterer(n_clusters=3, n_pca=4, method="kmeans", random_state=123)
    result = model.fit(X)
    assert result is model
    assert hasattr(model, "labels_")
    assert hasattr(model, "X_pca")
    assert hasattr(model, "X_tsne")
    assert model.X.shape == X.shape
    assert len(model.labels_) == X.shape[0]
    assert model.X_pca.shape[1] == model.n_pca
    assert model.X_tsne.shape[1] == model.n_tsne

def test_fit_errors(blob_data):
    X, y, _ = blob_data
    # Too few samples
    model = CharlieClusterer(n_clusters=10)
    with pytest.raises(ValueError):
        model.fit(X[:5])
    # Too many pca components
    model = CharlieClusterer(n_clusters=2, n_pca=20)
    with pytest.raises(ValueError):
        model.fit(X)

def test_label_clusters(blob_data):
    X, _, _ = blob_data
    model = CharlieClusterer(n_clusters=3, n_pca=3, random_state=99)
    model.fit(X)
    labels = model.label_clusters()
    assert isinstance(labels, list)
    assert len(labels) == 3
    assert all(label.startswith("PCA-") for label in labels)

def test_explain_clusters_with_and_without_names(blob_data):
    X, _, names = blob_data
    model = CharlieClusterer(n_clusters=3, n_pca=3)
    model.fit(X)
    # with names
    expl = model.explain_clusters(feature_names=names)
    assert isinstance(expl, list)
    assert len(expl) == 3
    assert all("top features" in e for e in expl)
    # without names
    expl2 = model.explain_clusters()
    assert all("f" in e for e in expl2)
    # mismatch feature names
    with pytest.raises(ValueError):
        model.explain_clusters(feature_names=["foo", "bar"])

def test_evaluate_and_ari(blob_data):
    X, y, _ = blob_data
    model = CharlieClusterer(n_clusters=3, n_pca=3)
    model.fit(X)
    results = model.evaluate()
    assert "silhouette_pca" in results
    assert "silhouette_tsne" in results
    assert "silhouette_orig" in results
    results_ari = model.evaluate(y_true=y)
    assert "ARI" in results_ari
    # y_true length mismatch
    with pytest.raises(ValueError):
        model.evaluate(y_true=np.arange(X.shape[0] - 1))

def test_get_labels_and_descriptions(blob_data):
    X, _, _ = blob_data
    model = CharlieClusterer(n_clusters=3, n_pca=3)
    model.fit(X)
    labels = model.get_labels()
    assert isinstance(labels, np.ndarray)
    model.label_clusters()
    descs = model.get_cluster_descriptions()
    assert isinstance(descs, list)
    # Not fitted case
    model2 = CharlieClusterer(n_clusters=2)
    with pytest.raises(RuntimeError):
        model2.get_labels()
    with pytest.raises(RuntimeError):
        model2.get_cluster_descriptions()

def test_method_agglo(blob_data):
    X, _, _ = blob_data
    model = CharlieClusterer(n_clusters=3, n_pca=3, method="agglo")
    model.fit(X)
    labels = model.get_labels()
    assert isinstance(labels, np.ndarray)
    assert len(labels) == X.shape[0]