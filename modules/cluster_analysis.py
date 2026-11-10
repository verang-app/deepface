import numpy as np
from typing import Any


def _kmeans_numpy(embeddings, n_clusters: int, max_iter: int = 100, random_state: int = 0):
    rng = np.random.RandomState(random_state)
    data = np.asarray(embeddings, dtype=np.float32)
    n_samples = data.shape[0]
    if n_samples <= n_clusters:
        return data, 0.0
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = data[indices].copy()
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            data[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    inertia = float(np.sum((data - centroids[labels]) ** 2))
    return centroids, inertia


def find_cluster_centroids(embeddings, max_k=10) -> Any:
    n_samples = len(embeddings)
    if n_samples == 0:
        raise ValueError("embeddings must not be empty")
    if max_k < 1:
        raise ValueError("max_k must be at least 1")

    max_k = min(max_k, n_samples)
    if max_k == 1:
        centroids, _ = _kmeans_numpy(embeddings, n_clusters=1, random_state=0)
        return centroids

    inertia = []
    cluster_centroids = []
    K = range(1, max_k+1)

    for k in K:
        centroids, inr = _kmeans_numpy(embeddings, n_clusters=k, random_state=0)
        inertia.append(inr)
        cluster_centroids.append({"k": k, "centroids": centroids})

    diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
    optimal_centroids = cluster_centroids[diffs.index(max(diffs)) + 1]['centroids']

    return optimal_centroids


def find_closest_centroid(centroids: list, normed_face_embedding) -> list:
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)
        
        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        return None
