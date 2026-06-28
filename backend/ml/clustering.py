"""
K-Means clustering for Vigilo.

Groups campaigns into performance-profile clusters, then maps the
raw integer cluster IDs to human-readable labels by inspecting each
cluster's centroid rather than hardcoding which index means what.

K selection — silhouette score (replaces fixed k=4 / elbow method):
  Silhouette score measures how similar each point is to its own
  cluster vs other clusters, ranging from -1 to +1. We try every
  valid k and pick the one with the highest average silhouette score.

  Why better than fixed k=4:
    - 5-campaign account → probably 2-3 real groups, not 4
    - 50-campaign account → may have 5-6 distinct profiles
    - Fixed k=4 forces artificial separation on small accounts
    - Silhouette score adapts to the actual data structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ml.feature_engineering import FEATURE_COLUMNS, build_feature_matrix

PROFILE_SIGNATURES = {
    "Underinvested Winner": {"ctr": 1,  "conversion_rate": 1,  "roas": 1,  "cost": -1, "cpc": -1},
    "Bleeding Budget":      {"ctr": -1, "conversion_rate": -1, "roas": -1, "cost": 1,  "cpc": 1},
    "Keyword Waste":        {"ctr": -1, "conversion_rate": -1, "roas": -1, "cost": 0,  "cpc": 1},
    "Steady Performer":     {"ctr": 0,  "conversion_rate": 0,  "roas": 0,  "cost": 0,  "cpc": 0},
}


def find_optimal_k(scaled_features: np.ndarray) -> tuple[int, dict[int, float]]:
    """
    Finds the best k using silhouette score.

    Tries k from 2 to min(6, n_samples-1). Returns the best k and
    the full scores dict so it can be plotted in the notebook.

    Silhouette score per sample i:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
        a(i) = mean distance to all other points in same cluster
        b(i) = mean distance to all points in nearest other cluster

    Average s across all samples — higher is better, max 1.0.
    """
    n_samples = len(scaled_features)
    k_max = min(6, n_samples - 1)

    if k_max < 2:
        return 1, {}

    scores = {}
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_features)
        if len(set(labels)) < 2:
            continue
        scores[k] = round(silhouette_score(scaled_features, labels), 4)

    if not scores:
        return 2, {}

    best_k = max(scores, key=scores.get)
    return best_k, scores


def find_optimal_k_elbow(scaled_features: np.ndarray, k_range=range(2, 7)) -> dict:
    """
    Kept for the Jupyter notebook — plots inertia alongside
    silhouette scores for comparison. Not used in the pipeline.
    """
    inertias = {}
    for k in k_range:
        if k >= len(scaled_features):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_features)
        inertias[k] = km.inertia_
    return inertias


def _label_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """
    Scores each cluster centroid against each profile signature and
    assigns the best-matching human-readable label. Most distinctive
    clusters claim their label first to avoid duplicates.
    """
    labels = {}
    used_labels = set()
    extremity = centroids.abs().sum(axis=1).sort_values(ascending=False)

    for cluster_id in extremity.index:
        centroid = centroids.loc[cluster_id]
        best_label, best_score = None, -np.inf

        for label, signature in PROFILE_SIGNATURES.items():
            if label in used_labels:
                continue
            score = sum(
                centroid[feat] * direction for feat, direction in signature.items()
            )
            if score > best_score:
                best_label, best_score = label, score

        if best_label is None:
            best_label = f"Cluster {cluster_id}"

        labels[cluster_id] = best_label
        used_labels.add(best_label)

    return labels


def run_clustering(df: pd.DataFrame, k: int = None) -> pd.DataFrame:
    """
    Main entry point. Takes the cleaned campaign DataFrame, returns it
    with cluster_id (int), cluster_label (str), and chosen_k (int).

    If k is provided it's used directly (e.g. from tests or notebook).
    If k is None, silhouette score picks the best k automatically.
    """
    scaled_df, scaler = build_feature_matrix(df)
    n_samples = len(scaled_df)

    if k is not None:
        # Explicit k — use as-is (tests, notebook)
        effective_k = min(k, n_samples - 1)
        chosen_k = effective_k
        silhouette_scores = {}
    else:
        # Automatic k selection via silhouette score
        chosen_k, silhouette_scores = find_optimal_k(scaled_df.values)
        if chosen_k == 1:
            # Too few samples — assign everything to one cluster
            result = df.copy()
            result["cluster_id"] = 0
            result["cluster_label"] = "Steady Performer"
            result["chosen_k"] = 1
            return result

    km = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(scaled_df)

    centroids = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLUMNS)
    label_map = _label_clusters(centroids)

    result = df.copy()
    result["cluster_id"] = cluster_ids
    result["cluster_label"] = result["cluster_id"].map(label_map)
    result["chosen_k"] = chosen_k

    if silhouette_scores:
        print(f"Silhouette scores: {silhouette_scores}")
        print(f"Best k: {chosen_k} (score: {silhouette_scores.get(chosen_k, 'N/A')})")

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from utils.csv_parser import parse_csv

    with open("../data/sample_campaigns.csv", "rb") as f:
        parsed = parse_csv(f.read())

    scaled_df, _ = build_feature_matrix(parsed.df)

    print("=== Silhouette Scores ===")
    best_k, scores = find_optimal_k(scaled_df.values)
    for k, s in scores.items():
        marker = " ← best" if k == best_k else ""
        print(f"  k={k}: silhouette={s}{marker}")

    print("\n=== Elbow Method (for comparison) ===")
    inertias = find_optimal_k_elbow(scaled_df.values)
    for k, inertia in inertias.items():
        print(f"  k={k}: inertia={inertia:.2f}")

    print("\n=== Clustering Result ===")
    clustered = run_clustering(parsed.df)
    print(clustered[["campaign_name", "cost", "ctr", "conversion_rate", "cluster_label", "chosen_k"]])