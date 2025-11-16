"""
Comparación entre DBSCAN, K-Means y Hierarchical Clustering

"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from core.my_dbscan import MyDBSCAN
from core.metrics import all_metrics
from core.datasets import load_moons, load_circles, load_blobs


def compare_clustering(X, y_true=None, eps=0.3, min_pts=5, k=3):

    # --- 1. Implementación de DBSCAN ---
    dbscan = MyDBSCAN(eps=eps, min_pts=min_pts)
    labels_db = dbscan.fit_predict(X)
    metrics_db = all_metrics(X, labels_db, y_true)

    # --- 2. K-Means ---
    kmeans = KMeans(n_clusters=k)
    labels_km = kmeans.fit_predict(X)
    metrics_km = all_metrics(X, labels_km, y_true)

    # --- 3. Hierarchical Clustering ---
    hac = AgglomerativeClustering(n_clusters=k)
    labels_hac = hac.fit_predict(X)
    metrics_hac = all_metrics(X, labels_hac, y_true)

    # --- 4. Gráficos ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(X[:, 0], X[:, 1], c=labels_db, s=10)
    axes[0].set_title("DBSCAN")

    axes[1].scatter(X[:, 0], X[:, 1], c=labels_km, s=10)
    axes[1].set_title("K-Means")

    axes[2].scatter(X[:, 0], X[:, 1], c=labels_hac, s=10)
    axes[2].set_title("Hierarchical")

    plt.show()

    # --- 5. Mostramos métricas ---
    print("\n=== MÉTRICAS DBSCAN ===")
    print(metrics_db)

    print("\n=== MÉTRICAS K-MEANS ===")
    print(metrics_km)

    print("\n=== MÉTRICAS HAC ===")
    print(metrics_hac)

