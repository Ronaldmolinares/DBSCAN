"""
Métricas para evaluar clustering

Incluye:
- Silhouette Score -> Qué tan bien se ajusta cada punto a su propio cluster comparado con otros clusters.
- Davies-Bouldin Index -> La relación entre la dispersión dentro de los clusters y la distancia entre ellos.
- Adjusted Rand Index (requiere etiquetas reales) -> Qué tan similar es el clustering obtenido con respecto a las etiquetas verdaderas.
- Proporción de outliers -> La fracción de puntos etiquetados como ruido (-1).
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score as sklearn_silhouette,
    davies_bouldin_score,
    adjusted_rand_score
)
from sklearn.neighbors import NearestNeighbors


# ===========================================================
# MÉTRICAS INDIVIDUALES
# ===========================================================

def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calcula Adjusted Rand Index ignorando ruido."""
    mask = (labels_pred != -1)
    if not mask.any():
        return -1.0
    return adjusted_rand_score(labels_true[mask], labels_pred[mask])


def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Calcula Davies-Bouldin Index ignorando ruido."""
    mask = (labels != -1)
    if not mask.any() or len(np.unique(labels[mask])) < 2:
        return np.inf

    return davies_bouldin_score(X[mask], labels[mask])


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette Score ignorando ruido."""
    mask = (labels != -1)
    if not mask.any() or len(np.unique(labels[mask])) < 2:
        return -1.0

    return sklearn_silhouette(X[mask], labels[mask])


def outlier_ratio(labels: np.ndarray) -> float:
    """Porcentaje de puntos etiquetados como ruido (-1)."""
    labels = np.array(labels)
    return np.sum(labels == -1) / len(labels)


# ===========================================================
# K-DISTANCE GRAPH (para selección de epsilon)
# ===========================================================

def compute_k_distance(X: np.ndarray, k: int) -> np.ndarray:
    """
    Calcula la distancia al k-ésimo vecino más cercano para cada punto.
    Fundamental para el k-distance graph.
    """
    if k >= len(X):
        raise ValueError(f"k ({k}) debe ser < número de puntos ({len(X)})")

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = distances[:, k]

    return np.sort(k_distances)


def find_optimal_eps(X: np.ndarray, k: int) -> float:
    """
    Detecta el epsilon óptimo usando:
    - máxima curvatura
    - fallback promedio (percentil 90%-95%) si la curvatura es dudosa
    """
    k_distances = compute_k_distance(X, k)
    n = len(k_distances)

    if n < 3:
        return k_distances[-1]

    # Segunda derivada → curvatura
    second_derivative = np.gradient(np.gradient(k_distances))
    idx = np.argmax(second_derivative)

    # Si la detección del codo ocurre al inicio → posiblemente incorrecta
    if idx < n * 0.1:
        # fallback robusto
        start = int(n * 0.90)
        end = int(n * 0.95)
        return float(np.mean(k_distances[start:end]))

    return float(k_distances[idx])


# ===========================================================
# MÉTRICAS COMPLETAS
# ===========================================================

def compute_all_metrics(X: np.ndarray, labels_pred: np.ndarray,
                        labels_true: np.ndarray = None) -> dict:
    """
    Retorna todas las métricas relevantes en un diccionario.
    Compatible con la GUI y con comparación entre algoritmos.
    """

    metrics = {
        "silhouette_score": silhouette_score(X, labels_pred),
        "davies_bouldin_index": davies_bouldin_index(X, labels_pred),
        "outlier_ratio": outlier_ratio(labels_pred),
        "n_clusters": len(set(labels_pred)) - (1 if -1 in labels_pred else 0),
        "n_noise": int(np.sum(labels_pred == -1)),
    }

    if labels_true is not None:
        metrics["adjusted_rand_index"] = adjusted_rand_index(labels_true, labels_pred)

    return metrics


# ===========================================================
# TABLA DE COMPARACIÓN
# ===========================================================

def compare_metrics(results: dict) -> None:
    """
    Imprime una tabla bonita con métricas de varios algoritmos.
    results = {
        "DBSCAN": {...},
        "KMeans": {...},
        "HAC": {...}
    }
    """

    print("\n" + "=" * 80)
    print("COMPARACIÓN DE MÉTRICAS DE CLUSTERING")
    print("=" * 80)

    methods = list(results.keys())

    # Obtener todos los nombres de métricas disponibles
    all_metric_names = set()
    for m in results.values():
        all_metric_names.update(m.keys())
    all_metric_names = sorted(all_metric_names)

    # Encabezado
    print(f"{'Métrica':<30}", end="")
    for method in methods:
        print(f"{method:>15}", end="")
    print("\n" + "-" * 80)

    # Filas
    for metric in all_metric_names:
        print(f"{metric:<30}", end="")
        for method in methods:
            value = results[method].get(metric, np.nan)

            if isinstance(value, (int, np.integer)):
                print(f"{value:>15d}", end="")
            elif isinstance(value, (float, np.floating)):
                print(f"{value:>15.4f}", end="")
            else:
                print(f"{str(value):>15}", end="")
        print()

    print("=" * 80)
    print("\nINTERPRETACIÓN:")
    print("  • Silhouette Score: [-1, 1] → mayor es mejor")
    print("  • Davies-Bouldin: [0, ∞) → menor es mejor")
    print("  • Adjusted Rand Index: [-1, 1] → mayor es mejor")
    print("  • Outlier Ratio: [0, 1] → menor suele ser mejor (depende del dataset)")
    print("=" * 80 + "\n")