"""
Métricas para evaluar clustering

Incluye:
- Silhouette Score -> Qué tan bien se ajusta cada punto a su propio cluster comparado con otros clusters.
- Davies-Bouldin Index -> La relación entre la dispersión dentro de los clusters y la distancia entre ellos.
- Adjusted Rand Index (requiere etiquetas reales) -> Qué tan similar es el clustering obtenido con respecto a las etiquetas verdaderas.
- Proporción de outliers -> La fracción de puntos etiquetados como ruido (-1).
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score


def outlier_ratio(labels):
    """
    Calcula el porcentaje de puntos etiquetados como ruido (-1).
    """
    labels = np.array(labels)
    total = len(labels)
    noise = np.sum(labels == -1)

    return noise / total

def compute_silhouette(X, labels):
    """
    Silhouette Score. Requiere al menos 2 clusters válidos.
    Retorna None si no puede calcularse.
    """
    unique = set(labels)
    unique = {u for u in unique if u != -1}  # despreciamos el ruido

    if len(unique) < 2:
        return None

    return silhouette_score(X, labels)


def compute_davies_bouldin(X, labels):
    """
    Davies–Bouldin Index.
    Requiere al menos 2 clusters válidos.
    """
    unique = set(labels)
    unique = {u for u in unique if u != -1}

    if len(unique) < 2:
        return None

    return davies_bouldin_score(X, labels)


def compute_ari(labels_pred, labels_true):
    """
    Adjusted Rand Index. Solo si se tienen etiquetas verdaderas.
    """
    return adjusted_rand_score(labels_true, labels_pred)


def all_metrics(X, labels, true_labels=None):
    """
    Retorna un diccionario con todas las métricas disponibles.
    Si true_labels no se proporciona, ARI será None.
    """

    metrics = {}

    # Silhouette
    sil = compute_silhouette(X, labels)
    metrics["silhouette"] = sil if sil is not None else "No aplicable"

    # Davies-Bouldin
    db = compute_davies_bouldin(X, labels)
    metrics["davies_bouldin"] = db if db is not None else "No aplicable"

    # Outliers
    metrics["outlier_ratio"] = outlier_ratio(labels)

    # ARI
    if true_labels is not None:
        metrics["ARI"] = compute_ari(labels, true_labels)
    else:
        metrics["ARI"] = "No disponible"

    return metrics