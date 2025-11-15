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