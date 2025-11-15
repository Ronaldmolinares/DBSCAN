"""
Análisis de sensibilidad del parámetro epsilon usando k-distance graph

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from core.datasets import load_moons, load_circles, load_blobs


def k_distance_graph(X, minPts):
    """
    Calcula el gráfico del k-ésimo vecino (k = minPts - 1)
    se hace para seleccionar epsilon en DBSCAN.
    """
    k = minPts - 1

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # Distancia al k-ésimo vecino
    k_dist = np.sort(distances[:, -1])

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(k_dist)
    plt.title(f"k-distance graph (k={k})")
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia al {k}-ésimo vecino")
    plt.grid(True)
    plt.show()

    return k_dist


if __name__ == "__main__":
    X, _ = load_moons()  # se puede cambiar
    minPts = 5

    print("Generando k-distance graph...")    
    k_distances = k_distance_graph(X, minPts)