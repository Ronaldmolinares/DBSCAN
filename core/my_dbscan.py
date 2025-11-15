"""
Implementación propia de DBSCAN (Density-Based Spatial Clustering)

"""

import numpy as np


class MyDBSCAN:

    def __init__(self, eps=0.5, min_pts=5):
        """
        Parámetros:
        eps (float): radio del vecindario
        min_pts (int): número mínimo de puntos para ser núcleo
        """
        if eps <= 0:
            raise ValueError("eps debe ser > 0")

        if min_pts < 2:
            raise ValueError("min_pts debe ser >= 2")

        self.eps = eps
        self.min_pts = min_pts
        
        # Resultados
        self.labels_ = None

# ----------------------------------------------------------
#  Función: limites de vecindad ε
# ----------------------------------------------------------
    def _region_query(self, X, idx):
        """
        Retorna los índices de los puntos dentro de eps del punto idx.
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors
    
# ----------------------------------------------------------
#  Expansión del cluster
# ----------------------------------------------------------
    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        """
        Expande el cluster actual agregando puntos densidad-alcanzables.
        """
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            # Si estaba marcado como ruido → lo reasignamos
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            # Si aún no está asignado a ningún cluster
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id

                # Revisar si este punto también es núcleo
                neighbor_neighbors = self._region_query(X, neighbor_idx)

                if len(neighbor_neighbors) >= self.min_pts:
                    # unir ambas listas
                    neighbors = np.concatenate((neighbors, neighbor_neighbors))

            i += 1

# ----------------------------------------------------------
#  Método principal fit()
# ----------------------------------------------------------
    def fit(self, X):
        """
        Ejecuta DBSCAN y genera las etiquetas de cluster

        labels_:
            -1 → ruido
             0 → sin asignar
             1..k → clusters
        """
        n = X.shape[0]
        labels = np.zeros(n, dtype=int)  # 0 = no visitado
        cluster_id = 0

        for point_idx in range(n):

            # Saltar si ya fue asignado
            if labels[point_idx] != 0:
                continue

            neighbors = self._region_query(X, point_idx)

            # No es núcleo → marcar como ruido
            if len(neighbors) < self.min_pts:
                labels[point_idx] = -1
                continue

            # Sí es núcleo → crear nuevo cluster
            cluster_id += 1
            self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)

        self.labels_ = labels
        return self

# ----------------------------------------------------------
#  fit_predict()
# ----------------------------------------------------------
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_