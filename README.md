# Implementación del algoritmo DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Características

- **Implementación propia** de DBSCAN en `core/my_dbscan.py`
- **GUI interactiva** con tkinter en `gui/main_gui.py`
- **Métricas de evaluación**: Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index
- **K-distance graph** para selección empírica de epsilon
- **Comparación** con K-Means y Hierarchical Clustering

## Instalación

```bash
pip install -r requirements.txt
```

## Para usar

```bash
python run.py
```

## Implementación DBSCAN

### Algoritmo Core

**Archivo**: `core/my_dbscan.py`

#### 1. Region Query
```python
def _region_query(self, X, idx):
    """Encuentra todos los puntos dentro del radio epsilon"""
    distances = np.linalg.norm(X - X[idx], axis=1)
    return np.where(distances <= self.eps)[0]
```

#### 2. Expand Cluster
```python
def _expand_cluster(self, X, labels, idx, neighbors, cluster_id):
    """Expande cluster siguiendo densidad alcanzable"""
    labels[idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:  # Era ruido
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:  # No visitado
            labels[neighbor_idx] = cluster_id
            new_neighbors = self._region_query(X, neighbor_idx)
            if len(new_neighbors) >= self.min_pts:
                neighbors = np.concatenate([neighbors, new_neighbors])
        i += 1
```

#### 3. Fit Predict
```python
def fit_predict(self, X):
    """Ejecuta DBSCAN completo"""
    labels = np.zeros(len(X), dtype=int)  # 0 = no visitado
    cluster_id = 0
    
    for idx in range(len(X)):
        if labels[idx] != 0:
            continue
            
        neighbors = self._region_query(X, idx)
        
        if len(neighbors) < self.min_pts:
            labels[idx] = -1  # Ruido
        else:
            cluster_id += 1
            self._expand_cluster(X, labels, idx, neighbors, cluster_id)
    
    return labels
```

### Clasificación de Puntos

- **Core Point**: Tiene ≥ `min_pts` vecinos dentro de `eps`
- **Border Point**: Está en vecindad de core point pero no es core
- **Noise**: No es core ni border (etiqueta = -1 ~ruido)

## Métricas Implementadas

### 1. Silhouette Score
Mide qué tan bien cada punto se ajusta a su cluster vs otros clusters.
- Rango: [-1, 1]
- Mejor: valores cercanos a 1

### 2. Davies-Bouldin Index
Relación entre dispersión intra-cluster y distancia inter-cluster.
- Rango: [0, ∞)
- Mejor: valores cercanos a 0

### 3. Adjusted Rand Index
Similaridad con etiquetas verdaderas (corregido por azar).
- Rango: [-1, 1]
- Mejor: valores cercanos a 1

### 4. K-Distance Graph
Gráfico para seleccionar epsilon óptimo:
- k = minPts − 1


## Estructura del Proyecto

```
DBSCAN/
├── core/
│   ├── my_dbscan.py        # Implementación DBSCAN
│   ├── metrics.py          # Métricas de evaluación
│   └── datasets.py         # Carga de datos
├── gui/
│   └── main_gui.py         # Interfaz gráfica
├── experiments/
│   └── run_experiments.py  # Experimentos comparativos
├── requirements.txt
└── run.py                  # Ejecutar GUI
```

## Funcionalidades GUI

1. **Selección de Dataset**: moons, circles, blobs
2. **Ajuste de Parámetros**: Sliders para epsilon (0.05-2.0) y minPts (2-20)
3. **Visualización**: Gráficos por cluster y outliers
4. **Comparación de Algoritmos**: DBSCAN vs K-Means vs Hierarchical
5. **K-Distance Graph**: Análisis para selección de epsilon


## Ventajas de DBSCAN

- Detecta clusters de **forma arbitraria**
- No requiere especificar número de clusters
- Identifica **outliers** 
- Robusto a ruido

## Parámetros

| Parámetro | Descripción | Rango típico |
|-----------|-------------|--------------|
| `eps` | Radio de vecindad -> epsilon | 0.1 - 2.0 |
| `min_pts` | Mínimo de vecinos para core point | 3 - 10 |

**Recomendación**: Usar k-distance graph para seleccionar `eps`, puede que el epsilon recomendado haga que en DBSCAN se produzca 1 solo cluster (epsilon grande para los datos). el epsilon recomendado no es perfecto.

## Dependencias

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn (solo para métricas y comparación)
- tkinter (incluido en Python)

## Autores

Molinares Sanabria Ronald Samir  
Peña Coronado Esteban Nicolas  
Sanchez Munevar Diego Armando  
Torres Fonseca Laura Katalina  

Proyecto desarrollado para el curso de Inteligencia Computacional.
