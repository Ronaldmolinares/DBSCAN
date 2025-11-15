"""
Funciones para:
- Generación de datasets (moons, circles, blobs)
- Carga de datasets desde CSV
- Validación y limpieza de datos

"""

import numpy as np
from sklearn import datasets


# --------------------------------------------------------
#  GENERADORES DE DATASETS
# --------------------------------------------------------

def load_moons(n_samples=500, noise=0.07):
    """
    Dataset de medias lunas.
    """
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise)
    return X, y


def load_circles(n_samples=500, noise=0.07):
    """
    Dataset de círculos concéntricos.
    """
    X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    return X, y


def load_blobs(n_samples=500, centers=3):
    """
    Dataset con clusters gaussianos (útil para comparar con K-Means).
    """
    X, y = datasets.make_blobs(n_samples=n_samples, centers=centers)
    return X, y


# --------------------------------------------------------
#  NORMALIZACIÓN
# --------------------------------------------------------

def normalize(X):
    """
    Normaliza cada feature al rango [0,1].
    """
    X = np.array(X, dtype=float)
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    denom = (max_val - min_val)
    denom[denom == 0] = 1  # evitar división por 0

    return (X - min_val) / denom


# --------------------------------------------------------
#  CARGA DE CSV
# --------------------------------------------------------

def load_csv(path, normalize_data=False):
    """
    Carga un archivo CSV con coordenadas numéricas.
    - Filtra filas con NaN
    - Convierte a float
    """

    try:
        raw = np.loadtxt(path, delimiter=",")
    except Exception as e:
        raise ValueError(f"No se pudo cargar el CSV: {e}")

    # Validar dimensión
    if raw.ndim != 2:
        raise ValueError("El CSV debe contener una matriz de 2 dimensiones")

    if raw.shape[1] < 2:
        raise ValueError("El CSV debe tener al menos 2 columnas")

    # Eliminar filas con NaN
    clean = raw[~np.isnan(raw).any(axis=1)]

    if clean.shape[0] == 0:
        raise ValueError("El CSV no contiene datos válidos")

    if normalize_data:
        clean = normalize(clean)

    # No hay etiquetas para CSV
    return clean, None


# --------------------------------------------------------
#  DISPATCH GENERAL DE DATASETS
# --------------------------------------------------------

def load_dataset_by_name(name):
    """
    Devuelve un dataset según el nombre recibido.
    Para uso rápido en la GUI.
    """

    name = name.lower()

    if name == "moons":
        return load_moons()
    elif name == "circles":
        return load_circles()
    elif name == "blobs":
        return load_blobs()
    else:
        raise ValueError(f"Dataset desconocido: {name}")
