"""
Análisis de sensibilidad del parámetro epsilon usando k-distance graph
"""

import numpy as np
import matplotlib.pyplot as plt
from core.metrics import compute_k_distance, find_optimal_eps
from core.datasets import load_moons, load_circles, load_blobs


def plot_k_distance_graph(X, minPts, show_optimal=True):
    """
    Visualiza el k-distance graph para análisis de sensibilidad de epsilon.
    
    Parámetros:
    -----------
    X : ndarray
        Dataset (n_samples, n_features)
    minPts : int
        Parámetro minPts de DBSCAN
    show_optimal : bool
        Si True, marca el epsilon óptimo detectado
    
    Retorna:
    --------
    k_distances : ndarray
        Distancias al k-ésimo vecino ordenadas
    """
    k = minPts - 1
    
    k_distances = compute_k_distance(X, k)
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, linewidth=2, color='steelblue')
    plt.title(f"K-Distance Graph (k={k})", fontsize=16, fontweight='bold')
    plt.xlabel("Puntos ordenados por distancia", fontsize=13)
    plt.ylabel(f"Distancia al {k}-ésimo vecino", fontsize=13)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Marcar epsilon óptimo si se solicita
    if show_optimal:
        eps_opt = find_optimal_eps(X, k)
        plt.axhline(y=eps_opt, color='red', linestyle='--', 
                   linewidth=2.5, label=f'ε óptimo = {eps_opt:.4f}')
        
        # Marcar el punto en la curva
        idx = np.where(k_distances >= eps_opt)[0][0]
        plt.scatter([idx], [eps_opt], color='red', s=150, 
                   zorder=5, marker='o', edgecolors='darkred', linewidths=2)
        
        plt.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return k_distances


def analyze_multiple_k(X, k_values=[3, 4, 5, 6, 7]):
    """
    Analiza sensibilidad para múltiples valores de k (minPts - 1).
    Útil para ver cómo cambia el epsilon óptimo.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    results = {}
    
    for idx, k in enumerate(k_values):
        if idx >= len(axes):
            break
            
        k_distances = compute_k_distance(X, k)
        eps_opt = find_optimal_eps(X, k)
        
        results[k] = {
            'k_distances': k_distances,
            'eps_optimal': eps_opt
        }
        
        # Graficar
        axes[idx].plot(k_distances, linewidth=2)
        axes[idx].axhline(y=eps_opt, color='red', linestyle='--', linewidth=2)
        axes[idx].set_title(f"k={k} (minPts={k+1})\nε óptimo={eps_opt:.4f}", 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Puntos ordenados", fontsize=10)
        axes[idx].set_ylabel(f"Distancia al {k}-ésimo vecino", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    # Ocultar subplots sobrantes
    for idx in range(len(k_values), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("ANÁLISIS DE SENSIBILIDAD DE EPSILON - DBSCAN")
    print("=" * 70 + "\n")
    
    # Cargar dataset de prueba
    print("Cargando dataset 'moons'...")
    X, _ = load_moons(n_samples=500, noise=0.07)
    print(f"  • Puntos: {len(X)}")
    print(f"  • Dimensiones: {X.shape[1]}\n")
    
    # Experimento 1: k-distance graph simple
    print("─" * 70)
    print("EXPERIMENTO 1: K-Distance Graph (minPts=5)")
    print("─" * 70)
    minPts = 5
    k_distances = plot_k_distance_graph(X, minPts, show_optimal=True)
    
    k = minPts - 1
    eps_opt = find_optimal_eps(X, k)
    
    print(f"\nRESULTADOS:")
    print(f"  • Epsilon óptimo detectado: {eps_opt:.4f}")
    print(f"  • Percentil 90: {np.percentile(k_distances, 90):.4f}")
    print(f"  • Percentil 95: {np.percentile(k_distances, 95):.4f}")
    print(f"  • Máximo: {k_distances[-1]:.4f}\n")
    
    # Experimento 2: Comparar múltiples valores de k
    print("─" * 70)
    print("EXPERIMENTO 2: Comparación con diferentes valores de minPts")
    print("─" * 70 + "\n")
    
    results = analyze_multiple_k(X, k_values=[3, 4, 5, 6, 7])
    
    print("\nRESUMEN DE EPSILON ÓPTIMO POR minPts:")
    print("-" * 40)
    for k, data in results.items():
        print(f"  k={k} (minPts={k+1}): ε={data['eps_optimal']:.4f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN:")
    print("  • El 'codo' en el gráfico sugiere el epsilon óptimo")
    print("  • Antes del codo: muchos puntos con baja densidad")
    print("  • Después del codo: puntos comienzan a fusionarse en clusters")
    print("  • Mayor minPts → mayor epsilon requerido")
    print("=" * 70)