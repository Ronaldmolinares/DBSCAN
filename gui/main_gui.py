import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Módulos propios
from core.datasets import load_dataset_by_name, load_csv
from core.my_dbscan import MyDBSCAN
from core.metrics import compute_all_metrics, compute_k_distance, find_optimal_eps


class DBSCANApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DBSCAN Clustering - GUI")
        self.root.geometry("1150x650")

        self.data = None

        # ===========================================================
        # PANEL IZQUIERDO
        # ===========================================================
        control_frame = tk.Frame(root, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # ---------- Selección de dataset ----------
        ttk.Label(control_frame, text="Dataset:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.dataset_var = tk.StringVar()

        dataset_menu = ttk.Combobox(
            control_frame, textvariable=self.dataset_var,
            values=["moons", "circles", "blobs", "csv"]
        )
        dataset_menu.pack(fill="x")
        dataset_menu.current(0)

        ttk.Button(control_frame, text="Cargar dataset", command=self.load_dataset).pack(
            pady=5, fill="x"
        )

        # ---------- Epsilon ----------
        ttk.Label(control_frame, text="ε (epsilon):", font=("Arial", 11, "bold")).pack(anchor="w")
        self.eps_var = tk.DoubleVar(value=0.3)

        eps_scale = ttk.Scale(
            control_frame, from_=0.05, to=1.0, variable=self.eps_var,
            command=lambda x: self.update_eps_label()
        )
        eps_scale.pack(fill="x")

        self.eps_label = ttk.Label(control_frame, text="Epsilon: 0.30")
        self.eps_label.pack(anchor="w")

        # ---------- minPts ----------
        ttk.Label(control_frame, text="minPts:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.minpts_var = tk.IntVar(value=5)

        minpts_scale = ttk.Scale(
            control_frame, from_=2, to=20, variable=self.minpts_var,
            command=lambda x: self.update_minpts_label()
        )
        minpts_scale.pack(fill="x")

        self.minpts_label = ttk.Label(control_frame, text="minPts: 5")
        self.minpts_label.pack(anchor="w")

        # ---------- Botones principales ----------
        ttk.Button(control_frame, text="Ejecutar DBSCAN",
                   command=self.run_dbscan).pack(pady=10, fill="x")

        ttk.Button(control_frame, text="Comparar Algoritmos",
                   command=self.run_comparison).pack(pady=5, fill="x")

        ttk.Button(control_frame, text="Analizar ε (k-distance graph)",
                   command=self.show_k_graph).pack(pady=5, fill="x")

        ttk.Label(control_frame, text="Información:", font=("Arial", 11, "bold")).pack(
            anchor="w", pady=10
        )

        self.info_label = ttk.Label(control_frame, text="", justify="left", wraplength=240)
        self.info_label.pack(pady=5)

        # ===========================================================
        # PANEL DERECHO – GRÁFICO
        # ===========================================================
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # ===========================================================
    # ACTUALIZACIÓN DE ETIQUETAS
    # ===========================================================
    def update_eps_label(self):
        self.eps_label.config(text=f"Epsilon: {self.eps_var.get():.2f}")

    def update_minpts_label(self):
        self.minpts_label.config(text=f"minPts: {int(self.minpts_var.get())}")

    # ===========================================================
    # CARGA DE DATOS
    # ===========================================================
    def load_dataset(self):
        option = self.dataset_var.get()

        try:
            if option in ["moons", "circles", "blobs"]:
                self.data, _ = load_dataset_by_name(option)
            else:
                file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
                if not file:
                    return
                self.data, _ = load_csv(file)

            self.plot_data(self.data)
            self.info_label.config(text="Dataset cargado correctamente.")

        except Exception as e:
            messagebox.showerror("Error al cargar dataset", str(e))

    # ===========================================================
    # EJECUCIÓN DE DBSCAN
    # ===========================================================
    def run_dbscan(self):
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero.")
            return

        eps = float(self.eps_var.get())
        minpts = int(self.minpts_var.get())

        model = MyDBSCAN(eps=eps, min_pts=minpts)
        labels = model.fit_predict(self.data)

        self.plot_clusters(self.data, labels)

        # Métricas
        metrics = compute_all_metrics(self.data, labels)

        self.info_label.config(
            text=(
                f"Clusters detectados: {metrics['n_clusters']}\n"
                f"Outliers: {metrics['n_noise']}\n"
                f"Silhouette: {metrics['silhouette_score']:.3f}\n"
                f"Davies-Bouldin: {metrics['davies_bouldin_index']:.3f}\n"
                f"Outlier Ratio: {metrics['outlier_ratio']:.3f}"
            )
        )

    # ===========================================================
    # COMPARACIÓN ENTRE ALGORITMOS
    # ===========================================================
    def run_comparison(self):
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero.")
            return

        from sklearn.cluster import KMeans, AgglomerativeClustering

        X = self.data
        eps = float(self.eps_var.get())
        minpts = int(self.minpts_var.get())

        # --- DBSCAN ---
        dbscan = MyDBSCAN(eps=eps, min_pts=minpts)
        labels_db = dbscan.fit_predict(X)
        m_db = compute_all_metrics(X, labels_db)

        # --- Elegir k automáticamente ---
        n_clusters_guess = max(2, m_db["n_clusters"])

        # --- K-Means ---
        kmeans = KMeans(n_clusters=n_clusters_guess)
        labels_km = kmeans.fit_predict(X)
        m_km = compute_all_metrics(X, labels_km)

        # --- Hierarchical ---
        hac = AgglomerativeClustering(n_clusters=n_clusters_guess)
        labels_hc = hac.fit_predict(X)
        m_hc = compute_all_metrics(X, labels_hc)

        # --- Graficación ---
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        titles = ["DBSCAN", "K-Means", "HAC"]
        labels_list = [labels_db, labels_km, labels_hc]
        metrics_list = [m_db, m_km, m_hc]

        for ax, lbls, title, metrics in zip(axes, labels_list, titles, metrics_list):
            ax.scatter(X[:, 0], X[:, 1], c=lbls, s=15, cmap="viridis")
            ax.set_title(
                f"{title}\n"
                f"Silh: {metrics['silhouette_score']:.3f} | "
                f"DBI: {metrics['davies_bouldin_index']:.3f} | "
                f"Outliers: {metrics['outlier_ratio']:.2f}"
            )

        plt.tight_layout()
        plt.show()

    # ===========================================================
    # ANÁLISIS DEL EPSILON (k-distance graph)
    # ===========================================================
    def show_k_graph(self):
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero.")
            return

        X = self.data
        minPts = int(self.minpts_var.get())
        k = minPts - 1

        try:
            k_distances = compute_k_distance(X, k)
            eps_recommended = find_optimal_eps(X, k)

            plt.figure(figsize=(6, 4))
            plt.plot(k_distances, label="k-distances")
            plt.scatter(np.argmax(np.gradient(np.gradient(k_distances))),
                        k_distances[np.argmax(np.gradient(np.gradient(k_distances)))],
                        color="red", s=50, label=f"Epsilon recomendado: {eps_recommended:.3f}")

            plt.title(f"k-distance graph (k={k})")
            plt.xlabel("Puntos ordenados")
            plt.ylabel(f"Distancia al {k}-ésimo vecino")
            plt.legend()
            plt.grid(True)
            plt.show()

            self.info_label.config(text=f"Epsilon recomendado: {eps_recommended:.3f}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ===========================================================
    # GRAFICACIÓN
    # ===========================================================
    def plot_data(self, X):
        self.ax.clear()
        self.ax.scatter(X[:, 0], X[:, 1], s=15, color="gray")
        self.ax.set_title("Dataset")
        self.canvas.draw()

    def plot_clusters(self, X, labels):
        self.ax.clear()
        unique = set(labels)

        for label in unique:
            mask = (labels == label)
            if label == -1:
                self.ax.scatter(X[mask, 0], X[mask, 1], s=20, c="black", label="Ruido")
            else:
                self.ax.scatter(X[mask, 0], X[mask, 1], s=20, label=f"Cluster {label}")

        self.ax.set_title("Resultado DBSCAN")
        self.ax.legend()
        self.canvas.draw()


# ===========================================================
# EJECUCIÓN PRINCIPAL
# ===========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DBSCANApp(root)
    root.mainloop()
