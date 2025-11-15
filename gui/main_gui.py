import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Módulos propios
from core.datasets import load_dataset_by_name, load_csv
from core.my_dbscan import MyDBSCAN
from core.metrics import all_metrics


class DBSCANApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DBSCAN Clustering - GUI")
        self.root.geometry("1100x650")

        self.data = None

        # ----- PANEL IZQUIERDO (Controles) -----
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

        load_btn = ttk.Button(control_frame, text="Cargar dataset", command=self.load_dataset)
        load_btn.pack(pady=5, fill="x")

        # ---------- Parámetro epsilon ----------
        ttk.Label(control_frame, text="ε (epsilon):", font=("Arial", 11, "bold")).pack(anchor="w")
        self.eps_var = tk.DoubleVar(value=0.3)
        eps_scale = ttk.Scale(control_frame, from_=0.05, to=2.0, variable=self.eps_var,
                              command=lambda x: self.update_eps_label())
        eps_scale.pack(fill="x")

        self.eps_label = ttk.Label(control_frame, text="Epsilon: 0.30")
        self.eps_label.pack(anchor="w")

        # ---------- Parámetro minPts ----------
        ttk.Label(control_frame, text="minPts:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.minpts_var = tk.IntVar(value=5)
        minpts_scale = ttk.Scale(control_frame, from_=2, to=20, variable=self.minpts_var,
                                 command=lambda x: self.update_minpts_label())
        minpts_scale.pack(fill="x")

        self.minpts_label = ttk.Label(control_frame, text="minPts: 5")
        self.minpts_label.pack(anchor="w")

        # ---------- Botones principales ----------
        run_btn = ttk.Button(control_frame, text="Ejecutar DBSCAN", command=self.run_dbscan)
        run_btn.pack(pady=10, fill="x")

        compare_btn = ttk.Button(control_frame, text="Comparar Algoritmos", command=self.run_comparison)
        compare_btn.pack(pady=5, fill="x")

        eps_btn = ttk.Button(control_frame, text="Analizar ε (k-distance graph)", command=self.show_k_graph)
        eps_btn.pack(pady=5, fill="x")

        # ---------- Información ----------
        ttk.Label(control_frame, text="Información:", font=("Arial", 11, "bold")).pack(anchor="w", pady=10)
        self.info_label = ttk.Label(control_frame, text="", justify="left", wraplength=240)
        self.info_label.pack(pady=5)

        # ----- PANEL DERECHO (Gráfica) -----
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

        metrics = all_metrics(self.data, labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        outliers = list(labels).count(-1)

        self.info_label.config(
            text=f"Clusters detectados: {n_clusters}\n"
                 f"Outliers: {outliers}\n"
                 f"Silhouette: {metrics['silhouette']}\n"
                 f"Davies-Bouldin: {metrics['davies_bouldin']}\n"
                 f"Outlier Ratio: {metrics['outlier_ratio']:.3f}"
        )


    # ===========================================================
    # COMPARACIÓN DE ALGORITMOS
    # ===========================================================

    def run_comparison(self):
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero.")
            return

        from sklearn.cluster import KMeans, AgglomerativeClustering

        X = self.data
        eps = float(self.eps_var.get())
        minpts = int(self.minpts_var.get())

        dbscan = MyDBSCAN(eps=eps, min_pts=minpts)
        labels_db = dbscan.fit_predict(X)
        m_db = all_metrics(X, labels_db)

        k = 2
        kmeans = KMeans(n_clusters=k)
        labels_km = kmeans.fit_predict(X)
        m_km = all_metrics(X, labels_km)

        hac = AgglomerativeClustering(n_clusters=k)
        labels_hc = hac.fit_predict(X)
        m_hc = all_metrics(X, labels_hc)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        titles = ["DBSCAN", "K-Means", "HAC"]
        metrics_list = [m_db, m_km, m_hc]
        labels_list = [labels_db, labels_km, labels_hc]

        for ax, labels, title, metrics in zip(axes, labels_list, titles, metrics_list):
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap="viridis")
            ax.set_title(
                f"{title}\n"
                f"Silhouette: {metrics['silhouette']:.3f} | "
                f"DBI: {metrics['davies_bouldin']:.3f} | "
                f"Outliers: {metrics['outlier_ratio']:.2f}"
            )

        plt.tight_layout()
        plt.show()


    # ===========================================================
    # K-DISTANCE GRAPH (SELECCIÓN DE EPS)
    # ===========================================================

    def show_k_graph(self):
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero.")
            return

        from sklearn.neighbors import NearestNeighbors

        X = self.data
        minPts = int(self.minpts_var.get())
        k = minPts - 1

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_distances = np.sort(distances[:, -1])

        # Detectar el codo
        y = k_distances
        x = np.arange(len(y))

        p1 = np.array([0, y[0]])
        p2 = np.array([len(y)-1, y[-1]])

        distances_to_line = np.abs(
            np.cross(p2 - p1, np.column_stack((x, y)) - p1)
        ) / np.linalg.norm(p2 - p1)

        elbow_idx = np.argmax(distances_to_line)
        epsilon_recomendado = y[elbow_idx]

        # Graficar
        plt.figure(figsize=(6, 4))
        plt.plot(y, label="k-distances")
        plt.scatter(elbow_idx, epsilon_recomendado, color="red", s=50,
                    label=f"Codo = {epsilon_recomendado:.3f}")

        plt.title(f"k-distance graph (k={k})")
        plt.xlabel("Puntos ordenados")
        plt.ylabel(f"Distancia al {k}-ésimo vecino")
        plt.legend()
        plt.grid(True)
        plt.show()

        self.info_label.config(
            text=f"Epsilon recomendado: {epsilon_recomendado:.3f}\n"
                 f"(detectado por método del codo)"
        )


    # ===========================================================
    # GRAFICACIÓN DE LA GUI
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
