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
        self.root.geometry("1200x700")

        self.data = None
        self.true_labels = None  # Etiquetas verdaderas para calcular ARI

        # ===========================================================
        # PANEL IZQUIERDO
        # ===========================================================
        control_frame = tk.Frame(root, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # ---------- Selección de dataset ----------
        ttk.Label(control_frame, text="Dataset:", font=("Arial", 13, "bold")).pack(anchor="w")
        self.dataset_var = tk.StringVar()

        dataset_menu = ttk.Combobox(
            control_frame, textvariable=self.dataset_var,
            values=["moons", "circles", "blobs", "csv"],
            font=("Arial", 11)
        )
        dataset_menu.pack(fill="x")
        dataset_menu.current(0)

        ttk.Button(control_frame, text="Cargar dataset", command=self.load_dataset).pack(
            pady=5, fill="x"
        )

        # ---------- Epsilon ----------
        ttk.Label(control_frame, text="ε (epsilon):", font=("Arial", 13, "bold")).pack(anchor="w")
        self.eps_var = tk.DoubleVar(value=0.3)

        eps_scale = ttk.Scale(
            control_frame, from_=0.05, to=2.0, variable=self.eps_var,
            command=lambda x: self.update_eps_label()
        )
        eps_scale.pack(fill="x")

        self.eps_label = ttk.Label(control_frame, text="Epsilon: 0.30", font=("Arial", 11))
        self.eps_label.pack(anchor="w")

        # ---------- minPts ----------
        ttk.Label(control_frame, text="minPts:", font=("Arial", 13, "bold")).pack(anchor="w")
        self.minpts_var = tk.IntVar(value=5)

        minpts_scale = ttk.Scale(
            control_frame, from_=2, to=20, variable=self.minpts_var,
            command=lambda x: self.update_minpts_label()
        )
        minpts_scale.pack(fill="x")

        self.minpts_label = ttk.Label(control_frame, text="minPts: 5", font=("Arial", 11))
        self.minpts_label.pack(anchor="w")

        # ---------- Botones principales ----------
        ttk.Button(control_frame, text="Ejecutar DBSCAN",
                   command=self.run_dbscan).pack(pady=10, fill="x")

        ttk.Button(control_frame, text="Comparar Algoritmos",
                   command=self.run_comparison).pack(pady=5, fill="x")

        ttk.Button(control_frame, text="Analizar ε (k-distance graph)",
                   command=self.show_k_graph).pack(pady=5, fill="x")
        
        ttk.Button(control_frame, text="Ver Tabla de Métricas",
                   command=self.show_metrics_table).pack(pady=5, fill="x")

        ttk.Label(control_frame, text="Información:", font=("Arial", 13, "bold")).pack(
            anchor="w", pady=10
        )

        self.info_label = ttk.Label(control_frame, text="", justify="left", wraplength=240, font=("Arial", 11))
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
                self.data, self.true_labels = load_dataset_by_name(option)
            else:
                file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
                if not file:
                    return
                self.data, self.true_labels = load_csv(file)

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
        metrics = compute_all_metrics(self.data, labels, self.true_labels)

        info_text = (
            f"Clusters: {metrics['n_clusters']}\n"
            f"Outliers: {metrics['n_noise']}\n"
            f"Silhouette: {metrics['silhouette_score']:.4f}\n"
            f"Davies-B: {metrics['davies_bouldin_index']:.4f}\n"
            f"Out. Ratio: {metrics['outlier_ratio']:.4f}"
        )
        
        # Agregar ARI si está disponible
        if 'adjusted_rand_index' in metrics:
            info_text += f"\nARI: {metrics['adjusted_rand_index']:.4f}"
        
        self.info_label.config(text=info_text)

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
        m_db = compute_all_metrics(X, labels_db, self.true_labels)

        # --- Elegir k automáticamente ---
        n_clusters_guess = max(2, m_db["n_clusters"])

        # --- K-Means ---
        kmeans = KMeans(n_clusters=n_clusters_guess)
        labels_km = kmeans.fit_predict(X)
        m_km = compute_all_metrics(X, labels_km, self.true_labels)

        # --- Hierarchical ---
        hac = AgglomerativeClustering(n_clusters=n_clusters_guess)
        labels_hc = hac.fit_predict(X)
        m_hc = compute_all_metrics(X, labels_hc, self.true_labels)

        # --- Graficación ---
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        titles = ["DBSCAN", "K-Means", "HAC"]
        labels_list = [labels_db, labels_km, labels_hc]
        metrics_list = [m_db, m_km, m_hc]

        for ax, lbls, title, metrics in zip(axes, labels_list, titles, metrics_list):
            ax.scatter(X[:, 0], X[:, 1], c=lbls, s=20, cmap="viridis", alpha=0.7)
            ax.set_title(
                f"{title}\n"
                f"Silh: {metrics['silhouette_score']:.4f} | "
                f"DBI: {metrics['davies_bouldin_index']:.4f} | "
                f"Out: {metrics['outlier_ratio']:.3f}",
                fontsize=11
            )
            ax.tick_params(labelsize=9)

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

            plt.figure(figsize=(8, 5))
            plt.plot(k_distances, label="k-distances", linewidth=2)
            plt.scatter(np.argmax(np.gradient(np.gradient(k_distances))),
                        k_distances[np.argmax(np.gradient(np.gradient(k_distances)))],
                        color="red", s=100, label=f"Epsilon recomendado: {eps_recommended:.4f}", zorder=5)

            plt.title(f"k-distance graph (k={k})", fontsize=14, fontweight='bold')
            plt.xlabel("Puntos ordenados", fontsize=12)
            plt.ylabel(f"Distancia al {k}-ésimo vecino", fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            self.info_label.config(text=f"Epsilon recomendado: {eps_recommended:.3f}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ===========================================================
    # GRAFICACIÓN
    # ===========================================================
    def plot_data(self, X):
        self.ax.clear()
        self.ax.scatter(X[:, 0], X[:, 1], s=25, color="gray", alpha=0.6)
        self.ax.set_title("Dataset", fontsize=14, fontweight='bold')
        self.ax.tick_params(labelsize=10)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_clusters(self, X, labels):
        self.ax.clear()
        unique = set(labels)

        for label in unique:
            mask = (labels == label)
            if label == -1:
                self.ax.scatter(X[mask, 0], X[mask, 1], s=30, c="black", marker='x', 
                              label="Ruido", linewidths=1.5)
            else:
                self.ax.scatter(X[mask, 0], X[mask, 1], s=30, label=f"Cluster {label}", alpha=0.7)

        self.ax.set_title("Resultado DBSCAN", fontsize=14, fontweight='bold')
        self.ax.legend(fontsize=10)
        self.ax.tick_params(labelsize=10)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def show_metrics_table(self):
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero.")
            return

        from sklearn.cluster import KMeans, AgglomerativeClustering

        X = self.data
        eps = float(self.eps_var.get())
        minpts = int(self.minpts_var.get())

        #======= DBSCAN =======
        dbscan = MyDBSCAN(eps=eps, min_pts=minpts)
        labels_db = dbscan.fit_predict(X)
        m_db = compute_all_metrics(X, labels_db, self.true_labels)

        # ======= K-Means =======
        # Usamos el número de clusters detectado por DBSCAN cuando es válido
        k = max(2, m_db["n_clusters"])
        kmeans = KMeans(n_clusters=k)
        labels_km = kmeans.fit_predict(X)
        m_km = compute_all_metrics(X, labels_km, self.true_labels)

        # ======= Hierarchical =======
        hac = AgglomerativeClustering(n_clusters=k)
        labels_hc = hac.fit_predict(X)
        m_hc = compute_all_metrics(X, labels_hc, self.true_labels)

        results = {
            "DBSCAN": m_db,
            "K-Means": m_km,
            "HAC": m_hc
        }

        # ======= CREAR VENTANA =======
        win = tk.Toplevel(self.root)
        win.title("Comparación de Métricas de Clustering")
        win.geometry("850x400")

        ttk.Label(win, text="Comparación de Métricas de Clustering",
              font=("Arial", 15, "bold")).pack(pady=15)

        # Crear estilo para la tabla con fuente más grande
        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 11), rowheight=30)
        style.configure("Treeview.Heading", font=("Arial", 12, "bold"))

        columns = ("metrica", "dbscan", "kmeans", "hac")
        table = ttk.Treeview(win, columns=columns, show="headings", height=10)

        table.heading("metrica", text="Métrica")
        table.heading("dbscan", text="DBSCAN")
        table.heading("kmeans", text="K-Means")
        table.heading("hac", text="Hierarchical")

        table.column("metrica", width=230)
        table.column("dbscan", width=180)
        table.column("kmeans", width=180)
        table.column("hac", width=180)

        table.pack(fill="both", expand=True, padx=10, pady=10)

        # Obtener el conjunto de métricas
        metric_names = sorted(set(list(m_db.keys()) +
                              list(m_km.keys()) +
                              list(m_hc.keys())))

        for name in metric_names:
            v_db = m_db.get(name, "")
            v_km = m_km.get(name, "")
            v_hc = m_hc.get(name, "")

            # Convertir floats a 4 decimales
            def fmt(x):
                return f"{x:.4f}" if isinstance(x, float) else str(x)

            table.insert("", tk.END, values=(name, fmt(v_db), fmt(v_km), fmt(v_hc)))

