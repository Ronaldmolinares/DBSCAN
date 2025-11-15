import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

from core.my_dbscan import MyDBSCAN
from core.datasets import load_dataset_by_name, load_csv
from core.metrics import all_metrics

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DBSCANApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DBSCAN Clustering - GUI")
        self.root.geometry("950x600")

        self.data = None

        # ----- Frame izquierda -----
        control_frame = tk.Frame(root, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Selección de dataset
        ttk.Label(control_frame, text="Dataset:").pack(anchor="w")
        self.dataset_var = tk.StringVar()
        dataset_menu = ttk.Combobox(
            control_frame, textvariable=self.dataset_var,
            values=["moons", "circles", "blobs", "csv"]
        )
        dataset_menu.pack(fill="x")
        dataset_menu.current(0)

        load_btn = ttk.Button(control_frame, text="Cargar dataset", command=self.load_dataset)
        load_btn.pack(pady=5, fill="x")

        # Parámetros
        ttk.Label(control_frame, text="ε (epsilon):").pack(anchor="w")
        self.eps_var = tk.DoubleVar(value=0.3)
        eps_scale = ttk.Scale(control_frame, from_=0.1, to=2.0, variable=self.eps_var)
        eps_scale.pack(fill="x")

        ttk.Label(control_frame, text="minPts:").pack(anchor="w")
        self.minpts_var = tk.IntVar(value=5)
        minpts_scale = ttk.Scale(control_frame, from_=2, to=20, variable=self.minpts_var)
        minpts_scale.pack(fill="x")

        run_btn = ttk.Button(control_frame, text="Ejecutar DBSCAN", command=self.run_dbscan)
        run_btn.pack(pady=15, fill="x")

        # Info
        self.info_label = ttk.Label(control_frame, text="", justify="left", wraplength=200)
        self.info_label.pack(pady=10)

        # ----- Frame derecha (gráfico) -----
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


    def load_dataset(self):
        option = self.dataset_var.get()

        if option in ["moons", "circles", "blobs"]:
            self.data, _ = load_dataset_by_name(option)
        else:
            file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if not file:
                return
            self.data, _ = load_csv(file)

        self.plot_data(self.data)
        self.info_label.config(text="Dataset cargado.")


    def run_dbscan(self):
        if self.data is None:
            messagebox.showerror("Error", "Cargue un dataset primero.")
            return

        eps = float(self.eps_var.get())
        minpts = int(self.minpts_var.get())

        model = MyDBSCAN(eps=eps, min_pts=minpts)
        labels = model.fit_predict(self.data)

        self.plot_clusters(self.data, labels)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        outliers = list(labels).count(-1)

        metrics = all_metrics(self.data, labels)

        self.info_label.config(
            text=f"Clusters: {n_clusters}\n"
                 f"Outliers: {outliers}\n"
                 f"Silhouette: {metrics['silhouette']}\n"
                 f"Davies-Bouldin: {metrics['davies_bouldin']}\n"
                 f"Outlier Ratio: {metrics['outlier_ratio']:.2f}"
        )


    def plot_data(self, X):
        self.ax.clear()
        self.ax.scatter(X[:, 0], X[:, 1], s=10, color="gray")
        self.ax.set_title("Dataset")
        self.canvas.draw()


    def plot_clusters(self, X, labels):
        self.ax.clear()
        unique_labels = set(labels)

        for label in unique_labels:
            mask = labels == label

            if label == -1:
                color = "black"
                label_name = "Ruido"
            else:
                color = None
                label_name = f"Cluster {label}"

            self.ax.scatter(X[mask, 0], X[mask, 1], s=20, label=label_name, c=color)

        self.ax.set_title("Resultado de DBSCAN")
        self.ax.legend()
        self.canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = DBSCANApp(root)
    root.mainloop()