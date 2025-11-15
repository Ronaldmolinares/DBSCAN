"""
run.py
Muestra la interfaz gráfica DBSCAN GUI.
"""

import tkinter as tk
from gui.main_gui import DBSCANApp

if __name__ == "__main__":
    root = tk.Tk()
    app = DBSCANApp(root)
    root.mainloop()