# app.py (main entry point)
import tkinter as tk
from tkinter import filedialog
from gui import MainApplication
from dataset_export import export_yolo_dataset

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Smart Polygon Annotator")
    root.geometry("1280x1280")
    app = MainApplication(root)
    root.mainloop()