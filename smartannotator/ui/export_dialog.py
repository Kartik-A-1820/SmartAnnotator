"""Export dialog: format, split ratios and destination; runs the export on a thread."""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..export import FORMATS, export_dataset

DEFAULT_SPLIT = (("train", 80), ("val", 15), ("test", 5))


class ExportDialog(tk.Toplevel):
    def __init__(self, parent, project, settings):
        super().__init__(parent)
        self.project, self.settings = project, settings
        self._results = queue.Queue()
        self.title("Export Dataset")
        self.transient(parent)
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        self.format = tk.StringVar(value=settings.get("export_format", "yolo-seg"))
        ttk.Label(frame, text="Format").grid(row=0, column=0, sticky=tk.NW, pady=2)
        for row, (key, label) in enumerate(FORMATS.items()):
            ttk.Radiobutton(frame, text=label, value=key, variable=self.format).grid(
                row=row, column=1, columnspan=3, sticky=tk.W)

        ttk.Label(frame, text="Split %").grid(row=3, column=0, sticky=tk.W, pady=(8, 2))
        self.ratios = []
        for col, (name, percent) in enumerate(DEFAULT_SPLIT):
            cell = ttk.Frame(frame)
            cell.grid(row=3, column=1 + col, sticky=tk.W, pady=(8, 2))
            ttk.Label(cell, text=name).pack(side=tk.LEFT)
            var = tk.IntVar(value=percent)
            ttk.Spinbox(cell, from_=0, to=100, width=4, textvariable=var).pack(side=tk.LEFT, padx=2)
            self.ratios.append(var)

        self.include_unlabeled = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Include images without annotations (as background)",
                        variable=self.include_unlabeled).grid(row=4, column=1, columnspan=3, sticky=tk.W, pady=4)

        default_dir = settings.get("export_dir") or os.path.join(os.path.dirname(project.path), "dataset")
        self.out_dir = tk.StringVar(value=default_dir)
        ttk.Label(frame, text="Output").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.out_dir, width=46).grid(row=5, column=1, columnspan=2, sticky=tk.EW)
        ttk.Button(frame, text="Browse...", command=self._browse).grid(row=5, column=3)

        self.progress = ttk.Progressbar(frame, length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=4, sticky=tk.EW, pady=8)
        ttk.Button(frame, text="Cancel", command=self.destroy).grid(row=7, column=2, sticky=tk.E)
        self.export_button = ttk.Button(frame, text="Export", command=self._start)
        self.export_button.grid(row=7, column=3, sticky=tk.E)

    def _browse(self):
        folder = filedialog.askdirectory(parent=self, initialdir=self.out_dir.get())
        if folder:
            self.out_dir.set(folder)

    def _start(self):
        try:
            ratios = tuple(max(0, v.get()) for v in self.ratios)
        except tk.TclError:
            messagebox.showerror("Export", "Split percentages must be numbers.", parent=self)
            return
        if ratios[0] <= 0:
            messagebox.showerror("Export", "The train split must be above 0%.", parent=self)
            return
        out_dir = self.out_dir.get().strip()
        if not out_dir:
            return
        self.settings.update(export_dir=out_dir, export_format=self.format.get())
        self.export_button.config(state=tk.DISABLED)

        args = (list(self.project.images), dict(self.project.annotations),
                [c["name"] for c in self.project.classes], out_dir, self.format.get(), ratios,
                self.include_unlabeled.get())
        threading.Thread(target=self._run, args=args, daemon=True).start()
        self._poll()

    def _run(self, *args):
        try:
            counts = export_dataset(*args, progress=lambda done, total: self._results.put(("progress", done / total)))
            self._results.put(("done", counts))
        except Exception as e:
            self._results.put(("error", f"{type(e).__name__}: {e}"))

    def _poll(self):
        if not self.winfo_exists():
            return
        while not self._results.empty():
            kind, value = self._results.get()
            if kind == "progress":
                self.progress["value"] = 100 * value
            elif kind == "error":
                messagebox.showerror("Export failed", value, parent=self)
                self.destroy()
                return
            else:
                summary = ", ".join(f"{split}: {n}" for split, n in value.items())
                messagebox.showinfo("Export complete", f"Dataset exported to\n{self.out_dir.get()}\n\n"
                                    f"Images per split: {summary}", parent=self.master)
                self.destroy()
                return
        self.after(50, self._poll)
