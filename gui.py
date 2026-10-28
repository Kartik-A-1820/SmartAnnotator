# gui.py (Tkinter interface: Roboflow-style smart polygon, box, polygon and edit tools)
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from annotation_processing import annotation_contains, detail_tolerance, make_annotation, polygons_bbox
from dataset_export import FORMATS, export_dataset
from image_io import IMAGE_EXTENSIONS, ImageCache, is_image_file, list_images
from project import Project, load_settings, save_settings
from sam_integration import (MODELS, InferenceWorker, backend_installed, candidate_models,
                             detect_device, install_hint)

SHORTCUTS = """Tools
  S  Smart polygon   click an object, SAM outlines it
  B  Bounding box    drag a box (saved instantly)
  P  Polygon         click vertices by hand
  V  Edit            move vertices / objects

Smart polygon
  Hover                 preview what a click would select
  Left click            include region (positive point)
  Right / Shift+click   exclude region (negative point)
  Left drag             box prompt around the object
  Enter / Space         accept with the current class
  Tab                   cycle alternative masks
  Ctrl+Z / Backspace    remove last point     Esc  clear

Polygon tool
  Click to add vertices; Enter, double-click or click the first vertex to close.
  Backspace / right-click removes the last vertex.

Edit tool
  Click an object to select it, drag vertices or the object.
  Shift+click or right-click a vertex deletes it; double-click an edge adds one.

Everywhere
  1-9, 0                select class 1-10
  D / Right / PgDn      next image        A / Left / PgUp   previous image
  N                     next image without annotations
  Del                   delete object under cursor (or selected)
  C                     set object under cursor to the current class
  H  show/hide annotations    F  fit to window
  Mouse wheel  zoom           Middle drag / Ctrl+drag  pan
  Ctrl+Z / Ctrl+Y       undo / redo
  Ctrl+O  open folder   Ctrl+S  save   Ctrl+E  export

Annotations autosave to smartannotator_project.json next to your images."""

TOOL_HINTS = {
    "smart": "Smart polygon: click an object (right/Shift-click excludes, drag for a box). "
             "Enter accepts, Tab cycles masks, Ctrl+Z removes the last point.",
    "box": "Box: drag to draw a bounding box. It is saved instantly with the current class.",
    "polygon": "Polygon: click to add vertices. Enter / double-click / click the first vertex closes it.",
    "edit": "Edit: click an object, then drag its vertices or the object itself. "
            "Shift-click a vertex deletes it, double-click an edge adds one.",
}
TOOL_KEYS = {"smart": "s", "box": "b", "polygon": "p", "edit": "v"}
TOOL_LABELS = {"smart": "Smart (S)", "box": "Box (B)", "polygon": "Polygon (P)", "edit": "Edit (V)"}
TOOL_CURSORS = {"smart": "crosshair", "box": "tcross", "polygon": "pencil", "edit": "arrow"}

PREFETCH_AHEAD = 1
PREVIEW_MAX_SIDE = 2560   # zoomed-out views render from a copy this size
HANDLE_RADIUS = 7         # vertex grab distance in screen pixels
DRAG_THRESHOLD = 6


def _hex_rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _text_color(color):
    r, g, b = _hex_rgb(color)
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#FFFFFF"


def _segment_distance(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


class MainApplication(ttk.Frame):
    def __init__(self, master=None, model=None, device="auto", precision="auto", images=None):
        super().__init__(master)
        self.master = master
        self.settings = load_settings()
        self.device_pref = device
        self.precision = precision
        self.project = Project()
        self.image_cache = ImageCache(max_items=4)

        self.index = 0
        self.cur_path = None
        self.cur_image = None
        self._preview = None         # (factor, downscaled copy) for zoomed-out rendering
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.auto_fit = True

        # smart polygon prompt
        self.points, self.labels, self.box = [], [], None
        self.candidates, self.cand_idx = [], 0
        self.request_id = 0
        # hover preview
        self.hover_id = 0
        self.hover_point = None
        self.hover_candidates = []
        self._hover_job = None
        self._pending_hover = None
        # manual polygon
        self.poly_pts = []
        self._cursor = None
        # edit tool
        self._edit = None            # active drag on the selected object
        self._edit_preview = None    # (index, annotation) while dragging

        self.current_class = 0
        self.hover = None
        self.selected_obj = None
        self.model_ready = False

        self._press = None
        self._drag_mode = None
        self._last = None
        self._base_key, self._base = None, None
        self._img_item = None
        self._photo = None
        self._render_job = None
        self._save_job = None
        self._detail_job = None
        self._save_error_shown = False
        self._export_widgets = None

        self.tool = tk.StringVar(value="smart")
        self.show_masks = tk.BooleanVar(value=True)
        self.hover_preview = tk.BooleanVar(value=self.settings.get("hover_preview", True))
        self.detail = tk.DoubleVar(value=detail_tolerance(self.settings.get("detail", "normal")))
        self.model_var = tk.StringVar(value=model or self.settings.get("model", "auto"))
        if self.model_var.get() != "auto" and self.model_var.get() not in MODELS:
            self.model_var.set("auto")

        self.events = queue.Queue()
        self.worker = InferenceWorker(lambda *e: self.events.put(e), self.image_cache)
        self.worker.start()

        self._build_menu()
        self._build_widgets()
        self._bind_keys()
        self.pack(fill=tk.BOTH, expand=True)
        self.master.protocol("WM_DELETE_WINDOW", self.quit_app)

        self._load_model()
        self._poll_job = self.after(20, self._poll_events)
        self._refresh_classes()
        if images:
            self.after(50, lambda: self._open_paths(images))
        else:
            self._schedule_render()

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        m = tk.Menu(menu, tearoff=0)
        m.add_command(label="Open Folder...", accelerator="Ctrl+O", command=self.open_folder)
        m.add_command(label="Open Images...", command=self.open_images)
        m.add_command(label="Save", accelerator="Ctrl+S", command=self.save_project)
        m.add_separator()
        m.add_command(label="Export Dataset...", accelerator="Ctrl+E", command=self.export_dialog)
        m.add_separator()
        m.add_command(label="Quit", command=self.quit_app)
        menu.add_cascade(label="File", menu=m)

        m = tk.Menu(menu, tearoff=0)
        m.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        m.add_command(label="Redo", accelerator="Ctrl+Y", command=self.redo)
        m.add_separator()
        m.add_command(label="Delete Object", accelerator="Del", command=self.delete_object)
        m.add_command(label="Clear Image Annotations", command=self.clear_image)
        menu.add_cascade(label="Edit", menu=m)

        m = tk.Menu(menu, tearoff=0)
        for tool, label in TOOL_LABELS.items():
            m.add_radiobutton(label=label, value=tool, variable=self.tool, command=self._on_tool_change)
        menu.add_cascade(label="Tools", menu=m)

        m = tk.Menu(menu, tearoff=0)
        m.add_command(label="Fit to Window", accelerator="F", command=self.fit_view)
        m.add_checkbutton(label="Show Annotations", accelerator="H", variable=self.show_masks,
                          command=self._schedule_render)
        m.add_checkbutton(label="Smart Hover Preview", variable=self.hover_preview,
                          command=self._on_hover_preview_toggle)
        menu.add_cascade(label="View", menu=m)

        m = tk.Menu(menu, tearoff=0)
        device = detect_device(self.device_pref)
        recommended = candidate_models(device)
        rec_label = MODELS[recommended[0]].label if recommended else "none installed"
        m.add_radiobutton(label=f"Auto - best for this PC ({rec_label})", value="auto",
                          variable=self.model_var, command=self._load_model)
        m.add_separator()
        for key, spec in MODELS.items():
            installed = backend_installed(spec.backend)
            label = f"{spec.label}  ({spec.size_mb} MB) - {spec.notes}"
            if not installed:
                label = f"{spec.label}  [not installed: {install_hint(spec.backend)}]"
            m.add_radiobutton(label=label, value=key, variable=self.model_var, command=self._load_model,
                              state=tk.NORMAL if installed else tk.DISABLED)
        menu.add_cascade(label="Model", menu=m)

        m = tk.Menu(menu, tearoff=0)
        m.add_command(label="Keyboard Shortcuts", accelerator="F1", command=self.show_help)
        m.add_command(label="Hardware Info", command=self.show_hardware)
        menu.add_cascade(label="Help", menu=m)

    def _build_widgets(self):
        style = ttk.Style(self.master)
        if self.master.tk.call("tk", "windowingsystem") == "x11" and "clam" in style.theme_names():
            style.theme_use("clam")  # the default X11 theme looks dated
        style.configure("Tool.Toolbutton", padding=(8, 3))
        # make the active tool obvious
        style.map("Tool.Toolbutton", background=[("selected", "#1a73e8"), ("active", "#d2e3fc")],
                  foreground=[("selected", "white")])

        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)

        def button(text, cmd, side=tk.LEFT):
            b = ttk.Button(bar, text=text, command=cmd, takefocus=0, width=len(text) + 2)
            b.pack(side=side, padx=2)
            return b

        def separator():
            ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        button("Open", self.open_folder)
        separator()
        for tool, label in TOOL_LABELS.items():
            ttk.Radiobutton(bar, text=label, value=tool, variable=self.tool, style="Tool.Toolbutton",
                            command=self._on_tool_change, takefocus=0).pack(side=tk.LEFT, padx=1)
        separator()
        button("Accept", self.accept)
        button("Undo", self.undo)
        button("Redo", self.redo)
        separator()
        ttk.Label(bar, text="Class").pack(side=tk.LEFT)
        self.class_combo = ttk.Combobox(bar, state="readonly", width=14, takefocus=0)
        self.class_combo.pack(side=tk.LEFT, padx=2)
        self.class_combo.bind("<<ComboboxSelected>>",
                              lambda e: self.select_class(self.class_combo.current()))
        separator()
        ttk.Label(bar, text="Polygon detail:  fine").pack(side=tk.LEFT)
        # Left = more vertices; the value is the simplification tolerance
        ttk.Scale(bar, from_=0.25, to=4.0, variable=self.detail, length=90, takefocus=0,
                  command=lambda v: self._on_detail_change()).pack(side=tk.LEFT, padx=2)
        ttk.Label(bar, text="simple").pack(side=tk.LEFT)

        button("Export", self.export_dialog, side=tk.RIGHT)

        status = ttk.Frame(self)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = ttk.Label(status, text="Ready", anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.model_status = ttk.Label(status, text="", anchor=tk.E)
        self.model_status.pack(side=tk.RIGHT, padx=4)
        self.progress = ttk.Progressbar(status, length=160, mode="determinate")

        side = ttk.Frame(self, width=250)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=2)
        side.pack_propagate(False)

        box = ttk.LabelFrame(side, text="Classes (keys 1-9)")
        box.pack(fill=tk.X, pady=2)
        self.class_list = tk.Listbox(box, height=7, exportselection=False, activestyle="none", takefocus=0)
        self.class_list.pack(fill=tk.X, padx=3, pady=3)
        self.class_list.bind("<<ListboxSelect>>", self._on_class_list_select)
        row = ttk.Frame(box)
        row.pack(fill=tk.X, padx=3, pady=(0, 3))
        for text, cmd in (("Add", self.add_class), ("Rename", self.rename_class), ("Delete", self.delete_class)):
            ttk.Button(row, text=text, command=cmd, width=7, takefocus=0).pack(side=tk.LEFT, padx=1)

        box = ttk.LabelFrame(side, text="Objects on this image")
        box.pack(fill=tk.X, pady=2)
        self.obj_list = tk.Listbox(box, height=7, exportselection=False, activestyle="none", takefocus=0)
        self.obj_list.pack(fill=tk.X, padx=3, pady=3)
        self.obj_list.bind("<<ListboxSelect>>", self._on_obj_select)

        box = ttk.LabelFrame(side, text="Images")
        box.pack(fill=tk.BOTH, expand=True, pady=2)
        row = ttk.Frame(box)
        row.pack(fill=tk.X, padx=3, pady=(3, 0))
        ttk.Button(row, text="<", width=3, command=self.prev_image, takefocus=0).pack(side=tk.LEFT)
        ttk.Button(row, text=">", width=3, command=self.next_image, takefocus=0).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Next unlabeled (N)", command=self.next_unlabeled,
                   takefocus=0).pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll = ttk.Scrollbar(box, orient=tk.VERTICAL)
        self.image_list = tk.Listbox(box, exportselection=False, activestyle="none",
                                     yscrollcommand=scroll.set, takefocus=0)
        scroll.config(command=self.image_list.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_list.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.image_list.bind("<<ListboxSelect>>", self._on_image_list_select)

        self.canvas = tk.Canvas(self, bg="#202124", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        aqua = self.master.tk.call("tk", "windowingsystem") == "aqua"
        right, middle = ("2", "3") if aqua else ("3", "2")
        c = self.canvas
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", self._on_double)
        c.bind(f"<ButtonPress-{right}>", self._on_right_click)
        c.bind(f"<ButtonPress-{middle}>", self._on_pan_start)
        c.bind(f"<B{middle}-Motion>", self._on_pan_move)
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", self._on_leave)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Button-4>", self._on_wheel)
        c.bind("<Button-5>", self._on_wheel)
        c.bind("<Configure>", self._on_resize)

    def _bind_keys(self):
        r = self.master
        bindings = {
            "<Return>": self.on_enter, "<KP_Enter>": self.on_enter, "<space>": self.on_enter,
            "<Escape>": self.on_escape, "<Tab>": self.cycle_candidate,
            "<d>": self.next_image, "<Right>": self.next_image, "<Next>": self.next_image,
            "<a>": self.prev_image, "<Left>": self.prev_image, "<Prior>": self.prev_image,
            "<n>": self.next_unlabeled, "<Delete>": self.delete_object, "<BackSpace>": self.on_backspace,
            "<c>": self.recolor_object, "<f>": self.fit_view, "<h>": self.toggle_masks,
            "<Control-z>": self.undo, "<Control-y>": self.redo, "<Control-Z>": self.redo,
            "<Control-s>": self.save_project, "<Control-o>": self.open_folder,
            "<Control-e>": self.export_dialog, "<F1>": self.show_help,
        }
        for tool, key in TOOL_KEYS.items():
            bindings[f"<{key}>"] = lambda tool=tool: self.set_tool(tool)
        for seq, fn in bindings.items():
            r.bind(seq, lambda e, fn=fn: (fn(), "break")[1])
        for i in range(10):
            r.bind(f"<Key-{i}>", lambda e, i=i: self.select_class((i - 1) % 10))
        if r.tk.call("tk", "windowingsystem") == "aqua":
            for seq, fn in (("<Command-z>", self.undo), ("<Command-Z>", self.redo),
                            ("<Command-s>", self.save_project), ("<Command-o>", self.open_folder)):
                r.bind(seq, lambda e, fn=fn: (fn(), "break")[1])

    def set_status(self, text):
        self.status.config(text=text)

    # ------------------------------------------------------------ tools
    def set_tool(self, tool):
        self.tool.set(tool)
        self._on_tool_change()

    def _on_tool_change(self):
        tool = self.tool.get()
        self._clear_prompt_state()
        self.poly_pts = []
        self._edit = self._edit_preview = None
        if tool != "edit":
            self.selected_obj = None
            self.obj_list.selection_clear(0, tk.END)
        self.canvas.config(cursor=TOOL_CURSORS[tool])
        self.set_status(TOOL_HINTS[tool])
        self._schedule_render()

    def on_enter(self):
        tool = self.tool.get()
        if tool == "polygon":
            self._close_polygon()
        elif tool == "smart":
            self.accept()

    def on_escape(self):
        tool = self.tool.get()
        if tool == "polygon":
            self.poly_pts = []
        elif tool == "edit":
            self.selected_obj = None
            self.obj_list.selection_clear(0, tk.END)
        self.clear_prompt()

    def on_backspace(self):
        if not self._undo_prompt_step():
            self.delete_object()

    def _undo_prompt_step(self):
        """Remove the last smart point / box or polygon vertex. False if none."""
        if self.tool.get() == "polygon" and self.poly_pts:
            self.poly_pts.pop()
            self._draw_canvas_items()
            return True
        if self.points or self.box:
            if self.points:
                self.points.pop()
                self.labels.pop()
            else:
                self.box = None
            self._request_prediction()
            return True
        return False

    # ------------------------------------------------------------ model
    def _load_model(self):
        key = self.model_var.get()
        self.settings["model"] = key
        save_settings(self.settings)
        self.model_ready = False
        self.model_status.config(text="Loading model...")
        self.worker.load_model(key, self.device_pref, self.precision)
        if self.cur_path:
            self.worker.set_current_image(self.cur_path, self._prefetch_list())

    def _poll_events(self):
        try:
            while True:
                event, *args = self.events.get_nowait()
                getattr(self, "_ev_" + event)(*args)
        except queue.Empty:
            pass
        self._poll_job = self.after(15, self._poll_events)

    def _ev_model_status(self, fraction, message):
        self.model_status.config(text=message)
        if fraction is None:
            self.progress.pack_forget()
        else:
            if not self.progress.winfo_ismapped():
                self.progress.pack(side=tk.RIGHT, padx=4)
            self.progress["value"] = fraction * 100

    def _ev_model_ready(self, key, device, precision):
        self.progress.pack_forget()
        self.model_ready = True
        self.model_status.config(text=f"{MODELS[key].label} | {device} | {precision}")
        if self.points or self.box:
            self._request_prediction()

    def _ev_model_error(self, message):
        self.progress.pack_forget()
        self.model_status.config(text="Model not loaded")
        messagebox.showerror("Model error", message + "\n\nPick another model from the Model menu.")

    def _ev_error(self, message):
        self.set_status(f"Error: {message}")

    def _ev_encoding(self, path):
        if path == self.cur_path:
            self.set_status("Preparing image for smart polygon...")
            self.canvas.config(cursor="watch")

    def _ev_encoded(self, path, seconds):
        if path == self.cur_path:
            self.canvas.config(cursor=TOOL_CURSORS[self.tool.get()])
            if not (self.points or self.box):
                self.set_status(f"Image ready ({seconds:.2f}s). {TOOL_HINTS[self.tool.get()]}")

    def _ev_prediction(self, request_id, path, candidates, seconds):
        if request_id != self.request_id or path != self.cur_path:
            return  # stale result: the user already moved on
        self.canvas.config(cursor=TOOL_CURSORS[self.tool.get()])
        self._show_candidates(candidates, seconds)

    def _show_candidates(self, candidates, seconds=None):
        self.candidates, self.cand_idx = candidates, 0
        if candidates:
            timing = f" ({seconds * 1000:.0f} ms)" if seconds else ""
            extra = " | Tab: other masks" if len(candidates) > 1 else ""
            self.set_status(f"Mask {candidates[0]['score']:.2f}{timing} | Enter: accept as "
                            f"'{self.project.class_name(self.current_class)}' | "
                            f"Right-click: exclude | Ctrl+Z: undo point{extra}")
        else:
            self.set_status("No mask here. Try another point, or drag a box.")
        self._schedule_render()

    def _ev_hover_prediction(self, request_id, path, point, candidates, seconds):
        if request_id != self.hover_id or path != self.cur_path or not self._hover_allowed():
            return
        self.hover_point, self.hover_candidates = point, candidates
        self._schedule_render()

    def _ev_export_progress(self, done, total):
        if self._export_widgets:
            self._export_widgets["progress"]["value"] = 100 * done / max(total, 1)

    def _ev_export_done(self, counts, error, out_dir):
        dialog = self._export_widgets and self._export_widgets["dialog"]
        self._export_widgets = None
        if dialog:
            dialog.destroy()
        if error:
            messagebox.showerror("Export failed", error)
        else:
            summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
            messagebox.showinfo("Export complete", f"Dataset exported to\n{out_dir}\n\n{summary} images")

    # ---------------------------------------------------------- images
    def open_folder(self):
        folder = filedialog.askdirectory(initialdir=self.settings.get("last_dir"))
        if folder:
            self._open_paths([folder])

    def open_images(self):
        pattern = " ".join("*" + e for e in IMAGE_EXTENSIONS)
        files = filedialog.askopenfilenames(initialdir=self.settings.get("last_dir"),
                                            filetypes=[("Images", pattern), ("All files", "*.*")])
        if files:
            self._open_paths(list(files))

    def _open_paths(self, paths):
        images = []
        for p in paths:
            if os.path.isdir(p):
                images.extend(list_images(p))
            elif is_image_file(p):
                images.append(p)
        if not images:
            messagebox.showerror("No images", "No images found (supported: " + ", ".join(IMAGE_EXTENSIONS) + ")")
            return
        self._save_now()
        try:
            self.project.open_images(images)
        except Exception as e:
            messagebox.showerror("Project error", f"Could not read the saved project file:\n{e}")
            return
        self.settings["last_dir"] = os.path.dirname(self.project.path)
        save_settings(self.settings)
        self.image_cache.clear()
        self.current_class = 0
        self._refresh_classes()
        self.image_list.delete(0, tk.END)
        self.image_list.insert(tk.END, *(self._image_row(p) for p in self.project.images))
        done = sum(self.project.is_annotated(p) for p in self.project.images)
        first = next((i for i, p in enumerate(self.project.images) if not self.project.is_annotated(p)), 0)
        self.show_image(first)
        self.set_status(f"Loaded {len(self.project.images)} images ({done} already annotated). "
                        f"{TOOL_HINTS[self.tool.get()]}  F1: all shortcuts.")

    def _image_row(self, path):
        n = len(self.project.get(path))
        return f"{'✓' if n else '  '} {os.path.basename(path)}" + (f"  ({n})" if n else "")

    def _prefetch_list(self):
        imgs = self.project.images
        return imgs[self.index + 1:self.index + 1 + PREFETCH_AHEAD]

    def show_image(self, index):
        if not self.project.images:
            return
        index = max(0, min(index, len(self.project.images) - 1))
        self._clear_prompt_state()
        self.poly_pts = []
        self._edit = self._edit_preview = None
        self.index = index
        path = self.project.images[index]
        try:
            self.cur_image = Image.fromarray(self.image_cache.get(path))
        except Exception as e:
            self.cur_image, self.cur_path = None, None
            self.set_status(f"Cannot open {os.path.basename(path)}: {e}")
            self._schedule_render()
            return
        self.cur_path = path
        self._base_key = None
        self._preview = None
        w, h = self.cur_image.size
        factor = math.ceil(max(w, h) / PREVIEW_MAX_SIDE)
        if factor >= 2:
            self._preview = (1.0 / factor, self.cur_image.reduce(factor))
        self.hover = self.selected_obj = None
        self.worker.set_current_image(path, self._prefetch_list())
        self.fit_view()
        self._refresh_objects()
        self.image_list.selection_clear(0, tk.END)
        self.image_list.selection_set(index)
        self.image_list.see(index)
        self.master.title(f"Smart Polygon Annotator - {os.path.basename(path)} ({w}x{h}) "
                          f"[{index + 1}/{len(self.project.images)}]")

    def next_image(self):
        if self.project.images and self.index < len(self.project.images) - 1:
            self.show_image(self.index + 1)

    def prev_image(self):
        if self.project.images and self.index > 0:
            self.show_image(self.index - 1)

    def next_unlabeled(self):
        imgs = self.project.images
        for offset in range(1, len(imgs) + 1):
            i = (self.index + offset) % len(imgs)
            if not self.project.is_annotated(imgs[i]):
                self.show_image(i)
                return
        self.set_status("All images are annotated.")

    def _on_image_list_select(self, event):
        sel = self.image_list.curselection()
        if sel and sel[0] != self.index:
            self.show_image(sel[0])

    # ------------------------------------------------------------ view
    def _to_image(self, cx, cy):
        return self.ox + cx / self.scale, self.oy + cy / self.scale

    def _to_canvas(self, ix, iy):
        return (ix - self.ox) * self.scale, (iy - self.oy) * self.scale

    def _inside(self, ix, iy):
        w, h = self.cur_image.size
        return 0 <= ix < w and 0 <= iy < h

    def _clamp(self, ix, iy):
        w, h = self.cur_image.size
        return min(max(ix, 0.0), w - 1.0), min(max(iy, 0.0), h - 1.0)

    def fit_view(self):
        self.auto_fit = True
        if self.cur_image is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        w, h = self.cur_image.size
        self.scale = min(cw / w, ch / h) * 0.98
        self.ox = (w - cw / self.scale) / 2
        self.oy = (h - ch / self.scale) / 2
        self._schedule_render()

    def _on_resize(self, event):
        if self.auto_fit:
            self.fit_view()
        else:
            self._schedule_render()

    def _on_wheel(self, event):
        if self.cur_image is None:
            return
        up = event.num == 4 or getattr(event, "delta", 0) > 0
        factor = 1.25 if up else 1 / 1.25
        w, h = self.cur_image.size
        fit = min(self.canvas.winfo_width() / w, self.canvas.winfo_height() / h)
        new_scale = min(max(self.scale * factor, fit * 0.25), 32.0)
        ix, iy = self._to_image(event.x, event.y)
        self.scale = new_scale
        self.ox, self.oy = ix - event.x / new_scale, iy - event.y / new_scale
        self.auto_fit = False
        self._schedule_render()

    def _on_pan_start(self, event):
        self._last = (event.x, event.y)

    def _on_pan_move(self, event):
        if self._last is None or self.cur_image is None:
            return
        self.ox -= (event.x - self._last[0]) / self.scale
        self.oy -= (event.y - self._last[1]) / self.scale
        self._last = (event.x, event.y)
        self.auto_fit = False
        self._schedule_render()

    def toggle_masks(self):
        self.show_masks.set(not self.show_masks.get())
        self._schedule_render()

    # ---------------------------------------------------------- render
    def _schedule_render(self):
        if self._render_job is None:
            self._render_job = self.after_idle(self._render)

    def _render(self):
        self._render_job = None
        c = self.canvas
        c.delete("hint")
        if self.cur_image is None:
            if self._img_item is not None:
                c.itemconfig(self._img_item, state=tk.HIDDEN)
            c.create_text(c.winfo_width() // 2, c.winfo_height() // 2, fill="#9aa0a6", tags="hint",
                          font=("TkDefaultFont", 14), justify=tk.CENTER,
                          text="Open a folder of images to start  (Ctrl+O)\n\nPress F1 for keyboard shortcuts")
            c.delete("overlay")
            return
        cw, ch = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        w, h = self.cur_image.size
        s = self.scale
        # Only the visible part of the image is resampled, and zoomed-out views
        # use a downscaled copy, so 20+ megapixel photos pan and zoom smoothly.
        x0, y0 = max(0.0, self.ox), max(0.0, self.oy)
        x1, y1 = min(float(w), self.ox + cw / s), min(float(h), self.oy + ch / s)
        if x1 <= x0 or y1 <= y0:
            if self._img_item is not None:
                c.itemconfig(self._img_item, state=tk.HIDDEN)
            self._draw_canvas_items()
            return
        dw, dh = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
        key = (self.cur_path, x0, y0, x1, y1, dw, dh)
        if key != self._base_key:
            src, f = self.cur_image, 1.0
            if self._preview and s <= self._preview[0]:
                f, src = self._preview
            resample = Image.NEAREST if s >= 4 else Image.BILINEAR
            self._base = src.resize((dw, dh), resample, box=(x0 * f, y0 * f, x1 * f, y1 * f),
                                    reducing_gap=2.0 if s / f < 0.5 else None).convert("RGBA")
            self._base_key = key
        frame = self._base
        overlay = self._draw_overlay(dw, dh, x0, y0, s)
        if overlay is not None:
            frame = Image.alpha_composite(frame, overlay)
        self._photo = ImageTk.PhotoImage(frame)
        px, py = (x0 - self.ox) * s, (y0 - self.oy) * s
        if self._img_item is None:
            self._img_item = c.create_image(px, py, anchor=tk.NW, image=self._photo)
        else:
            c.coords(self._img_item, px, py)
            c.itemconfig(self._img_item, image=self._photo, state=tk.NORMAL)
        self._draw_canvas_items()

    def _draw_overlay(self, dw, dh, x0, y0, s):
        anns = list(self.project.get(self.cur_path)) if self.show_masks.get() else []
        if self._edit_preview is not None and self._edit_preview[0] < len(anns):
            anns[self._edit_preview[0]] = self._edit_preview[1]
        cand = self.candidates[self.cand_idx] if self.candidates else None
        hover = self.hover_candidates[0] if self.hover_candidates and self._hover_allowed() else None
        if not anns and not cand and not hover:
            return None
        overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        offset = np.array([x0, y0], dtype=np.float32)

        def flat(poly):
            return ((np.asarray(poly, dtype=np.float32) - offset) * s).ravel().tolist()

        for i, ann in enumerate(anns):
            rgb = _hex_rgb(self.project.class_color(ann["class_id"]))
            focus = i in (self.hover, self.selected_obj)
            for poly in ann["polygons"]:
                if len(poly) >= 3:
                    draw.polygon(flat(poly), fill=rgb + (130 if focus else 60,), outline=rgb + (255,),
                                 width=3 if focus else 2)
        rgb = _hex_rgb(self.project.class_color(self.current_class))
        if cand:
            for poly in cand["polygons"]:
                draw.polygon(flat(poly), fill=rgb + (120,), outline=(255, 255, 255, 255), width=2)
        elif hover:
            for poly in hover["polygons"]:
                draw.polygon(flat(poly), fill=rgb + (70,), outline=(255, 255, 255, 200), width=1)
        return overlay

    def _draw_canvas_items(self):
        """Prompts, polygon-in-progress and edit handles are cheap canvas items
        drawn over the rendered image."""
        c = self.canvas
        c.delete("overlay")
        if self.cur_image is None:
            return
        if self.box:
            ax, ay = self._to_canvas(self.box[0], self.box[1])
            bx, by = self._to_canvas(self.box[2], self.box[3])
            c.create_rectangle(ax, ay, bx, by, outline="#00E5FF", width=2, dash=(6, 4), tags="overlay")
        for (x, y), label in zip(self.points, self.labels):
            cx, cy = self._to_canvas(x, y)
            c.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#00C853" if label else "#FF1744",
                          outline="white", width=2, tags="overlay")
            c.create_text(cx, cy - 1, text="+" if label else "−", fill="white",
                          font=("TkDefaultFont", 9, "bold"), tags="overlay")
        if self.poly_pts:
            color = self.project.class_color(self.current_class)
            pts = [self._to_canvas(x, y) for x, y in self.poly_pts]
            if self._cursor:
                pts_line = pts + [self._cursor]
            else:
                pts_line = pts
            if len(pts_line) >= 2:
                c.create_line(*[v for p in pts_line for v in p], fill=color, width=2, tags="overlay")
            for i, (cx, cy) in enumerate(pts):
                r = 6 if i == 0 else 4
                c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="white" if i == 0 else color,
                              outline=color, width=2, tags="overlay")
        ann = self._selected_annotation()
        if ann is not None and self.tool.get() == "edit":
            color = self.project.class_color(ann["class_id"])
            for poly in ann["polygons"]:
                for x, y in poly:
                    cx, cy = self._to_canvas(x, y)
                    c.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline=color,
                                       width=2, tags="overlay")

    # ---------------------------------------------------------- mouse
    def _on_press(self, event):
        self.canvas.focus_set()
        if self.cur_image is None:
            return
        self._press = (event.x, event.y)
        self._last = (event.x, event.y)
        if event.state & 0x0004:
            self._drag_mode = "pan"
        elif self.tool.get() == "edit":
            self._edit_press(event)
        else:
            self._drag_mode = "click"

    def _on_drag(self, event):
        if self._press is None:
            return
        mode = self._drag_mode
        if mode == "pan":
            self._on_pan_move(event)
            return
        if mode in ("vertex", "move"):
            self._edit_drag(event)
            return
        px, py = self._press
        far = abs(event.x - px) + abs(event.y - py) > DRAG_THRESHOLD
        if mode == "click" and far and self.tool.get() in ("smart", "box"):
            self._drag_mode = mode = "box"
        if mode == "box":
            self.canvas.delete("rubber")
            color = "#00E5FF" if self.tool.get() == "smart" else self.project.class_color(self.current_class)
            self.canvas.create_rectangle(px, py, event.x, event.y, outline=color, width=2,
                                         dash=(4, 3) if self.tool.get() == "smart" else None, tags="rubber")

    def _on_release(self, event):
        mode, press = self._drag_mode, self._press
        self._press = self._drag_mode = None
        self.canvas.delete("rubber")
        if press is None or self.cur_image is None:
            return
        tool = self.tool.get()
        shift = bool(event.state & 0x0001)
        if mode == "box":
            x0, y0 = self._clamp(*self._to_image(*press))
            x1, y1 = self._clamp(*self._to_image(event.x, event.y))
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 - x0 < 2 or y1 - y0 < 2:
                return
            if tool == "box":
                self._add_annotation([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
            else:
                self.box = [x0, y0, x1, y1]
                self._clear_hover()
                self._request_prediction()
        elif mode == "click":
            if tool == "smart":
                self._add_point(event.x, event.y, 0 if shift else 1)
            elif tool == "polygon":
                self._add_vertex(event.x, event.y)
        elif mode in ("vertex", "move"):
            self._edit_commit()

    def _on_double(self, event):
        self._press = self._drag_mode = None
        if self.cur_image is None:
            return
        tool = self.tool.get()
        if tool == "polygon":
            self._close_polygon()
        elif tool == "edit":
            self._insert_vertex(event.x, event.y)

    def _on_right_click(self, event):
        if self.cur_image is None:
            return
        tool = self.tool.get()
        if tool == "polygon":
            self._undo_prompt_step()
        elif tool == "edit":
            self._delete_vertex_at(event.x, event.y)
        elif tool == "smart":
            if not any(self.labels) and self.box is None:
                self.set_status("Right-click excludes a region from the current mask. Left-click an object first.")
                return
            self._add_point(event.x, event.y, 0)

    def _on_motion(self, event):
        if self.cur_image is None or self._press is not None:
            return
        ix, iy = self._to_image(event.x, event.y)
        tool = self.tool.get()
        if tool == "polygon" and self.poly_pts:
            self._cursor = (event.x, event.y)
            self._draw_canvas_items()
        hit = None
        if self.show_masks.get():
            anns = self.project.get(self.cur_path)
            for i in range(len(anns) - 1, -1, -1):
                if annotation_contains(anns[i], ix, iy):
                    hit = i
                    break
        self._set_hover(hit)
        if tool == "edit":
            on_vertex = self._vertex_at(event.x, event.y) is not None
            self.canvas.config(cursor="fleur" if on_vertex or (hit is not None and hit == self.selected_obj)
                               else "arrow")
        elif tool == "smart" and self._hover_allowed() and self._inside(ix, iy):
            self._queue_hover(ix, iy)

    def _on_leave(self, event):
        self._cursor = None
        self._set_hover(None)
        if self.hover_candidates:
            self._clear_hover()
            self._schedule_render()
        if self.poly_pts:
            self._draw_canvas_items()

    def _set_hover(self, index):
        if index != self.hover:
            self.hover = index
            self._schedule_render()

    # ------------------------------------------------- smart polygon
    def _hover_allowed(self):
        return (self.hover_preview.get() and self.tool.get() == "smart" and self.model_ready
                and not self.points and self.box is None)

    def _queue_hover(self, ix, iy):
        # Sent at most every 25 ms; the worker keeps only the newest anyway
        self._pending_hover = (ix, iy)
        if self._hover_job is None:
            self._hover_job = self.after(25, self._send_hover)

    def _send_hover(self):
        self._hover_job = None
        if self._hover_allowed() and self.cur_path:
            self.hover_id += 1
            self.worker.hover(self.hover_id, self.cur_path, self._pending_hover, self.detail.get())

    def _clear_hover(self):
        self.hover_id += 1
        self.hover_point, self.hover_candidates = None, []

    def _on_hover_preview_toggle(self):
        self.settings["hover_preview"] = self.hover_preview.get()
        save_settings(self.settings)
        self._clear_hover()
        self._schedule_render()

    def _add_point(self, cx, cy, label):
        ix, iy = self._to_image(cx, cy)
        if not self._inside(ix, iy):
            return
        # A click where the hover preview was computed is answered instantly
        hp = self.hover_point
        reuse = (label == 1 and not self.points and self.box is None and self.hover_candidates and hp
                 and math.hypot((hp[0] - ix) * self.scale, (hp[1] - iy) * self.scale) <= 4)
        candidates = self.hover_candidates
        self._clear_hover()
        self.points.append((ix, iy))
        self.labels.append(label)
        if reuse:
            self.points[-1] = tuple(hp)
            self.request_id += 1
            self._draw_canvas_items()
            self._show_candidates(candidates)
        else:
            self._request_prediction()

    def _request_prediction(self):
        self._draw_canvas_items()
        if not any(self.labels) and self.box is None:
            self.candidates = []
            self.request_id += 1
            self._schedule_render()
            return
        self.request_id += 1
        self.worker.predict(self.request_id, self.cur_path, list(self.points), list(self.labels),
                            self.box, self.detail.get())
        if not self.model_ready:
            self.set_status("Model is still loading; your prompt will run as soon as it is ready...")

    def _on_detail_change(self):
        if self._detail_job is not None:
            self.after_cancel(self._detail_job)
        self._detail_job = self.after(150, self._apply_detail)

    def _apply_detail(self):
        self._detail_job = None
        self.settings["detail"] = round(self.detail.get(), 2)
        save_settings(self.settings)
        self._clear_hover()
        if self.points or self.box:
            self._request_prediction()

    def _clear_prompt_state(self):
        self.points, self.labels, self.box = [], [], None
        self.candidates, self.cand_idx = [], 0
        self.request_id += 1  # drop in-flight predictions
        self._clear_hover()

    def clear_prompt(self):
        self._clear_prompt_state()
        self._schedule_render()

    def cycle_candidate(self):
        if len(self.candidates) > 1:
            self.cand_idx = (self.cand_idx + 1) % len(self.candidates)
            self.set_status(f"Mask {self.cand_idx + 1}/{len(self.candidates)} "
                            f"(score {self.candidates[self.cand_idx]['score']:.2f})")
            self._schedule_render()

    def accept(self):
        if self.tool.get() == "polygon":
            self._close_polygon()
            return
        if not self.candidates or self.cur_path is None:
            return
        cand = self.candidates[self.cand_idx]
        self._clear_prompt_state()
        self._add_annotation(cand["polygons"], cand["score"])

    def _add_annotation(self, polygons, score=None):
        ann = make_annotation(polygons, self.current_class, score)
        self.project.add_annotation(self.cur_path, ann)
        self._annotations_changed()
        self.set_status(f"Added '{self.project.class_name(self.current_class)}' "
                        f"({len(self.project.get(self.cur_path))} on this image).")

    # --------------------------------------------------- polygon tool
    def _add_vertex(self, cx, cy):
        ix, iy = self._to_image(cx, cy)
        if not self._inside(ix, iy):
            return
        if len(self.poly_pts) >= 3:
            fx, fy = self._to_canvas(*self.poly_pts[0])
            if math.hypot(fx - cx, fy - cy) <= HANDLE_RADIUS + 2:
                self._close_polygon()
                return
        self.poly_pts.append((ix, iy))
        self._cursor = (cx, cy)
        self._draw_canvas_items()

    def _close_polygon(self):
        pts = self.poly_pts
        if len(pts) < 3:
            if pts:
                self.set_status("A polygon needs at least 3 vertices.")
            return
        self.poly_pts = []
        self._add_annotation([[list(p) for p in pts]])

    # ------------------------------------------------------ edit tool
    def _selected_annotation(self):
        if self._edit_preview is not None:
            return self._edit_preview[1]
        anns = self.project.get(self.cur_path)
        if self.selected_obj is not None and self.selected_obj < len(anns):
            return anns[self.selected_obj]
        return None

    def _vertex_at(self, cx, cy):
        ann = self._selected_annotation()
        if ann is None:
            return None
        best = None
        for pi, poly in enumerate(ann["polygons"]):
            for vi, (x, y) in enumerate(poly):
                vx, vy = self._to_canvas(x, y)
                d = math.hypot(vx - cx, vy - cy)
                if d <= HANDLE_RADIUS and (best is None or d < best[0]):
                    best = (d, pi, vi)
        return best and best[1:]

    def _edit_press(self, event):
        vertex = self._vertex_at(event.x, event.y)
        if vertex is not None:
            if event.state & 0x0001:
                self._press = None
                self._delete_vertex_at(event.x, event.y)
                return
            self._drag_mode = "vertex"
            self._edit = {"vertex": vertex, "start": self._to_image(event.x, event.y),
                          "orig": self._selected_annotation()}
            return
        hit = self._hit_object(event.x, event.y)
        if hit is None and self._edge_at(event.x, event.y) is not None:
            hit = self.selected_obj  # on the outline of the selection: keep it (e.g. double-click to add a vertex)
        self.selected_obj = hit
        self.obj_list.selection_clear(0, tk.END)
        if hit is None:
            self._drag_mode = "pan"  # dragging empty space pans in edit mode
        else:
            self.obj_list.selection_set(hit)
            self._drag_mode = "move"
            self._edit = {"start": self._to_image(event.x, event.y), "orig": self._selected_annotation()}
        self._schedule_render()

    def _hit_object(self, cx, cy):
        ix, iy = self._to_image(cx, cy)
        anns = self.project.get(self.cur_path)
        for i in range(len(anns) - 1, -1, -1):
            if annotation_contains(anns[i], ix, iy):
                return i
        return None

    def _edit_drag(self, event):
        if not self._edit:
            return
        orig = self._edit["orig"]
        ix, iy = self._to_image(event.x, event.y)
        if self._drag_mode == "vertex":
            pi, vi = self._edit["vertex"]
            polys = [list(p) for p in orig["polygons"]]
            polys[pi] = list(polys[pi])
            polys[pi][vi] = list(self._clamp(ix, iy))
        else:
            w, h = self.cur_image.size
            x0, y0, x1, y1 = orig["bbox"]
            # keep the whole object inside the image
            dx = min(max(ix - self._edit["start"][0], -x0), w - 1 - x1)
            dy = min(max(iy - self._edit["start"][1], -y0), h - 1 - y1)
            polys = [[[x + dx, y + dy] for x, y in p] for p in orig["polygons"]]
        self._edit_preview = (self.selected_obj, dict(orig, polygons=polys, bbox=polygons_bbox(polys)))
        self._schedule_render()

    def _edit_commit(self):
        preview, self._edit_preview, self._edit = self._edit_preview, None, None
        if preview is None:
            self._schedule_render()
            return
        idx, ann = preview
        ann["polygons"] = [[[round(x, 1), round(y, 1)] for x, y in p] for p in ann["polygons"]]
        anns = list(self.project.get(self.cur_path))
        anns[idx] = ann
        self.project.set_annotations(self.cur_path, anns)
        self._annotations_changed(keep_selection=True)

    def _replace_selected_polygons(self, polys):
        anns = list(self.project.get(self.cur_path))
        anns[self.selected_obj] = dict(anns[self.selected_obj], polygons=polys, bbox=polygons_bbox(polys))
        self.project.set_annotations(self.cur_path, anns)
        self._annotations_changed(keep_selection=True)

    def _delete_vertex_at(self, cx, cy):
        vertex = self._vertex_at(cx, cy)
        if vertex is None:
            return
        pi, vi = vertex
        polys = [list(p) for p in self._selected_annotation()["polygons"]]
        if len(polys[pi]) <= 3:
            self.set_status("A polygon needs at least 3 vertices (Del removes the whole object).")
            return
        polys[pi] = polys[pi][:vi] + polys[pi][vi + 1:]
        self._replace_selected_polygons(polys)

    def _edge_at(self, cx, cy):
        """(polygon index, index of the edge's first vertex) of the selected
        object's outline under the cursor, or None."""
        ann = self._selected_annotation()
        if ann is None:
            return None
        best = None
        for pi, poly in enumerate(ann["polygons"]):
            pts = [self._to_canvas(x, y) for x, y in poly]
            for vi in range(len(pts)):
                d = _segment_distance((cx, cy), pts[vi], pts[(vi + 1) % len(pts)])
                if d <= HANDLE_RADIUS and (best is None or d < best[0]):
                    best = (d, pi, vi)
        return best and best[1:]

    def _insert_vertex(self, cx, cy):
        edge = self._edge_at(cx, cy)
        if edge is None:
            return
        pi, vi = edge
        polys = [list(p) for p in self._selected_annotation()["polygons"]]
        polys[pi] = polys[pi][:vi + 1] + [list(self._clamp(*self._to_image(cx, cy)))] + polys[pi][vi + 1:]
        self._replace_selected_polygons(polys)

    # ------------------------------------------------------- objects
    def _on_obj_select(self, event):
        sel = self.obj_list.curselection()
        self.selected_obj = sel[0] if sel else None
        self._schedule_render()

    def _target_object(self):
        if self.tool.get() == "edit" and self.selected_obj is not None:
            return self.selected_obj
        return self.hover if self.hover is not None else self.selected_obj

    def delete_object(self):
        idx = self._target_object()
        if self.cur_path is None or idx is None or idx >= len(self.project.get(self.cur_path)):
            return
        self.project.remove_annotation(self.cur_path, idx)
        self.hover = self.selected_obj = None
        self._annotations_changed()
        self.set_status("Object deleted (Ctrl+Z to undo).")

    def recolor_object(self):
        idx = self._target_object()
        if self.cur_path is None or idx is None or idx >= len(self.project.get(self.cur_path)):
            return
        self.project.change_class(self.cur_path, idx, self.current_class)
        self._annotations_changed(keep_selection=True)

    def clear_image(self):
        if self.cur_path and self.project.get(self.cur_path):
            self.project.set_annotations(self.cur_path, [])
            self.hover = self.selected_obj = None
            self._annotations_changed()

    def _annotations_changed(self, keep_selection=False):
        if not keep_selection:
            self.selected_obj = None
        self._refresh_objects()
        if self.cur_path:
            self.image_list.delete(self.index)
            self.image_list.insert(self.index, self._image_row(self.cur_path))
            self.image_list.selection_set(self.index)
        self._refresh_classes()
        self._schedule_render()
        self._schedule_save()

    def _refresh_objects(self):
        self.obj_list.delete(0, tk.END)
        for i, ann in enumerate(self.project.get(self.cur_path)):
            score = f"  ({ann['score']:.2f})" if "score" in ann else ""
            self.obj_list.insert(tk.END, f"#{i + 1}  {self.project.class_name(ann['class_id'])}{score}")
            color = self.project.class_color(ann["class_id"])
            self.obj_list.itemconfig(i, background=color, foreground=_text_color(color),
                                     selectbackground="#202124", selectforeground=color)
        if self.selected_obj is not None and self.selected_obj < self.obj_list.size():
            self.obj_list.selection_set(self.selected_obj)

    # ------------------------------------------------------- classes
    def _refresh_classes(self):
        counts = self.project.class_counts()
        self.class_list.delete(0, tk.END)
        for i, cls in enumerate(self.project.classes):
            key = f"[{(i + 1) % 10}] " if i < 10 else "    "
            self.class_list.insert(tk.END, f"{key}{cls['name']}  ({counts[i]})")
            # Selected row is drawn inverted so the active class stands out
            self.class_list.itemconfig(i, background=cls["color"], foreground=_text_color(cls["color"]),
                                       selectbackground="#202124", selectforeground=cls["color"])
        self.class_combo["values"] = [c["name"] for c in self.project.classes]
        self.current_class = min(self.current_class, len(self.project.classes) - 1)
        self.select_class(self.current_class)

    def select_class(self, class_id):
        if not 0 <= class_id < len(self.project.classes):
            return
        self.current_class = class_id
        self.class_list.selection_clear(0, tk.END)
        self.class_list.selection_set(class_id)
        self.class_combo.current(class_id)
        if self.candidates or self.poly_pts or self.hover_candidates:
            self._schedule_render()
            self._draw_canvas_items()

    def _on_class_list_select(self, event):
        sel = self.class_list.curselection()
        if sel:
            self.select_class(sel[0])

    def add_class(self):
        name = simpledialog.askstring("Add class", "Class name:", parent=self.master)
        if name and name.strip():
            self.select_class(self.project.add_class(name.strip()))
            self._refresh_classes()
            self._schedule_save()

    def rename_class(self):
        cid = self.current_class
        name = simpledialog.askstring("Rename class", "New name:", parent=self.master,
                                      initialvalue=self.project.class_name(cid))
        if name and name.strip():
            self.project.rename_class(cid, name.strip())
            self._refresh_classes()
            self._refresh_objects()
            self._schedule_save()

    def delete_class(self):
        if len(self.project.classes) <= 1:
            messagebox.showinfo("Delete class", "A project needs at least one class. Rename it instead.")
            return
        cid = self.current_class
        n = self.project.class_counts()[cid]
        if not messagebox.askyesno("Delete class", f"Delete class '{self.project.class_name(cid)}'"
                                   f" and its {n} annotation(s)?\nHigher class ids shift down by one."):
            return
        self.project.delete_class(cid)
        self.current_class = max(0, cid - 1)
        self._after_bulk_change()

    def _after_bulk_change(self):
        self._refresh_classes()
        self.image_list.delete(0, tk.END)
        self.image_list.insert(tk.END, *(self._image_row(p) for p in self.project.images))
        if self.project.images:
            self.image_list.selection_set(self.index)
        self.hover = self.selected_obj = None
        self._refresh_objects()
        self._schedule_render()
        self._schedule_save()

    # ------------------------------------------------------- history
    def undo(self):
        # While building a prompt or polygon, Ctrl+Z steps back through it first
        if not self._undo_prompt_step():
            self._history(self.project.undo)

    def redo(self):
        self._history(self.project.redo)

    def _history(self, action):
        changed, path = action()
        if not changed:
            return
        if path is None:
            self._after_bulk_change()
            return
        if path != self.cur_path and path in self.project.images:
            self.show_image(self.project.images.index(path))
        self.hover = None
        if self.selected_obj is not None and self.selected_obj >= len(self.project.get(self.cur_path)):
            self.selected_obj = None  # keep the selection while that object still exists
        if path in self.project.images:
            i = self.project.images.index(path)
            self.image_list.delete(i)
            self.image_list.insert(i, self._image_row(path))
            self.image_list.selection_set(self.index)
        self._refresh_objects()
        self._refresh_classes()
        self._schedule_render()
        self._schedule_save()

    # ---------------------------------------------------- persistence
    def _schedule_save(self):
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(1000, self._save_now)

    def _save_now(self):
        self._save_job = None
        if not self.project.dirty or not self.project.path:
            return
        try:
            self.project.save()
        except OSError as e:
            if not self._save_error_shown:
                self._save_error_shown = True
                messagebox.showwarning("Autosave failed", f"Could not save {self.project.path}:\n{e}\n"
                                       "Use File > Save to choose another location (Ctrl+S).")

    def save_project(self):
        if not self.project.images:
            return
        try:
            self.project.save()
            self.set_status(f"Saved {self.project.path}")
        except OSError:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Project", "*.json")])
            if path:
                self.project.save(path)
                self.set_status(f"Saved {path}")

    def quit_app(self):
        self._save_now()
        self.after_cancel(self._poll_job)
        self.worker.stop()
        self.master.destroy()
        # Let an in-flight encode finish; killing torch mid-op aborts the interpreter
        self.worker.join(timeout=15)

    # --------------------------------------------------------- export
    def export_dialog(self):
        if not any(self.project.get(p) for p in self.project.images):
            messagebox.showerror("Export", "No annotations to export yet.")
            return
        if self._export_widgets:
            return
        self._save_now()
        dlg = tk.Toplevel(self.master)
        dlg.title("Export Dataset")
        dlg.transient(self.master)
        dlg.resizable(False, False)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        fmt = tk.StringVar(value=self.settings.get("export_format", "yolo-seg"))
        ttk.Label(frm, text="Format").grid(row=0, column=0, sticky=tk.NW, pady=2)
        for i, (key, label) in enumerate(FORMATS.items()):
            ttk.Radiobutton(frm, text=label, value=key, variable=fmt).grid(row=i, column=1, columnspan=3,
                                                                            sticky=tk.W)
        ttk.Label(frm, text="Split %").grid(row=3, column=0, sticky=tk.W, pady=(8, 2))
        ratios = []
        for i, (name, default) in enumerate((("train", 80), ("val", 15), ("test", 5))):
            cell = ttk.Frame(frm)
            cell.grid(row=3, column=1 + i, sticky=tk.W, pady=(8, 2))
            ttk.Label(cell, text=name).pack(side=tk.LEFT)
            var = tk.IntVar(value=default)
            ttk.Spinbox(cell, from_=0, to=100, width=4, textvariable=var).pack(side=tk.LEFT, padx=2)
            ratios.append(var)
        unlabeled = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Include images without annotations (as background)",
                        variable=unlabeled).grid(row=4, column=1, columnspan=3, sticky=tk.W, pady=4)
        default_out = self.settings.get("export_dir") or os.path.join(os.path.dirname(self.project.path), "dataset")
        out = tk.StringVar(value=default_out)
        ttk.Label(frm, text="Output").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(frm, textvariable=out, width=46).grid(row=5, column=1, columnspan=2, sticky=tk.EW)
        ttk.Button(frm, text="Browse...", command=lambda: out.set(
            filedialog.askdirectory(parent=dlg, initialdir=out.get()) or out.get())).grid(row=5, column=3)
        progress = ttk.Progressbar(frm, length=300, mode="determinate")
        progress.grid(row=6, column=0, columnspan=4, sticky=tk.EW, pady=8)

        def run():
            try:
                split = tuple(max(0, v.get()) for v in ratios)
            except tk.TclError:
                messagebox.showerror("Export", "Split percentages must be numbers", parent=dlg)
                return
            if split[0] <= 0:
                messagebox.showerror("Export", "Train split must be above 0%", parent=dlg)
                return
            out_dir = out.get().strip()
            if not out_dir:
                return
            self.settings.update(export_dir=out_dir, export_format=fmt.get())
            save_settings(self.settings)
            go.config(state=tk.DISABLED)
            names = [c["name"] for c in self.project.classes]
            images, anns = list(self.project.images), dict(self.project.annotations)
            kind = fmt.get()

            def work():
                try:
                    counts = export_dataset(images, anns, names, out_dir, kind, split, unlabeled.get(),
                                            progress=lambda d, t: self.events.put(("export_progress", d, t)))
                    self.events.put(("export_done", counts, None, out_dir))
                except Exception as e:
                    self.events.put(("export_done", None, f"{type(e).__name__}: {e}", out_dir))

            threading.Thread(target=work, daemon=True).start()

        go = ttk.Button(frm, text="Export", command=run)
        go.grid(row=7, column=3, sticky=tk.E)
        ttk.Button(frm, text="Cancel", command=lambda: self._close_export(dlg)).grid(row=7, column=2, sticky=tk.E)
        dlg.protocol("WM_DELETE_WINDOW", lambda: self._close_export(dlg))
        self._export_widgets = {"dialog": dlg, "progress": progress}

    def _close_export(self, dlg):
        if self._export_widgets and str(self._export_widgets["dialog"]) == str(dlg):
            self._export_widgets = None
        dlg.destroy()

    # ----------------------------------------------------------- help
    def show_help(self):
        messagebox.showinfo("Keyboard Shortcuts", SHORTCUTS)

    def show_hardware(self):
        info = detect_device(self.device_pref)
        ranked = candidate_models(info)
        lines = [f"Device: {info.describe()}", "",
                 "Installed models, best for this PC first:"]
        lines += [f"  {i + 1}. {MODELS[k].label} - {MODELS[k].notes}" for i, k in enumerate(ranked)]
        missing = sorted({install_hint(m.backend) for m in MODELS.values() if not backend_installed(m.backend)})
        if missing:
            lines += ["", "More models available after:"] + [f"  {cmd}" for cmd in missing]
        messagebox.showinfo("Hardware Info", "\n".join(lines))
