"""Main window: layout, navigation, classes, history and model status.

Per-tool interaction lives in tools.py; this module routes input to the active
tool, renders the image with its annotations, and keeps the side panels in sync
with the project.
"""
import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .. import models
from ..geometry import DEFAULT_TOLERANCE, annotation_contains, make_annotation
from ..image_io import IMAGE_EXTENSIONS, ImageCache, is_image_file, list_images
from ..project import Project
from ..worker import InferenceWorker
from .export_dialog import ExportDialog
from .help import SHORTCUTS
from .style import CANVAS_BG, apply_theme, hex_to_rgb, readable_text_color
from .tools import TOOL_CLASSES
from .viewport import FrameRenderer, Viewport

PREFETCH_AHEAD = 1
AUTOSAVE_DELAY_MS = 1000
POLL_INTERVAL_MS = 15
CONTROL = 0x0004


class MainWindow(ttk.Frame):
    def __init__(self, root, settings, model=None, device="auto", precision="auto", images=None):
        super().__init__(root)
        self.root = root
        self.settings = settings
        self.device = device
        self.precision = precision
        self.project = Project()
        self.image_cache = ImageCache(max_items=4)
        self.view = Viewport()

        self.index = 0
        self.cur_path = None
        self.renderer = None
        self.current_class = 0
        self.hover = None          # index of the annotation under the cursor
        self.selected_obj = None   # index selected in the list or with the Edit tool
        self.model_ready = False

        self._panning = None
        self._photo = None
        self._image_item = None
        self._render_job = None
        self._save_job = None
        self._tolerance_job = None
        self._save_error_shown = False

        self.tool_var = tk.StringVar(value="smart")
        self.show_annotations = tk.BooleanVar(value=True)
        self.hover_preview = tk.BooleanVar(value=settings.get("hover_preview", True))
        self.tolerance_var = tk.DoubleVar(value=float(settings.get("tolerance", DEFAULT_TOLERANCE)))
        model = model or settings.get("model", "auto")
        self.model_var = tk.StringVar(value=model if model in models.MODELS else "auto")

        self.tools = {cls.name: cls(self) for cls in TOOL_CLASSES}
        self.events = queue.Queue()
        self.worker = InferenceWorker(lambda *event: self.events.put(event), self.image_cache)
        self.worker.start()
        self._event_handlers = {
            "model_status": self._on_model_status, "model_ready": self._on_model_ready,
            "model_error": self._on_model_error, "error": self._on_worker_error,
            "encoding": self._on_encoding, "encoded": self._on_encoded,
            "prediction": self._on_prediction, "hover_prediction": self._on_hover_prediction,
        }

        apply_theme(root)
        self._build_menu()
        self._build_toolbar()
        self._build_status_bar()
        self._build_side_panel()
        self._build_canvas()
        self._bind_keys()
        self.pack(fill=tk.BOTH, expand=True)
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.load_model()
        self._refresh_classes()
        self._poll_job = self.after(POLL_INTERVAL_MS, self._poll_events)
        if images:
            self.after(50, lambda: self.open_paths(images))
        else:
            self.request_render()

    # ------------------------------------------------------------------ layout
    def _build_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        def submenu(label, items):
            m = tk.Menu(menu, tearoff=0)
            for item in items:
                if item is None:
                    m.add_separator()
                else:
                    text, accel, command = item
                    m.add_command(label=text, accelerator=accel, command=command)
            menu.add_cascade(label=label, menu=m)
            return m

        submenu("File", [("Open Folder...", "Ctrl+O", self.open_folder), ("Open Images...", "", self.open_images),
                         ("Save", "Ctrl+S", self.save), None,
                         ("Export Dataset...", "Ctrl+E", self.export), None, ("Quit", "", self.quit)])
        submenu("Edit", [("Undo", "Ctrl+Z", self.undo), ("Redo", "Ctrl+Y", self.redo), None,
                         ("Delete Object", "Del", self.delete_object),
                         ("Clear Image Annotations", "", self.clear_image)])

        tools_menu = submenu("Tools", [])
        for tool in self.tools.values():
            tools_menu.add_radiobutton(label=tool.label, value=tool.name, variable=self.tool_var,
                                       command=self._on_tool_selected)

        view_menu = submenu("View", [("Fit to Window", "F", self.fit_view)])
        view_menu.add_checkbutton(label="Show Annotations", accelerator="H", variable=self.show_annotations,
                                  command=self.request_render)
        view_menu.add_checkbutton(label="Smart Hover Preview", variable=self.hover_preview,
                                  command=self._on_hover_preview_toggled)

        model_menu = submenu("Model", [])
        ranked = models.candidate_models(models.detect_device(self.device))
        auto_label = models.MODELS[ranked[0]].label if ranked else "nothing installed"
        model_menu.add_radiobutton(label=f"Auto - best for this PC ({auto_label})", value="auto",
                                   variable=self.model_var, command=self.load_model)
        for experimental in (False, True):
            model_menu.add_separator()
            if experimental:
                model_menu.add_command(label="Experimental (distilled):", state=tk.DISABLED)
            for key, spec in models.MODELS.items():
                if spec.experimental != experimental:
                    continue
                installed = models.backend_installed(spec.backend)
                label = (f"{spec.label}  ({spec.size_mb} MB) - {spec.notes}" if installed
                         else f"{spec.label}  [not installed: {models.INSTALL_HINTS[spec.backend]}]")
                model_menu.add_radiobutton(label=label, value=key, variable=self.model_var,
                                           command=self.load_model, state=tk.NORMAL if installed else tk.DISABLED)

        submenu("Help", [("Keyboard Shortcuts", "F1", self.show_help), ("Hardware Info", "", self.show_hardware)])

    def _build_toolbar(self):
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)

        def button(text, command, side=tk.LEFT):
            ttk.Button(bar, text=text, command=command, takefocus=0, width=len(text) + 2).pack(side=side, padx=2)

        def separator():
            ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        button("Open", self.open_folder)
        separator()
        for tool in self.tools.values():
            ttk.Radiobutton(bar, text=tool.label, value=tool.name, variable=self.tool_var, style="Tool.Toolbutton",
                            command=self._on_tool_selected, takefocus=0).pack(side=tk.LEFT, padx=1)
        separator()
        button("Accept", self.on_enter)
        button("Undo", self.undo)
        button("Redo", self.redo)
        separator()
        ttk.Label(bar, text="Class").pack(side=tk.LEFT)
        self.class_combo = ttk.Combobox(bar, state="readonly", width=14, takefocus=0)
        self.class_combo.pack(side=tk.LEFT, padx=2)
        self.class_combo.bind("<<ComboboxSelected>>", lambda e: self.select_class(self.class_combo.current()))
        separator()
        # The slider value is the simplification tolerance: left keeps more vertices
        ttk.Label(bar, text="Polygon detail:  fine").pack(side=tk.LEFT)
        ttk.Scale(bar, from_=0.25, to=4.0, variable=self.tolerance_var, length=90, takefocus=0,
                  command=lambda _: self._on_tolerance_changed()).pack(side=tk.LEFT, padx=2)
        ttk.Label(bar, text="simple").pack(side=tk.LEFT)
        button("Export", self.export, side=tk.RIGHT)

    def _build_status_bar(self):
        bar = ttk.Frame(self)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = ttk.Label(bar, text="Ready", anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.model_status = ttk.Label(bar, anchor=tk.E)
        self.model_status.pack(side=tk.RIGHT, padx=4)
        self.download_progress = ttk.Progressbar(bar, length=160, mode="determinate")

    def _build_side_panel(self):
        panel = ttk.Frame(self, width=250)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=2)
        panel.pack_propagate(False)

        def listbox(parent, **options):
            return tk.Listbox(parent, exportselection=False, activestyle="none", takefocus=0, **options)

        group = ttk.LabelFrame(panel, text="Classes (keys 1-9)")
        group.pack(fill=tk.X, pady=2)
        self.class_list = listbox(group, height=7)
        self.class_list.pack(fill=tk.X, padx=3, pady=3)
        self.class_list.bind("<<ListboxSelect>>", lambda e: self._on_list_select(self.class_list, self.select_class))
        row = ttk.Frame(group)
        row.pack(fill=tk.X, padx=3, pady=(0, 3))
        for text, command in (("Add", self.add_class), ("Rename", self.rename_class), ("Delete", self.delete_class)):
            ttk.Button(row, text=text, command=command, width=7, takefocus=0).pack(side=tk.LEFT, padx=1)

        group = ttk.LabelFrame(panel, text="Objects on this image")
        group.pack(fill=tk.X, pady=2)
        self.object_list = listbox(group, height=7)
        self.object_list.pack(fill=tk.X, padx=3, pady=3)
        self.object_list.bind("<<ListboxSelect>>",
                              lambda e: self._on_list_select(self.object_list, self.select_object))

        group = ttk.LabelFrame(panel, text="Images")
        group.pack(fill=tk.BOTH, expand=True, pady=2)
        row = ttk.Frame(group)
        row.pack(fill=tk.X, padx=3, pady=(3, 0))
        ttk.Button(row, text="<", width=3, command=self.prev_image, takefocus=0).pack(side=tk.LEFT)
        ttk.Button(row, text=">", width=3, command=self.next_image, takefocus=0).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Next unlabeled (N)", command=self.next_unlabeled,
                   takefocus=0).pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll = ttk.Scrollbar(group, orient=tk.VERTICAL)
        self.image_list = listbox(group, yscrollcommand=scroll.set)
        scroll.config(command=self.image_list.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_list.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.image_list.bind("<<ListboxSelect>>", lambda e: self._on_list_select(
            self.image_list, lambda i: i != self.index and self.show_image(i)))

    def _build_canvas(self):
        self.canvas = c = tk.Canvas(self, bg=CANVAS_BG, highlightthickness=0, cursor=self.tool.cursor)
        c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Tk on macOS numbers the right and middle buttons the other way round
        right, middle = ("2", "3") if self.root.tk.call("tk", "windowingsystem") == "aqua" else ("3", "2")
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", lambda e: self.cur_path and self.tool.on_double(e))
        c.bind(f"<ButtonPress-{right}>", lambda e: self.cur_path and self.tool.on_right(e))
        c.bind(f"<ButtonPress-{middle}>", lambda e: setattr(self, "_panning", (e.x, e.y)))
        c.bind(f"<B{middle}-Motion>", self._pan_to)
        c.bind(f"<ButtonRelease-{middle}>", lambda e: setattr(self, "_panning", None))
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", self._on_leave)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(sequence, self._on_wheel)
        c.bind("<Configure>", lambda e: self.fit_view() if self.view.auto_fit else self.request_render())

    def _bind_keys(self):
        keys = {
            "<Return>": self.on_enter, "<KP_Enter>": self.on_enter, "<space>": self.on_enter,
            "<Escape>": self.on_escape, "<Tab>": lambda: self.tool.cycle(),
            "<d>": self.next_image, "<Right>": self.next_image, "<Next>": self.next_image,
            "<a>": self.prev_image, "<Left>": self.prev_image, "<Prior>": self.prev_image,
            "<n>": self.next_unlabeled, "<Delete>": self.delete_object, "<BackSpace>": self.on_backspace,
            "<c>": self.recolor_object, "<f>": self.fit_view, "<h>": self.toggle_annotations,
            "<Control-z>": self.undo, "<Control-y>": self.redo, "<Control-Z>": self.redo,
            "<Control-s>": self.save, "<Control-o>": self.open_folder, "<Control-e>": self.export,
            "<F1>": self.show_help,
        }
        if self.root.tk.call("tk", "windowingsystem") == "aqua":
            keys.update({"<Command-z>": self.undo, "<Command-Z>": self.redo,
                         "<Command-s>": self.save, "<Command-o>": self.open_folder})
        for tool in self.tools.values():
            keys[f"<{tool.key}>"] = lambda name=tool.name: self.set_tool(name)
        for sequence, action in keys.items():
            # "break" stops Tk's defaults, e.g. Tab moving the keyboard focus
            self.root.bind(sequence, lambda e, action=action: (action(), "break")[1])
        for digit in range(10):
            self.root.bind(f"<Key-{digit}>", lambda e, d=digit: self.select_class((d - 1) % 10))

    # -------------------------------------------------- interface used by tools
    @property
    def tool(self):
        return self.tools[self.tool_var.get()]

    @property
    def tolerance(self):
        return round(self.tolerance_var.get(), 2)

    @property
    def hover_preview_enabled(self):
        return self.hover_preview.get()

    @property
    def image_size(self):
        return self.renderer.size

    @property
    def annotations(self):
        return self.project.get(self.cur_path)

    def inside(self, ix, iy):
        w, h = self.image_size
        return 0 <= ix < w and 0 <= iy < h

    def clamp(self, ix, iy):
        w, h = self.image_size
        return min(max(ix, 0.0), w - 1.0), min(max(iy, 0.0), h - 1.0)

    def image_box(self, a, b):
        """Image-space box spanned by two canvas points, or None if tiny."""
        x0, y0 = self.clamp(*self.view.to_image(*a))
        x1, y1 = self.clamp(*self.view.to_image(*b))
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return [x0, y0, x1, y1] if x1 - x0 >= 2 and y1 - y0 >= 2 else None

    def class_color(self, class_id):
        return self.project.class_color(class_id)

    def object_at(self, ix, iy):
        anns = self.annotations
        return next((i for i in range(len(anns) - 1, -1, -1) if annotation_contains(anns[i], ix, iy)), None)

    def add_annotation(self, polygons, score=None):
        self.project.add_annotation(self.cur_path, make_annotation(polygons, self.current_class, score))
        self._annotations_changed()
        self.set_status(f"Added '{self.project.class_name(self.current_class)}' "
                        f"({len(self.annotations)} on this image).")

    def replace_annotation(self, index, ann):
        self.project.replace_annotation(self.cur_path, index, ann)
        self._annotations_changed(keep_selection=True)

    def select_object(self, index):
        self.selected_obj = index
        self.object_list.selection_clear(0, tk.END)
        if index is not None:
            self.object_list.selection_set(index)
        self.request_render()

    def set_status(self, text):
        self.status.config(text=text)

    # -------------------------------------------------------------- tools
    def set_tool(self, name):
        self.tool_var.set(name)
        self._on_tool_selected()

    def _on_tool_selected(self):
        for tool in self.tools.values():
            tool.reset()
        if self.tool.name != "edit":
            self.select_object(None)
        self.canvas.config(cursor=self.tool.cursor)
        self.set_status(self.tool.hint)
        self.request_render()

    def on_enter(self):
        self.tool.on_enter()

    def on_escape(self):
        self.tool.on_escape()

    def on_backspace(self):
        if not self.tool.undo_step():
            self.delete_object()

    # ------------------------------------------------------- model / worker
    def load_model(self):
        self.settings.update(model=self.model_var.get())
        self.model_ready = False
        self.model_status.config(text="Loading model...")
        self.worker.load_model(self.model_var.get(), self.device, self.precision)
        if self.cur_path:
            self.worker.set_current_image(self.cur_path, self._prefetch_paths())

    def _poll_events(self):
        while True:
            try:
                event, *args = self.events.get_nowait()
            except queue.Empty:
                break
            self._event_handlers[event](*args)
        self._poll_job = self.after(POLL_INTERVAL_MS, self._poll_events)

    def _on_model_status(self, fraction, message):
        self.model_status.config(text=message)
        if fraction is None:
            self.download_progress.pack_forget()
        else:
            if not self.download_progress.winfo_ismapped():
                self.download_progress.pack(side=tk.RIGHT, padx=4)
            self.download_progress["value"] = fraction * 100

    def _on_model_ready(self, key, device, precision):
        self.download_progress.pack_forget()
        self.model_ready = True
        self.model_status.config(text=f"{models.MODELS[key].label} | {device} | {precision}")
        smart = self.tools["smart"]
        if smart.has_prompt:
            smart.request_prediction()

    def _on_model_error(self, message):
        self.download_progress.pack_forget()
        self.model_status.config(text="Model not loaded")
        messagebox.showerror("Model error", f"{message}\n\nPick another model from the Model menu.")

    def _on_worker_error(self, message):
        self.set_status(f"Error: {message}")

    def _on_encoding(self, path):
        if path == self.cur_path:
            self.set_status("Preparing image for smart polygon...")
            self.canvas.config(cursor="watch")

    def _on_encoded(self, path, seconds):
        if path == self.cur_path:
            self.canvas.config(cursor=self.tool.cursor)
            if not self.tools["smart"].has_prompt:
                self.set_status(f"Image ready ({seconds:.2f}s). {self.tool.hint}")

    def _on_prediction(self, request_id, path, candidates, seconds):
        if path == self.cur_path:
            self.canvas.config(cursor=self.tool.cursor)
            self.tools["smart"].on_prediction(request_id, candidates, seconds)

    def _on_hover_prediction(self, request_id, path, point, candidates):
        if path == self.cur_path and self.tool.name == "smart":
            self.tools["smart"].on_hover_result(request_id, point, candidates)

    def _on_tolerance_changed(self):
        if self._tolerance_job is not None:
            self.after_cancel(self._tolerance_job)
        self._tolerance_job = self.after(150, self.apply_tolerance)

    def apply_tolerance(self):
        """Re-trace the current mask with the new polygon detail."""
        self._tolerance_job = None
        self.settings.update(tolerance=self.tolerance)
        smart = self.tools["smart"]
        smart.clear_hover()
        if smart.has_prompt:
            smart.request_prediction()

    def _on_hover_preview_toggled(self):
        self.settings.update(hover_preview=self.hover_preview.get())
        self.tools["smart"].clear_hover()
        self.request_render()

    # -------------------------------------------------------------- images
    def open_folder(self):
        folder = filedialog.askdirectory(initialdir=self.settings.get("last_dir"))
        if folder:
            self.open_paths([folder])

    def open_images(self):
        pattern = " ".join("*" + ext for ext in IMAGE_EXTENSIONS)
        files = filedialog.askopenfilenames(initialdir=self.settings.get("last_dir"),
                                            filetypes=[("Images", pattern), ("All files", "*.*")])
        if files:
            self.open_paths(files)

    def open_paths(self, paths):
        images = []
        for path in paths:
            if os.path.isdir(path):
                images.extend(list_images(path))
            elif is_image_file(path):
                images.append(path)
        if not images:
            messagebox.showerror("No images", f"No images found. Supported: {', '.join(IMAGE_EXTENSIONS)}")
            return
        self._save_now()
        try:
            self.project = Project.open(images)
        except (OSError, ValueError, KeyError) as e:
            messagebox.showerror("Project error", f"Could not read the saved project file:\n{e}")
            return
        self.settings.update(last_dir=os.path.dirname(self.project.path))
        self.image_cache.clear()
        self.current_class = 0
        self._refresh_classes()
        self._refresh_image_list()
        done = sum(self.project.is_annotated(p) for p in self.project.images)
        first_unlabeled = next((i for i, p in enumerate(self.project.images)
                                if not self.project.is_annotated(p)), 0)
        self.show_image(first_unlabeled)
        self.set_status(f"Loaded {len(self.project.images)} images ({done} already annotated). "
                        f"{self.tool.hint}  F1: all shortcuts.")

    def show_image(self, index):
        images = self.project.images
        if not images:
            return
        self.index = index = max(0, min(index, len(images) - 1))
        for tool in self.tools.values():
            tool.reset()
        path = images[index]
        try:
            self.renderer = FrameRenderer(Image.fromarray(self.image_cache.get(path)))
        except OSError as e:
            self.renderer, self.cur_path = None, None
            self.set_status(f"Cannot open {os.path.basename(path)}: {e}")
            self.request_render()
            return
        self.cur_path = path
        self.hover = None
        self.select_object(None)
        self.worker.set_current_image(path, self._prefetch_paths())
        self.fit_view()
        self._refresh_objects()
        self.image_list.selection_clear(0, tk.END)
        self.image_list.selection_set(index)
        self.image_list.see(index)
        w, h = self.image_size
        self.root.title(f"Smart Polygon Annotator - {os.path.basename(path)} ({w}x{h}) [{index + 1}/{len(images)}]")

    def next_image(self):
        if self.index < len(self.project.images) - 1:
            self.show_image(self.index + 1)

    def prev_image(self):
        if self.index > 0:
            self.show_image(self.index - 1)

    def next_unlabeled(self):
        images = self.project.images
        for offset in range(1, len(images) + 1):
            i = (self.index + offset) % len(images)
            if not self.project.is_annotated(images[i]):
                self.show_image(i)
                return
        self.set_status("All images are annotated.")

    def _prefetch_paths(self):
        return self.project.images[self.index + 1:self.index + 1 + PREFETCH_AHEAD]

    # ---------------------------------------------------------------- view
    def fit_view(self):
        if self.renderer:
            self.view.fit(self.image_size, self._canvas_size())
        self.request_render()

    def toggle_annotations(self):
        self.show_annotations.set(not self.show_annotations.get())
        self.request_render()

    def _canvas_size(self):
        return max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)

    def _on_wheel(self, event):
        if self.renderer:
            zoom_in = event.num == 4 or event.delta > 0
            self.view.zoom_at(event.x, event.y, 1.25 if zoom_in else 0.8, self.image_size, self._canvas_size())
            self.request_render()

    def _pan_to(self, event):
        if self._panning and self.renderer:
            self.view.pan(event.x - self._panning[0], event.y - self._panning[1])
            self._panning = (event.x, event.y)
            self.request_render()

    # -------------------------------------------------------------- render
    def request_render(self):
        if self._render_job is None:
            self._render_job = self.after_idle(self._render)

    def _render(self):
        self._render_job = None
        c = self.canvas
        c.delete("hint")
        region = self.renderer and self.view.visible_region(self.image_size, self._canvas_size())
        if not region:
            if self._image_item is not None:
                c.itemconfig(self._image_item, state=tk.HIDDEN)
            if not self.renderer:
                c.create_text(c.winfo_width() // 2, c.winfo_height() // 2, fill="#9aa0a6", tags="hint",
                              font=("TkDefaultFont", 14), justify=tk.CENTER,
                              text="Open a folder of images to start  (Ctrl+O)\n\nF1: keyboard shortcuts")
            self.redraw_canvas_items()
            return

        x0, y0, x1, y1 = region
        s = self.view.scale
        size = (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))
        frame = self.renderer.frame(region, size, s)
        overlay = self._draw_overlay(size, (x0, y0), s)
        if overlay is not None:
            frame = Image.alpha_composite(frame, overlay)
        self._photo = ImageTk.PhotoImage(frame)  # must stay referenced while shown
        position = self.view.to_canvas(x0, y0)
        if self._image_item is None:
            self._image_item = c.create_image(*position, anchor=tk.NW, image=self._photo)
        else:
            c.coords(self._image_item, *position)
            c.itemconfig(self._image_item, image=self._photo, state=tk.NORMAL)
        self.redraw_canvas_items()

    def _draw_overlay(self, size, origin, scale):
        anns = list(self.annotations) if self.show_annotations.get() else []
        override = self.tool.annotation_override()
        if override and override[0] < len(anns):
            anns[override[0]] = override[1]
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        offset = np.array(origin, dtype=np.float32)

        def to_frame(polygon):
            return ((np.asarray(polygon, dtype=np.float32) - offset) * scale).ravel().tolist()

        for i, ann in enumerate(anns):
            rgb = hex_to_rgb(self.class_color(ann["class_id"]))
            focused = i in (self.hover, self.selected_obj)
            for poly in ann["polygons"]:
                draw.polygon(to_frame(poly), fill=rgb + (130 if focused else 60,), outline=rgb + (255,),
                             width=3 if focused else 2)
        self.tool.draw_overlay(draw, to_frame)
        return overlay if overlay.getbbox() else None

    def redraw_canvas_items(self):
        self.canvas.delete("overlay")
        if self.renderer:
            self.tool.draw_canvas(self.canvas)

    # --------------------------------------------------------------- mouse
    def _on_press(self, event):
        self.canvas.focus_set()
        if not self.renderer:
            return
        if event.state & CONTROL or self.tool.on_press(event) == "pan":
            self._panning = (event.x, event.y)

    def _on_drag(self, event):
        if self._panning:
            self._pan_to(event)
        elif self.renderer:
            self.tool.on_drag(event)

    def _on_release(self, event):
        if self._panning:
            self._panning = None
        elif self.renderer:
            self.tool.on_release(event)

    def _on_motion(self, event):
        if not self.renderer:
            return
        ix, iy = self.view.to_image(event.x, event.y)
        hit = self.object_at(ix, iy) if self.show_annotations.get() else None
        if hit != self.hover:
            self.hover = hit
            self.request_render()
        self.tool.on_motion(event, ix, iy)

    def _on_leave(self, event):
        if self.hover is not None:
            self.hover = None
            self.request_render()
        self.tool.on_leave()

    # ------------------------------------------------------------- objects
    def _target_object(self):
        """The object Del / C act on: the Edit selection, else the one under the cursor."""
        if self.tool.name == "edit" and self.selected_obj is not None:
            return self.selected_obj
        return self.hover if self.hover is not None else self.selected_obj

    def delete_object(self):
        index = self._target_object()
        if index is not None and index < len(self.annotations):
            self.project.remove_annotation(self.cur_path, index)
            self.hover = None
            self._annotations_changed()
            self.set_status("Object deleted (Ctrl+Z to undo).")

    def recolor_object(self):
        index = self._target_object()
        if index is not None and index < len(self.annotations):
            self.replace_annotation(index, dict(self.annotations[index], class_id=self.current_class))

    def clear_image(self):
        if self.annotations:
            self.project.set_annotations(self.cur_path, [])
            self.hover = None
            self._annotations_changed()

    def _annotations_changed(self, keep_selection=False):
        if not keep_selection:
            self.selected_obj = None
        self._refresh_objects()
        self._refresh_image_row(self.index)
        self._refresh_classes()
        self.request_render()
        self._schedule_save()

    def _refresh_objects(self):
        self.object_list.delete(0, tk.END)
        for i, ann in enumerate(self.annotations):
            score = f"  ({ann['score']:.2f})" if "score" in ann else ""
            self.object_list.insert(tk.END, f"#{i + 1}  {self.project.class_name(ann['class_id'])}{score}")
            self._color_row(self.object_list, i, self.class_color(ann["class_id"]))
        if self.selected_obj is not None and self.selected_obj < self.object_list.size():
            self.object_list.selection_set(self.selected_obj)

    def _refresh_image_list(self):
        self.image_list.delete(0, tk.END)
        self.image_list.insert(tk.END, *(self._image_row(p) for p in self.project.images))
        if self.project.images:
            self.image_list.selection_set(self.index)

    def _refresh_image_row(self, index):
        if 0 <= index < len(self.project.images):
            self.image_list.delete(index)
            self.image_list.insert(index, self._image_row(self.project.images[index]))
            self.image_list.selection_set(self.index)

    def _image_row(self, path):
        n = len(self.project.get(path))
        return f"{'✓' if n else '  '} {os.path.basename(path)}" + (f"  ({n})" if n else "")

    @staticmethod
    def _color_row(listbox, index, color):
        # Selected rows are drawn inverted so the selection stands out
        listbox.itemconfig(index, background=color, foreground=readable_text_color(color),
                           selectbackground=CANVAS_BG, selectforeground=color)

    @staticmethod
    def _on_list_select(listbox, action):
        selection = listbox.curselection()
        if selection:
            action(selection[0])

    # ------------------------------------------------------------- classes
    def _refresh_classes(self):
        counts = self.project.class_counts()
        self.class_list.delete(0, tk.END)
        for i, cls in enumerate(self.project.classes):
            shortcut = f"[{(i + 1) % 10}] " if i < 10 else "    "
            self.class_list.insert(tk.END, f"{shortcut}{cls['name']}  ({counts[i]})")
            self._color_row(self.class_list, i, cls["color"])
        self.class_combo["values"] = [c["name"] for c in self.project.classes]
        self.select_class(min(self.current_class, len(self.project.classes) - 1))

    def select_class(self, class_id):
        if not 0 <= class_id < len(self.project.classes):
            return
        self.current_class = class_id
        self.class_list.selection_clear(0, tk.END)
        self.class_list.selection_set(class_id)
        self.class_combo.current(class_id)
        self.request_render()  # the mask preview uses the class color

    def add_class(self):
        name = simpledialog.askstring("Add class", "Class name:", parent=self.root)
        if name and name.strip():
            self.current_class = self.project.add_class(name.strip())
            self._refresh_classes()
            self._schedule_save()

    def rename_class(self):
        name = simpledialog.askstring("Rename class", "New name:", parent=self.root,
                                      initialvalue=self.project.class_name(self.current_class))
        if name and name.strip():
            self.project.rename_class(self.current_class, name.strip())
            self._refresh_classes()
            self._refresh_objects()
            self._schedule_save()

    def delete_class(self):
        if len(self.project.classes) <= 1:
            messagebox.showinfo("Delete class", "A project needs at least one class. Rename it instead.")
            return
        class_id = self.current_class
        count = self.project.class_counts()[class_id]
        if messagebox.askyesno("Delete class", f"Delete class '{self.project.class_name(class_id)}' and its "
                               f"{count} annotation(s)?\nHigher class ids shift down by one."):
            self.project.delete_class(class_id)
            self.current_class = max(0, class_id - 1)
            self._refresh_everything()

    def _refresh_everything(self):
        self.hover = None
        self.select_object(None)
        self._refresh_classes()
        self._refresh_image_list()
        self._refresh_objects()
        self._schedule_save()

    # ------------------------------------------------------------- history
    def undo(self):
        # While building a prompt or polygon, Ctrl+Z steps back through it first
        if not self.tool.undo_step():
            self._apply_history(self.project.undo())

    def redo(self):
        self._apply_history(self.project.redo())

    def _apply_history(self, path):
        if path is False:
            return
        if path is None:  # a class edit, which can touch every image
            self._refresh_everything()
            return
        if path != self.cur_path and path in self.project.images:
            self.show_image(self.project.images.index(path))
        self.hover = None
        if self.selected_obj is not None and self.selected_obj >= len(self.annotations):
            self.selected_obj = None
        if path in self.project.images:
            self._refresh_image_row(self.project.images.index(path))
        self._refresh_objects()
        self._refresh_classes()
        self._schedule_save()

    # --------------------------------------------------------- persistence
    def _schedule_save(self):
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(AUTOSAVE_DELAY_MS, self._save_now)

    def _save_now(self):
        self._save_job = None
        if not (self.project.dirty and self.project.path):
            return
        try:
            self.project.save()
        except OSError as e:
            if not self._save_error_shown:
                self._save_error_shown = True
                messagebox.showwarning("Autosave failed", f"Could not save {self.project.path}:\n{e}\n"
                                       "Use File > Save to choose another location.")

    def save(self):
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

    def export(self):
        if not any(self.project.get(p) for p in self.project.images):
            messagebox.showerror("Export", "No annotations to export yet.")
            return
        self._save_now()
        ExportDialog(self.root, self.project, self.settings)

    def quit(self):
        self._save_now()
        self.after_cancel(self._poll_job)
        self.worker.stop()
        self.root.destroy()
        # Let an in-flight encode finish: stopping torch mid-operation aborts the process
        self.worker.join(timeout=15)

    # ---------------------------------------------------------------- help
    def show_help(self):
        messagebox.showinfo("Keyboard Shortcuts", SHORTCUTS)

    def show_hardware(self):
        info = models.detect_device(self.device)
        ranked = models.candidate_models(info)
        lines = [f"Device: {info.describe()}", "", "Installed models, best for this PC first:"]
        lines += [f"  {i + 1}. {models.MODELS[k].label} - {models.MODELS[k].notes}" for i, k in enumerate(ranked)]
        missing = sorted({models.INSTALL_HINTS[m.backend] for m in models.MODELS.values()
                          if not models.backend_installed(m.backend)})
        if missing:
            lines += ["", "More models are available after:"] + [f"  {cmd}" for cmd in missing]
        messagebox.showinfo("Hardware Info", "\n".join(lines))
