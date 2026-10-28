"""Annotation tools.

Each tool owns its in-progress state (prompt points, unfinished polygon, a
vertex being dragged) and turns mouse and keyboard input into edits through
the main window. The window routes events to the active tool and draws what
the tool returns.
"""
import math

from ..geometry import segment_distance, with_polygons
from .style import PROMPT_BOX_COLOR, hex_to_rgb

HANDLE_RADIUS = 7    # vertex grab distance, screen pixels
DRAG_THRESHOLD = 6   # movement that turns a click into a drag
HOVER_INTERVAL_MS = 25
SHIFT = 0x0001

POSITIVE_COLOR = "#00C853"
NEGATIVE_COLOR = "#FF1744"


class Tool:
    name = label = key = hint = ""
    cursor = "arrow"

    def __init__(self, win):
        self.win = win

    def reset(self):
        """Drop in-progress work (image change or tool switch)."""

    # Mouse. on_press may return "pan" to let the window pan for this drag.
    def on_press(self, event):
        pass

    def on_drag(self, event):
        pass

    def on_release(self, event):
        pass

    def on_double(self, event):
        pass

    def on_right(self, event):
        pass

    def on_motion(self, event, ix, iy):
        pass

    def on_leave(self):
        pass

    # Keyboard
    def on_enter(self):
        pass

    def on_escape(self):
        self.reset()
        self.win.request_render()

    def undo_step(self):
        """Step back within the in-progress work. False if there is none."""
        return False

    def cycle(self):
        pass

    # Drawing
    def annotation_override(self):
        """(index, annotation) to draw instead of a stored one, or None."""
        return None

    def draw_overlay(self, draw, to_frame):
        """Filled shapes, drawn into the RGBA overlay with PIL."""

    def draw_canvas(self, canvas):
        """Markers and handles, drawn as Tk canvas items tagged "overlay"."""


class _ClickOrDragTool(Tool):
    """A short press is a click; dragging draws a rectangle."""
    rubber_dash = None

    def __init__(self, win):
        super().__init__(win)
        self._press = None
        self._dragging = False

    def on_press(self, event):
        self._press, self._dragging = (event.x, event.y), False

    def on_drag(self, event):
        if self._press is None:
            return
        px, py = self._press
        if abs(event.x - px) + abs(event.y - py) > DRAG_THRESHOLD:
            self._dragging = True
        if self._dragging:
            canvas = self.win.canvas
            canvas.delete("rubber")
            canvas.create_rectangle(px, py, event.x, event.y, outline=self.rubber_color(), width=2,
                                    dash=self.rubber_dash, tags="rubber")

    def on_release(self, event):
        press, self._press = self._press, None
        self.win.canvas.delete("rubber")
        if press is None:
            return
        if self._dragging:
            box = self.win.image_box(press, (event.x, event.y))
            if box:
                self.on_box(box)
        else:
            self.on_click(event)

    def on_double(self, event):
        self._press = None

    def rubber_color(self):
        return self.win.class_color(self.win.current_class)

    def on_click(self, event):
        pass

    def on_box(self, box):
        pass


class SmartTool(_ClickOrDragTool):
    """SAM-assisted polygons from positive/negative points and box prompts."""
    name, label, key, cursor = "smart", "Smart (S)", "s", "crosshair"
    hint = ("Smart polygon: click an object (right/Shift-click excludes, drag for a box). "
            "Enter accepts, Tab cycles masks, Ctrl+Z removes the last point.")
    rubber_dash = (4, 3)

    def __init__(self, win):
        super().__init__(win)
        self.request_id = 0
        self.hover_id = 0
        self._hover_job = None
        self._pending_hover = None
        self.reset()

    def reset(self):
        self.points, self.labels, self.box = [], [], None
        self.candidates, self.index = [], 0
        self.request_id += 1  # results of in-flight requests are now stale
        self.clear_hover()

    @property
    def has_prompt(self):
        return bool(self.points) or self.box is not None

    def rubber_color(self):
        return PROMPT_BOX_COLOR

    # -- prompts --
    def on_click(self, event):
        self._add_point(event, 0 if event.state & SHIFT else 1)

    def on_right(self, event):
        if not any(self.labels) and self.box is None:
            self.win.set_status("Right-click excludes a region from the current mask. Left-click an object first.")
            return
        self._add_point(event, 0)

    def on_box(self, box):
        self.box = box
        self.clear_hover()
        self.request_prediction()

    def _add_point(self, event, label):
        ix, iy = self.win.view.to_image(event.x, event.y)
        if not self.win.inside(ix, iy):
            return
        # A click where the hover preview was computed is answered from it
        hover = self.hover_point
        reuse = (label == 1 and not self.has_prompt and self.hover_candidates and hover is not None
                 and math.dist(self.win.view.to_canvas(*hover), (event.x, event.y)) <= 4)
        candidates = self.hover_candidates
        self.clear_hover()
        self.labels.append(label)
        if reuse:
            self.points.append(tuple(hover))
            self.request_id += 1
            self.win.redraw_canvas_items()
            self.show_candidates(candidates)
        else:
            self.points.append((ix, iy))
            self.request_prediction()

    def request_prediction(self):
        self.request_id += 1
        self.win.redraw_canvas_items()
        if not any(self.labels) and self.box is None:
            self.candidates = []
            self.win.request_render()
            return
        self.win.worker.predict(self.request_id, self.win.cur_path, list(self.points), list(self.labels),
                                self.box, self.win.tolerance)
        if not self.win.model_ready:
            self.win.set_status("Model is still loading; your prompt will run when it is ready...")

    def on_prediction(self, request_id, candidates, seconds):
        if request_id == self.request_id:  # otherwise the user has moved on
            self.show_candidates(candidates, seconds)

    def show_candidates(self, candidates, seconds=None):
        self.candidates, self.index = candidates, 0
        if candidates:
            timing = f" ({seconds * 1000:.0f} ms)" if seconds else ""
            more = " | Tab: other masks" if len(candidates) > 1 else ""
            self.win.set_status(f"Mask {candidates[0]['score']:.2f}{timing} | Enter: accept as "
                                f"'{self.win.project.class_name(self.win.current_class)}' | "
                                f"Right-click: exclude | Ctrl+Z: undo point{more}")
        else:
            self.win.set_status("No mask here. Try another point, or drag a box.")
        self.win.request_render()

    def on_enter(self):
        if self.candidates:
            candidate = self.candidates[self.index]
            self.reset()
            self.win.add_annotation(candidate["polygons"], candidate["score"])

    def undo_step(self):
        if not self.has_prompt:
            return False
        if self.points:
            self.points.pop()
            self.labels.pop()
        else:
            self.box = None
        self.request_prediction()
        return True

    def cycle(self):
        if len(self.candidates) > 1:
            self.index = (self.index + 1) % len(self.candidates)
            self.win.set_status(f"Mask {self.index + 1}/{len(self.candidates)} "
                                f"(score {self.candidates[self.index]['score']:.2f})")
            self.win.request_render()

    # -- hover preview --
    def hover_allowed(self):
        return self.win.hover_preview_enabled and self.win.model_ready and not self.has_prompt

    def on_motion(self, event, ix, iy):
        if self.hover_allowed() and self.win.inside(ix, iy):
            self._pending_hover = (ix, iy)
            if self._hover_job is None:
                self._hover_job = self.win.after(HOVER_INTERVAL_MS, self._send_hover)

    def _send_hover(self):
        self._hover_job = None
        if self.hover_allowed() and self.win.cur_path:
            self.hover_id += 1
            self.win.worker.hover(self.hover_id, self.win.cur_path, self._pending_hover, self.win.tolerance)

    def on_hover_result(self, request_id, point, candidates):
        if request_id == self.hover_id and self.hover_allowed():
            self.hover_point, self.hover_candidates = point, candidates
            self.win.request_render()

    def clear_hover(self):
        self.hover_id += 1
        self.hover_point, self.hover_candidates = None, []

    def on_leave(self):
        if self.hover_candidates:
            self.clear_hover()
            self.win.request_render()

    # -- drawing --
    def draw_overlay(self, draw, to_frame):
        rgb = hex_to_rgb(self.win.class_color(self.win.current_class))
        if self.candidates:
            for poly in self.candidates[self.index]["polygons"]:
                draw.polygon(to_frame(poly), fill=rgb + (120,), outline=(255, 255, 255, 255), width=2)
        elif self.hover_candidates and self.hover_allowed():
            for poly in self.hover_candidates[0]["polygons"]:
                draw.polygon(to_frame(poly), fill=rgb + (70,), outline=(255, 255, 255, 200), width=1)

    def draw_canvas(self, canvas):
        to_canvas = self.win.view.to_canvas
        if self.box:
            (ax, ay), (bx, by) = to_canvas(*self.box[:2]), to_canvas(*self.box[2:])
            canvas.create_rectangle(ax, ay, bx, by, outline=PROMPT_BOX_COLOR, width=2, dash=(6, 4),
                                    tags="overlay")
        for (x, y), label in zip(self.points, self.labels, strict=True):
            cx, cy = to_canvas(x, y)
            canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill=POSITIVE_COLOR if label else NEGATIVE_COLOR,
                               outline="white", width=2, tags="overlay")
            canvas.create_text(cx, cy - 1, text="+" if label else "−", fill="white",
                               font=("TkDefaultFont", 9, "bold"), tags="overlay")


class BoxTool(_ClickOrDragTool):
    name, label, key, cursor = "box", "Box (B)", "b", "tcross"
    hint = "Box: drag to draw a bounding box. It is saved immediately with the current class."

    def on_box(self, box):
        x0, y0, x1, y1 = box
        self.win.add_annotation([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])


class PolygonTool(_ClickOrDragTool):
    name, label, key, cursor = "polygon", "Polygon (P)", "p", "pencil"
    hint = "Polygon: click to add vertices. Enter, double-click or clicking the first vertex closes it."

    def __init__(self, win):
        super().__init__(win)
        self.vertices = []
        self._cursor = None

    def reset(self):
        self.vertices = []
        self._cursor = None

    def on_drag(self, event):
        pass  # no rubber box: a sloppy click still places a vertex

    def on_click(self, event):
        ix, iy = self.win.view.to_image(event.x, event.y)
        if not self.win.inside(ix, iy):
            return
        if len(self.vertices) >= 3:
            first = self.win.view.to_canvas(*self.vertices[0])
            if math.dist(first, (event.x, event.y)) <= HANDLE_RADIUS + 2:
                self.on_enter()
                return
        self.vertices.append((ix, iy))
        self._cursor = (event.x, event.y)
        self.win.redraw_canvas_items()

    def on_double(self, event):
        super().on_double(event)
        self.on_enter()

    def on_right(self, event):
        self.undo_step()

    def on_motion(self, event, ix, iy):
        if self.vertices:
            self._cursor = (event.x, event.y)
            self.win.redraw_canvas_items()

    def on_leave(self):
        self._cursor = None
        self.win.redraw_canvas_items()

    def on_enter(self):
        if len(self.vertices) < 3:
            if self.vertices:
                self.win.set_status("A polygon needs at least 3 vertices.")
            return
        vertices, self.vertices = self.vertices, []
        self.win.add_annotation([[list(p) for p in vertices]])

    def undo_step(self):
        if not self.vertices:
            return False
        self.vertices.pop()
        self.win.redraw_canvas_items()
        return True

    def draw_canvas(self, canvas):
        if not self.vertices:
            return
        color = self.win.class_color(self.win.current_class)
        points = [self.win.view.to_canvas(x, y) for x, y in self.vertices]
        line = points + [self._cursor] if self._cursor else points
        if len(line) >= 2:
            canvas.create_line(*[v for p in line for v in p], fill=color, width=2, tags="overlay")
        for i, (cx, cy) in enumerate(points):
            r = 6 if i == 0 else 4  # the first vertex is larger: clicking it closes the polygon
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="white" if i == 0 else color,
                               outline=color, width=2, tags="overlay")


class EditTool(Tool):
    """Select an object, then move it or edit its vertices."""
    name, label, key, cursor = "edit", "Edit (V)", "v", "arrow"
    hint = ("Edit: click an object, then drag its vertices or the object itself. "
            "Shift-click a vertex deletes it, double-click an edge adds one.")

    def __init__(self, win):
        super().__init__(win)
        self._drag = None      # {"vertex": (poly, index) or None, "start": (x, y), "orig": annotation}
        self._preview = None   # (index, annotation) while dragging

    def reset(self):
        self._drag = self._preview = None

    def selected(self):
        if self._preview is not None:
            return self._preview[1]
        anns, index = self.win.annotations, self.win.selected_obj
        return anns[index] if index is not None and index < len(anns) else None

    def on_press(self, event):
        vertex = self._vertex_at(event.x, event.y)
        if vertex is not None:
            if event.state & SHIFT:
                self._delete_vertex(vertex)
            else:
                self._drag = {"vertex": vertex, "start": None, "orig": self.selected()}
            return None
        hit = self.win.object_at(*self.win.view.to_image(event.x, event.y))
        if hit is None and self._edge_at(event.x, event.y) is not None:
            hit = self.win.selected_obj  # on the selection's outline, e.g. double-clicking to add a vertex
        self.win.select_object(hit)
        if hit is None:
            return "pan"
        self._drag = {"vertex": None, "start": self.win.view.to_image(event.x, event.y), "orig": self.selected()}
        return None

    def on_drag(self, event):
        if not self._drag:
            return
        orig = self._drag["orig"]
        ix, iy = self.win.view.to_image(event.x, event.y)
        if self._drag["vertex"] is not None:
            pi, vi = self._drag["vertex"]
            polygons = [list(p) for p in orig["polygons"]]
            polygons[pi][vi] = list(self.win.clamp(ix, iy))
        else:
            w, h = self.win.image_size
            x0, y0, x1, y1 = orig["bbox"]
            sx, sy = self._drag["start"]
            dx = min(max(ix - sx, -x0), w - 1 - x1)  # keep the object inside the image
            dy = min(max(iy - sy, -y0), h - 1 - y1)
            polygons = [[[x + dx, y + dy] for x, y in p] for p in orig["polygons"]]
        self._preview = (self.win.selected_obj, with_polygons(orig, polygons))
        self.win.request_render()

    def on_release(self, event):
        preview, self._preview, self._drag = self._preview, None, None
        if preview is not None:
            self.win.replace_annotation(*preview)
        else:
            self.win.request_render()

    def on_double(self, event):
        edge = self._edge_at(event.x, event.y)
        if edge is None:
            return
        pi, vi = edge
        polygons = [list(p) for p in self.selected()["polygons"]]
        polygons[pi].insert(vi + 1, list(self.win.clamp(*self.win.view.to_image(event.x, event.y))))
        self.win.replace_annotation(self.win.selected_obj, with_polygons(self.selected(), polygons))

    def on_right(self, event):
        vertex = self._vertex_at(event.x, event.y)
        if vertex is not None:
            self._delete_vertex(vertex)

    def on_motion(self, event, ix, iy):
        over = self._vertex_at(event.x, event.y) is not None or (
            self.win.hover is not None and self.win.hover == self.win.selected_obj)
        self.win.canvas.config(cursor="fleur" if over else self.cursor)

    def on_escape(self):
        self.reset()
        self.win.select_object(None)

    def annotation_override(self):
        return self._preview

    def draw_canvas(self, canvas):
        ann = self.selected()
        if ann is None:
            return
        color = self.win.class_color(ann["class_id"])
        for poly in ann["polygons"]:
            for x, y in poly:
                cx, cy = self.win.view.to_canvas(x, y)
                canvas.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline=color,
                                        width=2, tags="overlay")

    def _delete_vertex(self, vertex):
        pi, vi = vertex
        ann = self.selected()
        polygons = [list(p) for p in ann["polygons"]]
        if len(polygons[pi]) <= 3:
            self.win.set_status("A polygon needs at least 3 vertices (Del removes the whole object).")
            return
        del polygons[pi][vi]
        self.win.replace_annotation(self.win.selected_obj, with_polygons(ann, polygons))

    def _vertex_at(self, cx, cy):
        """(polygon index, vertex index) of the selected object's vertex under the cursor."""
        ann = self.selected()
        if ann is None:
            return None
        best = None
        for pi, poly in enumerate(ann["polygons"]):
            for vi, (x, y) in enumerate(poly):
                d = math.dist(self.win.view.to_canvas(x, y), (cx, cy))
                if d <= HANDLE_RADIUS and (best is None or d < best[0]):
                    best = (d, pi, vi)
        return best and best[1:]

    def _edge_at(self, cx, cy):
        """(polygon index, index of the edge's first vertex) under the cursor."""
        ann = self.selected()
        if ann is None:
            return None
        best = None
        for pi, poly in enumerate(ann["polygons"]):
            points = [self.win.view.to_canvas(x, y) for x, y in poly]
            for vi in range(len(points)):
                d = segment_distance((cx, cy), points[vi], points[(vi + 1) % len(points)])
                if d <= HANDLE_RADIUS and (best is None or d < best[0]):
                    best = (d, pi, vi)
        return best and best[1:]


TOOL_CLASSES = (SmartTool, BoxTool, PolygonTool, EditTool)
