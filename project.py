# project.py (annotation project: images, classes, annotations, autosave and undo/redo)
import colorsys
import json
import os

from annotation_processing import normalize_annotation

PROJECT_FILENAME = "smartannotator_project.json"
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".smartannotator", "settings.json")

_PALETTE = [
    "#E6194B", "#3CB44B", "#4363D8", "#FFE119", "#F58231", "#911EB4", "#42D4F4", "#F032E6",
    "#BFEF45", "#FABED4", "#469990", "#DCBEFF", "#9A6324", "#FFFAC8", "#800000", "#AAFFC3",
]


def class_color(index):
    if index < len(_PALETTE):
        return _PALETTE[index]
    hue = (index * 0.618033988749895) % 1.0  # golden-angle hues stay distinct
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return "#%02X%02X%02X" % (int(r * 255), int(g * 255), int(b * 255))


class Project:
    """All annotation state. Annotation lists are treated as immutable: every
    edit stores a new list, so undo history can share them without deep copies."""

    def __init__(self):
        self.images = []
        self.classes = [{"name": "object", "color": class_color(0)}]
        self.annotations = {}  # image path -> list of annotation dicts
        self.path = None       # where the project file lives
        self.dirty = False
        self._undo = []
        self._redo = []

    # -- loading / saving --
    @staticmethod
    def default_path(image_paths):
        folder = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in image_paths])
        return os.path.join(folder, PROJECT_FILENAME)

    def open_images(self, image_paths, project_path=None):
        """Use these images, merging in any project file saved next to them."""
        self.__init__()
        self.images = [os.path.abspath(p) for p in image_paths]
        self.path = project_path or self.default_path(self.images)
        if os.path.isfile(self.path):
            self._read(self.path)

    def _read(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = os.path.dirname(path)
        if data.get("classes"):
            self.classes = [{"name": c["name"], "color": c.get("color") or class_color(i)}
                            for i, c in enumerate(data["classes"])]
        for rel, anns in data.get("annotations", {}).items():
            full = os.path.normpath(os.path.join(base, rel))
            self.annotations[full] = [normalize_annotation(a) for a in anns]

    def save(self, path=None):
        path = path or self.path
        base = os.path.dirname(path)
        data = {
            "version": 2,
            "classes": self.classes,
            "annotations": {os.path.relpath(p, base).replace(os.sep, "/"): anns
                            for p, anns in self.annotations.items() if anns},
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written project
        self.path = path
        self.dirty = False

    # -- queries --
    def get(self, image_path):
        return self.annotations.get(image_path, [])

    def is_annotated(self, image_path):
        return bool(self.annotations.get(image_path))

    def class_name(self, class_id):
        return self.classes[class_id]["name"] if 0 <= class_id < len(self.classes) else f"class {class_id}"

    def class_color(self, class_id):
        return self.classes[class_id]["color"] if 0 <= class_id < len(self.classes) else "#FFFFFF"

    # -- edits (all undoable) --
    def _snapshot(self):
        return ([dict(c) for c in self.classes], dict(self.annotations))

    def _commit(self, entry):
        self._undo.append(entry)
        if len(self._undo) > 500:
            self._undo.pop(0)
        self._redo.clear()
        self.dirty = True

    def set_annotations(self, image_path, new_list):
        old = self.annotations.get(image_path, [])
        self.annotations[image_path] = list(new_list)
        self._commit(("image", image_path, old, self.annotations[image_path]))

    def add_annotation(self, image_path, ann):
        self.set_annotations(image_path, self.get(image_path) + [ann])

    def remove_annotation(self, image_path, index):
        anns = self.get(image_path)
        self.set_annotations(image_path, anns[:index] + anns[index + 1:])

    def change_class(self, image_path, index, class_id):
        anns = list(self.get(image_path))
        anns[index] = dict(anns[index], class_id=class_id)
        self.set_annotations(image_path, anns)

    def _edit_project(self, mutate):
        before = self._snapshot()
        mutate()
        self._commit(("project", before, self._snapshot()))

    def add_class(self, name):
        def mutate():
            self.classes.append({"name": name, "color": class_color(len(self.classes))})
        self._edit_project(mutate)
        return len(self.classes) - 1

    def rename_class(self, class_id, name):
        def mutate():
            self.classes[class_id] = dict(self.classes[class_id], name=name)
        self._edit_project(mutate)

    def delete_class(self, class_id):
        """Remove a class and its annotations; higher class ids shift down by one."""
        def mutate():
            del self.classes[class_id]
            for path, anns in list(self.annotations.items()):
                if any(a["class_id"] >= class_id for a in anns):
                    self.annotations[path] = [
                        dict(a, class_id=a["class_id"] - (a["class_id"] > class_id))
                        for a in anns if a["class_id"] != class_id]
        self._edit_project(mutate)

    def class_counts(self):
        counts = [0] * len(self.classes)
        for anns in self.annotations.values():
            for a in anns:
                if 0 <= a["class_id"] < len(counts):
                    counts[a["class_id"]] += 1
        return counts

    # -- history --
    def _apply(self, entry, forward):
        if entry[0] == "image":
            _, path, old, new = entry
            self.annotations[path] = new if forward else old
            return path
        _, before, after = entry
        classes, annotations = after if forward else before
        self.classes = [dict(c) for c in classes]
        self.annotations = dict(annotations)
        return None

    def undo(self):
        """Undo the last edit. Returns (True, image path it touched or None)."""
        if not self._undo:
            return False, None
        entry = self._undo.pop()
        self._redo.append(entry)
        self.dirty = True
        return True, self._apply(entry, forward=False)

    def redo(self):
        if not self._redo:
            return False, None
        entry = self._redo.pop()
        self._undo.append(entry)
        self.dirty = True
        return True, self._apply(entry, forward=True)


def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(settings):
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        pass
