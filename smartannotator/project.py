"""Annotation project: image list, classes, annotations and undo history.

Annotation lists are treated as immutable values: every edit stores a new list,
so history entries can reference old lists directly instead of deep-copying
the whole project on each change.
"""
import colorsys
import contextlib
import json
import os
from collections import deque
from typing import NamedTuple

PROJECT_FILENAME = "smartannotator_project.json"
FORMAT_VERSION = 2
HISTORY_LIMIT = 500

_PALETTE = [
    "#E6194B", "#3CB44B", "#4363D8", "#FFE119", "#F58231", "#911EB4", "#42D4F4", "#F032E6",
    "#BFEF45", "#FABED4", "#469990", "#DCBEFF", "#9A6324", "#FFFAC8", "#800000", "#AAFFC3",
]


def class_color(index):
    if index < len(_PALETTE):
        return _PALETTE[index]
    hue = (index * 0.618033988749895) % 1.0  # golden-ratio steps keep hues apart
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


class _ImageEdit(NamedTuple):
    path: str
    before: list
    after: list


class _ProjectEdit(NamedTuple):
    """Class changes can touch every image, so they snapshot classes and the
    (shallow) annotation map."""
    before: tuple
    after: tuple


class Project:
    def __init__(self, images=(), path=None):
        self.images = [os.path.abspath(p) for p in images]
        self.path = path
        self.classes = [{"name": "object", "color": class_color(0)}]
        self.annotations = {}  # image path -> list of annotations
        self.dirty = False
        self._undo = deque(maxlen=HISTORY_LIMIT)
        self._redo = []

    @classmethod
    def open(cls, images, path=None):
        """Project for these images, merged with the project file next to them if any."""
        project = cls(images)
        project.path = path or default_project_path(project.images)
        if os.path.isfile(project.path):
            project._load(project.path)
        return project

    def _load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("classes"):
            self.classes = [{"name": c["name"], "color": c.get("color") or class_color(i)}
                            for i, c in enumerate(data["classes"])]
        base = os.path.dirname(path)
        for rel, anns in data.get("annotations", {}).items():
            self.annotations[os.path.normpath(os.path.join(base, rel))] = anns

    def save(self, path=None):
        path = path or self.path
        base = os.path.dirname(path)
        data = {
            "version": FORMAT_VERSION,
            "classes": self.classes,
            "annotations": {os.path.relpath(p, base).replace(os.sep, "/"): anns
                            for p, anns in self.annotations.items() if anns},
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, path)  # atomic, so a crash never leaves a half-written file
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

    def class_counts(self):
        counts = [0] * len(self.classes)
        for anns in self.annotations.values():
            for ann in anns:
                if 0 <= ann["class_id"] < len(counts):
                    counts[ann["class_id"]] += 1
        return counts

    # -- annotation edits --
    def set_annotations(self, image_path, anns):
        before = self.get(image_path)
        self.annotations[image_path] = list(anns)
        self._record(_ImageEdit(image_path, before, self.annotations[image_path]))

    def add_annotation(self, image_path, ann):
        self.set_annotations(image_path, self.get(image_path) + [ann])

    def replace_annotation(self, image_path, index, ann):
        anns = list(self.get(image_path))
        anns[index] = ann
        self.set_annotations(image_path, anns)

    def remove_annotation(self, image_path, index):
        anns = self.get(image_path)
        self.set_annotations(image_path, anns[:index] + anns[index + 1:])

    # -- class edits --
    def add_class(self, name):
        with self._project_edit():
            self.classes.append({"name": name, "color": class_color(len(self.classes))})
        return len(self.classes) - 1

    def rename_class(self, class_id, name):
        with self._project_edit():
            self.classes[class_id] = dict(self.classes[class_id], name=name)

    def delete_class(self, class_id):
        """Remove a class and its annotations; higher class ids shift down by one."""
        with self._project_edit():
            del self.classes[class_id]
            for path, anns in self.annotations.items():
                if any(a["class_id"] >= class_id for a in anns):
                    self.annotations[path] = [dict(a, class_id=a["class_id"] - (a["class_id"] > class_id))
                                              for a in anns if a["class_id"] != class_id]

    # -- history --
    def undo(self):
        """Revert the last edit. Returns the image path it touched, None for a
        project-wide edit, or False if there is nothing to undo."""
        if not self._undo:
            return False
        edit = self._undo.pop()
        self._redo.append(edit)
        return self._apply(edit, edit.before)

    def redo(self):
        if not self._redo:
            return False
        edit = self._redo.pop()
        self._undo.append(edit)
        return self._apply(edit, edit.after)

    def _apply(self, edit, state):
        self.dirty = True
        if isinstance(edit, _ImageEdit):
            self.annotations[edit.path] = state
            return edit.path
        classes, annotations = state
        self.classes = [dict(c) for c in classes]
        self.annotations = dict(annotations)
        return None

    def _record(self, edit):
        self._undo.append(edit)
        self._redo.clear()
        self.dirty = True

    def _snapshot(self):
        return [dict(c) for c in self.classes], dict(self.annotations)

    @contextlib.contextmanager
    def _project_edit(self):
        before = self._snapshot()
        yield
        self._record(_ProjectEdit(before, self._snapshot()))

def default_project_path(image_paths):
    folder = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in image_paths])
    return os.path.join(folder, PROJECT_FILENAME)
