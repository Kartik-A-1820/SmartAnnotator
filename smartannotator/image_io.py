"""Image loading shared by the UI, the inference worker and the exporter.

All coordinates in the app refer to the EXIF-corrected image, which is also
what OpenCV and Ultralytics see when training.
"""
import os
import re
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

_EXIF_ORIENTATION = 0x0112
_TRANSPOSED_ORIENTATIONS = (5, 6, 7, 8)  # orientations that swap width and height


def is_image_file(path):
    return path.lower().endswith(IMAGE_EXTENSIONS)


def list_images(folder):
    """Image files in `folder` (non-recursive) in natural order: img2 before img10."""
    names = [n for n in os.listdir(folder) if is_image_file(n)]
    return [os.path.join(folder, n) for n in sorted(names, key=_natural_key)]


def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def exif_orientation(path):
    try:
        with Image.open(path) as img:
            return img.getexif().get(_EXIF_ORIENTATION, 1)
    except OSError:
        return 1


def oriented_size(path):
    """(width, height) after EXIF rotation. Reads only the file header."""
    with Image.open(path) as img:
        w, h = img.size
        if img.getexif().get(_EXIF_ORIENTATION, 1) in _TRANSPOSED_ORIENTATIONS:
            w, h = h, w
    return w, h


def read_rgb(path):
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


class ImageCache:
    """Thread-safe LRU cache of decoded images, so the display and the model
    worker never decode the same file twice."""

    def __init__(self, max_items=4):
        self.max_items = max_items
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path):
        with self._lock:
            if path in self._items:
                self._items.move_to_end(path)
                return self._items[path]
        image = read_rgb(path)  # decode outside the lock
        with self._lock:
            self._items[path] = image
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return image

    def clear(self):
        with self._lock:
            self._items.clear()
