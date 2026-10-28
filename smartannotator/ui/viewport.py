"""View transform and frame rendering, independent of Tk.

Only the visible part of the image is ever resampled, and large images keep a
downscaled copy for zoomed-out views, so a 24 MP photo pans as smoothly as a
small one.
"""
import math

from PIL import Image

PREVIEW_MAX_SIDE = 2560
MAX_SCALE = 32.0
MIN_SCALE_OF_FIT = 0.25


class Viewport:
    """canvas = (image - origin) * scale"""

    def __init__(self):
        self.scale = 1.0
        self.ox = self.oy = 0.0
        self.auto_fit = True  # refit when the window is resized

    def to_image(self, cx, cy):
        return self.ox + cx / self.scale, self.oy + cy / self.scale

    def to_canvas(self, ix, iy):
        return (ix - self.ox) * self.scale, (iy - self.oy) * self.scale

    def fit(self, image_size, canvas_size, margin=0.98):
        (w, h), (cw, ch) = image_size, canvas_size
        self.scale = min(cw / w, ch / h) * margin
        self.ox = (w - cw / self.scale) / 2
        self.oy = (h - ch / self.scale) / 2
        self.auto_fit = True

    def zoom_at(self, cx, cy, factor, image_size, canvas_size):
        """Zoom keeping the image point under (cx, cy) fixed."""
        (w, h), (cw, ch) = image_size, canvas_size
        fit = min(cw / w, ch / h)
        ix, iy = self.to_image(cx, cy)
        self.scale = min(max(self.scale * factor, fit * MIN_SCALE_OF_FIT), MAX_SCALE)
        self.ox, self.oy = ix - cx / self.scale, iy - cy / self.scale
        self.auto_fit = False

    def pan(self, dx, dy):
        """Move the view by a canvas-pixel delta."""
        self.ox -= dx / self.scale
        self.oy -= dy / self.scale
        self.auto_fit = False

    def visible_region(self, image_size, canvas_size):
        """(x0, y0, x1, y1) of the image that is on screen, or None."""
        (w, h), (cw, ch) = image_size, canvas_size
        x0, y0 = max(0.0, self.ox), max(0.0, self.oy)
        x1, y1 = min(float(w), self.ox + cw / self.scale), min(float(h), self.oy + ch / self.scale)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


class FrameRenderer:
    """Produces the RGBA background frame for the visible region. The result is
    cached, so overlay-only redraws (hover, mask preview) skip resampling."""

    def __init__(self, image):
        self.image = image
        self.size = image.size
        self._preview = None  # (scale, downscaled copy)
        factor = math.ceil(max(image.size) / PREVIEW_MAX_SIDE)
        if factor >= 2:
            self._preview = (1.0 / factor, image.reduce(factor))
        self._key = None
        self._frame = None

    def frame(self, region, size, scale):
        key = (region, size)
        if key != self._key:
            src, f = self.image, 1.0
            if self._preview and scale <= self._preview[0]:
                f, src = self._preview
            x0, y0, x1, y1 = region
            resample = Image.NEAREST if scale >= 4 else Image.BILINEAR
            self._frame = src.resize(size, resample, box=(x0 * f, y0 * f, x1 * f, y1 * f),
                                     reducing_gap=2.0 if scale / f < 0.5 else None).convert("RGBA")
            self._key = key
        return self._frame
