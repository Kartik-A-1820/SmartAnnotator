"""SAM inference with cached image embeddings.

The image encoder (the expensive part) runs once per image and its output is
kept in an LRU cache. Each click then only runs the mask decoder, and masks are
produced at a working resolution instead of full photo size.
"""
import contextlib
import threading
import time
from collections import OrderedDict

import numpy as np

from . import models
from .geometry import work_resolution

torch = models.torch


class _Sam1Adapter:
    """segment_anything and MobileSAM share this predictor API."""

    def __init__(self, spec, checkpoint, device):
        module = __import__(models.BACKEND_MODULES[spec.backend])
        model = module.sam_model_registry[spec.arch](checkpoint=checkpoint)
        model.to(device=device).eval()
        self.predictor = module.SamPredictor(model)

    def set_image(self, image):
        self.predictor.set_image(image)

    def get_state(self):
        p = self.predictor
        return p.features, p.original_size, p.input_size

    def set_state(self, state):
        p = self.predictor
        p.features, p.original_size, p.input_size = state
        p.is_image_set = True

    def features_finite(self):
        return bool(torch.isfinite(self.predictor.features).all())

    def predict(self, points, labels, box, multimask, out_hw):
        # Prompts arrive in out_hw coordinates. Telling the predictor the image is
        # out_hw makes it map prompts correctly and upsample masks only to out_hw.
        with _swapped(self.predictor, "original_size", tuple(out_hw)):
            masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, box=box,
                                                      multimask_output=multimask)
        return masks, scores


class _EdgeSamAdapter(_Sam1Adapter):
    """EdgeSAM takes a mask count instead of a multimask flag."""

    def predict(self, points, labels, box, multimask, out_hw):
        with _swapped(self.predictor, "original_size", tuple(out_hw)):
            masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, box=box,
                                                      num_multimask_outputs=3 if multimask else 1)
        return masks, scores


class _Sam2Adapter:
    """SAM 2 / SAM 2.1, and EdgeTAM which keeps the same API."""

    def __init__(self, spec, checkpoint, device):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(build_sam2(spec.arch, checkpoint, device=device))

    def set_image(self, image):
        self.predictor.set_image(image)

    def get_state(self):
        return self.predictor._features, self.predictor._orig_hw

    def set_state(self, state):
        p = self.predictor
        p._features, p._orig_hw = state
        p._is_image_set = True
        p._is_batch = False

    def features_finite(self):
        f = self.predictor._features
        return all(bool(torch.isfinite(t).all()) for t in [f["image_embed"], *f["high_res_feats"]])

    def predict(self, points, labels, box, multimask, out_hw):
        with _swapped(self.predictor, "_orig_hw", [tuple(out_hw)]):
            masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, box=box,
                                                      multimask_output=multimask)
        return masks.astype(bool), scores


ADAPTERS = {
    "sam2": _Sam2Adapter,
    "edgetam": _Sam2Adapter,
    "segment_anything": _Sam1Adapter,
    "mobile_sam": _Sam1Adapter,
    "edge_sam": _EdgeSamAdapter,
}


@contextlib.contextmanager
def _swapped(obj, attr, value):
    saved = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, saved)


class SamEngine:
    def __init__(self, model_key, device_info=None, precision="auto", progress=None, cache_size=8):
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        self.spec = models.MODELS[model_key]
        if not models.backend_installed(self.spec.backend):
            raise RuntimeError(f"{self.spec.label} needs: {models.INSTALL_HINTS[self.spec.backend]}")
        self.device_info = device_info or models.detect_device()
        self.device = self.device_info.device
        self.use_fp16 = self.device == "cuda" and (
            precision == "fp16" or (precision == "auto" and self.device_info.fp16))
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        checkpoint = models.ensure_checkpoint(self.spec, progress)
        if progress:
            progress(None, f"Loading {self.spec.label} on {self.device_info.describe()}...")
        self._adapter = ADAPTERS[self.spec.backend](self.spec, checkpoint, self.device)
        self._cache = OrderedDict()  # image key -> (encoder state, (h, w))
        self._cache_size = cache_size
        self._lock = threading.RLock()
        self.current_key = None

    @property
    def precision(self):
        return "fp16" if self.use_fp16 else "fp32"

    def is_cached(self, key):
        return key in self._cache

    def set_image(self, key, image):
        """Make `image` the active one. Returns the encoder time in seconds (0 if cached)."""
        with self._lock:
            if key == self.current_key:
                return 0.0
            if key in self._cache:
                self._cache.move_to_end(key)
                self._adapter.set_state(self._cache[key][0])
                self.current_key = key
                return 0.0

            start = time.perf_counter()
            self.current_key = None
            self._encode(image)
            if self.use_fp16 and not self._adapter.features_finite():
                self.use_fp16 = False  # fp16 overflowed on this model/GPU; stay on fp32
                self._encode(image)
            if self.device == "cuda":
                torch.cuda.synchronize()
            self._cache[key] = (self._adapter.get_state(), image.shape[:2])
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            self.current_key = key
            return time.perf_counter() - start

    def predict(self, points=None, labels=None, box=None, multimask=None, max_side=None):
        """Run the mask decoder on the active image.

        Prompts are in image pixels. With `max_side`, masks are computed at a
        reduced resolution whose longest side is at most `max_side`.
        Returns (masks [N, h, w] bool, scores [N], (sx, sy)), best score first,
        where (sx, sy) scales image coordinates to mask coordinates."""
        with self._lock:
            if self.current_key is None:
                raise RuntimeError("No image set")
            h, w = self._cache[self.current_key][1]
            out_h, out_w = work_resolution(h, w, max_side) if max_side else (h, w)
            sx, sy = out_w / w, out_h / h
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) * (sx, sy) if points else None
            lbl = np.asarray(labels, dtype=np.int32) if points else None
            bx = np.asarray(box, dtype=np.float32) * (sx, sy, sx, sy) if box is not None else None
            if multimask is None:
                # One click is ambiguous (part or whole?), so ask for several candidates
                multimask = bx is None and pts is not None and len(pts) == 1
            with torch.inference_mode(), self._autocast():
                masks, scores = self._adapter.predict(pts, lbl, bx, multimask, (out_h, out_w))
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            order = np.argsort(-scores)
            return np.asarray(masks, dtype=bool)[order], scores[order], (sx, sy)

    def release(self):
        with self._lock:
            self._cache.clear()
            self._adapter = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode(self, image):
        with torch.inference_mode(), self._autocast():
            self._adapter.set_image(image)

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()
