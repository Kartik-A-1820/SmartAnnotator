"""Background inference thread.

The UI thread never touches the model. It posts requests here and receives
results through an `emit(event, *args)` callback, which the UI turns into Tk
events. Requests are coalesced rather than queued: only the newest prediction
and the newest hover preview are kept, so a burst of mouse movement costs
one decoder run, not fifty.

Priority: load model > click prediction > encode current image > hover preview
> pre-encode the next image. An encode cannot be interrupted, so pre-encoding
waits until the user has paused for about twice the encoder's running time:
almost immediately on a GPU, longer on a slow CPU.
"""
import threading
import time
import traceback
from collections import deque

from . import models
from .engine import SamEngine
from .geometry import mask_to_polygons, work_side
from .image_io import ImageCache

MIN_PREFETCH_IDLE_SECONDS = 0.8


class InferenceWorker(threading.Thread):
    def __init__(self, emit, image_cache=None):
        super().__init__(daemon=True, name="sam-worker")
        self.emit = emit
        self.images = image_cache or ImageCache()
        self.engine = None
        self._cond = threading.Condition()
        self._load = None
        self._predict = None
        self._hover = None
        self._encode = None
        self._prefetch = deque()
        self._last_interaction = 0.0
        self._encode_seconds = 0.0  # last measured encoder time
        self._stopping = False

    # -- requests, called from the UI thread --
    def load_model(self, model_key, device="auto", precision="auto"):
        with self._cond:
            self._load = (model_key, device, precision)
            self._predict = self._hover = None
            self._cond.notify()

    def set_current_image(self, path, prefetch=()):
        with self._cond:
            self._encode = path
            self._hover = None
            self._prefetch = deque(p for p in prefetch if p != path)
            self._cond.notify()

    def predict(self, request_id, path, points, labels, box, tolerance):
        with self._cond:
            self._predict = (request_id, path, points, labels, box, tolerance)
            self._hover = None
            self._last_interaction = time.monotonic()
            self._cond.notify()

    def hover(self, request_id, path, point, tolerance):
        """Preview what a click at `point` would select. Skipped unless the
        image is already encoded, so hovering never triggers an encode."""
        with self._cond:
            self._hover = (request_id, path, point, tolerance)
            self._last_interaction = time.monotonic()
            self._cond.notify()

    @property
    def busy_predicting(self):
        return self._predict is not None

    def stop(self):
        with self._cond:
            self._stopping = True
            self._cond.notify()

    # -- worker thread --
    def run(self):
        while (job := self._next_job()) is not None:
            handler, args = job
            try:
                handler(*args)
            except Exception as e:
                traceback.print_exc()
                event = "model_error" if handler == self._run_load else "error"
                self.emit(event, f"{type(e).__name__}: {e}")

    def _next_job(self):
        with self._cond:
            while not self._stopping:
                if self._load:
                    job, self._load = (self._run_load, self._load), None
                    return job
                if self.engine is None:
                    self._cond.wait()
                    continue
                if self._predict:
                    job, self._predict = (self._run_predict, self._predict), None
                    return job
                if self._encode:
                    job, self._encode = (self._run_encode, (self._encode, False)), None
                    return job
                if self._hover:
                    hover, self._hover = self._hover, None
                    if self.engine.is_cached(hover[1]):
                        return self._run_hover, hover
                if self._prefetch:
                    wait = max(MIN_PREFETCH_IDLE_SECONDS, 2 * self._encode_seconds)
                    idle = time.monotonic() - self._last_interaction
                    if idle < wait:
                        self._cond.wait(wait - idle)
                        continue
                    path = self._prefetch.popleft()
                    if not self.engine.is_cached(path):
                        return self._run_encode, (path, True)
                    continue
                self._cond.wait()
            return None

    def _run_load(self, model_key, device, precision):
        if self.engine is not None:
            self.engine.release()
            self.engine = None
        info = models.detect_device(device)
        keys = models.candidate_models(info) if model_key == "auto" else [model_key]
        if not keys:
            raise RuntimeError("No SAM backend installed. Run: pip install sam2")
        errors = []
        for key in keys:
            self.emit("model_status", None, f"Preparing {models.MODELS[key].label}...")
            try:
                self.engine = SamEngine(key, info, precision,
                                        progress=lambda f, m: self.emit("model_status", f, m))
                break
            except Exception as e:
                if model_key != "auto":
                    raise
                errors.append(f"{models.MODELS[key].label}: {e}")
        if self.engine is None:
            raise RuntimeError("No model could be loaded:\n" + "\n".join(errors))
        self.emit("model_ready", self.engine.spec.key, info.describe(), self.engine.precision)

    def _run_encode(self, path, prefetch):
        if not prefetch:
            self.emit("encoding", path)
        seconds = self.engine.set_image(path, self.images.get(path))
        self._encode_seconds = seconds or self._encode_seconds
        if not prefetch:
            self.emit("encoded", path, seconds)

    def _run_predict(self, request_id, path, points, labels, box, tolerance):
        if not self.engine.is_cached(path):
            self.emit("encoding", path)
        seconds = self.engine.set_image(path, self.images.get(path))
        if seconds:
            self.emit("encoded", path, seconds)
        start = time.perf_counter()
        candidates = self._segment(points, labels, box, tolerance)
        self.emit("prediction", request_id, path, candidates, time.perf_counter() - start)

    def _run_hover(self, request_id, path, point, tolerance):
        self.engine.set_image(path, self.images.get(path))
        self.emit("hover_prediction", request_id, path, point, self._segment([point], [1], None, tolerance))

    def _segment(self, points, labels, box, tolerance):
        """Candidate masks as polygons, best first."""
        masks, scores, scale = self.engine.predict(points, labels, box, max_side=work_side(tolerance))
        candidates = []
        for mask, score in zip(masks, scores, strict=True):
            polygons = mask_to_polygons(mask, tolerance, scale=scale)
            if polygons:
                candidates.append({"polygons": polygons, "score": float(score)})
        return candidates
