# sam_integration.py (SAM model handling: model zoo, hardware-aware selection,
# checkpoint download, cached image embeddings and a background inference worker)
import contextlib
import importlib.util
import os
import threading
import time
import traceback
import urllib.request
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from annotation_processing import mask_to_polygons, work_resolution, work_side
from image_io import ImageCache

try:
    import torch
except ImportError:  # the GUI still starts and explains what is missing
    torch = None

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.environ.get("SMARTANNOTATOR_CHECKPOINTS", os.path.join(PROJECT_DIR, "checkpoints"))

_SAM1_URL = "https://dl.fbaipublicfiles.com/segment_anything/"
_SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    backend: str      # "sam2", "segment_anything" or "mobile_sam"
    arch: str         # registry key (SAM 1 / MobileSAM) or hydra config (SAM 2)
    checkpoint: str
    url: str
    size_mb: int
    notes: str


MODELS = OrderedDict((m.key, m) for m in [
    ModelSpec("mobile_sam", "MobileSAM", "mobile_sam", "vit_t", "mobile_sam.pt",
              "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
              39, "Fastest. Best choice for CPU-only machines."),
    ModelSpec("sam2.1_tiny", "SAM 2.1 Tiny", "sam2", "configs/sam2.1/sam2.1_hiera_t.yaml",
              "sam2.1_hiera_tiny.pt", _SAM2_URL + "sam2.1_hiera_tiny.pt",
              149, "Very fast, good quality. Great on CPU and 2-4 GB GPUs."),
    ModelSpec("sam2.1_small", "SAM 2.1 Small", "sam2", "configs/sam2.1/sam2.1_hiera_s.yaml",
              "sam2.1_hiera_small.pt", _SAM2_URL + "sam2.1_hiera_small.pt",
              176, "Recommended for 4 GB GPUs (GTX 1650 / 1650 Ti, RTX 3050 4GB)."),
    ModelSpec("sam2.1_base_plus", "SAM 2.1 Base+", "sam2", "configs/sam2.1/sam2.1_hiera_b+.yaml",
              "sam2.1_hiera_base_plus.pt", _SAM2_URL + "sam2.1_hiera_base_plus.pt",
              309, "High quality. 6 GB+ GPUs."),
    ModelSpec("sam2.1_large", "SAM 2.1 Large", "sam2", "configs/sam2.1/sam2.1_hiera_l.yaml",
              "sam2.1_hiera_large.pt", _SAM2_URL + "sam2.1_hiera_large.pt",
              857, "Best quality. 10 GB+ GPUs."),
    ModelSpec("sam_vit_b", "SAM ViT-B", "segment_anything", "vit_b",
              "sam_vit_b_01ec64.pth", _SAM1_URL + "sam_vit_b_01ec64.pth",
              358, "Original SAM, base. Fits 4 GB GPUs."),
    ModelSpec("sam_vit_l", "SAM ViT-L", "segment_anything", "vit_l",
              "sam_vit_l_0b3195.pth", _SAM1_URL + "sam_vit_l_0b3195.pth",
              1191, "Original SAM, large. 6 GB+ GPUs."),
    ModelSpec("sam_vit_h", "SAM ViT-H", "segment_anything", "vit_h",
              "sam_vit_h_4b8939.pth", _SAM1_URL + "sam_vit_h_4b8939.pth",
              2445, "Original SAM, huge. 8 GB+ GPUs; slow."),
])

# Preference order per hardware tier (minimum VRAM in GB -> models, best first).
# A GTX 1650 Ti reports ~3.8 GB, so it lands in the 3 GB tier.
_GPU_TIERS = [
    (10.0, ["sam2.1_large", "sam2.1_base_plus", "sam_vit_h", "sam_vit_l", "sam2.1_small", "sam_vit_b"]),
    (6.0, ["sam2.1_base_plus", "sam2.1_small", "sam_vit_l", "sam_vit_b", "sam2.1_tiny"]),
    (3.0, ["sam2.1_small", "sam2.1_tiny", "sam_vit_b", "mobile_sam"]),
    (0.0, ["sam2.1_tiny", "mobile_sam", "sam_vit_b"]),
]
_CPU_ORDER = ["mobile_sam", "sam2.1_tiny", "sam2.1_small", "sam_vit_b"]

_BACKEND_MODULES = {"sam2": "sam2", "segment_anything": "segment_anything", "mobile_sam": "mobile_sam"}
_BACKEND_INSTALL = {
    "sam2": "pip install sam2",
    "segment_anything": "pip install segment-anything",
    "mobile_sam": "pip install git+https://github.com/ChaoningZhang/MobileSAM.git",
}


# --- Hardware detection -----------------------------------------------------

@dataclass
class DeviceInfo:
    device: str           # "cuda", "mps" or "cpu"
    name: str
    vram_gb: float = 0.0
    fp16: bool = False    # fast half precision available

    def describe(self):
        if self.device == "cuda":
            return f"{self.name} ({self.vram_gb:.1f} GB{', fp16' if self.fp16 else ''})"
        return self.name


def detect_device(preferred="auto"):
    if torch is None:
        return DeviceInfo("cpu", "CPU (PyTorch missing)")
    if preferred in ("auto", "cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Turing (GTX 16xx, RTX 20xx) and newer run fp16 at full speed; Pascal does not.
        return DeviceInfo("cuda", props.name, props.total_memory / 1024 ** 3, fp16=props.major >= 7)
    if preferred in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("mps", "Apple GPU (MPS)")
    return DeviceInfo("cpu", f"CPU ({os.cpu_count()} threads)")


def backend_installed(backend):
    return importlib.util.find_spec(_BACKEND_MODULES[backend]) is not None


def install_hint(backend):
    return _BACKEND_INSTALL[backend]


def available_models():
    return [k for k, m in MODELS.items() if backend_installed(m.backend)]


def candidate_models(device_info):
    """Installed models ordered from most to least suitable for this hardware."""
    if device_info.device == "cuda":
        order = next(models for min_gb, models in _GPU_TIERS if device_info.vram_gb >= min_gb)
    elif device_info.device == "mps":
        order = _GPU_TIERS[2][1]
    else:
        order = _CPU_ORDER
    installed = available_models()
    ranked = [k for k in order if k in installed]
    # Models already on disk are a better fallback than ones needing a download
    rest = sorted((k for k in installed if k not in ranked), key=lambda k: find_checkpoint(MODELS[k]) is None)
    return ranked + rest


def recommend_model(device_info=None):
    candidates = candidate_models(device_info or detect_device())
    return candidates[0] if candidates else None


# --- Checkpoints ----------------------------------------------------------

def find_checkpoint(spec):
    for folder in (CHECKPOINT_DIR, PROJECT_DIR, os.getcwd()):
        path = os.path.join(folder, spec.checkpoint)
        if os.path.isfile(path):
            return path
    return None


def ensure_checkpoint(spec, progress=None):
    """Return a local checkpoint path, downloading it if needed.
    `progress(fraction, message)` is called during the download."""
    path = find_checkpoint(spec)
    if path:
        return path
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, spec.checkpoint)
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(spec.url, timeout=30) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done, last = 0, 0.0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress and time.time() - last > 0.2:
                    last = time.time()
                    frac = done / total if total else 0.0
                    progress(frac, f"Downloading {spec.label}: {done >> 20} / {total >> 20 or '?'} MB")
        os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Could not download {spec.checkpoint} from {spec.url} ({e}). "
                           f"Download it manually into {CHECKPOINT_DIR}") from e
    return path


# --- Backend adapters -----------------------------------------------------

class _Sam1Adapter:
    """segment_anything and MobileSAM share the same predictor API."""

    def __init__(self, spec, checkpoint, device):
        module = __import__(_BACKEND_MODULES[spec.backend])
        model = module.sam_model_registry[spec.arch](checkpoint=checkpoint)
        model.to(device=device).eval()
        self.predictor = module.SamPredictor(model)

    def set_image(self, image):
        self.predictor.set_image(image)

    def get_state(self):
        p = self.predictor
        return (p.features, p.original_size, p.input_size)

    def set_state(self, state):
        p = self.predictor
        p.features, p.original_size, p.input_size = state
        p.is_image_set = True

    def features_finite(self):
        return bool(torch.isfinite(self.predictor.features).all())

    def predict(self, points, labels, box, multimask, out_hw):
        # Prompts arrive in out_hw coordinates. Pretending the original image is
        # out_hw makes SAM map them correctly and upsample masks only to out_hw
        # instead of the full photo, which is what makes large images fast.
        p = self.predictor
        full = p.original_size
        p.original_size = tuple(out_hw)
        try:
            masks, scores, _ = p.predict(point_coords=points, point_labels=labels, box=box,
                                         multimask_output=multimask)
        finally:
            p.original_size = full
        return masks, scores


class _Sam2Adapter:
    def __init__(self, spec, checkpoint, device):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(spec.arch, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image):
        self.predictor.set_image(image)

    def get_state(self):
        p = self.predictor
        return (p._features, p._orig_hw)

    def set_state(self, state):
        p = self.predictor
        p._features, p._orig_hw = state
        p._is_image_set = True
        p._is_batch = False

    def features_finite(self):
        f = self.predictor._features
        return all(bool(torch.isfinite(t).all()) for t in [f["image_embed"], *f["high_res_feats"]])

    def predict(self, points, labels, box, multimask, out_hw):
        p = self.predictor  # same trick as _Sam1Adapter.predict
        full = p._orig_hw
        p._orig_hw = [tuple(out_hw)]
        try:
            masks, scores, _ = p.predict(point_coords=points, point_labels=labels, box=box,
                                         multimask_output=multimask)
        finally:
            p._orig_hw = full
        return masks > 0.5 if masks.dtype != bool else masks, scores


ADAPTERS = {"sam2": _Sam2Adapter, "segment_anything": _Sam1Adapter, "mobile_sam": _Sam1Adapter}


class SamEngine:
    """A loaded SAM model with an LRU cache of image embeddings.

    The heavy image encoder runs once per image; every click afterwards only
    runs the lightweight mask decoder (milliseconds, even on a GTX 1650)."""

    def __init__(self, model_key, device_info=None, precision="auto", checkpoint=None,
                 progress=None, cache_size=8):
        if torch is None:
            raise RuntimeError("PyTorch is not installed. See README for installation steps.")
        self.spec = MODELS[model_key]
        if not backend_installed(self.spec.backend):
            raise RuntimeError(f"{self.spec.label} needs: {install_hint(self.spec.backend)}")
        self.device_info = device_info or detect_device()
        self.device = self.device_info.device
        self.use_fp16 = precision == "fp16" or (precision == "auto" and self.device_info.fp16)
        if self.device != "cuda":
            self.use_fp16 = False
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        if checkpoint is None:
            checkpoint = ensure_checkpoint(self.spec, progress)
        if progress:
            progress(None, f"Loading {self.spec.label} on {self.device_info.describe()}...")
        self._adapter = ADAPTERS[self.spec.backend](self.spec, checkpoint, self.device)
        self._cache = OrderedDict()
        self._sizes = {}
        self._cache_size = cache_size
        self._lock = threading.RLock()
        self.current_key = None

    @property
    def precision(self):
        return "fp16" if self.use_fp16 else "fp32"

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def is_cached(self, key):
        return key in self._cache

    def set_image(self, key, image):
        """Make `image` the active image. Returns encoder time in seconds (0 if cached)."""
        with self._lock:
            if key == self.current_key:
                return 0.0
            state = self._cache.get(key)
            if state is not None:
                self._cache.move_to_end(key)
                self._adapter.set_state(state)
                self.current_key = key
                return 0.0
            start = time.perf_counter()
            self.current_key = None
            with torch.inference_mode(), self._autocast():
                self._adapter.set_image(image)
            if self.use_fp16 and not self._adapter.features_finite():
                # fp16 overflowed on this model/GPU: fall back to fp32 for good
                self.use_fp16 = False
                with torch.inference_mode():
                    self._adapter.set_image(image)
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.current_key = key
            self._cache[key] = self._adapter.get_state()
            self._sizes[key] = image.shape[:2]
            while len(self._cache) > self._cache_size:
                self._sizes.pop(self._cache.popitem(last=False)[0], None)
            return time.perf_counter() - start

    def predict(self, points=None, labels=None, box=None, multimask=None, max_side=None):
        """Run the mask decoder on the active image.

        Prompts are in full-image pixels. With `max_side`, masks come back at a
        reduced working resolution (longest side <= max_side).
        Returns (masks[N, h, w] bool, scores[N], (sx, sy)) sorted best first, where
        (sx, sy) maps image coordinates to mask coordinates."""
        with self._lock:
            if self.current_key is None:
                raise RuntimeError("No image set")
            h, w = self._sizes[self.current_key]
            out_h, out_w = work_resolution(h, w, max_side) if max_side else (h, w)
            sx, sy = out_w / w, out_h / h
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) * (sx, sy) if points else None
            lbl = np.asarray(labels, dtype=np.int32) if points else None
            bx = np.asarray(box, dtype=np.float32) * (sx, sy, sx, sy) if box is not None else None
            if multimask is None:
                # A single click is ambiguous (part vs whole), so ask for 3 candidates
                multimask = bx is None and pts is not None and len(pts) == 1
            with torch.inference_mode(), self._autocast():
                masks, scores = self._adapter.predict(pts, lbl, bx, multimask, (out_h, out_w))
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            order = np.argsort(-scores)
            return np.asarray(masks)[order].astype(bool), scores[order], (sx, sy)

    def release(self):
        with self._lock:
            self._cache.clear()
            self._adapter = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


# Legacy API ---------------------------------------------------------------

def load_sam_model(device=None, model_key=None):
    info = detect_device(device or "auto")
    return SamEngine(model_key or recommend_model(info), info)


def generate_masks(engine, image_array, input_point, input_label=np.array([1])):
    engine.set_image(id(image_array), image_array)
    masks, _, _ = engine.predict(np.asarray(input_point).tolist(), np.asarray(input_label).tolist())
    return masks[:1]


# --- Background worker ----------------------------------------------------

PREFETCH_IDLE_SECONDS = 0.8


class InferenceWorker(threading.Thread):
    """Runs model loading, image encoding and mask prediction off the UI thread.

    Requests are coalesced: only the newest prediction (and newest hover
    preview) is kept, and prefetches for images the user has already skipped
    past are dropped. Results are sent through `emit(event, *payload)` from this
    thread; the GUI marshals them to Tk."""

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
        self._stopping = False
        self._last_interaction = 0.0

    # -- requests (called from the UI thread) --
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

    def predict(self, request_id, path, points, labels, box, detail):
        with self._cond:
            self._predict = (request_id, path, points, labels, box, detail)
            self._hover = None
            self._last_interaction = time.monotonic()
            self._cond.notify()

    def hover(self, request_id, path, point, detail):
        """Preview what a click at `point` would select. Only runs when the
        worker is otherwise idle and the image is already encoded."""
        with self._cond:
            self._hover = (request_id, path, point, detail)
            self._last_interaction = time.monotonic()
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stopping = True
            self._cond.notify()

    # -- worker loop --
    def _next_job(self):
        with self._cond:
            while True:
                if self._stopping:
                    return None
                if self._load:
                    job, self._load = ("load", self._load), None
                    return job
                if self.engine is not None:
                    if self._predict:
                        job, self._predict = ("predict", self._predict), None
                        return job
                    if self._encode:
                        job, self._encode = ("encode", (self._encode, False)), None
                        return job
                    if self._hover:
                        job, self._hover = self._hover, None
                        if self.engine.is_cached(job[1]):
                            return ("hover", job)
                    # Pre-encode the next image only once the user pauses, so a
                    # long encode never delays clicks or hover previews.
                    idle = time.monotonic() - self._last_interaction
                    if self._prefetch and idle < PREFETCH_IDLE_SECONDS:
                        self._cond.wait(PREFETCH_IDLE_SECONDS - idle)
                        continue
                    while self._prefetch:
                        path = self._prefetch.popleft()
                        if not self.engine.is_cached(path):
                            return ("encode", (path, True))
                self._cond.wait()

    def run(self):
        while True:
            job = self._next_job()
            if job is None:
                return
            kind, args = job
            try:
                getattr(self, "_do_" + kind)(*args)
            except Exception as e:
                traceback.print_exc()
                self.emit("model_error" if kind == "load" else "error", f"{type(e).__name__}: {e}")

    def _do_load(self, model_key, device, precision):
        if self.engine is not None:
            self.engine.release()
            self.engine = None
        info = detect_device(device)
        keys = candidate_models(info) if model_key == "auto" else [model_key]
        if not keys:
            raise RuntimeError("No SAM backend installed. Run: pip install sam2  (or segment-anything)")
        errors = []
        for key in keys:
            self.emit("model_status", None, f"Preparing {MODELS[key].label}...")
            try:
                self.engine = SamEngine(key, info, precision,
                                        progress=lambda f, m: self.emit("model_status", f, m))
                break
            except Exception as e:
                errors.append(f"{MODELS[key].label}: {e}")
                if model_key != "auto":
                    raise
        if self.engine is None:
            raise RuntimeError("No model could be loaded:\n" + "\n".join(errors))
        self.emit("model_ready", self.engine.spec.key, info.describe(), self.engine.precision)
        with self._cond:  # re-encode whatever image is on screen with the new model
            self._cond.notify()

    def _do_encode(self, path, prefetch):
        if not prefetch:
            self.emit("encoding", path)
        image = self.images.get(path)
        seconds = self.engine.set_image(path, image)
        if not prefetch:
            self.emit("encoded", path, seconds)

    def _segment(self, path, points, labels, box, detail, multimask=None, best_only=False):
        masks, scores, scale = self.engine.predict(points, labels, box, multimask=multimask,
                                                   max_side=work_side(detail))
        candidates = []
        for mask, score in zip(masks[:1] if best_only else masks, scores):
            polygons = mask_to_polygons(mask, detail, scale=scale)
            if polygons:
                candidates.append({"polygons": polygons, "score": float(score)})
        return candidates

    def _do_predict(self, request_id, path, points, labels, box, detail):
        if not self.engine.is_cached(path) and self.engine.current_key != path:
            self.emit("encoding", path)
        seconds = self.engine.set_image(path, self.images.get(path))
        if seconds:
            self.emit("encoded", path, seconds)
        start = time.perf_counter()
        candidates = self._segment(path, points, labels, box, detail)
        self.emit("prediction", request_id, path, candidates, time.perf_counter() - start)

    def _do_hover(self, request_id, path, point, detail):
        start = time.perf_counter()
        self.engine.set_image(path, self.images.get(path))
        candidates = self._segment(path, [point], [1], None, detail)
        self.emit("hover_prediction", request_id, path, point, candidates, time.perf_counter() - start)
