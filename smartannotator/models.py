"""Model registry, hardware detection and checkpoint management."""
import importlib.util
import os
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass

try:
    import torch
except ImportError:  # the UI still starts and reports what is missing
    torch = None

CHECKPOINT_DIR = os.environ.get(
    "SMARTANNOTATOR_CHECKPOINTS", os.path.join(os.path.expanduser("~"), ".cache", "smartannotator"))

_SAM1_URL = "https://dl.fbaipublicfiles.com/segment_anything/"
_SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    backend: str        # key into BACKEND_MODULES / engine.ADAPTERS
    arch: str           # registry key (SAM 1 style) or Hydra config (SAM 2 style)
    checkpoint: str
    url: str
    size_mb: int
    notes: str
    experimental: bool = False  # never chosen by "auto"


MODELS = OrderedDict((m.key, m) for m in [
    ModelSpec("mobile_sam", "MobileSAM", "mobile_sam", "vit_t", "mobile_sam.pt",
              "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
              39, "Fastest of the stable models. Good for CPU-only machines."),
    ModelSpec("sam2.1_tiny", "SAM 2.1 Tiny", "sam2", "configs/sam2.1/sam2.1_hiera_t.yaml",
              "sam2.1_hiera_tiny.pt", _SAM2_URL + "sam2.1_hiera_tiny.pt",
              149, "Very fast, good quality. CPU and 2-4 GB GPUs."),
    ModelSpec("sam2.1_small", "SAM 2.1 Small", "sam2", "configs/sam2.1/sam2.1_hiera_s.yaml",
              "sam2.1_hiera_small.pt", _SAM2_URL + "sam2.1_hiera_small.pt",
              176, "Default for 4 GB GPUs (GTX 1650, RTX 3050 4 GB)."),
    ModelSpec("sam2.1_base_plus", "SAM 2.1 Base+", "sam2", "configs/sam2.1/sam2.1_hiera_b+.yaml",
              "sam2.1_hiera_base_plus.pt", _SAM2_URL + "sam2.1_hiera_base_plus.pt",
              309, "High quality. 6 GB+ GPUs."),
    ModelSpec("sam2.1_large", "SAM 2.1 Large", "sam2", "configs/sam2.1/sam2.1_hiera_l.yaml",
              "sam2.1_hiera_large.pt", _SAM2_URL + "sam2.1_hiera_large.pt",
              857, "Best quality. 10 GB+ GPUs."),
    ModelSpec("sam_vit_b", "SAM ViT-B", "segment_anything", "vit_b",
              "sam_vit_b_01ec64.pth", _SAM1_URL + "sam_vit_b_01ec64.pth",
              358, "Original SAM. Fits 4 GB GPUs."),
    ModelSpec("sam_vit_l", "SAM ViT-L", "segment_anything", "vit_l",
              "sam_vit_l_0b3195.pth", _SAM1_URL + "sam_vit_l_0b3195.pth",
              1191, "Original SAM. 6 GB+ GPUs."),
    ModelSpec("sam_vit_h", "SAM ViT-H", "segment_anything", "vit_h",
              "sam_vit_h_4b8939.pth", _SAM1_URL + "sam_vit_h_4b8939.pth",
              2445, "Original SAM. 8 GB+ GPUs, slow."),
    # Distilled: small encoders trained to imitate SAM / SAM 2
    ModelSpec("edge_sam", "EdgeSAM", "edge_sam", "edge_sam", "edge_sam_3x.pth",
              "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth",
              38, "Distilled SAM. Fastest encoder.", experimental=True),
    ModelSpec("edgetam", "EdgeTAM", "edgetam", "configs/edgetam.yaml", "edgetam.pt",
              "https://github.com/facebookresearch/EdgeTAM/raw/refs/heads/main/checkpoints/edgetam.pt",
              54, "Distilled SAM 2.", experimental=True),
])

# Auto-selection: preference order per minimum VRAM (GB), best first.
# A GTX 1650 Ti reports ~3.8 GB and lands in the 3 GB tier.
GPU_TIERS = [
    (10.0, ["sam2.1_large", "sam2.1_base_plus", "sam_vit_h", "sam_vit_l", "sam2.1_small", "sam_vit_b"]),
    (6.0, ["sam2.1_base_plus", "sam2.1_small", "sam_vit_l", "sam_vit_b", "sam2.1_tiny"]),
    (3.0, ["sam2.1_small", "sam2.1_tiny", "sam_vit_b", "mobile_sam"]),
    (0.0, ["sam2.1_tiny", "mobile_sam", "sam_vit_b"]),
]
CPU_ORDER = ["mobile_sam", "sam2.1_tiny", "sam2.1_small", "sam_vit_b"]

BACKEND_MODULES = {
    "sam2": "sam2",
    "segment_anything": "segment_anything",
    "mobile_sam": "mobile_sam",
    "edge_sam": "edge_sam",
    "edgetam": "sam2",  # EdgeTAM ships as a fork of the sam2 package
}
INSTALL_HINTS = {
    "sam2": "pip install sam2",
    "segment_anything": "pip install segment-anything",
    "mobile_sam": "pip install git+https://github.com/ChaoningZhang/MobileSAM.git",
    "edge_sam": "pip install git+https://github.com/chongzhou96/EdgeSAM.git",
    "edgetam": "pip install git+https://github.com/facebookresearch/EdgeTAM.git",
}


@dataclass
class DeviceInfo:
    device: str           # "cuda", "mps" or "cpu"
    name: str
    vram_gb: float = 0.0
    fp16: bool = False    # half precision is fast on this device

    def describe(self):
        if self.device == "cuda":
            return f"{self.name} ({self.vram_gb:.1f} GB{', fp16' if self.fp16 else ''})"
        return self.name


def detect_device(preferred="auto"):
    if torch is None:
        return DeviceInfo("cpu", "CPU (PyTorch missing)")
    if preferred in ("auto", "cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Turing (GTX 16xx, RTX 20xx) and newer run fp16 at full rate; Pascal does not
        return DeviceInfo("cuda", props.name, props.total_memory / 1024 ** 3, fp16=props.major >= 7)
    if preferred in ("auto", "mps") and torch.backends.mps.is_available():
        return DeviceInfo("mps", "Apple GPU (MPS)")
    return DeviceInfo("cpu", f"CPU ({os.cpu_count()} threads)")


def backend_installed(backend):
    if backend == "edgetam":
        # Same module name as SAM 2, so look for EdgeTAM's config instead
        spec = importlib.util.find_spec("sam2")
        return spec is not None and any(os.path.isfile(os.path.join(d, MODELS["edgetam"].arch))
                                        for d in spec.submodule_search_locations or [])
    return importlib.util.find_spec(BACKEND_MODULES[backend]) is not None


def available_models():
    return [k for k, m in MODELS.items() if backend_installed(m.backend)]


def candidate_models(device_info):
    """Installed, non-experimental models, most suitable for this hardware first."""
    if device_info.device == "cuda":
        order = next(keys for min_gb, keys in GPU_TIERS if device_info.vram_gb >= min_gb)
    elif device_info.device == "mps":
        order = GPU_TIERS[2][1]
    else:
        order = CPU_ORDER
    installed = [k for k in available_models() if not MODELS[k].experimental]
    ranked = [k for k in order if k in installed]
    # Outside the preferred list, models already on disk beat ones needing a download
    rest = sorted((k for k in installed if k not in ranked), key=lambda k: find_checkpoint(MODELS[k]) is None)
    return ranked + rest


def find_checkpoint(spec):
    for folder in (CHECKPOINT_DIR, os.path.join(os.getcwd(), "checkpoints"), os.getcwd()):
        path = os.path.join(folder, spec.checkpoint)
        if os.path.isfile(path):
            return path
    return None


def ensure_checkpoint(spec, progress=None):
    """Local path of the checkpoint, downloading it first if needed.
    `progress(fraction, message)` is called while downloading."""
    path = find_checkpoint(spec)
    if path:
        return path
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, spec.checkpoint)
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(spec.url, timeout=30) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done, last_report = 0, 0.0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if progress and time.monotonic() - last_report > 0.2:
                    last_report = time.monotonic()
                    progress(done / total if total else 0.0,
                             f"Downloading {spec.label}: {done >> 20} / {total >> 20 or '?'} MB")
        os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Could not download {spec.url} ({e}). "
                           f"Download it manually into {CHECKPOINT_DIR}") from e
    return path
