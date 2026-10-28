"""The real SAM and SAM 2 architectures with random weights: latency and
coordinate handling are exercised without downloading checkpoints."""
import queue
import sys
import types

import cv2
import numpy as np
import pytest
from PIL import Image

from smartannotator import models
from smartannotator.engine import SamEngine
from smartannotator.models import DeviceInfo
from smartannotator.worker import InferenceWorker

torch = pytest.importorskip("torch")
CPU = DeviceInfo("cpu", "cpu")


def require(key):
    if not models.backend_installed(models.MODELS[key].backend):
        pytest.skip(f"{models.MODELS[key].backend} not installed")


@pytest.mark.parametrize("key", ["sam_vit_b", "sam2.1_tiny"])
def test_embeddings_are_cached(key, no_download):
    require(key)
    engine = SamEngine(key, CPU)
    image_a = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    image_b = np.random.randint(0, 255, (90, 70, 3), np.uint8)
    assert engine.set_image("a", image_a) > 0
    masks, scores, _ = engine.predict([[50, 40]], [1])
    assert masks.shape == (3, 120, 160) and masks.dtype == bool
    assert list(scores) == sorted(scores, reverse=True)
    engine.set_image("b", image_b)
    assert engine.set_image("a", image_a) == 0.0  # served from the cache
    masks, _, _ = engine.predict([[10, 10], [60, 50]], [1, 0], box=[5, 5, 100, 80])
    assert masks.shape == (1, 120, 160)
    engine.set_image("b", image_b)
    masks, _, _ = engine.predict(None, None, box=[5, 5, 60, 80])
    assert masks.shape[1:] == (90, 70)


@pytest.mark.parametrize("key", ["sam_vit_b", "sam2.1_tiny"])
def test_working_resolution_matches_full_resolution(key, no_download):
    """Masks decoded at reduced resolution must line up with full-resolution
    ones; this checks the prompt and size remapping in the adapters."""
    require(key)
    torch.manual_seed(0)
    engine = SamEngine(key, CPU)
    image = np.random.default_rng(0).integers(0, 255, (900, 1500, 3), dtype=np.uint8)
    engine.set_image("img", image)
    for prompt in (dict(points=[[700, 300], [200, 800]], labels=[1, 0]), dict(box=[300, 200, 1200, 700])):
        full, _, scale = engine.predict(**prompt)
        assert scale == (1.0, 1.0)
        small, _, (sx, _) = engine.predict(**prompt, max_side=600)
        assert small.shape[1:] == (360, 600) and sx == pytest.approx(0.4)
        reference = cv2.resize(full[0].astype(np.uint8), (600, 360), interpolation=cv2.INTER_AREA) > 0
        union = (reference | small[0]).sum()
        assert union == 0 or (reference & small[0]).sum() / union > 0.9


def test_edge_sam_adapter(monkeypatch, no_download):
    """Against a stand-in with EdgeSAM's predictor signature: `features` first,
    and a mask count instead of multimask_output."""
    calls = []

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    class Predictor:
        def __init__(self, model):
            self.original_size = None

        def set_image(self, image):
            self.features = torch.zeros(1)
            self.original_size, self.input_size = image.shape[:2], (1024, 1024)

        def predict(self, features=None, point_coords=None, point_labels=None, box=None, mask_input=None,
                    num_multimask_outputs=3, return_logits=False, use_stability_score=False):
            calls.append((num_multimask_outputs, tuple(self.original_size), point_coords.tolist()))
            h, w = self.original_size
            n = num_multimask_outputs
            return np.ones((n, h, w), bool), np.linspace(0.9, 0.5, n), None

    module = types.SimpleNamespace(sam_model_registry={"edge_sam": lambda checkpoint: Model()},
                                   SamPredictor=Predictor)
    monkeypatch.setitem(sys.modules, "edge_sam", module)
    monkeypatch.setattr(models, "backend_installed", lambda backend: True)
    engine = SamEngine("edge_sam", CPU)
    engine.set_image("a", np.zeros((400, 800, 3), np.uint8))
    masks, _, scale = engine.predict([[400, 200]], [1], max_side=400)
    assert masks.shape == (3, 200, 400) and scale == (0.5, 0.5)
    engine.predict([[400, 200], [10, 10]], [1, 0])
    assert calls == [(3, (200, 400), [[200.0, 100.0]]), (1, (400, 800), [[400.0, 200.0], [10.0, 10.0]])]
    assert engine._adapter.predictor.original_size == (400, 800)  # restored after each call


def test_worker_end_to_end(tmp_path, no_download):
    require("sam_vit_b")
    path = str(tmp_path / "x.png")
    Image.fromarray(np.random.randint(0, 255, (100, 100, 3), np.uint8)).save(path)
    events = queue.Queue()
    worker = InferenceWorker(lambda *event: events.put(event))
    worker.start()
    worker.load_model("sam_vit_b", "cpu")
    worker.set_current_image(path)
    worker.predict(1, path, [(50, 50)], [1], None, 1.0)
    seen = {}
    while "prediction" not in seen:
        event = events.get(timeout=120)
        assert event[0] not in ("error", "model_error"), event
        seen[event[0]] = event
    worker.stop()
    assert seen["model_ready"][1] == "sam_vit_b"
    assert seen["prediction"][1:3] == (1, path)
