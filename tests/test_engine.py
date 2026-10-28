"""Runs the real SAM / SAM 2 architectures with random weights (no download needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import sam_integration as si


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    monkeypatch.setattr(si, "ensure_checkpoint", lambda spec, progress=None: None)


def test_recommendations_for_gtx_1650_ti(monkeypatch):
    monkeypatch.setattr(si, "available_models", lambda: list(si.MODELS))
    gtx = si.DeviceInfo("cuda", "NVIDIA GeForce GTX 1650 Ti", 3.81, fp16=True)
    assert si.candidate_models(gtx)[0] == "sam2.1_small"
    assert "sam_vit_h" not in si.candidate_models(gtx)[:4]
    assert si.candidate_models(si.DeviceInfo("cpu", "cpu"))[0] == "mobile_sam"
    assert si.candidate_models(si.DeviceInfo("cuda", "4090", 24.0, True))[0] == "sam2.1_large"


@pytest.mark.parametrize("key", ["sam_vit_b", "sam2.1_tiny"])
def test_engine_caches_embeddings(key):
    if not si.backend_installed(si.MODELS[key].backend):
        pytest.skip("backend not installed")
    engine = si.SamEngine(key, si.DeviceInfo("cpu", "cpu"))
    img_a = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    img_b = np.random.randint(0, 255, (90, 70, 3), np.uint8)
    assert engine.set_image("a", img_a) > 0
    masks, scores, _ = engine.predict([[50, 40]], [1])
    assert masks.shape == (3, 120, 160) and masks.dtype == bool
    assert list(scores) == sorted(scores, reverse=True)
    engine.set_image("b", img_b)
    assert engine.set_image("a", img_a) == 0.0  # cached: no re-encode
    masks, _, _ = engine.predict([[10, 10], [60, 50]], [1, 0], box=[5, 5, 100, 80])
    assert masks.shape == (1, 120, 160)
    engine.set_image("b", img_b)
    masks, _, _ = engine.predict(None, None, box=[5, 5, 60, 80])
    assert masks.shape[1:] == (90, 70)


def test_worker_end_to_end(tmp_path):
    if not si.backend_installed("segment_anything"):
        pytest.skip("segment_anything not installed")
    import queue
    from PIL import Image
    path = str(tmp_path / "x.png")
    Image.fromarray(np.random.randint(0, 255, (100, 100, 3), np.uint8)).save(path)
    events = queue.Queue()
    worker = si.InferenceWorker(lambda *e: events.put(e))
    worker.start()
    worker.load_model("sam_vit_b", "cpu")
    worker.set_current_image(path)
    worker.predict(1, path, [(50, 50)], [1], None, "normal")
    seen = {}
    while "prediction" not in seen:
        event = events.get(timeout=120)
        assert event[0] not in ("error", "model_error"), event
        seen[event[0]] = event
    worker.stop()
    assert seen["model_ready"][1] == "sam_vit_b"
    _, rid, p, candidates, _ = seen["prediction"]
    assert rid == 1 and p == path


@pytest.mark.parametrize("key", ["sam_vit_b", "sam2.1_tiny"])
def test_working_resolution_matches_full_resolution(key):
    """Masks computed at reduced resolution must line up with full-res masks:
    this validates the prompt/size remapping in the adapters."""
    if not si.backend_installed(si.MODELS[key].backend):
        pytest.skip("backend not installed")
    import cv2
    torch.manual_seed(0)
    engine = si.SamEngine(key, si.DeviceInfo("cpu", "cpu"))
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (900, 1500, 3), dtype=np.uint8)
    engine.set_image("img", image)
    for prompt in [dict(points=[[700, 300], [200, 800]], labels=[1, 0]),
                   dict(box=[300, 200, 1200, 700])]:
        full, _, scale = engine.predict(**prompt)
        assert scale == (1.0, 1.0)
        small, _, (sx, sy) = engine.predict(**prompt, max_side=600)
        assert small.shape[1:] == (360, 600) and abs(sx - 0.4) < 1e-6
        ref = cv2.resize(full[0].astype(np.uint8), (600, 360), interpolation=cv2.INTER_AREA) > 0
        inter, union = (ref & small[0]).sum(), (ref | small[0]).sum()
        assert union == 0 or inter / union > 0.9


def test_experimental_models_never_auto_selected(monkeypatch):
    monkeypatch.setattr(si, "available_models", lambda: list(si.MODELS))
    for info in [si.DeviceInfo("cpu", "cpu"), si.DeviceInfo("cuda", "GTX 1650 Ti", 3.81, True),
                 si.DeviceInfo("cuda", "4090", 24.0, True)]:
        ranked = si.candidate_models(info)
        assert "edge_sam" not in ranked and "edgetam" not in ranked


def test_edgetam_detected_by_its_config(monkeypatch, tmp_path):
    """EdgeTAM installs as a `sam2` fork, so it is detected by its config file."""
    import types
    (tmp_path / "configs").mkdir()
    fake = types.SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(si.importlib.util, "find_spec", lambda name: fake if name == "sam2" else None)
    assert si.backend_installed("sam2") and not si.backend_installed("edgetam")
    (tmp_path / "configs" / "edgetam.yaml").write_text("")
    assert si.backend_installed("edgetam")


def test_edge_sam_adapter_uses_mask_count_api(monkeypatch):
    """Stand-in with EdgeSAM's predictor signature: features first, and
    num_multimask_outputs instead of multimask_output."""
    import sys
    import types
    calls = []

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FakePredictor:
        def __init__(self, model):
            self.original_size = None

        def set_image(self, image):
            self.features = torch.zeros(1)
            self.original_size, self.input_size = image.shape[:2], (1024, 1024)

        def predict(self, features=None, point_coords=None, point_labels=None, box=None, mask_input=None,
                    num_multimask_outputs=3, return_logits=False, use_stability_score=False):
            calls.append((num_multimask_outputs, tuple(self.original_size), point_coords.tolist()))
            h, w = self.original_size
            return (np.ones((num_multimask_outputs, h, w), bool), np.linspace(0.9, 0.5, num_multimask_outputs),
                    None)

    module = types.SimpleNamespace(sam_model_registry={"edge_sam": lambda checkpoint: FakeModel()},
                                   SamPredictor=FakePredictor)
    monkeypatch.setitem(sys.modules, "edge_sam", module)
    monkeypatch.setattr(si, "backend_installed", lambda backend: True)
    engine = si.SamEngine("edge_sam", si.DeviceInfo("cpu", "cpu"))
    engine.set_image("a", np.zeros((400, 800, 3), np.uint8))
    masks, scores, scale = engine.predict([[400, 200]], [1], max_side=400)
    assert masks.shape == (3, 200, 400) and scale == (0.5, 0.5)
    engine.predict([[400, 200], [10, 10]], [1, 0])
    assert calls == [(3, (200, 400), [[200.0, 100.0]]), (1, (400, 800), [[400.0, 200.0], [10.0, 10.0]])]
    assert engine._adapter.predictor.original_size == (400, 800)  # restored after the call
