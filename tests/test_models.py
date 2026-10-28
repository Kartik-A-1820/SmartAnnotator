import types

from smartannotator import models
from smartannotator.models import DeviceInfo, candidate_models

CPU = DeviceInfo("cpu", "cpu")
GTX_1650_TI = DeviceInfo("cuda", "NVIDIA GeForce GTX 1650 Ti", 3.81, fp16=True)
RTX_4090 = DeviceInfo("cuda", "NVIDIA GeForce RTX 4090", 24.0, fp16=True)


def test_auto_selection_per_hardware(monkeypatch):
    monkeypatch.setattr(models, "available_models", lambda: list(models.MODELS))
    assert candidate_models(GTX_1650_TI)[0] == "sam2.1_small"
    assert "sam_vit_h" not in candidate_models(GTX_1650_TI)[:4]
    assert candidate_models(CPU)[0] == "mobile_sam"
    assert candidate_models(RTX_4090)[0] == "sam2.1_large"


def test_auto_selection_falls_back_to_installed_backends(monkeypatch):
    monkeypatch.setattr(models, "available_models", lambda: ["sam_vit_b", "sam_vit_h"])
    assert candidate_models(GTX_1650_TI) == ["sam_vit_b", "sam_vit_h"]


def test_experimental_models_never_auto_selected(monkeypatch):
    monkeypatch.setattr(models, "available_models", lambda: list(models.MODELS))
    for device in (CPU, GTX_1650_TI, RTX_4090):
        ranked = candidate_models(device)
        assert "edge_sam" not in ranked and "edgetam" not in ranked


def test_edgetam_detected_by_its_config(monkeypatch, tmp_path):
    # EdgeTAM installs as a fork of the sam2 package, so its config file is the tell
    (tmp_path / "configs").mkdir()
    sam2 = types.SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: sam2 if name == "sam2" else None)
    assert models.backend_installed("sam2") and not models.backend_installed("edgetam")
    (tmp_path / "configs" / "edgetam.yaml").write_text("")
    assert models.backend_installed("edgetam")
