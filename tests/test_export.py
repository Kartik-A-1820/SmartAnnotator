import json

import numpy as np
import pytest
import yaml
from PIL import Image

from smartannotator.export import export_dataset, split_items


@pytest.mark.parametrize("n", [1, 2, 3, 5, 20, 100])
def test_split_any_size(n):
    s = split_items(range(n))
    assert sorted(s["train"] + s["val"] + s["test"]) == list(range(n))
    assert len(s["train"]) >= 1
    if n >= 2:
        assert len(s["val"]) >= 1


def make_images(tmp_path, n=4, size=(200, 100)):
    paths = []
    for i in range(n):
        p = tmp_path / "imgs" / f"img{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(p)
        paths.append(str(p))
    return paths


ANN = {"class_id": 1, "polygons": [[[20, 10], [120, 10], [120, 60], [20, 60]]], "bbox": [20, 10, 120, 60]}


def read_labels(out):
    lines = []
    for split in ("train", "val", "test"):
        d = out / split / "labels"
        if d.exists():
            for f in d.iterdir():
                lines += f.read_text().split("\n")
    return [line for line in lines if line]


def test_yolo_seg_coordinates(tmp_path):
    paths = make_images(tmp_path)
    out = tmp_path / "out"
    counts = export_dataset(paths, {p: [ANN] for p in paths}, ["a", "b"], str(out))
    assert sum(counts.values()) == 4
    line = read_labels(out)[0].split()
    assert line[0] == "1"
    coords = list(map(float, line[1:]))
    # x, y pairs for every vertex (the old exporter wrote only one value per vertex)
    assert coords == pytest.approx([0.1, 0.1, 0.6, 0.1, 0.6, 0.6, 0.1, 0.6])
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["nc"] == 2 and data["names"] == {0: "a", 1: "b"}


def test_yolo_det(tmp_path):
    paths = make_images(tmp_path, 2)
    out = tmp_path / "out"
    export_dataset(paths, {p: [ANN] for p in paths}, ["a", "b"], str(out), fmt="yolo-det")
    assert list(map(float, read_labels(out)[0].split()[1:])) == pytest.approx([0.35, 0.35, 0.5, 0.5])


def test_coco(tmp_path):
    paths = make_images(tmp_path, 3)
    out = tmp_path / "out"
    counts = export_dataset(paths, {p: [ANN] for p in paths}, ["a", "b"], str(out), fmt="coco")
    total = 0
    for split, n in counts.items():
        if n:
            coco = json.loads((out / split / "annotations.json").read_text())
            total += len(coco["images"])
            assert coco["annotations"][0]["category_id"] == 2
            assert coco["annotations"][0]["bbox"] == [20, 10, 100, 50]
            assert coco["annotations"][0]["area"] == 5000
    assert total == 3


def test_include_unlabeled_images(tmp_path):
    paths = make_images(tmp_path, 3)
    annotations = {paths[0]: [ANN]}
    assert sum(export_dataset(paths, annotations, ["a", "b"], str(tmp_path / "o1")).values()) == 1
    counts = export_dataset(paths, annotations, ["a", "b"], str(tmp_path / "o2"), include_unlabeled=True)
    assert sum(counts.values()) == 3


def test_exif_rotation_baked_in(tmp_path):
    p = tmp_path / "rot.jpg"
    img = Image.fromarray(np.zeros((100, 200, 3), np.uint8))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90: displayed as 100x200
    img.save(p, exif=exif)
    out = tmp_path / "out"
    export_dataset([str(p)], {str(p): [ANN]}, ["a", "b"], str(out))
    exported = next((out / "train" / "images").iterdir())
    assert Image.open(exported).size == (100, 200)
