"""Dataset export: YOLO segmentation, YOLO detection and COCO.

Output layout (all formats):
    <out>/{train,val,test}/images/...
YOLO adds   <out>/{split}/labels/*.txt and <out>/data.yaml
COCO adds   <out>/{split}/annotations.json
"""
import json
import os
import random
import shutil

import yaml
from PIL import Image, ImageOps

from .geometry import merge_polygons, polygon_area, polygons_bbox
from .image_io import exif_orientation, oriented_size

FORMATS = {
    "yolo-seg": "YOLO segmentation (polygons)",
    "yolo-det": "YOLO detection (boxes)",
    "coco": "COCO JSON (polygons + boxes)",
}
SPLITS = ("train", "val", "test")


def split_items(items, ratios=(0.8, 0.15, 0.05), seed=42):
    """Deterministic shuffle and split that works for any dataset size.

    With two or more items, val always gets at least one (YOLO training needs a
    val set) and train always keeps at least one."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    total = sum(ratios) or 1.0
    n_val = round(n * ratios[1] / total)
    n_test = round(n * ratios[2] / total)
    if ratios[1] > 0 and n >= 2 and n_val == 0:
        n_val = 1
    while n > 0 and n_val + n_test >= n:
        if n_test > 0:
            n_test -= 1
        else:
            n_val -= 1
    return {"train": items[n_val + n_test:], "val": items[:n_val], "test": items[n_val:n_val + n_test]}


def export_dataset(images, annotations, class_names, export_dir, fmt="yolo-seg",
                   ratios=(0.8, 0.15, 0.05), include_unlabeled=False, seed=42, progress=None):
    """Write a dataset and return the number of images per split.

    annotations: {image path: [annotation, ...]}; class_names[i] is class id i.
    progress: optional callable(done, total), called after each image."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}")
    items = [p for p in images if include_unlabeled or annotations.get(p)]
    if not items:
        raise ValueError("No annotated images to export")

    splits = split_items(items, ratios, seed)
    writer = _CocoWriter(class_names) if fmt == "coco" else _YoloWriter(fmt)
    used_names = set()
    done = 0
    for split in SPLITS:
        if not splits[split]:
            continue
        image_dir = os.path.join(export_dir, split, "images")
        os.makedirs(image_dir, exist_ok=True)
        writer.begin_split(os.path.join(export_dir, split))
        for path in splits[split]:
            name = _unique_name(path, used_names)
            _copy_image(path, os.path.join(image_dir, name))
            writer.add_image(name, oriented_size(path), annotations.get(path, []))
            done += 1
            if progress:
                progress(done, len(items))
        writer.end_split()

    counts = {split: len(splits[split]) for split in SPLITS}
    if fmt != "coco":
        _write_data_yaml(export_dir, class_names, counts)
    return counts


class _YoloWriter:
    def __init__(self, fmt):
        self.boxes = fmt == "yolo-det"

    def begin_split(self, split_dir):
        self.label_dir = os.path.join(split_dir, "labels")
        os.makedirs(self.label_dir, exist_ok=True)

    def add_image(self, name, size, anns):
        lines = [line for ann in anns if (line := self._line(ann, *size))]
        with open(os.path.join(self.label_dir, os.path.splitext(name)[0] + ".txt"), "w") as f:
            f.write("".join(line + "\n" for line in lines))

    def end_split(self):
        pass

    def _line(self, ann, w, h):
        polys = [p for p in ann["polygons"] if len(p) >= 3]
        if not polys:
            return None
        if self.boxes:
            x0, y0, x1, y1 = polygons_bbox(polys)
            x0, x1 = max(0.0, x0), min(float(w), x1)
            y0, y1 = max(0.0, y0), min(float(h), y1)
            values = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
        else:
            # one instance per line, so multi-part objects are bridged into one polygon
            poly = merge_polygons(polys) if len(polys) > 1 else polys[0]
            values = [min(max(v / d, 0.0), 1.0) for x, y in poly for v, d in ((x, w), (y, h))]
        return f"{ann['class_id']} " + " ".join(f"{v:.6f}" for v in values)


class _CocoWriter:
    def __init__(self, class_names):
        # COCO category ids conventionally start at 1
        self.categories = [{"id": i + 1, "name": n, "supercategory": "none"}
                           for i, n in enumerate(class_names)]

    def begin_split(self, split_dir):
        self.path = os.path.join(split_dir, "annotations.json")
        self.data = {"images": [], "annotations": [], "categories": self.categories}

    def add_image(self, name, size, anns):
        image_id = len(self.data["images"]) + 1
        self.data["images"].append({"id": image_id, "file_name": name, "width": size[0], "height": size[1]})
        for ann in anns:
            polys = [p for p in ann["polygons"] if len(p) >= 3]
            if not polys:
                continue
            x0, y0, x1, y1 = polygons_bbox(polys)
            self.data["annotations"].append({
                "id": len(self.data["annotations"]) + 1,
                "image_id": image_id,
                "category_id": ann["class_id"] + 1,
                "segmentation": [[round(c, 2) for pt in p for c in pt] for p in polys],
                "area": round(sum(polygon_area(p) for p in polys), 2),
                "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                "iscrowd": 0,
            })

    def end_split(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)


def _write_data_yaml(export_dir, class_names, counts):
    data = {
        "train": "train/images",
        "val": "val/images" if counts["val"] else "train/images",
        "nc": len(class_names),
        "names": dict(enumerate(class_names)),
    }
    if counts["test"]:
        data["test"] = "test/images"
    with open(os.path.join(export_dir, "data.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _unique_name(path, used):
    """File name for the export; images with the same name from different
    folders get the folder name as a prefix."""
    name = os.path.basename(path)
    if name in used:
        stem, ext = os.path.splitext(name)
        parent = os.path.basename(os.path.dirname(path))
        name = f"{parent}_{stem}{ext}"
        i = 1
        while name in used:
            name = f"{parent}_{stem}_{i}{ext}"
            i += 1
    used.add(name)
    return name


def _copy_image(src, dst):
    # Bake EXIF rotation into the pixels so every consumer sees what was annotated
    if exif_orientation(src) != 1:
        with Image.open(src) as img:
            ImageOps.exif_transpose(img).save(dst, quality=95)
    else:
        shutil.copy2(src, dst)
