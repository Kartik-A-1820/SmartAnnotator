# dataset_export.py (YOLO segmentation / YOLO detection / COCO export)
import json
import os
import random
import shutil

import yaml
from PIL import Image, ImageOps

from annotation_processing import merge_polygons, normalize_annotation, polygon_area, polygons_bbox
from image_io import exif_orientation, oriented_size

FORMATS = {
    "yolo-seg": "YOLO segmentation (polygons)",
    "yolo-det": "YOLO detection (boxes)",
    "coco": "COCO JSON (polygons + boxes)",
}
SPLITS = ("train", "val", "test")


def split_items(items, ratios=(0.8, 0.15, 0.05), seed=42):
    """Shuffle and split deterministically. Works for any dataset size: with two
    or more images, val always gets at least one so YOLO training can run."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    total = sum(ratios) or 1.0
    n_val = round(n * ratios[1] / total)
    n_test = round(n * ratios[2] / total)
    if ratios[1] > 0 and n >= 2 and n_val == 0:
        n_val = 1
    while n_val + n_test >= n and n > 0:  # keep at least one training image
        if n_test > 0:
            n_test -= 1
        else:
            n_val -= 1
    return {
        "train": items[n_val + n_test:],
        "val": items[:n_val],
        "test": items[n_val:n_val + n_test],
    }


def _unique_name(path, used):
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


def _yolo_lines(anns, w, h, fmt):
    lines = []
    for ann in anns:
        ann = normalize_annotation(ann)
        polys = [p for p in ann["polygons"] if len(p) >= 3]
        if not polys:
            continue
        cid = ann["class_id"]
        if fmt == "yolo-det":
            x0, y0, x1, y1 = polygons_bbox(polys)
            x0, x1 = max(0.0, x0), min(float(w), x1)
            y0, y1 = max(0.0, y0), min(float(h), y1)
            lines.append(f"{cid} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                         f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        else:
            poly = merge_polygons(polys) if len(polys) > 1 else polys[0]
            coords = []
            for x, y in poly:
                coords.append(f"{min(max(x / w, 0.0), 1.0):.6f}")
                coords.append(f"{min(max(y / h, 0.0), 1.0):.6f}")
            lines.append(f"{cid} " + " ".join(coords))
    return lines


def export_dataset(images, annotations, class_names, export_dir, fmt="yolo-seg",
                   ratios=(0.8, 0.15, 0.05), include_unlabeled=False, seed=42, progress=None):
    """Export a dataset.

    images: list of image paths to consider
    annotations: {image path: [annotation, ...]}
    class_names: list of names, index == class id
    progress: optional callable(done, total)
    Returns {"train": n, "val": n, "test": n}."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}")
    items = [p for p in images if include_unlabeled or annotations.get(p)]
    if not items:
        raise ValueError("No annotated images to export")
    splits = split_items(items, ratios, seed)
    total = len(items)
    done = 0
    used_names = set()
    counts = {}

    for split in SPLITS:
        paths = splits[split]
        counts[split] = len(paths)
        if not paths:
            continue
        img_dir = os.path.join(export_dir, split, "images")
        os.makedirs(img_dir, exist_ok=True)
        if fmt == "coco":
            coco = {"images": [], "annotations": [],
                    "categories": [{"id": i + 1, "name": n, "supercategory": "none"}
                                   for i, n in enumerate(class_names)]}
        else:
            label_dir = os.path.join(export_dir, split, "labels")
            os.makedirs(label_dir, exist_ok=True)

        for path in paths:
            name = _unique_name(path, used_names)
            _copy_image(path, os.path.join(img_dir, name))
            w, h = oriented_size(path)
            anns = annotations.get(path, [])
            if fmt == "coco":
                image_id = len(coco["images"]) + 1
                coco["images"].append({"id": image_id, "file_name": name, "width": w, "height": h})
                for ann in anns:
                    ann = normalize_annotation(ann)
                    polys = [p for p in ann["polygons"] if len(p) >= 3]
                    if not polys:
                        continue
                    x0, y0, x1, y1 = polygons_bbox(polys)
                    coco["annotations"].append({
                        "id": len(coco["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": ann["class_id"] + 1,
                        "segmentation": [[round(c, 2) for pt in p for c in pt] for p in polys],
                        "area": round(sum(polygon_area(p) for p in polys), 2),
                        "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                        "iscrowd": 0,
                    })
            else:
                txt = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
                with open(txt, "w") as f:
                    lines = _yolo_lines(anns, w, h, fmt)
                    f.write("\n".join(lines) + ("\n" if lines else ""))
            done += 1
            if progress:
                progress(done, total)

        if fmt == "coco":
            with open(os.path.join(export_dir, split, "annotations.json"), "w") as f:
                json.dump(coco, f)

    if fmt != "coco":
        data_yaml = {
            "train": "train/images",
            "val": "val/images" if counts["val"] else "train/images",
            "nc": len(class_names),
            "names": {i: n for i, n in enumerate(class_names)},
        }
        if counts["test"]:
            data_yaml["test"] = "test/images"
        with open(os.path.join(export_dir, "data.yaml"), "w") as f:
            yaml.safe_dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    return counts


def export_yolo_dataset(annotations, export_dir, class_names=None):
    """Legacy API: YOLO segmentation export of every annotated image.
    class_names may be a {id: name} dict or a list."""
    if isinstance(class_names, dict):
        names = [class_names.get(i, f"class_{i}") for i in range(max(class_names) + 1)]
    else:
        names = list(class_names or ["object"])
    return export_dataset(list(annotations), annotations, names, export_dir)
