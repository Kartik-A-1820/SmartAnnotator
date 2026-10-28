"""Mask-to-polygon conversion and small polygon utilities.

Polygons are lists of [x, y] points in image pixels. An annotation holds one or
more polygons (an object can be split into parts) plus its bounding box.
"""
import math

import cv2
import numpy as np

# Simplification tolerance, in pixels per 1000 px of the image's longest side.
DEFAULT_TOLERANCE = 1.0


def work_resolution(h, w, max_side):
    """Image size capped at `max_side` on the longest side, aspect ratio kept."""
    scale = min(1.0, max_side / max(h, w))
    return max(1, round(h * scale)), max(1, round(w * scale))


def work_side(tolerance):
    """Mask resolution that is just fine enough for a given tolerance.

    Simplification discards detail below roughly tolerance * size / 1000 px,
    so computing the mask at full resolution for a 24 MP photo is wasted work.
    """
    return int(min(2560, max(1024, 1600 / tolerance)))


def clean_mask(mask, min_component_frac=0.01, min_pixels=25):
    """Remove speckles: components smaller than 1% of the largest are dropped."""
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= max(min_pixels, areas.max() * min_component_frac)
    return keep[labels]


def mask_to_polygons(mask, tolerance=DEFAULT_TOLERANCE, scale=(1.0, 1.0)):
    """Outer contours of a binary mask, simplified, largest first.

    `scale` is the mask size relative to the image (sx, sy); points are mapped
    back to image pixels. Holes are dropped since YOLO polygons cannot express them.
    """
    mask = np.ascontiguousarray(clean_mask(mask), dtype=np.uint8)
    if not mask.any():
        return []
    epsilon = max(0.3, tolerance * max(mask.shape) / 1000.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 4:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            approx = contour.reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append((approx.astype(float) / scale).tolist())
    return polygons


def polygons_bbox(polygons):
    pts = np.concatenate([np.asarray(p, dtype=float).reshape(-1, 2) for p in polygons])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0), float(y0), float(x1), float(y1)]


def polygon_area(polygon):
    return float(cv2.contourArea(np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)))


def merge_polygons(polygons):
    """Join several polygons into one using zero-width bridges between their
    closest points, so a multi-part object stays a single YOLO instance."""
    polys = [np.asarray(p, dtype=float).reshape(-1, 2) for p in polygons if len(p) >= 3]
    if not polys:
        return []
    merged = polys[0]
    for poly in polys[1:]:
        dists = ((merged[:, None, :] - poly[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        rotated = np.roll(poly, -j, axis=0)
        merged = np.concatenate([merged[:i + 1], rotated, rotated[:1], merged[i:]])
    return merged.tolist()


def make_annotation(polygons, class_id, score=None):
    polygons = [[[round(x, 1), round(y, 1)] for x, y in poly] for poly in polygons]
    ann = {"class_id": int(class_id), "polygons": polygons, "bbox": polygons_bbox(polygons)}
    if score is not None:
        ann["score"] = round(float(score), 4)
    return ann


def with_polygons(ann, polygons):
    """Copy of `ann` with new geometry (annotations are never mutated in place)."""
    polygons = [[[round(x, 1), round(y, 1)] for x, y in poly] for poly in polygons]
    return dict(ann, polygons=polygons, bbox=polygons_bbox(polygons))


def annotation_contains(ann, x, y):
    x0, y0, x1, y1 = ann["bbox"]
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return False
    return any(cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2),
                                    (float(x), float(y)), False) >= 0
               for poly in ann["polygons"])


def segment_distance(p, a, b):
    """Distance from point p to segment ab."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else min(1.0, max(0.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))
