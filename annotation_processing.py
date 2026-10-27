# annotation_processing.py (mask -> polygon conversion and polygon helpers)
import cv2
import numpy as np

# Polygon detail presets: simplification tolerance in pixels per 1000 px of image size
DETAIL_LEVELS = {"fine": 0.5, "normal": 1.0, "coarse": 2.5}


def clean_mask(mask, min_component_frac=0.01, min_pixels=25):
    """Drop the speckles SAM sometimes produces next to the main object.
    Components smaller than `min_component_frac` of the largest one are removed."""
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_min = max(min_pixels, areas.max() * min_component_frac)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= keep_min
    return keep[labels]


def mask_to_polygons(mask, detail="normal", clean=True):
    """Convert a binary mask into simplified outer polygons.

    Returns a list of polygons, each a list of [x, y] points in image pixels,
    largest first. Holes are not represented (YOLO polygons cannot hold them)."""
    if clean:
        mask = clean_mask(mask)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    if not mask.any():
        return []
    h, w = mask.shape[:2]
    tolerance = DETAIL_LEVELS.get(detail, detail if isinstance(detail, (int, float)) else 1.0)
    epsilon = max(0.3, tolerance * max(h, w) / 1000.0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 4:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            approx = contour.reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append(approx.astype(float).tolist())
    return polygons


def polygons_bbox(polygons):
    pts = np.concatenate([np.asarray(p, dtype=float).reshape(-1, 2) for p in polygons])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0), float(y0), float(x1), float(y1)]


def polygon_area(polygon):
    return float(cv2.contourArea(np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)))


def merge_polygons(polygons):
    """Join several polygons into one by zero-width bridges between their
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
    ann = {
        "class_id": int(class_id),
        "polygons": [[[round(x, 1), round(y, 1)] for x, y in poly] for poly in polygons],
    }
    ann["bbox"] = polygons_bbox(ann["polygons"])
    if score is not None:
        ann["score"] = round(float(score), 4)
    return ann


def normalize_annotation(ann):
    """Accept both the current format ({'polygons': [...]}) and the legacy
    single flat polygon format ({'polygon': [[x, y], ...]})."""
    if "polygons" in ann:
        return ann
    polygon = ann.get("polygon") or []
    out = dict(ann)
    out.pop("polygon", None)
    out["polygons"] = [polygon] if len(polygon) >= 3 else []
    if out["polygons"] and "bbox" not in out:
        out["bbox"] = polygons_bbox(out["polygons"])
    return out


def annotation_contains(ann, x, y):
    """True if image point (x, y) lies inside any polygon of the annotation."""
    bbox = ann.get("bbox")
    if bbox and not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
        return False
    for poly in ann.get("polygons", []):
        contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
        if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
            return True
    return False


# --- Backward-compatible helpers ------------------------------------------

def create_mask_annotation(mask):
    """Legacy API: largest polygon of the mask as a list of [x, y] points."""
    polygons = mask_to_polygons(mask)
    return polygons[0] if polygons else []


def polygons_to_bboxes(polygons):
    return [polygons_bbox([p]) for p in polygons]
