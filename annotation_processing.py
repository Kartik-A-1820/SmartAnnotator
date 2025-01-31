# annotation_processing.py (updated with mask smoothing)
import cv2
import numpy as np

def create_mask_annotation(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = []
    for contour in contours:
        contour = contour.squeeze()
        if len(contour) >= 3:
            polygon.extend(contour.tolist())
    return polygon

def polygons_to_bboxes(polygons):
    bboxes = []
    for polygon in polygons:
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes