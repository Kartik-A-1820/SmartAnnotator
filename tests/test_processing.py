import numpy as np

from annotation_processing import (annotation_contains, clean_mask, make_annotation, mask_to_polygons,
                                   merge_polygons, normalize_annotation, polygon_area)


def square_mask(h=200, w=300):
    mask = np.zeros((h, w), bool)
    mask[50:150, 60:160] = True
    return mask


def test_mask_to_polygons_square():
    polys = mask_to_polygons(square_mask())
    assert len(polys) == 1
    assert 4 <= len(polys[0]) <= 8
    assert abs(polygon_area(polys[0]) - 99 * 99) < 300


def test_speckles_removed_but_real_parts_kept():
    mask = square_mask()
    mask[5:7, 5:7] = True            # 4 px speckle
    mask[160:190, 200:280] = True    # a real second part
    polys = mask_to_polygons(mask)
    assert len(polys) == 2
    assert polygon_area(polys[0]) > polygon_area(polys[1])
    assert not clean_mask(mask)[5, 5]


def test_empty_mask():
    assert mask_to_polygons(np.zeros((10, 10), bool)) == []


def test_merge_polygons_keeps_all_points():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[20, 0], [30, 0], [30, 10], [20, 10]]
    merged = merge_polygons([a, b])
    assert len(merged) == len(a) + len(b) + 2
    for p in a + b:
        assert p in merged


def test_annotation_roundtrip_and_hit_test():
    ann = make_annotation(mask_to_polygons(square_mask()), class_id=2, score=0.91234)
    assert ann["class_id"] == 2 and ann["score"] == 0.9123
    assert annotation_contains(ann, 100, 100)
    assert not annotation_contains(ann, 10, 10)


def test_legacy_annotation_format():
    ann = normalize_annotation({"polygon": [[0, 0], [5, 0], [5, 5]], "class_id": 0})
    assert ann["polygons"] == [[[0, 0], [5, 0], [5, 5]]]
    assert ann["bbox"] == [0, 0, 5, 5]
