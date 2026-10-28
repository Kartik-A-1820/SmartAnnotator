import numpy as np

from smartannotator.geometry import (
    annotation_contains,
    clean_mask,
    make_annotation,
    mask_to_polygons,
    merge_polygons,
    polygon_area,
    segment_distance,
    with_polygons,
)


def square_mask(h=200, w=300):
    mask = np.zeros((h, w), bool)
    mask[50:150, 60:160] = True
    return mask


def test_square_becomes_a_few_vertices():
    polygons = mask_to_polygons(square_mask())
    assert len(polygons) == 1
    assert 4 <= len(polygons[0]) <= 8
    assert abs(polygon_area(polygons[0]) - 99 * 99) < 300


def test_speckles_removed_but_real_parts_kept():
    mask = square_mask()
    mask[5:7, 5:7] = True            # 4 px speckle
    mask[160:190, 200:280] = True    # a genuine second part
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 2
    assert polygon_area(polygons[0]) > polygon_area(polygons[1])
    assert not clean_mask(mask)[5, 5]


def test_scale_maps_back_to_image_pixels():
    polygons = mask_to_polygons(square_mask(), scale=(0.5, 0.5))
    assert abs(polygon_area(polygons[0]) - 4 * 99 * 99) < 1200


def test_coarser_tolerance_gives_fewer_vertices():
    yy, xx = np.mgrid[:400, :400]
    disk = (xx - 200) ** 2 + (yy - 200) ** 2 < 150 ** 2
    assert len(mask_to_polygons(disk, tolerance=4.0)[0]) < len(mask_to_polygons(disk, tolerance=0.5)[0])


def test_empty_mask():
    assert mask_to_polygons(np.zeros((10, 10), bool)) == []


def test_merge_keeps_every_vertex():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[20, 0], [30, 0], [30, 10], [20, 10]]
    merged = merge_polygons([a, b])
    assert len(merged) == len(a) + len(b) + 2  # plus the two bridge points
    assert all(p in merged for p in a + b)


def test_annotation_helpers():
    ann = make_annotation(mask_to_polygons(square_mask()), class_id=2, score=0.91234)
    assert ann["class_id"] == 2 and ann["score"] == 0.9123
    assert annotation_contains(ann, 100, 100) and not annotation_contains(ann, 10, 10)
    moved = with_polygons(ann, [[[0, 0], [5, 0], [5, 5]]])
    assert moved["bbox"] == [0, 0, 5, 5] and moved["class_id"] == 2
    assert ann["bbox"] != moved["bbox"]  # the original is untouched


def test_segment_distance():
    assert segment_distance((5, 3), (0, 0), (10, 0)) == 3
    assert segment_distance((13, 4), (0, 0), (10, 0)) == 5
