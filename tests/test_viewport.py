import pytest
from PIL import Image

from smartannotator.ui.viewport import FrameRenderer, Viewport


def test_round_trip_and_fit():
    view = Viewport()
    view.fit((2000, 1000), (1000, 1000), margin=1.0)
    assert view.scale == 0.5
    assert view.to_canvas(*view.to_image(123, 456)) == pytest.approx((123, 456))
    assert view.to_canvas(0, 0) == pytest.approx((0, 250))  # centered vertically


def test_zoom_keeps_point_under_cursor():
    view = Viewport()
    view.fit((2000, 1000), (1000, 1000))
    before = view.to_image(300, 400)
    view.zoom_at(300, 400, 2.0, (2000, 1000), (1000, 1000))
    assert view.to_image(300, 400) == pytest.approx(before)
    assert not view.auto_fit


def test_visible_region_is_clipped():
    view = Viewport()
    view.scale, view.ox, view.oy = 2.0, -100.0, 50.0
    assert view.visible_region((1000, 1000), (800, 600)) == (0.0, 50.0, 300.0, 350.0)
    view.ox = 5000
    assert view.visible_region((1000, 1000), (800, 600)) is None


def test_large_images_use_a_downscaled_copy_when_zoomed_out():
    renderer = FrameRenderer(Image.new("RGB", (6000, 4000), (10, 20, 30)))
    assert renderer._preview[1].size == (2000, 1334)
    frame = renderer.frame((0, 0, 6000, 4000), (600, 400), 0.1)
    assert frame.size == (600, 400) and frame.getpixel((10, 10)) == (10, 20, 30, 255)
    assert renderer.frame((0, 0, 6000, 4000), (600, 400), 0.1) is frame  # cached
