"""End-to-end UI tests: drive the real Tk app with synthetic mouse/keyboard
events. Needs a display (on Linux CI: xvfb-run python -m pytest tests)."""
import os
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

tk = pytest.importorskip("tkinter")
pytest.importorskip("torch")

from annotation_processing import polygon_area  # noqa: E402
from tests import fake_sam  # noqa: E402

BG, RED, BLUE, GREEN = (128, 128, 128), (220, 40, 40), (40, 60, 220), (40, 200, 60)


def make_scene(path, size=(1200, 800)):
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    d.ellipse((150, 250, 450, 550), fill=RED)          # circle, r=150 at (300, 400)
    d.rectangle((650, 200, 849, 599), fill=BLUE)       # 200 x 400
    d.rectangle((850, 200, 1049, 599), fill=GREEN)     # touching the blue one
    img.save(path)


@pytest.fixture
def app(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    fake_sam.install(monkeypatch)
    import gui
    monkeypatch.setattr(gui, "load_settings", lambda: {"hover_preview": True})
    monkeypatch.setattr(gui, "save_settings", lambda s: None)
    for i in range(2):
        make_scene(tmp_path / f"scene_{i}.png")
    root.geometry("1400x900")
    application = gui.MainApplication(root, model="fake", device="cpu", images=[str(tmp_path)])
    driver = Driver(root, application)
    driver.wait(lambda: application.model_ready and application.cur_path)
    driver.wait(lambda: application.worker.engine.is_cached(application.cur_path))
    yield driver
    application.quit_app()


class Driver:
    def __init__(self, root, app):
        self.root, self.app, self.c = root, app, app.canvas
        self.t = 1000  # synthetic event clock (ms), so Tk can tell clicks from double-clicks

    def tick(self, ms=600):
        self.t += ms
        return self.t

    def pump(self, seconds=0.05):
        end = time.time() + seconds
        while time.time() < end:
            self.root.update()
            time.sleep(0.005)

    def wait(self, cond, timeout=20):
        end = time.time() + timeout
        while not cond():
            assert time.time() < end, "timed out"
            self.root.update()
            time.sleep(0.005)

    def xy(self, ix, iy):
        cx, cy = self.app._to_canvas(ix, iy)
        return int(round(cx)), int(round(cy))

    def move(self, ix, iy):
        x, y = self.xy(ix, iy)
        self.c.event_generate("<Motion>", x=x, y=y)
        self.pump(0.01)

    def click(self, ix, iy, button=1, state=0):
        x, y = self.xy(ix, iy)
        self.c.event_generate(f"<ButtonPress-{button}>", x=x, y=y, state=state, time=self.tick())
        if button == 1:
            self.c.event_generate("<ButtonRelease-1>", x=x, y=y, state=state, time=self.tick(30))
        self.pump(0.01)

    def double_click(self, ix, iy):
        # two real clicks in quick succession; Tk turns the second into <Double-Button-1>
        x, y = self.xy(ix, iy)
        for gap in (600, 50):
            self.c.event_generate("<ButtonPress-1>", x=x, y=y, time=self.tick(gap))
            self.c.event_generate("<ButtonRelease-1>", x=x, y=y, time=self.tick(30))
        self.pump(0.01)

    def drag(self, a, b):
        (ax, ay), (bx, by) = self.xy(*a), self.xy(*b)
        self.c.event_generate("<ButtonPress-1>", x=ax, y=ay, time=self.tick())
        for t in np.linspace(0, 1, 6)[1:]:
            self.c.event_generate("<B1-Motion>", x=int(ax + (bx - ax) * t), y=int(ay + (by - ay) * t),
                                  time=self.tick(20))
        self.c.event_generate("<ButtonRelease-1>", x=bx, y=by, time=self.tick(20))
        self.pump(0.01)

    def key(self, keysym):
        self.root.event_generate(keysym)
        self.pump(0.01)

    def predicted(self):
        """Wait for the in-flight prediction, return the current candidate's area."""
        app = self.app
        rid = app.request_id
        self.wait(lambda: app.candidates and app.request_id == rid and not app.worker._predict)
        self.pump(0.05)
        return sum(polygon_area(p) for p in app.candidates[app.cand_idx]["polygons"])

    def anns(self):
        return self.app.project.get(self.app.cur_path)


def area(ann):
    return sum(polygon_area(p) for p in ann["polygons"])


def test_hover_preview_then_instant_click(app):
    a = app.app
    app.move(300, 400)
    app.wait(lambda: a.hover_candidates)
    rid = a.request_id
    app.click(300, 400)
    # answered from the hover preview: no new request reaches the worker
    assert a.candidates and a.worker._predict is None and a.request_id == rid + 1
    best = a.candidates[0]
    assert abs(sum(polygon_area(p) for p in best["polygons"]) - np.pi * 150 ** 2) < 1500
    app.key("<Tab>")
    assert a.cand_idx == 1
    app.key("<Tab>")
    app.key("<Return>")
    assert len(app.anns()) == 1 and a.candidates == [] and a.points == []


def test_positive_and_negative_points(app):
    a = app.app
    a.hover_preview.set(False)
    app.click(750, 400)
    assert abs(app.predicted() - 200 * 400) < 2000
    app.click(950, 400)                       # second positive point: union
    assert abs(app.predicted() - 400 * 400) < 3000
    app.click(950, 400, button=3)             # right-click: exclude green
    assert a.labels == [1, 1, 0]
    assert abs(app.predicted() - 200 * 400) < 2000
    app.key("<Control-z>")                    # undo last point only
    assert a.labels == [1, 1]
    assert abs(app.predicted() - 400 * 400) < 3000
    app.click(950, 400, state=1)              # Shift+click is negative too
    assert a.labels == [1, 1, 0]
    assert abs(app.predicted() - 200 * 400) < 2000
    app.key("<BackSpace>")
    app.key("<BackSpace>")
    assert a.labels == [1]
    app.key("<Escape>")
    assert a.points == [] and a.candidates == []
    assert app.anns() == []                   # nothing was accepted


def test_smart_box_prompt_and_classes(app):
    a = app.app
    a.project.add_class("blue")
    a._refresh_classes()
    app.key("<Key-2>")
    assert a.current_class == 1
    app.drag((630, 180), (870, 620))
    assert a.box is not None
    assert abs(app.predicted() - 200 * 400) < 3000
    app.key("<space>")
    assert app.anns()[0]["class_id"] == 1


def test_box_and_polygon_tools(app):
    a = app.app
    app.key("<b>")
    assert a.tool.get() == "box"
    app.drag((100, 100), (300, 200))
    assert len(app.anns()) == 1 and abs(area(app.anns()[0]) - 200 * 100) < 400

    app.key("<p>")
    for p in [(500, 600), (600, 600), (600, 700)]:
        app.click(*p)
    app.key("<BackSpace>")                    # remove the last vertex
    assert len(a.poly_pts) == 2
    app.click(600, 700)
    app.click(500, 700)
    app.click(500, 600)                       # click the first vertex closes it
    assert len(app.anns()) == 2 and abs(area(app.anns()[1]) - 100 * 100) < 300
    for p in [(700, 650), (750, 650)]:
        app.click(*p)
    app.double_click(750, 700)                # double-click closes too
    assert len(app.anns()) == 3 and len(app.anns()[2]["polygons"][0]) == 3


def test_edit_tool(app):
    a = app.app
    app.key("<b>")
    app.drag((100, 100), (300, 200))
    app.key("<v>")
    app.click(200, 150)                       # select
    assert a.selected_obj == 0
    app.drag((300, 200), (350, 260))          # drag a corner vertex
    assert any(abs(x - 350) < 2 and abs(y - 260) < 2 for x, y in app.anns()[0]["polygons"][0])
    app.drag((180, 140), (230, 190))          # move the whole object
    x0, y0, _, _ = app.anns()[0]["bbox"]
    assert abs(x0 - 150) < 2 and abs(y0 - 150) < 2
    app.double_click(200, 150)                # click on the top edge inserts a vertex
    assert len(app.anns()[0]["polygons"][0]) == 5
    app.click(200, 150, state=1)              # Shift+click deletes it again
    assert len(app.anns()[0]["polygons"][0]) == 4
    app.key("<Control-z>")
    app.key("<Control-z>")
    app.key("<Control-z>")
    app.key("<Control-z>")
    x0, y0, x1, y1 = app.anns()[0]["bbox"]
    assert np.allclose([x0, y0, x1, y1], [100, 100, 300, 200], atol=2)
    app.key("<Delete>")
    assert app.anns() == []


def test_detail_slider_and_navigation_and_autosave(app, tmp_path):
    a = app.app
    a.hover_preview.set(False)
    app.click(300, 400)
    app.predicted()
    fine = len(a.candidates[0]["polygons"][0])
    a.detail.set(4.0)
    a._apply_detail()
    app.predicted()
    coarse = len(a.candidates[0]["polygons"][0])
    assert coarse < fine
    app.key("<Return>")
    app.key("<n>")                            # next unlabeled image
    assert a.index == 1
    app.pump(1.3)                             # autosave is debounced by 1 s
    assert os.path.exists(tmp_path / "smartannotator_project.json")
