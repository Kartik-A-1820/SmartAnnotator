"""End-to-end UI tests: the real Tk app driven by synthetic mouse and keyboard
events, with a stand-in model that segments flat-colored shapes.
Needs a display; on headless Linux run under xvfb-run."""
import os
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from smartannotator.geometry import polygon_area
from smartannotator.settings import Settings

tk = pytest.importorskip("tkinter")
pytest.importorskip("torch")

BACKGROUND, RED, BLUE, GREEN = (128, 128, 128), (220, 40, 40), (40, 60, 220), (40, 200, 60)


def make_scene(path):
    image = Image.new("RGB", (1200, 800), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.ellipse((150, 250, 450, 550), fill=RED)        # r=150 circle at (300, 400)
    draw.rectangle((650, 200, 849, 599), fill=BLUE)     # 200 x 400
    draw.rectangle((850, 200, 1049, 599), fill=GREEN)   # 200 x 400, touching the blue one
    image.save(path)


@pytest.fixture
def ui(tmp_path, fake_model):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    from smartannotator.ui import MainWindow

    for i in range(2):
        make_scene(tmp_path / f"scene_{i}.png")
    root.geometry("1400x900")
    window = MainWindow(root, Settings(str(tmp_path / "settings.json")), model=fake_model, device="cpu",
                        images=[str(tmp_path)])
    driver = Driver(root, window)
    driver.wait(lambda: window.model_ready and window.cur_path)
    driver.wait(lambda: window.worker.engine.is_cached(window.cur_path))
    yield driver
    window.quit()


class Driver:
    def __init__(self, root, window):
        self.root, self.window, self.canvas = root, window, window.canvas
        self.smart = window.tools["smart"]
        self.clock = 1000  # synthetic event time in ms, so Tk can tell clicks from double-clicks

    def tick(self, ms=600):
        self.clock += ms
        return self.clock

    def pump(self, seconds=0.05):
        end = time.time() + seconds
        while time.time() < end:
            self.root.update()
            time.sleep(0.005)

    def wait(self, condition, timeout=20):
        end = time.time() + timeout
        while not condition():
            assert time.time() < end, "timed out"
            self.root.update()
            time.sleep(0.005)

    def at(self, ix, iy):
        cx, cy = self.window.view.to_canvas(ix, iy)
        return int(round(cx)), int(round(cy))

    def move(self, ix, iy):
        x, y = self.at(ix, iy)
        self.canvas.event_generate("<Motion>", x=x, y=y)
        self.pump(0.01)

    def click(self, ix, iy, button=1, state=0):
        x, y = self.at(ix, iy)
        self.canvas.event_generate(f"<ButtonPress-{button}>", x=x, y=y, state=state, time=self.tick())
        if button == 1:
            self.canvas.event_generate("<ButtonRelease-1>", x=x, y=y, state=state, time=self.tick(30))
        self.pump(0.01)

    def double_click(self, ix, iy):
        x, y = self.at(ix, iy)
        for gap in (600, 50):
            self.canvas.event_generate("<ButtonPress-1>", x=x, y=y, time=self.tick(gap))
            self.canvas.event_generate("<ButtonRelease-1>", x=x, y=y, time=self.tick(30))
        self.pump(0.01)

    def drag(self, start, end):
        (ax, ay), (bx, by) = self.at(*start), self.at(*end)
        self.canvas.event_generate("<ButtonPress-1>", x=ax, y=ay, time=self.tick())
        for t in np.linspace(0, 1, 6)[1:]:
            self.canvas.event_generate("<B1-Motion>", x=int(ax + (bx - ax) * t), y=int(ay + (by - ay) * t),
                                       time=self.tick(20))
        self.canvas.event_generate("<ButtonRelease-1>", x=bx, y=by, time=self.tick(20))
        self.pump(0.01)

    def key(self, sequence):
        self.root.event_generate(sequence)
        self.pump(0.01)

    def mask_area(self):
        """Wait for the in-flight prediction; return the shown candidate's area."""
        smart, request_id = self.smart, self.smart.request_id
        self.wait(lambda: smart.candidates and smart.request_id == request_id
                  and not self.window.worker.busy_predicting)
        self.pump(0.05)
        return sum(polygon_area(p) for p in smart.candidates[smart.index]["polygons"])

    def annotations(self):
        return self.window.annotations


def area(ann):
    return sum(polygon_area(p) for p in ann["polygons"])


def test_hover_preview_makes_the_click_instant(ui):
    smart = ui.smart
    ui.move(300, 400)
    ui.wait(lambda: smart.hover_candidates)
    request_id = smart.request_id
    ui.click(300, 400)
    # answered from the hover result: nothing new is sent to the worker
    assert smart.candidates and not ui.window.worker.busy_predicting
    assert smart.request_id == request_id + 1
    assert abs(area(smart.candidates[0]) - np.pi * 150 ** 2) < 1500
    ui.key("<Tab>")
    assert smart.index == 1
    ui.key("<Tab>")
    ui.key("<Return>")
    assert len(ui.annotations()) == 1 and smart.candidates == [] and smart.points == []


def test_positive_and_negative_points(ui):
    smart = ui.smart
    ui.window.hover_preview.set(False)
    ui.click(750, 400)
    assert abs(ui.mask_area() - 200 * 400) < 2000
    ui.click(950, 400)                       # second positive point: both rectangles
    assert abs(ui.mask_area() - 400 * 400) < 3000
    ui.click(950, 400, button=3)             # right-click excludes the green one
    assert smart.labels == [1, 1, 0]
    assert abs(ui.mask_area() - 200 * 400) < 2000
    ui.key("<Control-z>")                    # undoes the last point only
    assert smart.labels == [1, 1]
    assert abs(ui.mask_area() - 400 * 400) < 3000
    ui.click(950, 400, state=1)              # Shift+click is negative too
    assert smart.labels == [1, 1, 0]
    assert abs(ui.mask_area() - 200 * 400) < 2000
    ui.key("<BackSpace>")
    ui.key("<BackSpace>")
    assert smart.labels == [1]
    ui.key("<Escape>")
    assert smart.points == [] and smart.candidates == []
    assert ui.annotations() == []


def test_box_prompt_with_selected_class(ui):
    window = ui.window
    window.project.add_class("blue")
    window._refresh_classes()
    ui.key("<Key-2>")
    assert window.current_class == 1
    ui.drag((630, 180), (870, 620))
    assert ui.smart.box is not None
    assert abs(ui.mask_area() - 200 * 400) < 3000
    ui.key("<space>")
    assert ui.annotations()[0]["class_id"] == 1


def test_box_and_polygon_tools(ui):
    ui.key("<b>")
    assert ui.window.tool.name == "box"
    ui.drag((100, 100), (300, 200))
    assert len(ui.annotations()) == 1 and abs(area(ui.annotations()[0]) - 200 * 100) < 400

    polygon = ui.window.tools["polygon"]
    ui.key("<p>")
    for point in [(500, 600), (600, 600), (600, 700)]:
        ui.click(*point)
    ui.key("<BackSpace>")                    # removes the last vertex
    assert len(polygon.vertices) == 2
    ui.click(600, 700)
    ui.click(500, 700)
    ui.click(500, 600)                       # clicking the first vertex closes it
    assert len(ui.annotations()) == 2 and abs(area(ui.annotations()[1]) - 100 * 100) < 300
    for point in [(700, 650), (750, 650)]:
        ui.click(*point)
    ui.double_click(750, 700)                # so does a double-click
    assert len(ui.annotations()) == 3 and len(ui.annotations()[2]["polygons"][0]) == 3


def test_edit_tool(ui):
    window = ui.window
    ui.key("<b>")
    ui.drag((100, 100), (300, 200))
    ui.key("<v>")
    ui.click(200, 150)
    assert window.selected_obj == 0
    ui.drag((300, 200), (350, 260))          # drag a corner
    assert any(abs(x - 350) < 2 and abs(y - 260) < 2 for x, y in ui.annotations()[0]["polygons"][0])
    ui.drag((180, 140), (230, 190))          # move the whole object
    x0, y0, _, _ = ui.annotations()[0]["bbox"]
    assert abs(x0 - 150) < 2 and abs(y0 - 150) < 2
    ui.double_click(200, 150)                # double-click on an edge adds a vertex
    assert len(ui.annotations()[0]["polygons"][0]) == 5
    ui.click(200, 150, state=1)              # Shift+click removes it
    assert len(ui.annotations()[0]["polygons"][0]) == 4
    for _ in range(4):
        ui.key("<Control-z>")
    assert np.allclose(ui.annotations()[0]["bbox"], [100, 100, 300, 200], atol=2)
    ui.key("<Delete>")                       # selection survives undo
    assert ui.annotations() == []


def test_detail_slider_navigation_and_autosave(ui, tmp_path):
    window, smart = ui.window, ui.smart
    window.hover_preview.set(False)
    ui.click(300, 400)
    ui.mask_area()
    fine = len(smart.candidates[0]["polygons"][0])
    window.tolerance_var.set(4.0)
    window.apply_tolerance()
    ui.mask_area()
    assert len(smart.candidates[0]["polygons"][0]) < fine
    ui.key("<Return>")
    ui.key("<n>")                            # next unlabeled image
    assert window.index == 1
    ui.pump(1.3)                             # autosave is debounced by one second
    assert os.path.exists(tmp_path / "smartannotator_project.json")
