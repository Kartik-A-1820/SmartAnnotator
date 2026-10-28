"""Render the README screenshots by driving the real app with synthetic input.

Runs headless under Xvfb (see .github/workflows/screenshots.yml):
    xvfb-run -a -s "-screen 0 1280x900x24" python scripts/render_screenshots.py --out docs/images

Sample photos come from scikit-image's bundled data (public domain / CC0).
"""
import argparse
import json
import os
import tempfile
import time
import tkinter as tk

import skimage.data
from PIL import Image, ImageGrab

from smartannotator.settings import Settings
from smartannotator.ui import MainWindow

WINDOW = (1280, 800)
CLASSES = ["person", "helmet", "flag", "rocket", "cup", "saucer", "spoon"]
SAMPLES = {"astronaut": skimage.data.astronaut, "coffee": skimage.data.coffee,
           "rocket": skimage.data.rocket, "chelsea": skimage.data.chelsea}


class Director:
    def __init__(self, root, window):
        self.root, self.window, self.canvas = root, window, window.canvas
        self.smart = window.tools["smart"]
        self.frames = []

    def pump(self, seconds=0.1):
        end = time.time() + seconds
        while time.time() < end:
            self.root.update()
            time.sleep(0.01)

    def wait(self, condition, timeout=300):
        end = time.time() + timeout
        while not condition():
            if time.time() > end:
                raise TimeoutError
            self.root.update()
            time.sleep(0.01)

    def wait_ready(self):
        w = self.window
        self.wait(lambda: w.model_ready and w.cur_path and w.worker.engine.is_cached(w.cur_path))
        self.pump(0.3)

    def at(self, ix, iy):
        x, y = self.window.view.to_canvas(ix, iy)
        return int(x), int(y)

    def hover(self, ix, iy):
        x, y = self.at(ix, iy)
        self.canvas.event_generate("<Motion>", x=x, y=y)
        self.wait(lambda: self.smart.hover_candidates, timeout=60)
        self.pump(0.2)

    def click(self, ix, iy, button=1):
        x, y = self.at(ix, iy)
        request = self.smart.request_id
        self.canvas.event_generate(f"<ButtonPress-{button}>", x=x, y=y)
        if button == 1:
            self.canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
        self.wait(lambda: self.smart.request_id > request and not self.window.worker.busy_predicting
                  and self.window.events.empty(), timeout=60)
        self.pump(0.3)

    def drag(self, start, end):
        (ax, ay), (bx, by) = self.at(*start), self.at(*end)
        request = self.smart.request_id
        self.canvas.event_generate("<ButtonPress-1>", x=ax, y=ay)
        for t in (0.25, 0.5, 0.75, 1.0):
            self.canvas.event_generate("<B1-Motion>", x=int(ax + (bx - ax) * t), y=int(ay + (by - ay) * t))
        self.canvas.event_generate("<ButtonRelease-1>", x=bx, y=by)
        self.wait(lambda: self.smart.request_id > request and not self.window.worker.busy_predicting
                  and self.window.events.empty(), timeout=60)
        self.pump(0.3)

    def key(self, sequence):
        self.root.event_generate(sequence)
        self.pump(0.2)

    def segment(self, class_key, clicks, accept=True):
        """Select a class, click the given (x, y, button) points, accept."""
        self.key(f"<Key-{class_key}>")
        for x, y, *button in clicks:
            self.click(x, y, *button)
        if accept:
            self.key("<Return>")

    def screenshot(self, path=None):
        self.pump(0.3)
        # The window sits at the screen origin; its menu bar is above winfo_rooty()
        bottom = self.root.winfo_rooty() + self.root.winfo_height()
        image = ImageGrab.grab(bbox=(0, 0, self.root.winfo_width(), bottom), xdisplay=os.environ.get("DISPLAY"))
        if path:
            image.save(path, optimize=True)
        return image

    def frame(self, duration_ms):
        image = self.screenshot()
        self.frames.append((image.resize((image.width * 3 // 4, image.height * 3 // 4), Image.LANCZOS),
                            duration_ms))

    def save_gif(self, path):
        images = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f, _ in self.frames]
        images[0].save(path, save_all=True, append_images=images[1:], loop=0, optimize=True,
                       duration=[d for _, d in self.frames])


def prepare_images(folder):
    for name, load in SAMPLES.items():
        Image.fromarray(load()[..., :3]).save(os.path.join(folder, f"{name}.jpg"), quality=92)
    project = {"version": 2, "classes": [{"name": n} for n in CLASSES], "annotations": {}}
    with open(os.path.join(folder, "smartannotator_project.json"), "w") as f:
        json.dump(project, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/images")
    parser.add_argument("--model", default="sam2.1_small")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = lambda name: os.path.join(args.out, name)  # noqa: E731

    work = tempfile.mkdtemp()
    prepare_images(work)
    root = tk.Tk()
    root.geometry(f"{WINDOW[0]}x{WINDOW[1]}+0+0")
    window = MainWindow(root, Settings(os.path.join(work, "settings.json")), model=args.model, device="cpu",
                        images=[work])
    d = Director(root, window)
    d.wait_ready()

    # Coffee: cup (rim, body and handle), saucer, spoon
    window.show_image(window.project.images.index(os.path.join(work, "coffee.jpg")))
    d.wait_ready()
    d.segment(5, [(200, 60), (240, 215), (225, 265)])
    d.segment(6, [(130, 300)])
    d.segment(7, [(365, 290)])

    window.show_image(window.project.images.index(os.path.join(work, "rocket.jpg")))
    d.wait_ready()
    d.segment(4, [(320, 250)])

    # Astronaut, recorded as an animation
    window.show_image(window.project.images.index(os.path.join(work, "astronaut.jpg")))
    d.wait_ready()
    d.frame(1200)
    d.key("<Key-2>")                       # helmet: hover preview, then one click
    d.hover(390, 440)
    d.frame(900)
    d.click(390, 440)
    d.frame(700)
    d.key("<Return>")
    d.frame(600)
    d.key("<Key-3>")                       # flag: box prompt
    d.drag((0, 0), (115, 365))
    d.frame(1000)
    d.key("<Return>")
    d.frame(600)
    d.key("<Key-4>")                       # model rocket: tank and orbiter, two positive clicks
    d.hover(420, 100)
    d.frame(700)
    d.click(420, 100)
    d.frame(700)
    d.click(385, 250)
    d.frame(900)
    d.key("<Return>")
    d.frame(600)
    d.key("<Key-1>")
    d.hover(200, 330)
    d.frame(900)
    d.click(200, 330)
    d.screenshot(out("smart_polygon.png"))  # mask preview with the + point, before accepting
    d.frame(1200)
    d.key("<Return>")
    d.frame(2000)
    d.save_gif(out("demo.gif"))
    d.screenshot(out("annotated.png"))

    # Refining with a negative point: saucer selected, cup excluded
    window.show_image(window.project.images.index(os.path.join(work, "coffee.jpg")))
    d.wait_ready()
    window.hover_preview.set(False)
    d.segment(6, [(130, 300), (240, 215, 3)], accept=False)
    d.screenshot(out("negative_point.png"))
    d.key("<Escape>")

    # Edit tool with vertex handles
    d.key("<v>")
    x, y = d.at(365, 290)
    d.canvas.event_generate("<ButtonPress-1>", x=x, y=y)
    d.canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
    d.screenshot(out("edit_tool.png"))

    window.quit()


if __name__ == "__main__":
    main()
