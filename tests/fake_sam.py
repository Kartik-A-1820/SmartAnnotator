"""A stand-in for SAM used to test the UI without model weights: it segments
the flat-colored region under each prompt, so clicks give predictable masks."""
import cv2
import numpy as np

import sam_integration as si


class ColorRegionAdapter:
    def __init__(self, spec, checkpoint, device):
        self.image = None

    def set_image(self, image):
        self.image = np.ascontiguousarray(image)

    def get_state(self):
        return self.image

    def set_state(self, state):
        self.image = state

    def features_finite(self):
        return True

    def predict(self, points, labels, box, multimask, out_hw):
        h, w = out_hw
        img = cv2.resize(self.image, (w, h), interpolation=cv2.INTER_NEAREST)

        def region(x, y):
            flood = np.zeros((h + 2, w + 2), np.uint8)
            seed = (int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1)))
            cv2.floodFill(img, flood, seed, 0, (8, 8, 8), (8, 8, 8),
                          4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
            return flood[1:-1, 1:-1].astype(bool)

        mask = np.zeros((h, w), bool)
        if box is not None:
            x0, y0, x1, y1 = box
            mask = region((x0 + x1) / 2, (y0 + y1) / 2)
            inside = np.zeros_like(mask)
            inside[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
            mask &= inside
        if points is not None:
            for (x, y), positive in zip(points, labels):
                if positive:
                    mask |= region(x, y)
            for (x, y), positive in zip(points, labels):
                if not positive:
                    mask &= ~region(x, y)
        if not multimask:
            return mask[None], np.array([0.95])
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        return np.stack([grown, mask]), np.array([0.7, 0.95])


def install(monkeypatch):
    spec = si.ModelSpec("fake", "Fake SAM", "fake", "", "", "", 0, "test stand-in")
    monkeypatch.setitem(si.MODELS, "fake", spec)
    monkeypatch.setitem(si.ADAPTERS, "fake", ColorRegionAdapter)
    monkeypatch.setitem(si._BACKEND_MODULES, "fake", "cv2")
    monkeypatch.setattr(si, "ensure_checkpoint", lambda spec, progress=None: None)
