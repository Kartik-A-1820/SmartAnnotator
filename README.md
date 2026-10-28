# Smart Polygon Annotator

**Smart Polygon Annotator** is a fast, keyboard-driven tool for polygon segmentation labeling.
Click an object and [SAM 2.1](https://github.com/facebookresearch/sam2), [SAM](https://github.com/facebookresearch/segment-anything)
or [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) outlines it instantly. Press Enter to keep it.
Export to **YOLO segmentation**, **YOLO detection** or **COCO**.

It is tuned to run well on consumer hardware, including **4 GB GPUs like the GTX 1650 / 1650 Ti** and CPU-only laptops.

---

## Table of Contents
- [Highlights](#highlights)
- [Installation](#installation)
- [Choosing a model (4 GB GPUs, CPU)](#choosing-a-model)
- [Usage](#usage)
- [Keyboard and mouse](#keyboard-and-mouse)
- [Exporting](#exporting)
- [Running tests](#running-tests)

---

## Highlights

- **Roboflow-style smart polygon:** hover to preview what SAM will select, click to include (positive point),
  right-click or Shift+click to exclude (negative point), drag a box for hard objects, and press **Enter** to accept.
  **Ctrl+Z** removes the last point and **Tab** cycles alternative masks. There are no pop-ups.
- **Four tools:** Smart polygon (S), Bounding box (B), Polygon by hand (P) and Edit (V). Edit lets you drag
  vertices or whole objects, add a vertex by double-clicking an edge, and Shift+click a vertex to delete it.
- **Instant masks:** each image is encoded by SAM once and cached, so every click only runs the lightweight mask
  decoder, at a working resolution matched to the polygon detail. On CPU, a click on a 24-megapixel photo returns
  its mask in about 60-180 ms (it was over 3 s). The next image is pre-encoded while you pause.
- **Live polygon detail slider:** move it and the current mask is re-traced with more or fewer vertices.
- **Hardware-aware model selection:** the app detects your GPU and VRAM and picks the best model that fits.
  It downloads checkpoints automatically and uses fp16 on GPUs that run it fast (GTX 16xx, RTX).
- **Never lose work:** annotations autosave to `smartannotator_project.json` next to your images.
  Reopen the folder and you resume at the first unlabeled image.
- **Undo/redo everything** (Ctrl+Z / Ctrl+Y), including class edits.
- **Unlimited classes** with distinct colors, number-key selection, rename/delete, and per-class counts.
- **Smooth zoom & pan** on large photos; only the visible part of the image is rendered.
- **Clean polygons:** speckles are removed and outlines simplified, with adjustable detail
  (View > Polygon detail). Multi-part objects are handled correctly.
- **Correct exports:** YOLO-seg, YOLO-det or COCO, with train/val/test splits that work for any dataset size,
  EXIF-rotated photos handled, and a ready-to-train `data.yaml`.

---

## Installation

Requires **Python 3.10+** and Tkinter (bundled with Python on Windows/macOS; on Linux: `sudo apt install python3-tk`).

1. **Clone**
   ```bash
   git clone https://github.com/Kartik-A-1820/SmartAnnotator.git
   cd SmartAnnotator
   ```

2. **Install PyTorch** for your hardware ([pytorch.org](https://pytorch.org/get-started/locally/)). For NVIDIA GPUs, for example:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install the rest**
   ```bash
   pip install -r requirements.txt
   ```
   This installs both SAM 2.1 (`sam2`) and the original SAM (`segment-anything`). Either one is enough. If `sam2`
   fails to install on your system, remove it from `requirements.txt` and the app will use SAM ViT-B.
   For MobileSAM (fastest on CPU): `pip install git+https://github.com/ChaoningZhang/MobileSAM.git`

4. **Model checkpoints** download automatically on first use into `./checkpoints/`.
   Offline? Download them manually (links below) and put them in `checkpoints/` (or the project folder).

---

## Choosing a model

Leave **Model > Auto** selected and the app picks for you. *Help > Hardware Info* shows what it detected.

| Model | Download | Speed | Quality | Best for |
|---|---|---|---|---|
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt) | 39 MB | fastest | good | CPU only |
| [SAM 2.1 Tiny](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | 149 MB | very fast | very good | CPU, 2-4 GB GPUs |
| [SAM 2.1 Small](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) | 176 MB | fast | very good | **4 GB GPUs (GTX 1650 / 1650 Ti)** |
| [SAM 2.1 Base+](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) | 309 MB | medium | excellent | 6 GB+ GPUs |
| [SAM 2.1 Large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | 857 MB | slower | best | 10 GB+ GPUs |
| [SAM ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | 358 MB | medium | good | 4 GB GPUs without `sam2` |
| [SAM ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | 1.2 GB | slow | very good | 6 GB+ GPUs |
| [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | 2.4 GB | slowest | very good | 8 GB+ GPUs |

### Experimental: distilled models (super fast)

Distilled models use a much smaller image encoder trained to imitate SAM or SAM 2. Encoding is several times faster,
which matters most on CPU and for flipping through images quickly. Mask quality is somewhat below the full models.
Auto never picks them. Choose them under **Model > Experimental**, or with `--model edge_sam` / `--model edgetam`.

| Model | Download | Notes | Install |
|---|---|---|---|
| [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) | ~38 MB | Distilled SAM. Reports 164 FPS vs 104 for MobileSAM on an RTX 2080 Ti, with better point-prompt accuracy. NTU S-Lab license (check before commercial use). | `pip install git+https://github.com/chongzhou96/EdgeSAM.git` |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | 54 MB | Meta's SAM 2 distilled for phones (reported 22x faster than SAM 2). Apache-2.0. | `pip install git+https://github.com/facebookresearch/EdgeTAM.git` |

EdgeTAM installs as a modified copy of the `sam2` package and replaces the official one. It still includes the
SAM 2.1 configs, so the SAM 2.1 models should keep working. To go back, run `pip install --force-reinstall sam2`.

### GTX 1650 / 1650 Ti (4 GB) tips
- Auto picks **SAM 2.1 Small** in **fp16**: a small model (46M parameters) that fits comfortably in 4 GB, with
  better masks than SAM ViT-B. If you want maximum speed, choose **SAM 2.1 Tiny**.
- The previous version of this tool always loaded SAM ViT-H on CUDA. That model is too heavy for 4 GB cards,
  and it re-ran the full encoder on every click.
- If fp16 ever produces bad output on your GPU, the app falls back to fp32 automatically.
  You can also force it: `python app.py --precision fp32`.
- Close other GPU-heavy apps (browsers with hardware acceleration, games) to keep VRAM free.

---

## Usage

```bash
python app.py                      # then File > Open Folder (Ctrl+O)
python app.py path/to/images       # open a folder directly
python app.py images --model sam2.1_tiny --device cpu
```

Options: `--model {auto, mobile_sam, sam2.1_tiny, sam2.1_small, sam2.1_base_plus, sam2.1_large, sam_vit_b, sam_vit_l, sam_vit_h}`,
`--device {auto, cuda, mps, cpu}`, `--precision {auto, fp16, fp32}`. The last chosen model is remembered.

### Typical workflow
1. **Open a folder.** Classes and annotations from earlier sessions load automatically.
2. **Add your classes** in the right panel (Add / Rename / Delete) and pick one with keys **1-9**.
3. **Smart polygon (S):** hover over an object to see the predicted mask, then click it.
   - Not quite right? **Left-click** more of the object, **right-click** (or Shift+click) to exclude a region,
     or **drag a box** around it. **Ctrl+Z** / Backspace removes the last point.
   - Press **Tab** to cycle the alternative masks SAM proposes for a single click.
4. **Press Enter** (or Space) to accept, then hover the next object.
5. Need a plain box or a hand-drawn shape? Press **B** and drag, or **P** and click the vertices.
6. Fix anything with **Edit (V)**: drag vertices or the whole object, double-click an edge to add a vertex.
7. Press **D** / **N** for the next / next unlabeled image, and **Export** (Ctrl+E) when done.

Hover an existing object to highlight it. Press **Del** to delete it, or **C** to change it to the current class.

---

## Keyboard and mouse

| Action | Input |
|---|---|
| Tools: smart polygon / box / polygon / edit | S / B / P / V |
| **Smart:** preview mask | Hover |
| **Smart:** include region (positive point) | Left click |
| **Smart:** exclude region (negative point) | Right click / Shift+click |
| **Smart:** box prompt | Left drag |
| **Smart:** remove last point | Ctrl+Z / Backspace |
| Accept mask / close polygon | Enter / Space |
| Clear prompt / cancel polygon | Esc |
| Alternative masks | Tab |
| **Box:** draw box (saved instantly) | Left drag |
| **Polygon:** add vertex / remove last / close | Click / Backspace or right click / Enter, double-click or click first vertex |
| **Edit:** move vertex or object | Drag |
| **Edit:** add / delete vertex | Double-click edge / Shift+click or right click vertex |
| Select class | 1-9, 0 |
| Next / previous image | D, Right, PgDn / A, Left, PgUp |
| Next unlabeled image | N |
| Delete object under cursor | Del / Backspace |
| Set object under cursor to current class | C |
| Zoom | Mouse wheel |
| Pan | Middle drag / Ctrl+drag |
| Fit to window | F |
| Show/hide annotations | H |
| Undo / Redo | Ctrl+Z / Ctrl+Y |
| Open / Save / Export | Ctrl+O / Ctrl+S / Ctrl+E |
| Help | F1 |

---

## Exporting

**File > Export Dataset** (Ctrl+E) lets you choose the format, train/val/test percentages, and whether to include
unlabeled images as background examples.

| Format | Output |
|---|---|
| YOLO segmentation | `{train,val,test}/images`, `{train,val,test}/labels/*.txt` (`class x1 y1 x2 y2 ...`, normalized), `data.yaml` |
| YOLO detection | same layout, labels as `class cx cy w h` |
| COCO | `{train,val,test}/images` and `{train,val,test}/annotations.json` |

Train a YOLO model right away:
```bash
yolo segment train data=path/to/export/data.yaml model=yolo11n-seg.pt
```

---

## Running tests

```bash
pip install pytest
python -m pytest tests            # on a headless Linux box: xvfb-run python -m pytest tests
```
The engine tests run the real SAM / SAM 2 architectures with random weights, so no checkpoint download is needed.
The UI tests drive the real app with synthetic mouse and keyboard events. They use a stand-in model that segments
flat-colored shapes, so every tool's result can be checked exactly.

<p align="center"><strong>Happy Annotating!</strong></p>
