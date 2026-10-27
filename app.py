# app.py (main entry point)
import argparse
import tkinter as tk

from sam_integration import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Polygon Annotator: SAM-assisted polygon labeling")
    parser.add_argument("images", nargs="*", help="image folder or image files to open")
    parser.add_argument("--model", choices=["auto", *MODELS], default=None,
                        help="SAM model (default: last used, or auto-picked for your GPU/CPU)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto",
                        help="auto uses fp16 on GPUs that run it fast (GTX 16xx / RTX and newer)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from gui import MainApplication

    root = tk.Tk()
    root.title("Smart Polygon Annotator")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{min(1600, int(sw * 0.9))}x{min(1000, int(sh * 0.85))}")
    root.minsize(900, 600)
    app = MainApplication(root, model=args.model, device=args.device, precision=args.precision,
                          images=args.images)
    root.mainloop()
