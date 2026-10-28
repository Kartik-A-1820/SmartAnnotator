"""Command-line entry point."""
import argparse
import tkinter as tk

from . import __version__
from .models import MODELS
from .settings import Settings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="smartannotator",
                                     description="SAM-assisted polygon and box annotation.")
    parser.add_argument("images", nargs="*", help="image folder or image files to open")
    parser.add_argument("--model", choices=["auto", *MODELS],
                        help="SAM model (default: last used, else picked for your hardware)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto",
                        help="auto uses fp16 on GPUs that run it at full speed (GTX 16xx, RTX and newer)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from .ui import MainWindow  # imported late so --help works without a display

    root = tk.Tk()
    root.title("Smart Polygon Annotator")
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{min(1600, int(width * 0.9))}x{min(1000, int(height * 0.85))}")
    root.minsize(900, 600)
    MainWindow(root, Settings(), model=args.model, device=args.device, precision=args.precision,
               images=args.images)
    root.mainloop()
