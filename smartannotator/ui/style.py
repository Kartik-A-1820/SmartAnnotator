"""Colors and ttk styling shared by the UI modules."""
from tkinter import ttk

CANVAS_BG = "#202124"
ACCENT = "#1a73e8"
PROMPT_BOX_COLOR = "#00E5FF"


def hex_to_rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def readable_text_color(background):
    r, g, b = hex_to_rgb(background)
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#FFFFFF"


def apply_theme(root):
    style = ttk.Style(root)
    if root.tk.call("tk", "windowingsystem") == "x11" and "clam" in style.theme_names():
        style.theme_use("clam")  # the default X11 theme looks dated
    style.configure("Tool.Toolbutton", padding=(8, 3))
    style.map("Tool.Toolbutton", background=[("selected", ACCENT), ("active", "#d2e3fc")],
              foreground=[("selected", "white")])
