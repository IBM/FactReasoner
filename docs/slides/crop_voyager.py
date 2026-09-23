"""Regenerate the Voyager-1 figure crops used by the coherence slides.

Source: docs/iclr2027/coherence/figures/motivating-example.png (the ICLR paper's
motivating figure). It is far too dense for one slide, so it is split into five
pieces -- each gets its own slide at roughly double the size it would have
two-up, which is what makes the text readable when projected.

Each piece is then auto-trimmed to its ink bounding box, so the drawing fills the
frame instead of carrying the figure's internal whitespace.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "iclr2027", "coherence", "figures",
                   "motivating-example.png")
OUT = os.path.join(HERE, "assets")

# Vertical fractions of the source height; the x split is the gutter between the
# two response panels, found at x=973 of 2005 (a fully blank column).
GUTTER = 973
BANDS = {
    "voy_resp_a":  (20,     0.048, GUTTER, 0.240),
    "voy_resp_b":  (GUTTER, 0.048, -20,    0.240),
    "voy_graph_a": (60,     0.288, GUTTER, 0.520),
    "voy_graph_b": (GUTTER, 0.288, -30,    0.520),
    "voy_table":   (0,      0.585, 0,      0.798),
}


def trim(im: Image.Image, pad: int = 8) -> Image.Image:
    """Crop to the ink bounding box, so the content fills the frame."""
    a = np.array(im.convert("L"))
    mask = a < 246
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return im
    t, b = max(0, rows[0] - pad), min(a.shape[0], rows[-1] + pad)
    l, r = max(0, cols[0] - pad), min(a.shape[1], cols[-1] + pad)
    return im.crop((l, t, r, b))


def main() -> None:
    im = Image.open(SRC)
    w, h = im.size
    os.makedirs(OUT, exist_ok=True)
    for name, (x0, t0, x1, t1) in BANDS.items():
        left = x0 if x0 >= 0 else w + x0
        right = (x1 if x1 > 0 else w + x1) if x1 != 0 else w
        piece = trim(im.crop((left, int(h * t0), right, int(h * t1))))
        piece.save(os.path.join(OUT, f"{name}.png"))
        print(f"wrote {name}.png  {piece.size[0]}x{piece.size[1]}")


if __name__ == "__main__":
    main()
