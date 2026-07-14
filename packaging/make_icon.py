"""Turn the icon artwork into the square PNG the .icns needs.

The artwork (``icon-source.png``) already carries transparency around the
badge; this script finds the badge through the alpha channel, crops a square
around it with a small margin, and writes a 1024x1024 ``icon.png``.
``just app`` then converts that into ``icon.icns`` with ``sips`` and
``iconutil``, both of which ship with macOS.

Re-run after changing the artwork:  python packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
SOURCE = HERE / "icon-source.png"
OUT = HERE / "icon.png"

# Fraction of the badge size left as transparent margin — Apple's icon grid
# expects the artwork not to bleed to the very edge.
MARGIN = 0.05


def main() -> None:
    image = Image.open(SOURCE).convert("RGBA")
    alpha = np.asarray(image)[..., 3]

    # The badge is wherever the artwork is not transparent.
    visible_rows = np.flatnonzero((alpha > 10).any(axis=1))
    visible_cols = np.flatnonzero((alpha > 10).any(axis=0))
    top, bottom = visible_rows[0], visible_rows[-1]
    left, right = visible_cols[0], visible_cols[-1]

    # Square crop centred on the badge. PIL fills anything outside the source
    # with transparent pixels, which is exactly the padding we want.
    size = int(max(bottom - top, right - left) * (1 + 2 * MARGIN))
    cy, cx = (top + bottom) // 2, (left + right) // 2
    box = (cx - size // 2, cy - size // 2, cx - size // 2 + size, cy - size // 2 + size)
    badge = image.crop(box).resize((1024, 1024), Image.LANCZOS)

    badge.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
