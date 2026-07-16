#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – icon generator
===========================
Draws the app icon from scratch and writes depthforge.png (256x256, for the
AppImage / .desktop) and depthforge.ico (multi-size, for the Windows installer).

The icon is generated rather than hand-drawn so it can be regenerated and tweaked
without a binary asset round-trip; the outputs are committed so a build never
depends on running this.

Motif: a figure rendered as a few discrete height terraces rather than a smooth
gradient — which is what tactile mode actually produces, and the thing that makes
a relief readable by fingertip.

    python packaging/assets/make_icon.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SIZE = 1024          # drawn large, downsampled for antialiasing
LEVELS = 5           # terraces — matches the 3–5 the tactile guidelines call for
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

# Deep teal base → warm cream crest. The hue shift (not just a lightness ramp)
# is what keeps the mid terraces from collapsing into grey.
DEEP = np.array([64, 42, 18], dtype=np.float32)      # BGR, shadowed base
HIGH = np.array([206, 238, 252], dtype=np.float32)   # BGR, lit crest


def height_field(n: int) -> np.ndarray:
    """A smooth 0..1 height field: an off-centre ridge, like a face in relief.

    Deliberately asymmetric — a centred blob reads as a bullseye, and the whole
    point is that this looks like a *relief*, not a target.
    """
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (xx - n * 0.5) / (n * 0.5)
    y = (yy - n * 0.5) / (n * 0.5)

    # Main mass, pushed up-left so the terraces bunch on one side.
    main = np.exp(-(((x + 0.18) ** 2) / 0.42 + ((y + 0.10) ** 2) / 0.55))
    # A second, lower shoulder to the lower-right, so contours read as landscape.
    shoulder = np.exp(-(((x - 0.52) ** 2) / 0.30 + ((y - 0.46) ** 2) / 0.22))
    # Broad tilt: keeps the outermost ring from hugging the frame.
    tilt = 0.14 * (0.6 - 0.5 * x - 0.4 * y)

    field = 1.02 * main + 0.55 * shoulder + tilt
    return np.clip(field / field.max(), 0.0, 1.0)


def terrace(field: np.ndarray, levels: int) -> np.ndarray:
    """Quantize to discrete levels — the tactile-mode signature."""
    q = np.round(field * (levels - 1)) / (levels - 1)
    return np.clip(q, 0.0, 1.0)


def shade(levels_f: np.ndarray, n: int) -> np.ndarray:
    """Colour-ramp the terraces, then light them from the top-left.

    The emboss is what sells the steps: a flat colour ramp of 5 quantized levels
    reads as a poster, whereas a lit riser per step reads as a physical relief —
    the same reason tactile mode keeps the level boundaries broad.
    """
    t = levels_f[..., None]
    bgr = DEEP * (1.0 - t) + HIGH * t

    # Gradient of the terraced field: non-zero only on the risers between levels.
    k = max(3, (n // 128) | 1)
    gx = cv2.Sobel(levels_f, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(levels_f, cv2.CV_32F, 0, 1, ksize=k)

    # Light from the top-left: risers facing it catch a highlight, the opposite
    # faces fall into shadow.
    lit = -(gx + gy)
    lit = cv2.GaussianBlur(lit, (0, 0), n * 0.006)
    scale = np.abs(lit).max() or 1.0
    lit = lit / scale

    bgr += np.clip(lit, 0, None)[..., None] * 150.0    # highlight
    bgr += np.clip(lit, None, 0)[..., None] * 110.0    # shadow
    return np.clip(bgr, 0, 255).astype(np.uint8)


def rounded_mask(n: int, radius_frac: float = 0.20) -> np.ndarray:
    """Squircle-ish mask so the icon sits well in both GNOME and Windows."""
    m = np.zeros((n, n), dtype=np.uint8)
    r = int(n * radius_frac)
    cv2.rectangle(m, (r, 0), (n - r, n), 255, -1)
    cv2.rectangle(m, (0, r), (n, n - r), 255, -1)
    for cx, cy in ((r, r), (n - r, r), (r, n - r), (n - r, n - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), n * 0.004)


def render(n: int) -> np.ndarray:
    """Return an RGBA image of the icon at size n."""
    field = height_field(n)
    bgr = shade(terrace(field, LEVELS), n)

    # Inner border, a half-step darker than the deepest tone.
    cv2.rectangle(bgr, (0, 0), (n - 1, n - 1), (34, 22, 10), max(1, n // 90))

    alpha = rounded_mask(n)
    return np.dstack([bgr, alpha])


def write_png(path: Path, bgra: np.ndarray) -> None:
    """Write RGBA PNG via stdlib zlib — cv2.imwrite handles alpha, but being
    explicit here keeps the byte layout obvious for the .ico packing below."""
    h, w = bgra.shape[:2]
    rgba = bgra[..., [2, 1, 0, 3]]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def write_ico(path: Path, master: np.ndarray) -> None:
    """Pack PNG-compressed entries into an .ico (Vista+ format, Inno Setup is fine
    with it)."""
    entries = []
    for s in ICO_SIZES:
        img = cv2.resize(master, (s, s), interpolation=cv2.INTER_AREA)
        tmp = path.with_suffix(f".{s}.tmp.png")
        write_png(tmp, img)
        entries.append((s, tmp.read_bytes()))
        tmp.unlink()

    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = 6 + 16 * len(entries)
    dir_blob, data_blob = b"", b""
    for s, blob in entries:
        dim = 0 if s >= 256 else s
        dir_blob += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset)
        data_blob += blob
        offset += len(blob)

    path.write_bytes(header + dir_blob + data_blob)


def main() -> None:
    master = render(SIZE)

    png = cv2.resize(master, (256, 256), interpolation=cv2.INTER_AREA)
    write_png(HERE / "depthforge.png", png)
    print(f"  wrote {HERE / 'depthforge.png'} (256x256)")

    write_ico(HERE / "depthforge.ico", master)
    print(f"  wrote {HERE / 'depthforge.ico'} ({', '.join(str(s) for s in ICO_SIZES)})")


if __name__ == "__main__":
    main()
