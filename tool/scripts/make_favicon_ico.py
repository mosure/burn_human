import argparse
import math
import os
import struct
from typing import List, Tuple


def _clamp_u8(x: float) -> int:
    if x <= 0:
        return 0
    if x >= 255:
        return 255
    return int(x + 0.5)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        _clamp_u8(_lerp(c0[0], c1[0], t)),
        _clamp_u8(_lerp(c0[1], c1[1], t)),
        _clamp_u8(_lerp(c0[2], c1[2], t)),
    )


def _blend_over(dst: List[int], idx: int, src_rgba: Tuple[int, int, int, int]) -> None:
    sr, sg, sb, sa = src_rgba
    if sa <= 0:
        return

    dr = dst[idx + 0]
    dg = dst[idx + 1]
    db = dst[idx + 2]
    da = dst[idx + 3]

    a = sa / 255.0
    ia = 1.0 - a

    out_r = sr * a + dr * ia
    out_g = sg * a + dg * ia
    out_b = sb * a + db * ia

    # keep destination alpha opaque-ish (we're making a favicon)
    out_a = 255.0 - (255.0 - da) * ia

    dst[idx + 0] = _clamp_u8(out_r)
    dst[idx + 1] = _clamp_u8(out_g)
    dst[idx + 2] = _clamp_u8(out_b)
    dst[idx + 3] = _clamp_u8(out_a)


def _draw_disc(img: List[int], size: int, cx: float, cy: float, radius: float, color: Tuple[int, int, int, int]) -> None:
    r2 = radius * radius
    x0 = max(0, int(math.floor(cx - radius - 1)))
    x1 = min(size - 1, int(math.ceil(cx + radius + 1)))
    y0 = max(0, int(math.floor(cy - radius - 1)))
    y1 = min(size - 1, int(math.ceil(cy + radius + 1)))

    for y in range(y0, y1 + 1):
        dy = (y + 0.5) - cy
        for x in range(x0, x1 + 1):
            dx = (x + 0.5) - cx
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                idx = (y * size + x) * 4
                _blend_over(img, idx, color)


def render_swirl_rgba(size: int) -> bytes:
    # RGBA, top-down
    img: List[int] = [0] * (size * size * 4)

    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    half = size / 2.0

    # Orange radial background
    c_inner = (255, 178, 74)  # #ffb24a
    c_mid = (255, 122, 0)     # #ff7a00
    c_outer = (232, 93, 0)    # #e85d00

    for y in range(size):
        for x in range(size):
            dx = (x + 0.5) - cx
            dy = (y + 0.5) - cy
            d = math.sqrt(dx * dx + dy * dy) / (half * 0.95)
            d = max(0.0, min(1.0, d))
            # two-stop gradient
            if d < 0.65:
                t = d / 0.65
                r, g, b = _lerp_rgb(c_inner, c_mid, t)
            else:
                t = (d - 0.65) / 0.35
                r, g, b = _lerp_rgb(c_mid, c_outer, t)

            idx = (y * size + x) * 4
            img[idx + 0] = r
            img[idx + 1] = g
            img[idx + 2] = b
            img[idx + 3] = 255

    # White swirl: stamp discs along a spiral curve
    base_thick = max(2.0, size * 0.09)
    steps = int(size * 22)

    for i in range(steps):
        t = i / max(1, steps - 1)
        angle = _lerp(-0.9 * math.pi, 2.6 * math.pi, t)
        radius = _lerp(0.46 * size, 0.10 * size, t)

        x = cx + math.cos(angle) * radius
        y = cy + math.sin(angle) * radius

        thick = base_thick * (1.0 - 0.35 * t)
        _draw_disc(img, size, x, y, thick, (255, 255, 255, 242))

        # secondary softer inner line
        _draw_disc(img, size, x + 0.25, y + 0.25, max(1.0, thick * 0.42), (255, 255, 255, 140))

    # circular clip (alpha outside circle)
    r_clip = (size * 0.5) - 1.0
    r2_clip = r_clip * r_clip
    for y in range(size):
        for x in range(size):
            dx = (x + 0.5) - cx
            dy = (y + 0.5) - cy
            if dx * dx + dy * dy > r2_clip:
                idx = (y * size + x) * 4
                img[idx + 3] = 0

    return bytes(img)


def _rgba_topdown_to_bgra_bottomup(rgba: bytes, size: int) -> bytes:
    # ICO DIB stores XOR bitmap as BGRA, bottom-up rows
    out = bytearray(size * size * 4)
    for y in range(size):
        src_row = y * size * 4
        dst_row = (size - 1 - y) * size * 4
        for x in range(size):
            si = src_row + x * 4
            di = dst_row + x * 4
            r = rgba[si + 0]
            g = rgba[si + 1]
            b = rgba[si + 2]
            a = rgba[si + 3]
            out[di + 0] = b
            out[di + 1] = g
            out[di + 2] = r
            out[di + 3] = a
    return bytes(out)


def _make_ico_entry(size: int) -> Tuple[bytes, bytes]:
    rgba = render_swirl_rgba(size)
    xor_bgra = _rgba_topdown_to_bgra_bottomup(rgba, size)

    # AND mask: 1-bit rows padded to 32 bits; all zeros = fully opaque
    mask_row_bytes = ((size + 31) // 32) * 4
    and_mask = bytes(mask_row_bytes * size)

    # BITMAPINFOHEADER (40 bytes)
    # height is doubled for ICO (XOR + AND)
    header = struct.pack(
        "<IIIHHIIIIII",
        40,                # biSize
        size,              # biWidth
        size * 2,          # biHeight
        1,                 # biPlanes
        32,                # biBitCount
        0,                 # biCompression (BI_RGB)
        len(xor_bgra) + len(and_mask),
        0, 0, 0, 0,
    )

    dib = header + xor_bgra + and_mask

    # ICONDIRENTRY (16 bytes) is written later, once offsets are known
    # Return dib and also a placeholder to simplify assembly
    return dib, rgba


def write_ico(path: str, sizes: List[int]) -> None:
    images: List[bytes] = []
    for s in sizes:
        dib, _ = _make_ico_entry(s)
        images.append(dib)

    # ICONDIR: reserved(0), type(1), count
    out = bytearray()
    out += struct.pack("<HHH", 0, 1, len(images))

    # Directory entries come next
    entry_offset = 6
    data_offset = entry_offset + 16 * len(images)

    entries = bytearray()
    data = bytearray()

    current = data_offset
    for dib, s in zip(images, sizes):
        w = 0 if s >= 256 else s
        h = 0 if s >= 256 else s
        color_count = 0
        reserved = 0
        planes = 1
        bit_count = 32
        bytes_in_res = len(dib)

        entries += struct.pack(
            "<BBBBHHII",
            w,
            h,
            color_count,
            reserved,
            planes,
            bit_count,
            bytes_in_res,
            current,
        )
        data += dib
        current += bytes_in_res

    out += entries
    out += data

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an orange/white swirl favicon.ico")
    parser.add_argument("--out", default="favicon.ico", help="Output .ico path")
    args = parser.parse_args()

    write_ico(args.out, sizes=[16, 32, 48, 64])


if __name__ == "__main__":
    main()
