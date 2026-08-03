# Castle Hills exhibit — image enhancement pipeline
# Reads originals from images-originals/, writes enhanced web copies to images/
# and 480px thumbnails to images/thumbs/.
#
# Philosophy: these are historical records. No AI generation, no inpainting.
# Only: EXIF orientation, conservative low-detail edge trim, gray-world white
# balance (damped), autocontrast, mild saturation, unsharp mask, resize.

import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

SRC = "images-originals"
DST = "images"
THUMBS = os.path.join(DST, "thumbs")
MAX_EDGE = 1600
THUMB_EDGE = 480
QUALITY = 82
TRIM_CAP = 0.14          # never trim more than 14% per side
TRIM_STEP = 0.02         # examine border strips in 2% increments
ENERGY_RATIO = 0.42      # strip must be this fraction of mean energy to be "content"


def gradient_energy_profiles(gray):
    """Return per-row and per-column mean absolute gradient energy."""
    w, h = gray.size
    px = list(gray.getdata())
    rows = [0.0] * h
    cols = [0.0] * w
    for y in range(h):
        base = y * w
        for x in range(1, w):
            d = abs(px[base + x] - px[base + x - 1])
            rows[y] += d
            cols[x] += d
    rows = [r / max(w - 1, 1) for r in rows]
    cols = [c / max(h, 1) for c in cols]
    return rows, cols


def conservative_trim(im):
    """Trim low-detail border strips (ceiling, carpet, wall) — capped, safe."""
    small = im.convert("L").resize((200, int(200 * im.height / im.width)) if im.width >= im.height
                                   else (int(200 * im.width / im.height), 200), Image.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(1))
    rows, cols = gradient_energy_profiles(small)
    mean_r = sum(rows) / len(rows)
    mean_c = sum(cols) / len(cols)
    h, w = len(rows), len(cols)

    def trim_amount(profile, mean, size):
        lo, hi = 0, 0
        step = max(1, int(size * TRIM_STEP))
        cap = int(size * TRIM_CAP)
        # from start
        i = 0
        while i + step <= cap:
            strip = profile[i:i + step]
            if sum(strip) / len(strip) < mean * ENERGY_RATIO:
                i += step
            else:
                break
        lo = i
        # from end
        i = 0
        while i + step <= cap:
            strip = profile[size - i - step:size - i]
            if sum(strip) / len(strip) < mean * ENERGY_RATIO:
                i += step
            else:
                break
        hi = i
        return lo, hi

    top, bottom = trim_amount(rows, mean_r, h)
    left, right = trim_amount(cols, mean_c, w)

    # scale back to full-size coordinates
    fx = im.width / w
    fy = im.height / h
    box = (int(left * fx), int(top * fy),
           im.width - int(right * fx), im.height - int(bottom * fy))
    if box[2] - box[0] < im.width * 0.6 or box[3] - box[1] < im.height * 0.6:
        return im  # safety: never crop away more than expected
    return im.crop(box)


def gray_world_wb(im, strength=0.55):
    """Damped gray-world white balance — nudges channels toward neutral."""
    r, g, b = im.split()
    means = [sum(ch.getdata()) / (im.width * im.height) for ch in (r, g, b)]
    gray = sum(means) / 3
    out = []
    for ch, m in zip((r, g, b), means):
        if m <= 0:
            out.append(ch)
            continue
        gain = 1 + strength * (gray / m - 1)
        gain = max(0.85, min(1.18, gain))  # clamp: gentle only
        out.append(ch.point(lambda v, g=gain: min(255, int(v * g))))
    return Image.merge("RGB", out)


def enhance(im):
    im = gray_world_wb(im)
    im = ImageOps.autocontrast(im, cutoff=1)
    im = ImageEnhance.Color(im).enhance(1.06)
    im = ImageEnhance.Contrast(im).enhance(1.04)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.6, percent=68, threshold=3))
    return im


def resize_max(im, edge):
    scale = edge / max(im.size)
    if scale >= 1:
        return im
    return im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)


def main():
    os.makedirs(THUMBS, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(".jpg"))
    for i, name in enumerate(files, 1):
        im = Image.open(os.path.join(SRC, name))
        im = ImageOps.exif_transpose(im).convert("RGB")
        before = im.size
        im = conservative_trim(im)
        im = enhance(im)
        full = resize_max(im, MAX_EDGE)
        full.save(os.path.join(DST, name), quality=QUALITY, optimize=True, progressive=True)
        thumb = resize_max(im, THUMB_EDGE)
        thumb.save(os.path.join(THUMBS, name), quality=78, optimize=True, progressive=True)
        print(f"[{i:2}/{len(files)}] {name}  {before} -> crop {im.size} -> web {full.size}")


if __name__ == "__main__":
    main()
