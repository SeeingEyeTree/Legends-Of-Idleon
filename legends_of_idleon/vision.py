"""Small OpenCV helpers shared by the bots."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(relative_path):
    """Load a template image from the package's ``templates`` folder as grayscale."""
    path = TEMPLATE_DIR / relative_path
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Template image not found: {path}")
    return image


def to_gray(image):
    """Convert a BGR/BGRA image to grayscale (grayscale input is returned as is)."""
    if image.ndim == 2:
        return image
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)


def match_template(haystack, needle, threshold):
    """Return ``(ys, xs)`` arrays of the top-left corners where ``needle`` matches."""
    scores = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(scores >= threshold)
    return ys, xs


def find_template(haystack, needle, threshold):
    """Return the ``(x, y)`` of the first match of ``needle``, or ``None``."""
    ys, xs = match_template(haystack, needle, threshold)
    if len(xs) == 0:
        return None
    return int(xs[0]), int(ys[0])


def dedupe_positions(positions, min_gap):
    """Collapse runs of nearby positions (neighbouring pixels of one match) into one."""
    unique = []
    for pos in sorted(int(p) for p in positions):
        if not unique or pos - unique[-1] >= min_gap:
            unique.append(pos)
    return unique


def color_lines(image, lower, upper, threshold1, threshold2):
    """Keep only pixels inside an HSV range, then return their Canny edges."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return cv2.Canny(masked, threshold1, threshold2)


def color_find(image, lower, upper):
    """Return ``image`` with everything outside the HSV range blacked out."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)


def find_color_blobs(image, lower, upper, min_area=50):
    """Find solid regions of a colour (HSV range) in a BGR image.

    Returns ``[(x, y, w, h, area), ...]`` for every connected blob of at least
    ``min_area`` pixels, largest first. Useful for buttons that are a known
    colour and don't need a template.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = [tuple(int(v) for v in stats[i]) for i in range(1, count) if stats[i][4] >= min_area]
    return sorted(blobs, key=lambda b: b[4], reverse=True)
