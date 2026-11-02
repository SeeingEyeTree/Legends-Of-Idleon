"""Finding UI elements on a normalised screenshot.

Screenshots are resized to ``REFERENCE`` (1920x1080) before matching, so
templates and coordinates are the same on every machine with a 16:9 display.
Each template is ``templates/ui/<name>.png`` with an optional ``<name>.json``
sidecar::

    {"threshold": 0.9, "region": [x1, y1, x2, y2], "color": false,
     "click_offset": [dx, dy]}

``region`` limits the search area (reference pixels) and ``click_offset`` moves
the click point away from the centre of the match.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..vision import TEMPLATE_DIR

REFERENCE = (1920, 1080)
UI_TEMPLATE_DIR = TEMPLATE_DIR / "ui"
DEFAULT_THRESHOLD = 0.9


@dataclass(frozen=True)
class Match:
    x: int
    y: int
    w: int
    h: int
    score: float
    click_offset: tuple[int, int] = (0, 0)

    @property
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def click_point(self):
        cx, cy = self.center
        return cx + self.click_offset[0], cy + self.click_offset[1]


@dataclass(frozen=True)
class Template:
    name: str
    image: np.ndarray
    threshold: float = DEFAULT_THRESHOLD
    region: tuple[int, int, int, int] | None = None
    color: bool = False
    click_offset: tuple[int, int] = (0, 0)


def load_template(name, directory=None):
    directory = Path(directory) if directory else UI_TEMPLATE_DIR
    png = directory / f"{name}.png"
    meta_path = directory / f"{name}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    color = bool(meta.get("color", False))
    image = cv2.imread(str(png), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"UI template not found: {png}")
    region = meta.get("region")
    return Template(
        name=name,
        image=image,
        threshold=float(meta.get("threshold", DEFAULT_THRESHOLD)),
        region=tuple(region) if region else None,
        color=color,
        click_offset=tuple(meta.get("click_offset", (0, 0))),
    )


def normalize(image, reference=REFERENCE):
    """Return a BGR copy of a BGR/BGRA screenshot resized to ``reference``."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    height, width = image.shape[:2]
    if (width, height) != tuple(reference):
        image = cv2.resize(image, tuple(reference), interpolation=cv2.INTER_AREA)
    return image


def find(screen, template, threshold=None):
    """Return the best :class:`Match` of ``template`` in ``screen``, or ``None``."""
    x0, y0 = 0, 0
    area = screen
    if template.region:
        x0, y0, x1, y1 = template.region
        area = screen[y0:y1, x0:x1]
    if not template.color and area.ndim == 3:
        area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    th, tw = template.image.shape[:2]
    if area.shape[0] < th or area.shape[1] < tw:
        return None
    scores = cv2.matchTemplate(area, template.image, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(scores)
    limit = template.threshold if threshold is None else threshold
    if best < limit:
        return None
    return Match(loc[0] + x0, loc[1] + y0, tw, th, float(best), template.click_offset)


def score(screen, template):
    """Best match score regardless of threshold (for tuning thresholds)."""
    return find(screen, template, threshold=-1.0)


def wait_until(predicate, timeout=5.0, interval=0.4):
    """Poll ``predicate`` until it returns something truthy; return it, or ``None``."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)
