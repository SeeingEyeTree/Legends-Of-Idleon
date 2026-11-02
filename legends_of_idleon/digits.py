"""Read the numbers shown in the game UI by template-matching digit images.

The digit templates in ``templates/inventory_digits`` were captured at
1920x1080. On other resolutions you need to take your own screenshots of the
digits and replace those files.
"""

from __future__ import annotations

import pyautogui

from . import keys
from .controls import click, press, scale_point
from .screen import grab_screen
from .vision import dedupe_positions, load_template, match_template, to_gray

UI_RESOLUTION = (1920, 1080)
MATCH_THRESHOLD = 0.93
# Matches closer together than this (reference pixels) are the same digit.
MIN_DIGIT_GAP = 15

# Screen regions (left, top, right, bottom) at UI_RESOLUTION.
WORSHIP_REGIONS = {
    "charge-rate": (615, 529, 678, 563),
    "cur-charge": (631, 628, 686, 658),
}


def _check_resolution():
    if tuple(pyautogui.size()) != UI_RESOLUTION:
        print("The digit templates were captured at 1920x1080.")
        print("You will likely need to screenshot the numbers at your own resolution")
        print("and replace the images in legends_of_idleon/templates/inventory_digits.")


def read_numbers(region):
    """Read the digits inside ``region`` (screen pixels) and return them as a string."""
    screen = to_gray(grab_screen(region))
    min_gap, _ = scale_point(MIN_DIGIT_GAP, 0, reference=UI_RESOLUTION)

    found = []  # (x position, digit)
    for digit in range(10):
        template = load_template(f"inventory_digits/{digit}.png")
        _, xs = match_template(screen, template, MATCH_THRESHOLD)
        found += [(x, digit) for x in dedupe_positions(xs, min_gap)]

    found.sort()
    return "".join(str(digit) for _, digit in found)


def read_worship(what):
    """Open the Worship UI and read ``"charge-rate"`` or ``"cur-charge"``."""
    if what not in WORSHIP_REGIONS:
        raise ValueError(f"what must be one of {sorted(WORSHIP_REGIONS)}")

    _check_resolution()
    press(keys.ESCAPE, delay=0.1)
    press(keys.I, delay=0.1)
    click(637, 137, reference=UI_RESOLUTION)
    click(688, 552, reference=UI_RESOLUTION)

    left, top, right, bottom = WORSHIP_REGIONS[what]
    x1, y1 = scale_point(left, top, reference=UI_RESOLUTION)
    x2, y2 = scale_point(right, bottom, reference=UI_RESOLUTION)
    return read_numbers((x1, y1, x2, y2))
