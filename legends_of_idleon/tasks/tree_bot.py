"""Experimental: jump (Space) based on edge detection of a screen strip.

UNFINISHED. This is the author's early prototype for a timing mini-game bot,
tidied up so it runs, but its "when to press" rule is still a placeholder.
Press ``q`` in one of the debug windows to stop.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageGrab

from .. import keys
from ..controls import press

# Screen strips (left, top, right, bottom) in pixels.
RED_REGION = (1200, 180, 1700, 240)
DARK_REGION = (1200, 180, 1700, 210)

RED_LOWER = np.array([120, 50, 50])
RED_UPPER = np.array([255, 250, 250])
DARK_LOWER = np.array([80, 80, 80])
DARK_UPPER = np.array([127, 167, 193])


def _edges(image, lower, upper, threshold1, threshold2):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return cv2.Canny(masked, threshold1, threshold2)


def red_edges(image):
    return _edges(image, RED_LOWER, RED_UPPER, 100, 200)


def dark_edges(image):
    return _edges(image, DARK_LOWER, DARK_UPPER, 2, 2)


def last_signal_columns(edge_image):
    """Columns of the last row of ``edge_image`` that contains any edge pixel."""
    columns = np.array([], dtype=int)
    for row in edge_image:
        if row.any():
            columns = np.flatnonzero(row)
    return columns


def run():
    while True:
        red_strip = np.array(ImageGrab.grab(bbox=RED_REGION))
        dark_strip = np.array(ImageGrab.grab(bbox=DARK_REGION))
        red = red_edges(red_strip)
        dark = dark_edges(dark_strip)
        cv2.imshow("red lines", red)
        cv2.imshow("dark", dark)

        # Placeholder rule carried over from the original prototype.
        if last_signal_columns(dark).all() != last_signal_columns(red).all():
            press(keys.SPACE)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
