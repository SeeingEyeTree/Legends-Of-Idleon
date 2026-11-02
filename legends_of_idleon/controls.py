"""Mouse and keyboard helpers.

All screen coordinates in this project were recorded on a 2560x1600 display
(``REFERENCE_RESOLUTION``) and are scaled to the actual screen size on use.
"""

from __future__ import annotations

import time

import pyautogui

from . import keys

REFERENCE_RESOLUTION = (2560, 1600)

# Inventory tab buttons (one per page of 16 slots) at the reference resolution.
_INVENTORY_TABS = [(1753, 313), (1937, 328), (2080, 329), (2283, 324), (2437, 328)]
_SLOTS_PER_PAGE = 16


def scale_point(x, y, reference=REFERENCE_RESOLUTION):
    """Scale a point recorded at ``reference`` resolution to this screen."""
    screen_w, screen_h = pyautogui.size()
    return round(x / reference[0] * screen_w), round(y / reference[1] * screen_h)


def move_to(x, y, duration=0.0, reference=REFERENCE_RESOLUTION):
    pyautogui.moveTo(*scale_point(x, y, reference), duration=duration)


def click(x, y, delay=0.1, reference=REFERENCE_RESOLUTION):
    """Move to a (reference-resolution) point and left-click it."""
    move_to(x, y, reference=reference)
    pyautogui.mouseDown()
    pyautogui.mouseUp()
    time.sleep(delay)


def hold_down(duration=0.5, x=None, y=None, reference=REFERENCE_RESOLUTION):
    """Hold the left mouse button for ``duration`` seconds.

    If ``x``/``y`` are given the mouse is moved there first.
    """
    if x is not None and y is not None:
        move_to(x, y, reference=reference)
    pyautogui.mouseDown()
    time.sleep(duration)
    pyautogui.mouseUp()


def press(key, delay=0.0):
    """Tap a key (scan code from :mod:`legends_of_idleon.keys`)."""
    keys.press_key(key)
    keys.release_key(key)
    if delay:
        time.sleep(delay)


def type_number(text):
    """Type a string of digits followed by Enter."""
    for char in str(text):
        press(keys.DIGITS[char])
    press(keys.ENTER)


def open_chest_page():
    """Open the chest UI on its first page."""
    press(keys.C)
    click(1492, 547)
    click(487, 347)


def select_inventory_slot(index):
    """Switch to the inventory page containing slot ``index`` and hover the slot.

    Slots are numbered 0-79: 16 per page, four per row, over five pages.
    """
    page, slot = divmod(int(index), _SLOTS_PER_PAGE)
    tab_x, tab_y = _INVENTORY_TABS[page]
    click(tab_x, tab_y, delay=0)
    move_to(1780 + (slot % 4) * 200, 520 + (slot // 4) * 200)
