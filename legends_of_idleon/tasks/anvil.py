"""Collect finished items from every anvil production line."""

from __future__ import annotations

import time

import pyautogui

from .. import keys
from ..controls import click, press
from ..screen import grab_screen
from ..vision import find_template, load_template, to_gray
from .player import find_player

ANVIL_COLUMNS = 3
ANVIL_COLUMN_SPACING = 334
ANVIL_FIRST_X = 666
# Anvil UI page buttons, indexed by page.
PAGE_BUTTONS = {0: (607, 765), 1: (1403, 767)}


def click_claim_and_collect():
    """Click the claim button, then walk the character across to pick up the loot.

    Raises ``LookupError`` if the claim button is not on screen.
    """
    screen = to_gray(grab_screen())
    claim = load_template("claim_button.png")
    match = find_template(screen, claim, 0.9)
    if match is None:
        raise LookupError("claim button not found")
    x, y = match
    click(x + 50, y + 50, reference=pyautogui.size())  # already screen pixels
    time.sleep(2)

    _, player_y = find_player()
    y = player_y - 50
    pyautogui.moveTo(120, y)
    pyautogui.mouseDown()
    pyautogui.moveTo(1800, y, 4)
    pyautogui.mouseUp()


def collect_anvil(start_x, start_y, page):
    """Open the anvil at ``(start_x, start_y)`` on ``page`` and collect from it."""
    press(keys.ESCAPE)
    click(1686, 641)
    click(*PAGE_BUTTONS[page])
    click(start_x, start_y)
    click(1000, 760)
    time.sleep(2.5)
    try:
        click_claim_and_collect()
    except (LookupError, NotImplementedError) as error:
        print(f"Skipping claim: {error}")
    press(keys.C)
    click(1170, 225)
    click(900, 400)
    click(1030, 230)
    click(890, 220)
    press(keys.ESCAPE)


def _collect_row(page, y):
    for column in range(ANVIL_COLUMNS):
        collect_anvil(ANVIL_FIRST_X + column * ANVIL_COLUMN_SPACING, y, page)


def collect_all():
    """Collect from all nine anvils (two rows on page 0, one row on page 1)."""
    _collect_row(page=0, y=360)
    _collect_row(page=0, y=666)
    time.sleep(0.2)
    _collect_row(page=1, y=360)
