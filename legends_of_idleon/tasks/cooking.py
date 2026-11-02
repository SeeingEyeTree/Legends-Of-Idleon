"""Fill the kitchens with meals and level them up, using ladles from the chest."""

from __future__ import annotations

import pyautogui

from .. import keys
from ..controls import (
    click,
    hold_down,
    open_chest_page,
    press,
    select_inventory_slot,
    type_number,
)

NUMBER_OF_MEALS = 67
SHELF_LEFT_CLICKS = 4
SHELVES = 3
ROWS_PER_SHELF = 4
MEALS_PER_ROW = 5
# If a meal needs more than this many level-ups, cooking was neglected for far too long.
MAX_LEVELUPS_PER_MEAL = 10
MENU_ROW_OFFSET = 180


def _close_menus():
    press(keys.C)
    press(keys.C)


def _open_first_kitchen():
    click(890, 785)
    click(1662, 318)  # go back


def get_ladles(ladles_per_meal):
    """Withdraw ``ladles_per_meal`` ladles for every meal from the chest."""
    open_chest_page()
    click(228, 377)  # quick-tap, so split-stack stays on for the rest of the sequence
    click(241, 730)  # split stack
    for _ in range(NUMBER_OF_MEALS):
        click(524, 520, 0.01)
        type_number(ladles_per_meal)
    _close_menus()


def cook(ladles_per_meal):
    """Run a full cooking pass using ``ladles_per_meal`` ladles per meal."""
    get_ladles(ladles_per_meal)
    _open_first_kitchen()
    for _ in range(SHELF_LEFT_CLICKS):
        click(1566, 1084)  # shelf left
    _close_menus()

    ladle_slots = list(range(NUMBER_OF_MEALS))
    flip = 0
    for _ in range(SHELVES):
        y_meal = 435
        for _ in range(ROWS_PER_SHELF):
            _cook_row(y_meal, ladle_slots, MENU_ROW_OFFSET * flip)
            flip = 1 - flip
            y_meal += 155
            click(1498, 879)  # menu
            pyautogui.scroll(-10)
            pyautogui.scroll(-10)
            if flip == 0:
                pyautogui.scroll(-10)
            _close_menus()
        _open_first_kitchen()
        click(1784, 1087)  # shelf right
        _close_menus()


def _cook_row(y_meal, ladle_slots, menu_offset):
    """Cook one row of meals, consuming ladle slots from the front of ``ladle_slots``."""
    x_meal = 1329
    y_menu = 281
    for _ in range(MEALS_PER_ROW):
        _open_first_kitchen()
        click(x_meal, y_meal)
        click(1706, 1036)  # cook all
        press(keys.I)
        select_inventory_slot(ladle_slots.pop(0))
        hold_down()
        press(keys.I)
        click(1498, 879)  # menu
        for _ in range(MAX_LEVELUPS_PER_MEAL):
            click(1722, y_menu + menu_offset)  # level up meal
        x_meal += 190
        y_menu += 180
        _close_menus()
