"""World 2 boss: attack-order macro and loot collection."""

from __future__ import annotations

import pyautogui

from ..controls import click
from .player import find_player

# Button for each digit of the attack order: "1", "2", anything else -> third button.
BOSS_BUTTONS = {"1": (1642, 881), "2": (1642, 1023)}
DEFAULT_BOSS_BUTTON = (1663, 1193)


def boss(order):
    """Click the boss buttons in sequence, e.g. ``boss("2232213332213")``."""
    for step in order:
        click(*BOSS_BUTTONS.get(step, DEFAULT_BOSS_BUTTON))


def active_collect():
    """Drag the mouse in front of the player to pick up nearby drops."""
    x, y = find_player()
    pyautogui.moveTo(x - 130, y - 50)  # find_player() returns screen pixels
    pyautogui.mouseDown()
    pyautogui.move(500, 1, 1.5)
    pyautogui.mouseUp()
