"""Repeatedly drag an item into the vials and collect the result."""

from __future__ import annotations

import time

import pyautogui

from .. import keys
from ..controls import click, move_to, press, scale_point


def run_vials(loops):
    """Run the vial routine ``loops`` times. Start with the inventory open."""
    time.sleep(2)
    for _ in range(loops):
        move_to(1320, 360)
        x, y = scale_point(280, 450)
        pyautogui.dragTo(x, y, 0.3, button="left")
        time.sleep(3)
        press(keys.ESCAPE)
        click(730, 780)
        press(keys.I)
        time.sleep(2)
