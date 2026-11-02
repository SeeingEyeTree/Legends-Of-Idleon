"""Buying from the shops."""

from __future__ import annotations

from ..controls import click, hold_down

SLOT_HEIGHT = 130
FIRST_SLOT_Y = 390


def buy_slot(hold_seconds, slot):
    """Hold the buy button for ``slot`` (0-based) for ``hold_seconds`` and confirm."""
    y = slot * SLOT_HEIGHT + FIRST_SLOT_Y
    hold_down(hold_seconds, 1047, y)
    click(689, y)


def buy_shards_and_teleports():
    click(1137, 800)
    click(657, 855)
    hold_down(2.5, 1047, 603)
    hold_down(1, 1047, 726)
    click(687, 783)
    click(687, 639)
