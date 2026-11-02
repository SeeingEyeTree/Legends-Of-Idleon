"""Polling for the keys the user is physically holding (Windows only)."""

from __future__ import annotations

import win32api

# Keys that can be reported, in priority order: if several are held at once,
# the first one in this string wins.
WATCHED_KEYS = "ABCD EFGHIJKLMOPQRS"


def key_check() -> str | None:
    """Return the highest-priority watched key currently held down, or ``None``."""
    for char in WATCHED_KEYS:
        if win32api.GetAsyncKeyState(ord(char)):
            return char
    return None
