"""Keyboard input through the Windows ``SendInput`` API using hardware scan codes.

Scan codes (rather than virtual-key codes) are used because many games ignore
virtual-key input. Windows only.

Not written by the project author. Adapted from:
  * http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
  * http://www.flint.jp/misc/?q=dik&lang=en
"""

from __future__ import annotations

import ctypes

# --- Scan codes -------------------------------------------------------------

# Digits
ZERO = 0x0B
ONE = 0x02
TWO = 0x03
THREE = 0x04
FOUR = 0x05
FIVE = 0x06
SIX = 0x07
SEVEN = 0x08
EIGHT = 0x09
NINE = 0x0A

DIGITS = {
    "0": ZERO,
    "1": ONE,
    "2": TWO,
    "3": THREE,
    "4": FOUR,
    "5": FIVE,
    "6": SIX,
    "7": SEVEN,
    "8": EIGHT,
    "9": NINE,
}

# Misc
ESCAPE = 0x01
MINUS = 0x0C
EQUALS = 0x0D
BACK = 0x0E
TAB = 0x0F
ENTER = 0x1C
LCTRL = 0x1D
LSHIFT = 0x2A
LALT = 0x38
SPACE = 0x39
CAPSLOCK = 0x3A
F4 = 0x3E
DELETE = 0xD3

# Letters
A = 0x1E
B = 0x30
C = 0x2E
D = 0x20
E = 0x12
F = 0x21
G = 0x22
H = 0x23
I = 0x17  # noqa: E741
J = 0x24
K = 0x25
L = 0x26
M = 0x32
N = 0x31
O = 0x18  # noqa: E741
P = 0x19
Q = 0x10
R = 0x13
S = 0x1F
T = 0x14
U = 0x16
V = 0x2F
W = 0x11
X = 0x2D
Y = 0x15
Z = 0x2C

# --- SendInput plumbing -----------------------------------------------------

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008

_PUL = ctypes.POINTER(ctypes.c_ulong)


class _KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", _PUL),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", _PUL),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyBdInput), ("mi", _MouseInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _InputUnion)]


def _send(scan_code: int, flags: int) -> None:
    extra = ctypes.c_ulong(0)
    union = _InputUnion()
    union.ki = _KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
    event = _Input(ctypes.c_ulong(_INPUT_KEYBOARD), union)
    # ``windll`` only exists on Windows, so it is looked up lazily to keep this
    # module importable (e.g. for linting) on other platforms.
    ctypes.windll.user32.SendInput(1, ctypes.pointer(event), ctypes.sizeof(event))


def press_key(scan_code: int) -> None:
    """Send a key-down event (the key stays down until :func:`release_key`)."""
    _send(scan_code, _KEYEVENTF_SCANCODE)


def release_key(scan_code: int) -> None:
    """Send a key-up event."""
    _send(scan_code, _KEYEVENTF_SCANCODE | _KEYEVENTF_KEYUP)
