"""Fast Windows screen capture using the GDI API.

Based on code by Frannecklp.
"""

from __future__ import annotations

import time

import numpy as np
import win32api
import win32con
import win32gui
import win32ui


def grab_screen(region: tuple[int, int, int, int] | None = None, timer: bool = False) -> np.ndarray:
    """Capture the screen and return it as a ``(height, width, 4)`` BGRA array.

    ``region`` is ``(left, top, right, bottom)`` in pixels. Without it the whole
    virtual desktop (all monitors) is captured.
    """
    start = time.time() if timer else None

    if region:
        left, top, right, bottom = region
        width = right - left + 1
        height = bottom - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    desktop = win32gui.GetDesktopWindow()
    desktop_dc = win32gui.GetWindowDC(desktop)
    src_dc = win32ui.CreateDCFromHandle(desktop_dc)
    mem_dc = src_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(src_dc, width, height)
    mem_dc.SelectObject(bitmap)
    mem_dc.BitBlt((0, 0), (width, height), src_dc, (left, top), win32con.SRCCOPY)

    pixels = np.frombuffer(bitmap.GetBitmapBits(True), dtype="uint8")
    image = pixels.reshape((height, width, 4))

    src_dc.DeleteDC()
    mem_dc.DeleteDC()
    win32gui.ReleaseDC(desktop, desktop_dc)
    win32gui.DeleteObject(bitmap.GetHandle())

    if start is not None:
        print("Loop time", time.time() - start)
    return image
