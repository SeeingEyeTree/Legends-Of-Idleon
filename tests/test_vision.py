import numpy as np

from legends_of_idleon.vision import dedupe_positions, find_template, match_template


def test_dedupe_merges_neighbouring_pixels():
    assert dedupe_positions([10, 11, 12, 40, 41, 90], min_gap=15) == [10, 40, 90]


def test_dedupe_empty():
    assert dedupe_positions([], min_gap=5) == []


def _scene():
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, (60, 80), dtype=np.uint8)
    needle = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    scene[20:28, 30:38] = needle
    return scene, needle


def test_find_template_locates_patch():
    scene, needle = _scene()
    assert find_template(scene, needle, 0.99) == (30, 20)


def test_find_template_returns_none_when_missing():
    scene, _ = _scene()
    other = np.full((8, 8), 7, dtype=np.uint8)
    other[::2] = 200
    assert find_template(scene, other, 0.99) is None


def test_match_template_returns_arrays():
    scene, needle = _scene()
    ys, xs = match_template(scene, needle, 0.99)
    assert list(ys) == [20] and list(xs) == [30]


def test_scan_code_mapping(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setitem(__import__("sys").modules, "pyautogui", MagicMock())
    from legends_of_idleon import keys
    from legends_of_idleon.controls import _scan_code

    assert _scan_code("T") == (keys.T, True)
    assert _scan_code("t") == (keys.T, False)
    assert _scan_code("7") == (keys.SEVEN, False)
    assert _scan_code(" ") == (keys.SPACE, False)
