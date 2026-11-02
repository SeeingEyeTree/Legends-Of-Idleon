import cv2
import numpy as np
import pytest

from legends_of_idleon.stack import devtools, ui
from legends_of_idleon.stack.runner import Context, StepFailed, click_template, run_flow
from legends_of_idleon.vision import find_color_blobs


def _noise(shape, seed):
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


def _scene(patch=None, at=(0, 0)):
    screen = np.full((1080, 1920, 3), 30, np.uint8)
    if patch is not None:
        x, y = at
        screen[y : y + patch.shape[0], x : x + patch.shape[1]] = patch
    return screen


@pytest.fixture()
def templates(tmp_path):
    folder = tmp_path / "ui"
    folder.mkdir()
    buttons = {"play": _noise((40, 120, 3), 1), "name_field": _noise((30, 200, 3), 2)}
    for name, patch in buttons.items():
        cv2.imwrite(str(folder / f"{name}.png"), patch)
    return folder, buttons


def test_find_returns_click_point(templates):
    folder, buttons = templates
    tpl = ui.load_template("play", folder)
    match = ui.find(_scene(buttons["play"], (700, 500)), tpl)
    assert match and match.center == (760, 520)
    assert ui.find(_scene(), tpl) is None


def test_normalize_scales_other_resolutions(templates):
    folder, buttons = templates
    big = cv2.resize(
        _scene(buttons["play"], (700, 500)), (2560, 1440), interpolation=cv2.INTER_AREA
    )
    shot = np.dstack([big, np.full(big.shape[:2], 255, np.uint8)])  # BGRA
    tpl = ui.load_template("play", folder)
    match = ui.find(ui.normalize(shot), tpl, threshold=0.7)
    assert match and abs(match.x - 700) <= 2 and abs(match.y - 500) <= 2


def test_flow_follows_the_screen(templates, tmp_path):
    folder, buttons = templates
    state = {"scene": "menu"}
    scenes = {
        "menu": _scene(buttons["play"], (700, 500)),
        "created": _scene(buttons["name_field"], (100, 100)),
    }
    clicks = []

    def click(x, y):
        clicks.append((x, y))
        if abs(x - 760) < 60 and abs(y - 520) < 20:
            state["scene"] = "created"

    ctx = Context(
        tmp_path / "run", grab=lambda: scenes[state["scene"]], click=click, template_dir=folder
    )
    steps = [click_template("play", expect="name_field", expect_gone="play", timeout=0.5)]
    run_flow(steps, ctx)
    assert clicks == [(760, 520)]
    assert (tmp_path / "run" / "log.jsonl").exists()


def test_failed_step_retries_then_saves_screenshot(templates, tmp_path):
    folder, buttons = templates
    scene = _scene(buttons["play"], (700, 500))
    clicks = []
    ctx = Context(
        tmp_path / "run",
        grab=lambda: scene,
        click=lambda x, y: clicks.append((x, y)),
        template_dir=folder,
    )
    step = click_template("play", expect="name_field", timeout=0.2, retries=2)
    step.settle = 0
    with pytest.raises(StepFailed):
        run_flow([step], ctx)
    assert len(clicks) == 3
    assert list((tmp_path / "run").glob("*_FAILED.png"))


def test_color_blobs():
    img = np.zeros((200, 300, 3), np.uint8)
    img[50:90, 100:180] = (0, 200, 0)  # green button (BGR)
    img[150:153, 10:13] = (0, 200, 0)  # too small
    blobs = find_color_blobs(img, (40, 100, 100), (80, 255, 255), min_area=100)
    assert len(blobs) == 1 and blobs[0][:4] == (100, 50, 80, 40)


def test_crop_then_find_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ui, "UI_TEMPLATE_DIR", tmp_path / "ui")
    shot = _scene(_noise((40, 120, 3), 5), (300, 300))
    path = tmp_path / "shot.png"
    cv2.imwrite(str(path), shot)
    devtools.main(["crop", str(path), "btn", "300", "300", "420", "340"])
    devtools.main(["find", str(path), "btn"])
    assert "MATCH" in capsys.readouterr().out
