import json

import cv2
import numpy as np

from legends_of_idleon import keys
from legends_of_idleon.stack.agent import process_pending, run_agent
from legends_of_idleon.stack.runner import Context


def _setup(tmp_path):
    bridge = tmp_path / "bridge"
    for name in ("jobs", "results", "done", "out"):
        (bridge / name).mkdir(parents=True)
    rng = np.random.default_rng(3)
    patch = rng.integers(0, 255, (40, 120, 3), dtype=np.uint8)
    screen = np.full((1080, 1920, 3), 20, np.uint8)
    screen[300:340, 500:620] = patch
    templates = tmp_path / "ui"
    templates.mkdir()
    cv2.imwrite(str(templates / "btn.png"), patch)
    log = []
    ctx = Context(
        tmp_path / "runs",
        grab=lambda: screen,
        click=lambda x, y: log.append(("click", x, y)),
        move=lambda x, y: log.append(("move", x, y)),
        press=lambda k: log.append(("key", k)),
        type_text=lambda t: log.append(("type", t)),
        template_dir=templates,
    )
    return bridge, ctx, log


def _job(bridge, job_id, **job):
    (bridge / "jobs" / f"{job_id}.json").write_text(json.dumps(job))


def _result(bridge, name):
    return json.loads((bridge / "results" / f"{name}.json").read_text())


def test_agent_runs_jobs_in_order_and_reports(tmp_path):
    bridge, ctx, log = _setup(tmp_path)
    _job(bridge, "001", op="click", x=10, y=20)
    _job(bridge, "002", op="key", key="escape")
    _job(bridge, "003", op="snap", name="a.png", box=[500, 300, 620, 340])
    _job(bridge, "004", op="find", template="btn")
    assert process_pending(bridge, ctx) == 4
    assert log == [("click", 10, 20), ("key", keys.ESCAPE)]
    assert _result(bridge, "003")["data"]["size"] == [120, 40]
    assert cv2.imread(str(bridge / "out" / "a.png")).shape[:2] == (40, 120)
    found = _result(bridge, "004")["data"]
    assert found["found"] and found["center"] == [560, 320]
    assert not list((bridge / "jobs").glob("*.json"))


def test_agent_rejects_unknown_ops_and_keys(tmp_path):
    bridge, ctx, log = _setup(tmp_path)
    _job(bridge, "a", op="shell", cmd="whoami")
    _job(bridge, "b", op="key", key="__class__")
    _job(bridge, "c", op="snap", name="../../evil.png")
    process_pending(bridge, ctx)
    assert not _result(bridge, "a")["ok"] and "unknown op" in _result(bridge, "a")["error"]
    assert not _result(bridge, "b")["ok"]
    assert _result(bridge, "c")["data"]["file"] == "out/evil.png"  # path stripped
    assert log == []


def test_run_agent_once(tmp_path):
    bridge, ctx, _ = _setup(tmp_path)
    _job(bridge, "p", op="ping")
    assert run_agent(bridge, ctx=ctx, once=True) == 1
    assert _result(bridge, "p")["ok"]


def test_type_op(tmp_path):
    bridge, ctx, log = _setup(tmp_path)
    _job(bridge, "t", op="type", text="TreeBot")
    process_pending(bridge, ctx)
    assert log == [("type", "TreeBot")]
    assert _result(bridge, "t")["data"] == {"typed": 7}
