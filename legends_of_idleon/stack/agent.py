"""File-based job agent: lets a remote operator drive the game through a shared folder.

Run it on the game machine in your own terminal::

    python -m legends_of_idleon agent C:\\dev\\Legends-Of-Idleon\\bridge

It watches ``<bridge>/jobs/*.json``, performs the job, and writes
``<bridge>/results/<job>.json`` (plus any images under ``<bridge>/out/``).

It is deliberately NOT a remote shell. Only the operations in ``OPS`` exist:
screenshots, mouse clicks/moves, key presses, template lookups and sleeping.
Press Ctrl+C in the terminal to stop it at any time.

Job format::

    {"op": "snap", "name": "menu.png", "box": [x1, y1, x2, y2]}
    {"op": "click", "x": 960, "y": 540}
    {"op": "key", "key": "ESCAPE"}
    {"op": "find", "template": "create_button"}

Coordinates are in 1920x1080 reference pixels.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import cv2

from .. import keys
from . import ui
from .runner import Context

MAX_SLEEP = 30.0


def _key_code(name):
    name = str(name).upper()
    code = getattr(keys, name, None) if name.isalnum() else None
    if not isinstance(code, int):
        raise ValueError(f"unknown key: {name!r}")
    return code


def op_ping(job, ctx, bridge):
    return {"pong": True, "reference": list(ui.REFERENCE)}


def op_snap(job, ctx, bridge):
    image = ctx.capture()
    box = job.get("box")
    if box:
        x1, y1, x2, y2 = (int(v) for v in box)
        image = image[y1:y2, x1:x2]
        if image.size == 0:
            raise ValueError("empty box")
    name = Path(job.get("name", "snap.png")).name  # no path traversal
    out_dir = bridge / "out"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / name
    cv2.imwrite(str(path), image)
    return {"file": f"out/{name}", "size": [image.shape[1], image.shape[0]]}


def op_click(job, ctx, bridge):
    ctx.click(int(job["x"]), int(job["y"]))
    return {}


def op_move(job, ctx, bridge):
    ctx.move(int(job["x"]), int(job["y"]))
    return {}


def op_key(job, ctx, bridge):
    ctx.press(_key_code(job["key"]))
    return {}


def op_find(job, ctx, bridge):
    template = ctx.template(job["template"])
    best = ui.score(ctx.capture(), template)
    if best is None:
        return {"found": False, "reason": "template larger than search area"}
    return {
        "found": best.score >= template.threshold,
        "score": round(best.score, 4),
        "threshold": template.threshold,
        "center": list(best.center),
        "click_point": list(best.click_point),
    }


def op_sleep(job, ctx, bridge):
    time.sleep(min(float(job.get("seconds", 1)), MAX_SLEEP))
    return {}


OPS = {
    "ping": op_ping,
    "snap": op_snap,
    "click": op_click,
    "move": op_move,
    "key": op_key,
    "find": op_find,
    "sleep": op_sleep,
}


def handle(job, ctx, bridge):
    op = OPS.get(job.get("op"))
    if op is None:
        raise ValueError(f"unknown op: {job.get('op')!r} (allowed: {sorted(OPS)})")
    return op(job, ctx, bridge)


def _write_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def process_pending(bridge, ctx):
    """Run every queued job once; return how many were handled."""
    jobs, results, done = bridge / "jobs", bridge / "results", bridge / "done"
    handled = 0
    for path in sorted(jobs.glob("*.json")):
        result = {"job": path.name}
        try:
            result["data"] = handle(json.loads(path.read_text()), ctx, bridge)
            result["ok"] = True
        except Exception as error:  # report every failure back to the operator
            result.update(ok=False, error=f"{type(error).__name__}: {error}")
            result["trace"] = traceback.format_exc(limit=3)
        _write_json(results / path.name, result)
        path.replace(done / path.name)
        print(f"{path.name}: {'ok' if result['ok'] else result['error']}")
        handled += 1
    return handled


def run_agent(bridge_dir, poll=0.3, ctx=None, once=False):
    bridge = Path(bridge_dir)
    for name in ("jobs", "results", "done", "out", "runs"):
        (bridge / name).mkdir(parents=True, exist_ok=True)
    ctx = ctx or Context(run_dir=bridge / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    print(f"Agent watching {bridge / 'jobs'}  (Ctrl+C to stop)")
    while True:
        handled = process_pending(bridge, ctx)
        if once:
            return handled
        time.sleep(poll)
