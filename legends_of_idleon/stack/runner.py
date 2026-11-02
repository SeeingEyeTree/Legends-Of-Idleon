"""Run automation flows one verified step at a time.

A :class:`Step` performs an action, then waits for the screen to show what it
expects (a template appearing and/or another disappearing). If it doesn't, the
action is retried; after the last retry :class:`StepFailed` is raised with the
screenshot saved so the template or coordinates can be fixed.

Every step saves ``before``/``after`` screenshots and a JSON log line under
``runs/<timestamp>/`` (git-ignored).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from . import ui


class StepFailed(RuntimeError):
    """A step's expectation was not met after all retries."""


class Context:
    """Everything a step needs: the screen, input, and a run log.

    ``grab``/``click``/``press`` are injectable so flows can be tested without
    a display; by default they use the real Windows implementations.
    """

    def __init__(
        self, run_dir=None, grab=None, click=None, press=None, template_dir=None, move=None
    ):
        self._grab = grab
        self._click = click
        self._press = press
        self._move = move
        self.template_dir = template_dir
        self._templates = {}
        base = Path(run_dir) if run_dir else Path("runs") / time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = base
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.step_no = 0

    # --- screen -----------------------------------------------------------
    def capture(self):
        if self._grab is None:
            from ..screen import grab_screen

            return ui.normalize(grab_screen())
        return ui.normalize(self._grab())

    def template(self, name):
        if name not in self._templates:
            self._templates[name] = ui.load_template(name, self.template_dir)
        return self._templates[name]

    def find(self, name, screen=None):
        screen = self.capture() if screen is None else screen
        return ui.find(screen, self.template(name))

    def wait_for(self, name, timeout=5.0):
        return ui.wait_until(lambda: self.find(name), timeout)

    def wait_gone(self, name, timeout=5.0):
        return ui.wait_until(lambda: self.find(name) is None, timeout)

    # --- input ------------------------------------------------------------
    def click(self, x, y):
        """Click a point given in reference (1920x1080) coordinates."""
        if self._click is None:
            from ..controls import click

            click(x, y, reference=ui.REFERENCE)
        else:
            self._click(x, y)

    def move(self, x, y):
        """Move the mouse to a point given in reference (1920x1080) coordinates."""
        if self._move is None:
            from ..controls import move_to

            move_to(x, y, reference=ui.REFERENCE)
        else:
            self._move(x, y)

    def press(self, key):
        if self._press is None:
            from ..controls import press

            press(key, delay=0.05)
        else:
            self._press(key)

    # --- logging ----------------------------------------------------------
    def save(self, image, label):
        path = self.run_dir / f"{self.step_no:03d}_{label}.png"
        cv2.imwrite(str(path), image)
        return path

    def log(self, **event):
        with (self.run_dir / "log.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")


@dataclass
class Step:
    name: str
    action: Callable[[Context], None]
    expect: str | None = None  # template that must appear afterwards
    expect_gone: str | None = None  # template that must disappear afterwards
    timeout: float = 5.0
    retries: int = 2
    settle: float = 0.3  # seconds to let the UI animate before checking
    extra: dict = field(default_factory=dict)

    def run(self, ctx):
        ctx.step_no += 1
        ctx.save(ctx.capture(), f"{self.name}_before")
        for attempt in range(1, self.retries + 2):
            self.action(ctx)
            time.sleep(self.settle)
            if self._satisfied(ctx):
                ctx.save(ctx.capture(), f"{self.name}_after")
                ctx.log(step=self.name, ok=True, attempt=attempt)
                return
            ctx.log(step=self.name, ok=False, attempt=attempt)
        shot = ctx.save(ctx.capture(), f"{self.name}_FAILED")
        raise StepFailed(f"step '{self.name}' failed after {self.retries + 1} tries (see {shot})")

    def _satisfied(self, ctx):
        if self.expect and not ctx.wait_for(self.expect, self.timeout):
            return False
        if self.expect_gone and not ctx.wait_gone(self.expect_gone, self.timeout):
            return False
        return True


# --- step builders --------------------------------------------------------


def click_template(template, name=None, **kwargs):
    """Click a template wherever it currently is on screen."""

    def action(ctx):
        match = ctx.wait_for(template, kwargs.get("timeout", 5.0))
        if match is None:
            return  # the expectation check will fail and trigger a retry
        ctx.click(*match.click_point)

    return Step(name or f"click_{template}", action, **kwargs)


def click_at(x, y, name, **kwargs):
    """Click a fixed point (reference 1920x1080 coordinates)."""
    return Step(name, lambda ctx: ctx.click(x, y), **kwargs)


def press_key(key, name, **kwargs):
    return Step(name, lambda ctx: ctx.press(key), **kwargs)


def run_flow(steps, ctx=None):
    """Run ``steps`` in order, stopping at the first failure."""
    ctx = ctx or Context()
    for step in steps:
        step.run(ctx)
    return ctx
