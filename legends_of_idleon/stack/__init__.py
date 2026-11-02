"""Screenshot-driven automation stack.

Every action is a :class:`~legends_of_idleon.stack.runner.Step`: look at the
screen, act, then confirm the screen changed the way we expected. Failures save
the screenshot so the template or coordinates can be fixed.
"""
