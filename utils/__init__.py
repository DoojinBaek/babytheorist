"""Auxiliary tools for BabyTheorist.

These are *optional* utilities — the core `babytheorist/` library has only `numpy`
and `scipy.ndimage` as dependencies. Anything imported from `utils.*` may
pull in extra dependencies (e.g., matplotlib for `viz`), so import only the
modules you actually need.
"""

from .viz import ARC_COLORS, render_dataset, render_task

__all__ = ["ARC_COLORS", "render_dataset", "render_task"]
