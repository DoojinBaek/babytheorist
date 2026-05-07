"""Selector resolution for BabyTheorist.

Given a Selector and a list of Shapes, return the indices of shapes
that match the selector criteria.
"""

from __future__ import annotations

from .core import MULTI_SELECTORS, Selector, Shape


def select(selector: Selector, shapes: list[Shape]) -> list[int]:
    """Return indices of shapes matching the selector.

    For multi-select selectors (ALL, by_color, is_hollow), returns all
    matching indices.  For single-select selectors (topmost, bottommost,
    leftmost, rightmost), returns a list with exactly one element.

    Returns an empty list if no shapes match or the list is empty.
    """
    if not shapes:
        return []

    sel = selector.type

    # -- Multi-select --
    if sel == "ALL":
        return list(range(len(shapes)))

    if sel == "by_color":
        color = selector.param
        return [i for i, s in enumerate(shapes) if s.primary_color == color]

    if sel == "is_hollow":
        return [i for i, s in enumerate(shapes) if s.is_hollow]

    # -- Single-select: raster index (1-based, stable from scene creation) --
    if sel == "by_index":
        k = selector.param
        return [k - 1] if 1 <= k <= len(shapes) else []

    # -- Single-select (positional) --
    if sel == "topmost":
        idx = min(range(len(shapes)),
                  key=lambda i: (shapes[i].position[0], shapes[i].position[1]))
        return [idx]

    if sel == "bottommost":
        idx = max(range(len(shapes)),
                  key=lambda i: (shapes[i].position[0], shapes[i].position[1]))
        return [idx]

    if sel == "leftmost":
        idx = min(range(len(shapes)),
                  key=lambda i: (shapes[i].position[1], shapes[i].position[0]))
        return [idx]

    if sel == "rightmost":
        idx = max(range(len(shapes)),
                  key=lambda i: (shapes[i].position[1], shapes[i].position[0]))
        return [idx]

    raise ValueError(f"Unknown selector type: {sel!r}")


def select_one(selector: Selector, shapes: list[Shape]) -> int | None:
    """Return the single index for a positional (single-select) selector.

    Returns None if *shapes* is empty.
    Raises ValueError if the selector is a multi-select type.
    """
    if selector.type in MULTI_SELECTORS:
        raise ValueError(
            f"select_one() requires a single-select selector, "
            f"got {selector.type!r}"
        )

    if not shapes:
        return None

    result = select(selector, shapes)
    return result[0] if result else None
