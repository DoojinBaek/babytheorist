"""Shared shape generation profile defaults.

This module is deliberately dependency-light so both `generate.py` and
`env/env.py` can import the same metadata defaults without importing the full
`babytheorist` package.
"""

from __future__ import annotations


SHAPE_PROFILE_STANDARD = "standard"
SHAPE_PROFILES = (SHAPE_PROFILE_STANDARD,)


def shape_profile_defaults(profile: str) -> dict[str, object]:
    """Default shape sampler knobs for a named generation profile."""
    if profile == SHAPE_PROFILE_STANDARD:
        return {
            "shape_size_range": (4, 16),
            "shape_compactness": 8.0,
            "shape_bordered_hollow_min_size": 9,
            "shape_bordered_hollow_random_inner_size": True,
            "shape_max_bbox_span": 5,
        }
    raise ValueError(f"unknown shape profile {profile!r}")
