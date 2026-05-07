"""BabyTheorist version identifiers.

Keep code, benchmark spec, and default dataset release naming together so
generated artifacts can record the exact development milestone they came from.
"""

from __future__ import annotations


__version__ = "0.5.5"
BENCHMARK_SPEC_VERSION = "0.5.5"


def default_dataset_release_version(mode: str | None) -> str:
    """Return the default dataset release name for a benchmark mode."""
    label = mode if mode is not None else "custom"
    return f"BabyTheorist-{label}-v{__version__}"


def default_dataset_dir_name(mode: str) -> str:
    """Return the default on-disk directory name for a benchmark mode."""
    return f"{mode}-v{__version__}"
