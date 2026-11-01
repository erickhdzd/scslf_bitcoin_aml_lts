"""Utilities for producing reviewer-safe artifact metadata."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def portable_path(path: Path | str) -> str:
    """Return a reviewer-safe path without exposing local user directories.

    Paths inside the project are represented relative to the project root.
    External paths are reduced to an ``external/<name>`` placeholder because
    their absolute location is machine-specific and may reveal user metadata.
    """
    candidate = Path(path).expanduser().resolve()
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external/{candidate.name}"
