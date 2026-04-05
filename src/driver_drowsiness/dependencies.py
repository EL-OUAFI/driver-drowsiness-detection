"""Helpers for importing optional runtime dependencies with clear errors."""

from __future__ import annotations

import importlib

from .exceptions import MissingDependencyError


def require_module(module_name: str, install_hint: str | None = None):
    """Import a module and raise a project-specific error when unavailable."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = install_hint or "pip install -e ."
        message = (
            f"Missing optional dependency '{module_name}'. Install the project dependencies "
            f"first, for example with: {hint}"
        )
        raise MissingDependencyError(message) from exc
