"""Minimal fail-closed Git provenance queries without scientific dependencies."""

from __future__ import annotations

import subprocess
from pathlib import Path


def require_git_output(code_root: Path, arguments: tuple[str, ...]) -> str:
    """Return one Git query result for ``code_root`` or raise on failure."""

    result = subprocess.run(
        ("/usr/bin/git", "-C", str(code_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


__all__ = ["require_git_output"]
