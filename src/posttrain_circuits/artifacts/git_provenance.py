"""Minimal fail-closed Git provenance queries without scientific dependencies."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def require_git_bytes(code_root: Path, arguments: tuple[str, ...]) -> bytes:
    """Return raw bytes from one fail-closed Git query."""

    result = subprocess.run(
        (
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(code_root),
            *arguments,
        ),
        check=True,
        capture_output=True,
        env=_git_environment(),
    )
    return result.stdout


def require_git_output(code_root: Path, arguments: tuple[str, ...]) -> str:
    """Return strict UTF-8 text from one fail-closed Git query."""

    return require_git_bytes(code_root, arguments).decode("utf-8", errors="strict").strip()


def unsafe_untracked_paths(code_root: Path) -> tuple[str, ...]:
    """List untracked paths even when ignore rules hide them from ``status``.

    The private scheduler entrypoints establish an isolated bytecode cache
    before project imports, so source-tree bytecode is inert.  The one local
    Codex configuration file is also non-importable.  Every other untracked
    path is unsafe for a hash-bound execution checkout.
    """

    raw = require_git_bytes(code_root, ("ls-files", "--others", "-z", "--"))
    if raw and not raw.endswith(b"\0"):
        raise ValueError("Git returned a malformed untracked-path stream")
    unsafe: list[str] = []
    for encoded in raw.split(b"\0")[:-1]:
        if not encoded:
            raise ValueError("Git returned an empty untracked path")
        path = encoded.decode("utf-8", errors="strict")
        parsed = Path(path)
        if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
            raise ValueError("Git returned a non-canonical untracked path")
        if path == ".codex/config.toml":
            continue
        if parsed.suffix == ".pyc" and "__pycache__" in parsed.parts:
            continue
        unsafe.append(path)
    return tuple(unsafe)


__all__ = ["require_git_bytes", "require_git_output", "unsafe_untracked_paths"]
