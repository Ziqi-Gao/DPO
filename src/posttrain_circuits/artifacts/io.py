"""Durable, atomic writes shared by project-owned artifact formats."""
from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.execution_safe_io import (
    atomic_torch_save,
    publish_path_no_clobber,
    publish_torch_once,
)


def read_regular_bytes_nofollow(
    path: Path,
    *,
    context: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    """Read one single-link regular file through a no-follow descriptor walk."""

    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
    ):
        raise ValueError(f"{context} path or size bound is invalid")
    parts = tuple(part for part in candidate.parts[1:] if part not in {"", "."})
    if not parts:
        raise ValueError(f"{context} must name a file below /")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    file_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        file_flags |= os.O_NONBLOCK
    directory = -1
    descriptor = -1
    try:
        directory = os.open("/", directory_flags)
        for component in parts[:-1]:
            following = os.open(component, directory_flags, dir_fd=directory)
            try:
                if not stat.S_ISDIR(os.fstat(following).st_mode):
                    raise ValueError(
                        f"{context} ancestor is not a no-follow directory"
                    )
            except BaseException:
                os.close(following)
                raise
            os.close(directory)
            directory = following
        descriptor = os.open(parts[-1], file_flags, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{context} must be one non-linked regular file")
        if before.st_size > max_bytes:
            raise ValueError(f"{context} exceeds {max_bytes} bytes")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > max_bytes:
                raise ValueError(f"{context} exceeds {max_bytes} bytes")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before or observed != before.st_size:
            raise ValueError(f"{context} changed while being read")
        return b"".join(chunks)
    except OSError as error:
        raise ValueError(f"cannot read no-follow {context}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory >= 0:
            os.close(directory)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_descriptor(descriptor: int, value: Any) -> None:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, value: Any) -> None:
    """Write project JSON atomically without changing its historical byte format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        _write_json_descriptor(descriptor, value)
        os.replace(temporary_name, path)
        _fsync_parent(path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _create_json_no_clobber(path: Path, value: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        _write_json_descriptor(descriptor, value)
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            os.unlink(temporary_name)
            _fsync_parent(path)
            return False
        os.unlink(temporary_name)
        _fsync_parent(path)
        return True
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _read_regular_json(path: Path) -> Any:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"existing JSON artifact is not a readable regular file: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"existing JSON artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            return json.load(handle)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing JSON artifact is invalid: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def publish_json_once(path: Path, value: Any) -> Any:
    """Atomically create JSON, reusing only an identical existing artifact.

    The destination is never replaced. A concurrent identical publication is
    idempotent; a different existing payload is a hard conflict.
    """

    if _create_json_no_clobber(path, value):
        return value
    existing = _read_regular_json(path)
    if existing == value:
        return existing
    raise FileExistsError(f"conflicting JSON artifact already exists: {path}")
