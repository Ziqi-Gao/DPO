"""Durable, atomic writes shared by project-owned artifact formats."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import json
import os
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Atomically publish a torch-serializable value from one foreground process."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_parent(path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def publish_path_no_clobber(source: Path, destination: Path) -> None:
    """Atomically publish one owned staging path without replacing a destination.

    Both paths must be direct children of the same real directory.  Linux
    ``renameat2(RENAME_NOREPLACE)`` gives files and directories the same
    publication-once contract and closes the exists-check/rename race.  A
    regular-file hard-link fallback remains safe on platforms without
    ``renameat2``; directory publication fails closed there.
    """

    source = Path(source).absolute()
    destination = Path(destination).absolute()
    if source.name in {"", ".", ".."} or destination.name in {"", ".", ".."}:
        raise ValueError("publication paths must name direct child entries")
    if source.parent != destination.parent:
        raise ValueError("staging and destination paths must share one parent")
    if source == destination:
        raise ValueError("staging and destination paths must be distinct")
    parent = source.parent
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(parent, flags)
    try:
        source_status = os.stat(
            source.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not (stat.S_ISREG(source_status.st_mode) or stat.S_ISDIR(source_status.st_mode)):
            raise ValueError("staging publication source must be a regular file or directory")

        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                parent_descriptor,
                os.fsencode(source.name),
                parent_descriptor,
                os.fsencode(destination.name),
                1,  # RENAME_NOREPLACE
            )
            if result != 0:
                observed_errno = ctypes.get_errno()
                if observed_errno == errno.EEXIST:
                    raise FileExistsError(
                        observed_errno,
                        os.strerror(observed_errno),
                        str(destination),
                    )
                if observed_errno not in {errno.ENOSYS, errno.EINVAL}:
                    raise OSError(
                        observed_errno,
                        os.strerror(observed_errno),
                        str(destination),
                    )
            else:
                os.fsync(parent_descriptor)
                return

        if stat.S_ISDIR(source_status.st_mode):
            raise RuntimeError(
                "atomic no-clobber directory publication requires renameat2(RENAME_NOREPLACE)"
            )
        try:
            os.link(
                source.name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise
        os.unlink(source.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def publish_torch_once(path: Path, payload: Any) -> None:
    """Serialize to a unique sibling and atomically publish without clobbering."""

    import torch

    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.stage-", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        publish_path_no_clobber(temporary_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        raise
