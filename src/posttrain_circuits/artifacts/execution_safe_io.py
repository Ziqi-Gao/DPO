"""Atomic publication primitives that are part of execution safety."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    """Publish one owned sibling path atomically without replacing a destination."""

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
        os.link(
            source.name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.unlink(source.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def publish_torch_once(path: Path, payload: Any) -> None:
    """Serialize to a unique sibling and atomically publish without clobbering."""

    import torch

    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", dir=path.parent
    )
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


__all__ = ["atomic_torch_save", "publish_path_no_clobber", "publish_torch_once"]
