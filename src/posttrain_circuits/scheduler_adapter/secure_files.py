"""Linux descriptor-walk primitives for adapter trust-boundary files."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError


def published_json_bytes(payload: object) -> bytes:
    """Exact durable JSON bytes used by descriptor-safe adapter publishers."""

    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _absolute_parts(path: Path, *, context: str) -> tuple[str, ...]:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise AdapterValidationError(f"{context} path must be absolute and normalized")
    parts = candidate.parts
    if not parts or parts[0] != "/":
        raise AdapterValidationError(f"{context} path must start at the filesystem root")
    return tuple(part for part in parts[1:] if part not in {"", "."})


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _path_flags() -> int:
    if not hasattr(os, "O_PATH"):
        raise AdapterValidationError("descriptor-safe adapter reads require Linux O_PATH")
    flags = os.O_PATH
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _read_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _open_root(*, context: str) -> int:
    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as error:
        raise AdapterValidationError(f"cannot open filesystem root for {context}: {error}") from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise AdapterValidationError(f"filesystem root is not a directory for {context}")
    return descriptor


def _open_directory_component(parent_fd: int, name: str, *, context: str) -> int:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise AdapterValidationError(
            f"cannot open no-follow directory component {name!r} for {context}: {error}"
        ) from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise AdapterValidationError(
            f"no-follow path component is not a directory for {context}: {name!r}"
        )
    return descriptor


def open_directory_nofollow(path: Path, *, context: str) -> int:
    """Open every ancestor relative to a held directory descriptor."""

    parts = _absolute_parts(Path(path), context=context)
    current = _open_root(context=context)
    try:
        for part in parts:
            following = _open_directory_component(current, part, context=context)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def open_parent_directory_nofollow(path: Path, *, context: str) -> tuple[int, str]:
    parts = _absolute_parts(Path(path), context=context)
    if not parts:
        raise AdapterValidationError(f"{context} requires a named target below /")
    parent = open_directory_nofollow(Path("/").joinpath(*parts[:-1]), context=context)
    return parent, parts[-1]


def open_regular_at_nofollow(parent_fd: int, name: str, *, context: str) -> int:
    """Open a final regular inode without ever opening a FIFO/device for reading."""

    try:
        path_descriptor = os.open(name, _path_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise AdapterValidationError(f"cannot open no-follow {context}: {error}") from error
    try:
        metadata = os.fstat(path_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AdapterValidationError(f"{context} must be a regular file")
        # /proc/self/fd resolves the already-held inode rather than performing a
        # second lookup of the untrusted final path component.
        try:
            descriptor = os.open(f"/proc/self/fd/{path_descriptor}", _read_flags())
        except OSError as error:
            raise AdapterValidationError(
                f"cannot obtain a readable descriptor for {context}: {error}"
            ) from error
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or (
            observed.st_dev,
            observed.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            os.close(descriptor)
            raise AdapterValidationError(f"{context} inode changed during descriptor acquisition")
        return descriptor
    finally:
        os.close(path_descriptor)


def open_regular_file_nofollow(path: Path, *, context: str) -> int:
    parent_fd, name = open_parent_directory_nofollow(Path(path), context=context)
    try:
        return open_regular_at_nofollow(parent_fd, name, context=context)
    finally:
        os.close(parent_fd)


def regular_file_exists_nofollow(path: Path, *, context: str) -> bool:
    try:
        descriptor = open_regular_file_nofollow(Path(path), context=context)
    except AdapterValidationError as error:
        cause: BaseException | None = error
        while cause is not None:
            if isinstance(cause, OSError) and cause.errno == errno.ENOENT:
                return False
            cause = cause.__cause__
        raise
    os.close(descriptor)
    return True


def read_descriptor_bytes(
    descriptor: int,
    *,
    context: str,
    max_bytes: int,
) -> bytes:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise AdapterValidationError(f"{context} must be a regular file")
    if metadata.st_size > max_bytes:
        raise AdapterValidationError(f"{context} exceeds {max_bytes} bytes")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise AdapterValidationError(f"cannot seek {context}: {error}") from error
    chunks: list[bytes] = []
    observed = 0
    while True:
        try:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
        except OSError as error:
            raise AdapterValidationError(f"cannot read {context}: {error}") from error
        if not chunk:
            break
        observed += len(chunk)
        if observed > max_bytes:
            raise AdapterValidationError(f"{context} exceeds {max_bytes} bytes")
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def read_regular_file_nofollow(
    path: Path,
    *,
    context: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    descriptor = open_regular_file_nofollow(Path(path), context=context)
    try:
        raw = read_descriptor_bytes(descriptor, context=context, max_bytes=max_bytes)
    finally:
        os.close(descriptor)
    return raw, hashlib.sha256(raw).hexdigest()


def sha256_descriptor(descriptor: int, *, context: str) -> str:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise AdapterValidationError(f"{context} must be a regular file")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise AdapterValidationError(f"cannot seek {context}: {error}") from error
    digest = hashlib.sha256()
    while True:
        try:
            chunk = os.read(descriptor, 1024 * 1024)
        except OSError as error:
            raise AdapterValidationError(f"cannot hash {context}: {error}") from error
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _relative_parts(path: Path, root: Path, *, context: str) -> tuple[str, ...]:
    candidate = Path(path)
    approved = Path(root)
    _absolute_parts(candidate, context=context)
    _absolute_parts(approved, context="approved OPD root")
    try:
        relative = candidate.relative_to(approved)
    except ValueError as error:
        raise AdapterValidationError(
            f"{context} is outside approved OPD root: {candidate}"
        ) from error
    if ".." in relative.parts:
        raise AdapterValidationError(f"{context} escapes approved OPD root")
    return tuple(part for part in relative.parts if part not in {"", "."})


def open_confined_directory(
    path: Path,
    *,
    approved_root: Path,
    context: str,
    create: bool,
) -> int:
    relative_parts = _relative_parts(Path(path), Path(approved_root), context=context)
    current = open_directory_nofollow(Path(approved_root), context="approved OPD root")
    try:
        for part in relative_parts:
            try:
                following = _open_directory_component(current, part, context=context)
            except AdapterValidationError as error:
                cause = error.__cause__
                if not create or not isinstance(cause, OSError) or cause.errno != errno.ENOENT:
                    raise
                try:
                    os.mkdir(part, mode=0o750, dir_fd=current)
                    os.fsync(current)
                except OSError as mkdir_error:
                    if mkdir_error.errno != errno.EEXIST:
                        raise AdapterValidationError(
                            f"cannot create confined directory {part!r}: {mkdir_error}"
                        ) from mkdir_error
                following = _open_directory_component(current, part, context=context)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def open_confined_parent(
    path: Path,
    *,
    approved_root: Path,
    context: str,
    create_parents: bool,
) -> tuple[int, str]:
    relative_parts = _relative_parts(Path(path), Path(approved_root), context=context)
    if not relative_parts:
        raise AdapterValidationError(f"{context} target must be below its approved root")
    parent_path = Path(approved_root).joinpath(*relative_parts[:-1])
    parent_fd = open_confined_directory(
        parent_path,
        approved_root=approved_root,
        context=f"{context} parent",
        create=create_parents,
    )
    return parent_fd, relative_parts[-1]


def publish_bytes_once(
    path: Path,
    raw: bytes,
    *,
    approved_root: Path,
    context: str,
) -> bool:
    """Create one exact byte artifact via held parent fd; never replace it."""

    parent_fd, name = open_confined_parent(
        Path(path),
        approved_root=Path(approved_root),
        context=context,
        create_parents=True,
    )
    temporary_name = f".{name}.{secrets.token_hex(16)}.tmp"
    temporary_fd = -1
    try:
        try:
            existing_fd = open_regular_at_nofollow(parent_fd, name, context=context)
        except AdapterValidationError as error:
            cause = error.__cause__
            if not isinstance(cause, OSError) or cause.errno != errno.ENOENT:
                raise
        else:
            try:
                existing = read_descriptor_bytes(
                    existing_fd,
                    context=context,
                    max_bytes=max(len(raw), 1),
                )
            finally:
                os.close(existing_fd)
            if existing == raw:
                return False
            raise FileExistsError(f"conflicting immutable artifact already exists: {path}")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            temporary_fd = os.open(
                temporary_name,
                flags,
                0o640,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise AdapterValidationError(f"cannot create temporary {context}: {error}") from error
        view = memoryview(raw)
        while view:
            written = os.write(temporary_fd, view)
            if written < 1:
                raise AdapterValidationError(f"short write while publishing {context}")
            view = view[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            created = True
        except FileExistsError:
            created = False
        if not created:
            existing_fd = open_regular_at_nofollow(parent_fd, name, context=context)
            try:
                existing = read_descriptor_bytes(
                    existing_fd,
                    context=context,
                    max_bytes=max(len(raw), 1),
                )
            finally:
                os.close(existing_fd)
            if existing != raw:
                raise FileExistsError(f"conflicting immutable artifact already exists: {path}")
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = ""
        os.fsync(parent_fd)
        final_fd = open_regular_at_nofollow(parent_fd, name, context=context)
        try:
            observed = read_descriptor_bytes(
                final_fd, context=context, max_bytes=max(len(raw), 1)
            )
        finally:
            os.close(final_fd)
        if observed != raw:
            raise AdapterValidationError(f"published {context} bytes changed")
        return created
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)


@dataclass
class HeldRegularFile:
    """One verified regular inode whose descriptor remains open across spawn."""

    path: Path
    descriptor: int
    sha256: str

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldRegularFile":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def hold_regular_file(path: Path, *, context: str, expected_sha256: str) -> HeldRegularFile:
    descriptor = open_regular_file_nofollow(Path(path), context=context)
    try:
        observed = sha256_descriptor(descriptor, context=context)
        if observed != expected_sha256:
            raise AdapterValidationError(f"{context} hash differs from its reviewed contract")
        return HeldRegularFile(path=Path(path), descriptor=descriptor, sha256=observed)
    except BaseException:
        os.close(descriptor)
        raise


@contextlib.contextmanager
def held_directory(path: Path, *, context: str) -> Iterator[int]:
    descriptor = open_directory_nofollow(Path(path), context=context)
    try:
        yield descriptor
    finally:
        os.close(descriptor)
