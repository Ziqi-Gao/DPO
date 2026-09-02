"""Descriptor-safe OPD content store and isolated output-attempt staging."""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import publish_path_no_clobber
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import IDENTIFIER, SHA256
from posttrain_circuits.scheduler_adapter.paths import (
    ATTEMPT_COMPLETION_NAME,
    CONTENT_KINDS,
    WorkflowLayout,
    ensure_directory_chain,
)
from posttrain_circuits.scheduler_adapter.secure_files import (
    open_confined_directory,
    open_regular_at_nofollow,
    open_regular_file_nofollow,
    publish_bytes_once,
    published_json_bytes,
    read_descriptor_bytes,
    sha256_descriptor,
)
from posttrain_circuits.scheduler_adapter.strict_json import parse_strict_json
from posttrain_circuits.workflows.contracts import ContentIdentity


TREE_INVENTORY_NAME = ".opd-tree-inventory.json"
TREE_SCHEMA_VERSION = 1
MAX_TREE_INVENTORY_BYTES = 16 * 1024 * 1024


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise AdapterValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_entry_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not IDENTIFIER.fullmatch(value)
        or value == TREE_INVENTORY_NAME
    ):
        raise AdapterValidationError(
            "tree entries must be non-reserved top-level identifier names"
        )
    return value


def _tree_content(file_hashes: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise AdapterValidationError("tree content must contain at least one top-level file")
    entries: list[dict[str, str]] = []
    for name in sorted(file_hashes):
        entry_name = _require_entry_name(name)
        entries.append(
            {
                "name": entry_name,
                "sha256": _require_sha256(
                    file_hashes[name], name=f"tree entry {entry_name!r} sha256"
                ),
            }
        )
    return {
        "files": entries,
        "kind": "tree",
        "schema_version": TREE_SCHEMA_VERSION,
    }


def tree_content_sha256(files: Mapping[str, bytes]) -> str:
    """Hash only the canonical top-level inventory and each file's bytes."""

    if not isinstance(files, Mapping) or not files:
        raise AdapterValidationError("tree content must contain at least one top-level file")
    file_hashes: dict[str, str] = {}
    for name, raw in files.items():
        entry_name = _require_entry_name(name)
        if not isinstance(raw, bytes):
            raise AdapterValidationError(
                f"tree entry {entry_name!r} content must be bytes"
            )
        file_hashes[entry_name] = hashlib.sha256(raw).hexdigest()
    return sha256_value(_tree_content(file_hashes))


def _tree_inventory(file_hashes: Mapping[str, str]) -> dict[str, Any]:
    content = _tree_content(file_hashes)
    return {**content, "sha256": sha256_value(content)}


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _write_bytes_at(parent_fd: int, name: str, raw: bytes, *, context: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(name, flags, 0o440, dir_fd=parent_fd)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise AdapterValidationError(f"short write while creating {context}")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise AdapterValidationError(f"cannot create {context}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass
class ReadOnlyContentHandle:
    """One hash-verified CAS inode held read-only across child execution."""

    name: str
    kind: str
    sha256: str
    path: Path
    descriptor: int

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "ReadOnlyContentHandle":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass
class HeldContentInputs:
    """Canonical input handles with deterministic argv and fd ordering."""

    handles: tuple[ReadOnlyContentHandle, ...]

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return tuple(handle.descriptor for handle in self.handles)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.close()

    def __enter__(self) -> "HeldContentInputs":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass
class OutputAttempt:
    """A newly-created, never-reused directory for exactly one scheduler attempt."""

    path: Path
    descriptor: int
    workflow_id: str
    plan_sha256: str
    unit_id: str
    attempt: int

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    @property
    def completion_name(self) -> str:
        return ATTEMPT_COMPLETION_NAME

    def assert_path_identity(self) -> None:
        held = os.fstat(self.descriptor)
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise AdapterValidationError(
                f"output attempt path changed during execution: {error}"
            ) from error
        if not stat.S_ISDIR(current.st_mode) or (
            held.st_dev,
            held.st_ino,
        ) != (current.st_dev, current.st_ino):
            raise AdapterValidationError("output attempt inode changed during execution")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "OutputAttempt":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ContentStore:
    """Resolve only ``(kind, sha256)`` identities at fixed OPD-owned locations."""

    def __init__(self, layout: WorkflowLayout) -> None:
        layout.validate()
        self.layout = layout

    def publish_file(self, *, sha256: str, raw: bytes) -> Path:
        expected = _require_sha256(sha256, name="file content sha256")
        if not isinstance(raw, bytes):
            raise AdapterValidationError("file content must be bytes")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise AdapterValidationError("file content differs from its requested SHA-256")
        path = self.layout.content_path(kind="file", sha256=expected)
        try:
            publish_bytes_once(
                path,
                raw,
                approved_root=self.layout.data_root,
                context="immutable OPD file content",
            )
        except (OSError, TypeError, ValueError) as error:
            raise AdapterValidationError(f"cannot publish OPD file content: {error}") from error
        with self.open(sha256=expected, kind="file", name="published_file"):
            pass
        return path

    def publish_tree(self, *, sha256: str, files: Mapping[str, bytes]) -> Path:
        expected = _require_sha256(sha256, name="tree content sha256")
        observed = tree_content_sha256(files)
        if observed != expected:
            raise AdapterValidationError("tree content differs from its requested SHA-256")
        file_hashes = {
            _require_entry_name(name): hashlib.sha256(raw).hexdigest()
            for name, raw in files.items()
        }
        inventory = _tree_inventory(file_hashes)
        parent_path = self.layout.content_path(kind="tree", sha256=expected).parent
        ensure_directory_chain(parent_path, approved_root=self.layout.data_root)
        parent_fd = open_confined_directory(
            parent_path,
            approved_root=self.layout.data_root,
            context="OPD tree content parent",
            create=False,
        )
        staging_name = f".{expected}.{secrets.token_hex(16)}.stage"
        staging_path = parent_path / staging_name
        final_path = self.layout.content_path(kind="tree", sha256=expected)
        staging_fd = -1
        created_names: list[str] = []
        published = False
        try:
            try:
                os.mkdir(staging_name, mode=0o750, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError as error:
                raise AdapterValidationError(
                    f"cannot create OPD tree staging directory: {error}"
                ) from error
            staging_fd = os.open(staging_name, _directory_flags(), dir_fd=parent_fd)
            for name in sorted(files):
                _write_bytes_at(
                    staging_fd,
                    name,
                    files[name],
                    context=f"OPD tree entry {name!r}",
                )
                created_names.append(name)
            _write_bytes_at(
                staging_fd,
                TREE_INVENTORY_NAME,
                published_json_bytes(inventory),
                context="OPD tree inventory",
            )
            created_names.append(TREE_INVENTORY_NAME)
            os.fchmod(staging_fd, 0o550)
            os.fsync(staging_fd)
            os.close(staging_fd)
            staging_fd = -1
            try:
                publish_path_no_clobber(staging_path, final_path)
                published = True
            except FileExistsError:
                # A concurrent identical publication is reusable only after a
                # complete descriptor-safe validation of the winning tree.
                with self.open(sha256=expected, kind="tree", name="published_tree"):
                    pass
            if published:
                with self.open(sha256=expected, kind="tree", name="published_tree"):
                    pass
            return final_path
        except (OSError, TypeError, ValueError) as error:
            if isinstance(error, AdapterValidationError):
                raise
            raise AdapterValidationError(f"cannot publish OPD tree content: {error}") from error
        finally:
            if staging_fd >= 0:
                os.close(staging_fd)
            if not published:
                cleanup_fd = -1
                with contextlib.suppress(OSError):
                    cleanup_fd = os.open(staging_name, _directory_flags(), dir_fd=parent_fd)
                if cleanup_fd >= 0:
                    for name in reversed(created_names):
                        with contextlib.suppress(FileNotFoundError):
                            os.unlink(name, dir_fd=cleanup_fd)
                    os.close(cleanup_fd)
                    with contextlib.suppress(FileNotFoundError):
                        os.rmdir(staging_name, dir_fd=parent_fd)
            os.close(parent_fd)

    def open(self, *, sha256: str, kind: str, name: str) -> ReadOnlyContentHandle:
        expected = _require_sha256(sha256, name="content sha256")
        if kind not in CONTENT_KINDS:
            raise AdapterValidationError("content kind must be 'file' or 'tree'")
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
            raise AdapterValidationError("content handle name must be an identifier")
        path = self.layout.content_path(kind=kind, sha256=expected)
        descriptor = -1
        try:
            if kind == "file":
                descriptor = open_regular_file_nofollow(
                    path, context=f"OPD file content {name!r}"
                )
                if sha256_descriptor(
                    descriptor, context=f"OPD file content {name!r}"
                ) != expected:
                    raise AdapterValidationError(
                        f"OPD file content {name!r} hash mismatch"
                    )
            else:
                descriptor = open_confined_directory(
                    path,
                    approved_root=self.layout.data_root,
                    context=f"OPD tree content {name!r}",
                    create=False,
                )
                self._validate_tree(descriptor, expected=expected, name=name)
            return ReadOnlyContentHandle(
                name=name,
                kind=kind,
                sha256=expected,
                path=path,
                descriptor=descriptor,
            )
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise

    def open_identity(self, identity: ContentIdentity) -> ReadOnlyContentHandle:
        if not isinstance(identity, ContentIdentity):
            raise AdapterValidationError("workflow content identity is invalid")
        try:
            identity.validate()
        except ValueError as error:
            raise AdapterValidationError(f"workflow content identity is invalid: {error}") from error
        return self.open(
            sha256=identity.sha256, kind=identity.kind, name=identity.name
        )

    def open_inputs(
        self, identities: tuple[ContentIdentity, ...]
    ) -> HeldContentInputs:
        if not isinstance(identities, tuple):
            raise AdapterValidationError("workflow content identities must be a tuple")
        handles: list[ReadOnlyContentHandle] = []
        try:
            for identity in identities:
                handles.append(self.open_identity(identity))
            return HeldContentInputs(tuple(handles))
        except BaseException:
            for handle in reversed(handles):
                handle.close()
            raise

    def _validate_tree(self, descriptor: int, *, expected: str, name: str) -> None:
        try:
            listed = os.listdir(descriptor)
        except OSError as error:
            raise AdapterValidationError(
                f"cannot list OPD tree content {name!r}: {error}"
            ) from error
        inventory_fd = open_regular_at_nofollow(
            descriptor,
            TREE_INVENTORY_NAME,
            context=f"OPD tree inventory {name!r}",
        )
        try:
            raw = read_descriptor_bytes(
                inventory_fd,
                context=f"OPD tree inventory {name!r}",
                max_bytes=MAX_TREE_INVENTORY_BYTES,
            )
        finally:
            os.close(inventory_fd)
        payload = parse_strict_json(raw, context=f"OPD tree inventory {name!r}")
        if not isinstance(payload, dict) or set(payload) != {
            "files",
            "kind",
            "schema_version",
            "sha256",
        }:
            raise AdapterValidationError("OPD tree inventory fields are invalid")
        files = payload.get("files")
        if (
            payload.get("kind") != "tree"
            or payload.get("schema_version") != TREE_SCHEMA_VERSION
            or isinstance(payload.get("schema_version"), bool)
            or not isinstance(files, list)
            or not files
        ):
            raise AdapterValidationError("OPD tree inventory contract is invalid")
        file_hashes: dict[str, str] = {}
        prior_name = ""
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != {"name", "sha256"}:
                raise AdapterValidationError("OPD tree inventory entry is invalid")
            entry_name = _require_entry_name(entry.get("name"))
            entry_sha256 = _require_sha256(
                entry.get("sha256"), name=f"tree entry {entry_name!r} sha256"
            )
            if entry_name <= prior_name:
                raise AdapterValidationError(
                    "OPD tree inventory entries must be sorted and unique"
                )
            prior_name = entry_name
            file_hashes[entry_name] = entry_sha256
        content = _tree_content(file_hashes)
        if payload.get("sha256") != sha256_value(content) or payload.get("sha256") != expected:
            raise AdapterValidationError("OPD tree inventory hash mismatch")
        expected_names = set(file_hashes) | {TREE_INVENTORY_NAME}
        if set(listed) != expected_names or len(listed) != len(expected_names):
            raise AdapterValidationError("OPD tree has missing or extra top-level entries")
        for entry_name, entry_sha256 in file_hashes.items():
            entry_fd = open_regular_at_nofollow(
                descriptor,
                entry_name,
                context=f"OPD tree entry {entry_name!r}",
            )
            try:
                observed = sha256_descriptor(
                    entry_fd, context=f"OPD tree entry {entry_name!r}"
                )
            finally:
                os.close(entry_fd)
            if observed != entry_sha256:
                raise AdapterValidationError(
                    f"OPD tree entry {entry_name!r} hash mismatch"
                )

    def create_output_attempt(
        self,
        *,
        workflow_id: str,
        plan_sha256: str,
        unit_id: str,
        attempt: int,
    ) -> OutputAttempt:
        self.layout.prepare_attempt_parents(
            workflow_id=workflow_id,
            plan_sha256=plan_sha256,
            unit_id=unit_id,
        )
        path = self.layout.attempt_directory(
            workflow_id=workflow_id,
            plan_sha256=plan_sha256,
            unit_id=unit_id,
            attempt=attempt,
        )
        parent_fd = open_confined_directory(
            path.parent,
            approved_root=self.layout.data_root,
            context="OPD output-attempt parent",
            create=False,
        )
        descriptor = -1
        try:
            try:
                os.mkdir(path.name, mode=0o750, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileExistsError as error:
                raise AdapterValidationError(
                    "output attempt staging already exists and cannot be reused"
                ) from error
            except OSError as error:
                raise AdapterValidationError(
                    f"cannot create output attempt staging: {error}"
                ) from error
            descriptor = os.open(path.name, _directory_flags(), dir_fd=parent_fd)
            return OutputAttempt(
                path=path,
                descriptor=descriptor,
                workflow_id=workflow_id,
                plan_sha256=plan_sha256,
                unit_id=unit_id,
                attempt=attempt,
            )
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        finally:
            os.close(parent_fd)
