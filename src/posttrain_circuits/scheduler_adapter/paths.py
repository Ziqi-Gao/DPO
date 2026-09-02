"""Fixed OPD-owned path derivation and no-symlink confinement checks."""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import IDENTIFIER, SHA256
from posttrain_circuits.scheduler_adapter.secure_files import (
    open_confined_directory,
    open_confined_parent,
    open_directory_nofollow,
    open_regular_at_nofollow,
    open_regular_file_nofollow,
)
PRODUCTION_CODE_ROOT = Path("/home/del6500/projects/OPD")
PRODUCTION_DATA_ROOT = Path("/data/del6500/OPD")
PRODUCTION_SCRATCH_ROOT = Path("/scr/del6500/OPD")
CONTENT_KINDS = frozenset({"file", "tree"})
ATTEMPT_COMPLETION_NAME = ".opd-scientific-completion.json"


def _validate_root(root: Path, *, name: str) -> Path:
    candidate = Path(root)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise AdapterValidationError(f"{name} must be an absolute normalized path")
    descriptor = open_directory_nofollow(candidate, context=name)
    os.close(descriptor)
    return candidate


def _containing_root(path: Path, roots: Iterable[Path]) -> tuple[Path, tuple[str, ...]]:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise AdapterValidationError("project path must be absolute and normalized")
    for raw_root in roots:
        root = Path(raw_root)
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        return root, relative.parts
    raise AdapterValidationError(f"path is outside approved OPD roots: {candidate}")


def validate_path_chain(
    path: Path,
    *,
    approved_roots: Iterable[Path],
    final_kind: str | None,
) -> Path:
    """Reject symlinks/non-directories along an existing confined path."""

    roots = tuple(
        _validate_root(Path(root), name="approved OPD root") for root in approved_roots
    )
    root, _relative_parts = _containing_root(Path(path), roots)
    if final_kind == "file":
        descriptor = open_regular_file_nofollow(Path(path), context="approved OPD file")
    elif final_kind == "directory":
        descriptor = open_confined_directory(
            Path(path),
            approved_root=root,
            context="approved OPD directory",
            create=False,
        )
    else:
        raise ValueError("final_kind must be 'file' or 'directory'")
    os.close(descriptor)
    return Path(path)


def ensure_directory_chain(path: Path, *, approved_root: Path) -> Path:
    """Create a confined project directory one checked component at a time."""

    root = _validate_root(Path(approved_root), name="approved OPD root")
    candidate = Path(path)
    _containing_root(candidate, (root,))
    descriptor = open_confined_directory(
        candidate,
        approved_root=root,
        context="approved OPD directory",
        create=True,
    )
    os.close(descriptor)
    return candidate


def validate_new_file_target(path: Path, *, approved_root: Path) -> Path:
    """Validate a confined parent and reject an existing non-regular target."""

    root = _validate_root(Path(approved_root), name="approved OPD root")
    candidate = Path(path)
    _containing_root(candidate, (root,))
    parent_fd, name = open_confined_parent(
        candidate,
        approved_root=root,
        context="approved OPD target",
        create_parents=False,
    )
    try:
        try:
            descriptor = open_regular_at_nofollow(
                parent_fd, name, context="approved OPD target"
            )
        except AdapterValidationError as error:
            cause = error.__cause__
            if not isinstance(cause, OSError) or cause.errno != errno.ENOENT:
                raise
        else:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    return candidate


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise AdapterValidationError(f"{name} is not a valid identifier")
    return value


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise AdapterValidationError(f"{name} is not a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class WorkflowLayout:
    """All project paths derived from content identities, never job parameters."""

    code_root: Path
    data_root: Path
    scratch_root: Path

    @classmethod
    def production(cls) -> "WorkflowLayout":
        return cls(
            code_root=PRODUCTION_CODE_ROOT,
            data_root=PRODUCTION_DATA_ROOT,
            scratch_root=PRODUCTION_SCRATCH_ROOT,
        )

    def validate(self) -> None:
        _validate_root(self.code_root, name="OPD code root")
        _validate_root(self.data_root, name="OPD data root")
        _validate_root(self.scratch_root, name="OPD scratch root")

    def plan_path(self, *, workflow_id: str, plan_sha256: str) -> Path:
        _identifier(workflow_id, name="workflow_id")
        _sha256(plan_sha256, name="plan_sha256")
        return self.data_root / "workflows" / "plans" / workflow_id / f"{plan_sha256}.json"

    def content_path(self, *, kind: str, sha256: str) -> Path:
        """Return the sole CAS location for one immutable content identity."""

        if kind not in CONTENT_KINDS:
            raise AdapterValidationError("content kind must be 'file' or 'tree'")
        _sha256(sha256, name="content sha256")
        collection = "files" if kind == "file" else "trees"
        return self.data_root / "workflows" / "inputs" / collection / sha256

    def output_directory(
        self, *, workflow_id: str, plan_sha256: str, unit_id: str
    ) -> Path:
        _identifier(workflow_id, name="workflow_id")
        _sha256(plan_sha256, name="plan_sha256")
        _identifier(unit_id, name="unit_id")
        return self.data_root / "workflows" / "outputs" / workflow_id / plan_sha256 / unit_id

    def output_parent_directory(self, *, workflow_id: str, plan_sha256: str) -> Path:
        _identifier(workflow_id, name="workflow_id")
        _sha256(plan_sha256, name="plan_sha256")
        return self.data_root / "workflows" / "outputs" / workflow_id / plan_sha256

    def attempt_directory(
        self,
        *,
        workflow_id: str,
        plan_sha256: str,
        unit_id: str,
        attempt: int,
    ) -> Path:
        """Derive one isolated staging directory, adjacent to its final output."""

        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise AdapterValidationError("attempt must be a positive integer")
        _identifier(unit_id, name="unit_id")
        return self.output_parent_directory(
            workflow_id=workflow_id, plan_sha256=plan_sha256
        ) / f".{unit_id}.attempt-{attempt}.stage"

    def output_path(
        self,
        *,
        workflow_id: str,
        plan_sha256: str,
        unit_id: str,
        output_name: str,
    ) -> Path:
        _identifier(output_name, name="output_name")
        return self.output_directory(
            workflow_id=workflow_id, plan_sha256=plan_sha256, unit_id=unit_id
        ) / output_name

    def completion_path(
        self, *, workflow_id: str, plan_sha256: str, unit_id: str
    ) -> Path:
        _identifier(workflow_id, name="workflow_id")
        _sha256(plan_sha256, name="plan_sha256")
        _identifier(unit_id, name="unit_id")
        return (
            self.data_root
            / "workflows"
            / "completions"
            / workflow_id
            / plan_sha256
            / f"{unit_id}.json"
        )

    def outbox_directory(self) -> Path:
        return self.scratch_root / "scheduler" / "outbox"

    def prepare_attempt_parents(
        self, *, workflow_id: str, plan_sha256: str, unit_id: str
    ) -> None:
        """Create only parents; final output and attempt entries remain no-clobber."""

        ensure_directory_chain(
            self.output_parent_directory(
                workflow_id=workflow_id,
                plan_sha256=plan_sha256,
            ),
            approved_root=self.data_root,
        )
        completion = self.completion_path(
            workflow_id=workflow_id,
            plan_sha256=plan_sha256,
            unit_id=unit_id,
        )
        ensure_directory_chain(completion.parent, approved_root=self.data_root)
        validate_new_file_target(completion, approved_root=self.data_root)
