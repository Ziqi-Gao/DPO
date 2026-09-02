"""Strict, no-follow JSON reads for scheduler-boundary files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.secure_files import read_regular_file_nofollow


DEFAULT_MAX_BYTES = 2 * 1024 * 1024


def _reject_constant(value: str) -> None:
    raise AdapterValidationError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdapterValidationError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def parse_strict_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AdapterValidationError(f"{context} is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except AdapterValidationError:
        raise
    except json.JSONDecodeError as error:
        raise AdapterValidationError(f"{context} is not valid strict JSON: {error}") from error


def read_strict_json(
    path: Path,
    *,
    context: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[Any, str]:
    """Read one regular file without following any path-component symlink.

    The returned digest covers the exact bytes observed through the validated
    descriptor, not a later path lookup.
    """

    raw, digest = read_regular_file_nofollow(
        Path(path), context=context, max_bytes=max_bytes
    )
    return parse_strict_json(raw, context=context), digest
