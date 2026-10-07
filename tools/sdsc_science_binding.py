#!/usr/bin/env python3
"""Candidate binding of a runtime initial-checkpoint hash to frozen science.

This does not change the existing scientific hash algorithm or finalizer. Only
production_safety.initial_checkpoint_hash may differ from the reviewed base,
and only after the independently expected checkpoint bytes have been verified.
Storage/scheduler exclusions remain exactly those of the scientific projection.
No execution certificate, ServerScheduler context, G0 decision, or pilot
authorization is produced. The CLI reads a real clean scientific checkout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

PROTOCOL = "prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_prompt_v3.yaml"
AMENDMENT = "prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml"
FROZEN_BASE_SHA256 = "6c3942f4d6cf87329e1c70ba32b0b8d4cdbd526a3bbcfdcb1afd3e1674662657"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
MAX_CHECKPOINT_BYTES = 32 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def real_path(value):
    path = Path(value)
    require(
        path.is_absolute() and ".." not in path.parts and str(path) == str(value),
        "path must be canonical and absolute",
    )
    require(not any(ord(char) < 32 for char in str(path)), "path contains control characters")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "path traverses a symlink")
    return path


def file_identity(path, *, limit=MAX_CHECKPOINT_BYTES, return_bytes=False):
    """Bind a regular file without following links or accepting a replaced inode."""
    path = real_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            "expected a nonempty regular file within the byte limit",
        )
        hasher, total, chunks = hashlib.sha256(), 0, []
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            require(total <= limit, "file grew beyond the byte limit")
            hasher.update(chunk)
            if return_bytes:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    current = real_path(path).lstat()

    def stable(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    require(
        stable(before) == stable(after) == stable(current) and total == before.st_size,
        "artifact changed during hashing",
    )
    identity = {"path": str(path), "size": total, "sha256": hasher.hexdigest()}
    return (identity, b"".join(chunks)) if return_bytes else identity


def json_document(path):
    identity, raw = file_identity(path, limit=8 * 1024 * 1024, return_bytes=True)

    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    require(isinstance(value, dict), "configuration/binding must be a JSON object")
    canonical(value)
    return value, identity


def producer_origin(bindings, *, expected_science_head, expected_formal_binding=None):
    """Never relabel historical teacher/run artifacts to a newer consumer HEAD."""
    require(COMMIT.fullmatch(expected_science_head), "expected science HEAD is invalid")
    require(isinstance(bindings, list) and bindings, "explicit producer bindings are required")
    result = []
    for binding in bindings:
        require(isinstance(binding, dict), "producer binding must be an object")
        producer = binding.get("code_commit")
        require(
            producer == expected_science_head,
            "producer code_commit differs; preserve its real origin and require separate cross-commit review",
        )
        if expected_formal_binding is not None:
            require(
                canonical({key: binding[key] for key in expected_formal_binding if key in binding})
                == canonical(expected_formal_binding),
                "producer formal model/tokenizer/preregistration/amendment binding differs",
            )
        result.append({"producer_code_commit": producer, "consumer_science_head": expected_science_head})
    return result


def verify_runtime_binding(
    base_config,
    runtime_config,
    *,
    expected_base_science_sha256,
    expected_checkpoint_sha256,
    checkpoint_path,
    science_projection,
    relocated_workspace=None,
):
    """Verify one runtime-derived field without changing canonical hash semantics.

    The caller must independently resolve the accepted base/hash and provide
    the original scientific projection function. Relocation is explicit and
    only maps source output_root/initial_checkpoint.pt to the identical name
    in an extracted workspace; arbitrary artifact remapping is refused.
    """
    require(isinstance(base_config, dict) and isinstance(runtime_config, dict), "configs must be objects")
    require(SHA256.fullmatch(expected_base_science_sha256 or ""), "invalid reviewed base hash")
    require(SHA256.fullmatch(expected_checkpoint_sha256 or ""), "invalid independent checkpoint hash")
    base = copy.deepcopy(base_config)
    runtime = copy.deepcopy(runtime_config)
    canonical(base)
    canonical(runtime)
    require(
        isinstance(base.get("production_safety"), dict)
        and base["production_safety"].get("initial_checkpoint_hash") == "",
        "reviewed base must contain the frozen empty initial-checkpoint hash",
    )
    base_projection = science_projection(base)
    base_hash = digest(canonical(base_projection))
    require(base_hash == expected_base_science_sha256, "base config differs from its accepted hash")
    require(isinstance(runtime.get("production_safety"), dict), "runtime safety section missing")
    safety = runtime["production_safety"]
    require(
        safety.get("initial_checkpoint_hash") == expected_checkpoint_sha256,
        "runtime config differs from independently expected checkpoint hash",
    )
    output_root = real_path(runtime.get("output_root", ""))
    source_locator = real_path(safety.get("initial_checkpoint_path", ""))
    require(
        source_locator == output_root / "initial_checkpoint.pt",
        "runtime initial checkpoint is outside the exact attempt locator",
    )
    actual = real_path(checkpoint_path)
    target_root = output_root if relocated_workspace is None else real_path(relocated_workspace)
    require(
        actual == target_root / "initial_checkpoint.pt",
        "checkpoint does not match explicit workspace mapping",
    )
    artifact = file_identity(actual)
    require(
        artifact["sha256"] == expected_checkpoint_sha256,
        "actual checkpoint bytes differ from runtime binding",
    )

    # Do not globally exclude the hash from science. Normalization is permitted
    # here only because the concrete artifact and independent expected SHA match.
    restored = copy.deepcopy(runtime)
    restored["production_safety"]["initial_checkpoint_hash"] = ""
    restored_projection = science_projection(restored)
    require(
        restored_projection == base_projection
        and digest(canonical(restored_projection)) == expected_base_science_sha256,
        "runtime changed scientific fields other than the bound initial-checkpoint hash",
    )
    runtime_hash = digest(canonical(science_projection(runtime)))
    require(runtime_hash != base_hash, "runtime hash unexpectedly equals the frozen empty-hash base")
    evidence = {
        "schema": "quest-sdsc-runtime-science-binding-candidate-v1",
        "scope": "initial_checkpoint_byte_binding_and_scientific_config_only",
        "configuration_binding_validated": True,
        "base_science_config_sha256": base_hash,
        "runtime_science_config_sha256": runtime_hash,
        "runtime_resolved_config_sha256": digest(canonical(runtime)),
        "normalized_fields": ["production_safety.initial_checkpoint_hash"],
        "initial_checkpoint": artifact,
        "configured_initial_checkpoint_path": str(source_locator),
        "explicit_relocated_workspace": str(target_root) if relocated_workspace is not None else None,
        "checkpoint_content_semantics_verified": False,
        "g0_passed": False,
        "pilot_passed": False,
        "factorial_ready": False,
        "execution_class_certified": False,
        "old_finalizer_modified": False,
    }
    evidence["sha256"] = digest(canonical(evidence))
    return evidence


def git(root, *arguments):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_OPTIONAL_LOCKS="0",
        GIT_NO_LAZY_FETCH="1",
    )
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "protocol.allow=never",
            "-C",
            str(root),
            *arguments,
        ],
        env=environment,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return result.stdout


def verify_source_bytes(root, head):
    # Reuse the existing restricted Git reader so assume-unchanged/skip-worktree
    # cannot make modified source appear clean. No raw Git metadata is copied.
    spec = importlib.util.spec_from_file_location(
        "_science_binding_provenance", Path(__file__).with_name("sdsc_provenance.py")
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    records, _contents = verifier.tree_records(root, head)
    verifier.verify_science_workspace(root, head, records)


def reviewed_science(root, expected_head, expected_protocol_sha256):
    root = real_path(root)
    require(root.is_dir() and (root / ".git").is_dir(), "restored real scientific Git checkout is required")
    real_path(root / ".git")
    require(
        COMMIT.fullmatch(expected_head or "") and SHA256.fullmatch(expected_protocol_sha256 or ""),
        "external science/protocol identities are required",
    )
    require(git(root, "rev-parse", "HEAD").decode().strip() == expected_head, "scientific HEAD differs")
    require(git(root, "rev-parse", "--show-toplevel").decode().strip() == str(root), "Git root differs")
    require(
        not git(root, "status", "--porcelain=v1", "--untracked-files=all")
        and not git(root, "ls-files", "--others", "-z"),
        "scientific checkout is not clean",
    )
    verify_source_bytes(root, expected_head)
    require(
        not any(
            name == "posttrain_circuits" or name.startswith("posttrain_circuits.") for name in sys.modules
        ),
        "scientific modules imported before provenance validation",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    protocol_api = importlib.import_module("posttrain_circuits.artifacts.execution_science_protocol")
    config_api = importlib.import_module("posttrain_circuits.core.config")
    resolved = protocol_api.resolve_accepted_execution_science_protocol(
        code_root=root, configured_path=PROTOCOL, expected_head=expected_head
    )
    require(resolved.sha256 == expected_protocol_sha256, "accepted science protocol bytes differ")
    require(
        resolved.binding.storage_neutral_resolved_config_sha256 == FROZEN_BASE_SHA256,
        "candidate supports only the existing frozen prompt-v3 base",
    )
    base = config_api.compose_config(
        [*resolved.binding.hydra_override_vector, "protocol_amendment_path=" + AMENDMENT],
        config_root=root / "configs",
    )
    return base, protocol_api.canonical_science_config_projection, resolved


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("science-root", "runtime-config", "initial-checkpoint"):
        result.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-science-head", "expected-protocol-sha256", "expected-initial-sha256"):
        result.add_argument("--" + name, required=True)
    result.add_argument(
        "--producer-binding",
        required=True,
        action="append",
        type=Path,
        help="JSON formal/protocol binding, or report containing it; repeat for each producer",
    )
    result.add_argument("--relocated-workspace", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    runtime, runtime_identity = json_document(args.runtime_config)
    producers, records = [], []
    for path in args.producer_binding:
        value, identity = json_document(path)
        producers.append(value.get("formal_binding", value.get("protocol_bindings", value)))
        records.append(identity)
    origins = producer_origin(producers, expected_science_head=args.expected_science_head)
    previous = Path.cwd()
    try:
        os.chdir(real_path(args.science_root))
        base, projection, reviewed = reviewed_science(
            args.science_root, args.expected_science_head, args.expected_protocol_sha256
        )
        runs = importlib.import_module("posttrain_circuits.artifacts.runs")
        formal_binding = runs.formal_artifact_binding(base)
        require(
            formal_binding["code_commit"] == args.expected_science_head
            and runs.formal_artifact_binding(runtime) == formal_binding,
            "runtime formal model/tokenizer/preregistration/amendment binding differs",
        )
        origins = producer_origin(
            producers,
            expected_science_head=args.expected_science_head,
            expected_formal_binding=formal_binding,
        )
        evidence = verify_runtime_binding(
            base,
            runtime,
            expected_base_science_sha256=FROZEN_BASE_SHA256,
            expected_checkpoint_sha256=args.expected_initial_sha256,
            checkpoint_path=args.initial_checkpoint,
            science_projection=projection,
            relocated_workspace=args.relocated_workspace,
        )
        require(
            not git(args.science_root, "status", "--porcelain=v1", "--untracked-files=all"),
            "scientific checkout changed while validating",
        )
        verify_source_bytes(args.science_root, args.expected_science_head)
    finally:
        os.chdir(previous)
    result = {
        "binding": evidence,
        "science_head": args.expected_science_head,
        "science_protocol_sha256": reviewed.sha256,
        "science_acceptance_commit": reviewed.acceptance_commit,
        "runtime_config_file": runtime_identity,
        "producer_binding_files": records,
        "origins": origins,
        "formal_binding": formal_binding,
        "g0_passed": False,
        "pilot_passed": False,
        "execution_class_certified": False,
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
