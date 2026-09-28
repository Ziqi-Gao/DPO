"""Pure protocol checks and actual temporary Git histories; no GPU/model imports."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from posttrain_circuits.artifacts import adapted_student_protocol as protocol
from posttrain_circuits.core.config import compose_config

ROOT = Path(__file__).resolve().parents[2]


def raw(value):
    return json.dumps(value, indent=2).encode() + b"\n"


def config():
    return compose_config(
        [
            "g0=qwen3_v2_eap_separation",
            "experiment=canonical_sft",
            "adapted_teacher=qwen3_accepted_v1",
            f"protocol_amendment_path={protocol.PROTOCOL_PATH}",
        ],
        config_root=ROOT / "configs",
    )


def accepted(implementation):
    value = protocol.proposed_adapted_student_protocol()
    value["review"] = {
        "status": "accepted",
        "reviewed_implementation_commit": implementation,
        "reviewer": "independent test reviewer",
        "reviewed_at_utc": "2026-09-28T12:00:00Z",
        "rationale": "Fixture review of the bounded adapted-teacher calibration.",
    }
    return value


def test_declared_protocol_and_config_match_actual_accepted_teacher():
    value = protocol.load_adapted_student_protocol((ROOT / protocol.PROTOCOL_PATH).read_bytes())
    assert protocol.adapted_student_protocol_sha256(value) == protocol.adapted_student_protocol_sha256(
        protocol.proposed_adapted_student_protocol()
    )
    assert value["scope"]["calibration_only"] is True
    assert value["scope"]["g0_pass_claim"] is False
    assert value["scope"]["execution_class_certified"] is False
    assert value["student"]["max_completion_length"] == 128
    assert value["producer"]["readiness_max_new_tokens"] == 256
    assert (
        yaml.safe_load((ROOT / "configs/adapted_teacher/qwen3_accepted_v1.yaml").read_text())
        == protocol.adapted_teacher_config()
    )
    protocol.validate_student_config(config())


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("schema_version",), True),
        (("scope", "calibration_only"), False),
        (("scope", "g0_pass_claim"), True),
        (("scope", "execution_class_certified"), True),
        (("student", "max_completion_length"), 256),
        (("student", "global_batch_size"), 32),
        (("student", "max_steps"), 1000),
        (("execution", "gpu_count"), 4),
        (("execution", "host_memory_gib"), 96),
        (("producer", "accepted_candidates"), 2047),
        (("accepted_teacher", "teacher_identity", "teacher_checkpoint_sha256"), "f" * 64),
        (("producer", "acceptance_commit"), "a" * 40),
        (("producer", "readiness_thresholds", "minimum_teacher_answer_accuracy"), 0.89),
    ],
)
def test_fixed_protocol_rejects_scientific_changes(path, replacement):
    value = protocol.proposed_adapted_student_protocol()
    target = value
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = replacement
    with pytest.raises(protocol.AdaptedStudentProtocolError):
        protocol.validate_adapted_student_protocol(value)


@pytest.mark.parametrize(
    "payload",
    [b"", b"[]", b'{"schema_version":1,"schema_version":1}', b'{"value":NaN}', b"x" * (2 * 1024 * 1024 + 1)],
)
def test_json_structure_and_duplicates_rejected(payload):
    with pytest.raises(protocol.AdaptedStudentProtocolError):
        protocol.load_adapted_student_protocol(payload)


def test_proposed_does_not_allow_execution_and_review_only_hash_is_stable():
    proposed = protocol.proposed_adapted_student_protocol()
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="proposed"):
        protocol.validate_adapted_student_protocol(proposed, require_accepted=True)
    reviewed = accepted("a" * 40)
    protocol.validate_adapted_student_protocol(reviewed, require_accepted=True)
    assert protocol.adapted_student_protocol_sha256(reviewed) == protocol.adapted_student_protocol_sha256(
        proposed
    )
    for field, value in [
        ("reviewed_implementation_commit", "short"),
        ("reviewer", " "),
        ("reviewed_at_utc", "2026-09-28"),
        ("reviewed_at_utc", "2026-99-99T12:00:00Z"),
        ("rationale", None),
    ]:
        changed = copy.deepcopy(reviewed)
        changed["review"][field] = value
        with pytest.raises(protocol.AdaptedStudentProtocolError):
            protocol.validate_adapted_student_protocol(changed)


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("seed",), 43),
        (("seed",), 42.0),
        (("trainer", "max_steps"), 119),
        (("trainer", "global_batch_size"), 32),
        (("trainer", "max_microbatch_size"), 8),
        (("trainer", "max_model_input_length"), 2048),
        (("trainer", "token_budget"), 2000001),
        (("trainer", "max_completion_length"), 256),
        (("trainer", "evaluation_every"), 10),
        (("trainer", "checkpoint_every"), 10),
        (("trainer", "batch_size"), 4),
        (("experiment", "name"), "offline_hard"),
        (("task", "num_examples"), 255),
        (("state_source", "num_candidates"), 9),
        (("state_source", "max_new_tokens"), 128),
        (("state_source", "require_exact_verifier_success"), False),
        (("model", "model_revision"), "a" * 40),
        (("model", "gradient_checkpointing"), False),
        (("model", "prompt_protocol", "enable_thinking"), True),
        (("model", "sampling_protocol", "temperature"), 1.0),
        (("teacher", "tokenizer_fingerprint"), "f" * 64),
        (("teacher", "trust_remote_code"), True),
        (("g0", "full_parameter_training"), False),
        (("adapted_teacher", "producer_science_head"), "a" * 40),
        (("adapted_teacher", "teacher_identity", "kind"), "pinned_hf_base"),
        (("adapted_teacher", "acceptance_inventory_sha256"), "f" * 64),
        (("teacher_readiness", "minimum_teacher_answer_accuracy"), 0.89),
        (("execution_class_certification_path",), "old-blackwell.yaml"),
    ],
)
def test_config_rejects_wrong_identity_resources_or_semantics(path, replacement):
    value = config()
    target = value
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = replacement
    with pytest.raises(protocol.AdaptedStudentProtocolError):
        protocol.validate_student_config(value)


def test_storage_locators_and_actual_initial_hash_are_runtime_bindings():
    value = config()
    value["state_source"]["store_path"] = "/scratch/job/verified/accepted-teacher"
    value["task"]["dataset_family_path"] = "/scratch/job/verified/family"
    value["production_safety"]["initial_checkpoint_path"] = "/scratch/job/initial.pt"
    value["production_safety"]["initial_checkpoint_hash"] = "b" * 64
    protocol.validate_student_config(value)


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "science"
    root.mkdir()

    def git(*arguments):
        return (
            subprocess.check_output(["/usr/bin/git", "-C", str(root), *arguments], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

    def write(path, content):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def commit(message):
        git("add", ".")
        git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            message,
        )
        return git("rev-parse", "HEAD")

    git("init", "-q", "--template=")
    for path in protocol.SCIENCE_PATHS:
        write(path, b"fixture source\n")
    write(protocol.PROTOCOL_PATH, raw(protocol.proposed_adapted_student_protocol()))
    implementation = commit("Fixture proposed implementation")
    # The producer resolver is independently tested against its genuine history.
    # Here only its fixed external binding is substituted; student Git is real.
    producer = SimpleNamespace(
        acceptance_commit=protocol.PRODUCER_HEAD,
        implementation_commit=protocol.PRODUCER_IMPLEMENTATION,
        protocol_sha256=protocol.PRODUCER_PROTOCOL_SHA256,
        science_file_sha256={str(index): "a" * 64 for index in range(47)},
    )
    calls = []

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return producer

    monkeypatch.setattr(protocol, "resolve_teacher_adaptation_protocol", upstream)
    return SimpleNamespace(
        root=root,
        git=git,
        write=write,
        commit=commit,
        implementation=implementation,
        producer=producer,
        upstream_calls=calls,
    )


def accept_history(history):
    history.write(protocol.PROTOCOL_PATH, raw(accepted(history.implementation)))
    return history.commit("Fixture independent review only")


def test_real_git_review_pair_and_later_unrelated_documentation(history, monkeypatch):
    acceptance = accept_history(history)
    history.write("docs/note.md", b"unrelated later documentation\n")
    head = history.commit("Fixture later docs")
    binding = protocol.resolve_adapted_student_protocol(history.root, expected_head=head)
    assert binding.git_commit == acceptance
    assert binding.reviewed_implementation_commit == history.implementation
    assert binding.head == head
    assert binding.path == history.root / protocol.PROTOCOL_PATH
    assert set(binding.science_file_sha256) == set(protocol.SCIENCE_PATHS)
    assert history.upstream_calls[-1][1]["expected_head"] == head
    monkeypatch.chdir(history.root)
    assert protocol.validate_student_protocol(config()).git_commit == acceptance
    (history.root / ".git").rename(history.root / ".opd-git")
    assert protocol.validate_student_protocol(config()).git_commit == acceptance


def test_real_git_proposed_and_wrong_head_rejected(history):
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="proposed"):
        protocol.resolve_adapted_student_protocol(history.root)
    proposed = protocol.resolve_adapted_student_protocol(history.root, require_accepted=False)
    assert proposed.git_commit is None
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="HEAD"):
        protocol.resolve_adapted_student_protocol(
            history.root, expected_head="0" * 40, require_accepted=False
        )


@pytest.mark.parametrize("mode", ["dirty", "staged", "committed", "symlink"])
def test_real_git_named_source_mismatch_rejected(history, mode):
    accept_history(history)
    path = protocol.SCIENCE_PATHS[0]
    history.write(path, b"unreviewed science\n")
    if mode == "staged":
        history.git("add", path)
    elif mode == "committed":
        history.commit("Fixture unreviewed source after acceptance")
    elif mode == "symlink":
        target = history.root / path
        target.unlink()
        target.symlink_to(history.root / protocol.SCIENCE_PATHS[1])
    with pytest.raises(ValueError):
        protocol.resolve_adapted_student_protocol(history.root)


def test_real_git_acceptance_cannot_mix_science_changes(history):
    history.write(protocol.SCIENCE_PATHS[0], b"unreviewed in acceptance\n")
    accept_history(history)
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="non-review"):
        protocol.resolve_adapted_student_protocol(history.root)


def test_real_git_relabelled_implementation_and_reedited_protocol_rejected(history):
    accept_history(history)
    value = accepted(history.implementation)
    value["review"]["rationale"] = "Another review would be a new provenance event."
    history.write(protocol.PROTOCOL_PATH, raw(value))
    history.commit("Fixture second acceptance change")
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="exactly one"):
        protocol.resolve_adapted_student_protocol(history.root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("acceptance_commit", "a" * 40),
        ("implementation_commit", "a" * 40),
        ("protocol_sha256", "a" * 64),
        ("science_file_sha256", {}),
    ],
)
def test_upstream_must_be_exact_teacher_origin(history, field, value):
    accept_history(history)
    setattr(history.producer, field, value)
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="producer lineage"):
        protocol.resolve_adapted_student_protocol(history.root)


def test_import_does_not_load_ml_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import posttrain_circuits.artifacts.adapted_student_protocol; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
