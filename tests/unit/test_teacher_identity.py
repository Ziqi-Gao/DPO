"""Identity is content metadata, never a substitute for actual checkpoint verification."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.teacher_identity import (
    TeacherCheckpointLocator,
    TeacherIdentity,
    adapted_teacher_identity,
    pinned_base_teacher_identity,
    teacher_content_cache_identity,
    teacher_content_cache_key,
    teacher_science_bindings,
    validate_checkpoint_locator,
)
from posttrain_circuits.core.config import compose_config


def manifest(weight_digest="1" * 64):
    value = {
        "artifact_kind": "adapted_dense_teacher",
        "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "adaptation_plan_sha256": "2" * 64,
        "files": [
            {"path": "merged/model.safetensors", "size": 10, "sha256": weight_digest},
            {"path": "adapter/adapter_model.safetensors", "size": 3, "sha256": "3" * 64},
        ],
        "formal_teacher_accepted": False,
        "training_provenance": {"optimizer_steps": 128, "dataset_manifest_sha256": "4" * 64},
    }
    return {**value, "sha256": sha256_value(value)}


def identity_from_manifest(value):
    return adapted_teacher_identity(value, expected_manifest_sha256=value["sha256"])


def rebind(value):
    value["sha256"] = sha256_value({key: item for key, item in value.items() if key != "sha256"})


def cache_key(identity, **kwargs):
    return teacher_content_cache_key(
        identity, **({"input_ids": [2, 3], "response_ids": [4, 5], "top_k": 128} | kwargs)
    )


def test_two_learned_checkpoints_and_hf_base_have_distinct_science_and_cache_identity():
    first, second = identity_from_manifest(manifest()), identity_from_manifest(manifest("5" * 64))
    base = pinned_base_teacher_identity()
    assert len({item.sha256 for item in (first, second, base)}) == 3
    assert len({cache_key(item) for item in (first, second, base)}) == 3
    assert first.base_revision == second.base_revision == base.base_revision
    assert base.kind == "pinned_hf_base" and base.teacher_checkpoint_sha256 is None
    assert first.kind == "learned_dense_checkpoint"
    assert first.teacher_checkpoint_sha256 != second.teacher_checkpoint_sha256
    assert not hasattr(first, "resolved_model_commit")
    binding = teacher_science_bindings(first)
    assert binding["teacher_checkpoint_sha256"] == first.teacher_checkpoint_sha256
    assert binding["teacher_base_revision"] == base.base_revision
    assert binding["teacher_identity_sha256"] == first.sha256
    assert binding["enable_thinking"] is False
    assert not {"teacher_model_revision", "resolved_teacher_commit", "passed", "accepted"} & binding.keys()
    cache = teacher_content_cache_identity(first)
    assert cache["teacher_identity"] == first.to_mapping()
    assert cache["cache_namespace"] == "opd_teacher_scores_with_weight_identity_v1"


def test_identity_roundtrip_and_pins_match_original_configuration():
    identity = pinned_base_teacher_identity()
    assert TeacherIdentity.from_mapping(json.loads(json.dumps(identity.to_mapping()))) == identity
    config = compose_config(["teacher=qwen3_v2_teacher_8b"])["teacher"]
    assert identity.base_model_id == identity.tokenizer_id == config["model_name_or_path"]
    assert identity.base_revision == identity.tokenizer_revision == config["model_revision"]
    assert identity.tokenizer_fingerprint == config["tokenizer_fingerprint"]
    assert identity.chat_template_sha256 == config["prompt_protocol"]["chat_template_sha256"]
    assert identity.enable_thinking is config["prompt_protocol"]["enable_thinking"] is False
    assert identity.prompt_protocol == config["prompt_protocol"]["name"]


@pytest.mark.parametrize(
    "change",
    [
        {"format_version": True},
        {"format_version": 2},
        {"kind": "local_hf_alias"},
        {"base_model_id": "Qwen/Qwen3-1.7B"},
        {"base_revision": "a" * 40},
        {"tokenizer_id": "different"},
        {"tokenizer_revision": "b" * 40},
        {"tokenizer_fingerprint": "c" * 64},
        {"chat_template_sha256": "d" * 64},
        {"prompt_protocol": "thinking"},
        {"enable_thinking": True},
        {"enable_thinking": 0},
        {"teacher_checkpoint_sha256": None},
        {"teacher_checkpoint_sha256": "a" * 40},
        {"teacher_checkpoint_sha256": "A" * 64},
        {"teacher_checkpoint_sha256": "not-a-hash"},
    ],
)
def test_invalid_learned_identity_fails_at_construction(change):
    value = identity_from_manifest(manifest()).to_mapping() | change
    with pytest.raises(ValueError):
        TeacherIdentity.from_mapping(value)


@pytest.mark.parametrize(
    "field", ["teacher_checkpoint_sha256", "enable_thinking", "kind", "tokenizer_fingerprint"]
)
def test_learned_identity_cannot_omit_its_required_fields(field):
    value = identity_from_manifest(manifest()).to_mapping()
    del value[field]
    with pytest.raises(ValueError, match="missing or unknown"):
        TeacherIdentity.from_mapping(value)


@pytest.mark.parametrize(
    "field",
    ["resolved_teacher_commit", "teacher_model_revision", "checkpoint_root", "formal_teacher_accepted"],
)
def test_revision_path_and_acceptance_aliases_are_not_identity_fields(field):
    value = identity_from_manifest(manifest()).to_mapping() | {field: "invented"}
    with pytest.raises(ValueError, match="missing or unknown"):
        TeacherIdentity.from_mapping(value)


def test_learned_weight_sha_cannot_be_attached_to_base_only_identity():
    learned = identity_from_manifest(manifest())
    with pytest.raises(ValueError, match="cannot stand in"):
        replace(learned, kind="pinned_hf_base")
    with pytest.raises(ValueError, match="actual dense checkpoint"):
        replace(pinned_base_teacher_identity(), kind="learned_dense_checkpoint")


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_expected",
        "changed_content",
        "fake_commit",
        "wrong_base",
        "unknown",
        "accepted",
        "unsafe_path",
        "duplicate",
        "bad_size",
        "no_dense",
    ],
)
def test_manifest_metadata_is_validated_without_self_acceptance(fault):
    value = manifest()
    expected = value["sha256"]
    if fault == "wrong_expected":
        expected = "0" * 64
    elif fault == "changed_content":
        value["files"][0]["sha256"] = "f" * 64
    elif fault == "fake_commit":
        expected = "a" * 40
    else:
        if fault == "wrong_base":
            value["base_revision"] = "b" * 40
        elif fault == "unknown":
            value["resolved_teacher_commit"] = value["base_revision"]
        elif fault == "accepted":
            value["formal_teacher_accepted"] = True
        elif fault == "unsafe_path":
            value["files"][0]["path"] = "merged/../../outside"
        elif fault == "duplicate":
            value["files"].append(copy.deepcopy(value["files"][0]))
        elif fault == "bad_size":
            value["files"][0]["size"] = True
        elif fault == "no_dense":
            value["files"] = value["files"][1:]
        rebind(value)
        expected = value["sha256"]
    with pytest.raises(ValueError):
        adapted_teacher_identity(value, expected_manifest_sha256=expected)


def test_manifest_binds_training_provenance_without_claiming_storage_verification():
    first = manifest()
    second = copy.deepcopy(first)
    second["training_provenance"]["optimizer_steps"] = 256
    rebind(second)
    one, two = identity_from_manifest(first), identity_from_manifest(second)
    assert one.sha256 != two.sha256
    assert cache_key(one) != cache_key(two)
    # Neither fabricated fixture files nor a model need to exist for metadata
    # identity. Actual file-byte verification remains the separate loader's job.
    assert not {"byte_verified", "passed", "formal_teacher_accepted"} & one.to_mapping().keys()


def test_cache_changes_with_tokens_and_topk_and_normalizes_list_tuple():
    identity = identity_from_manifest(manifest())
    assert cache_key(identity) == cache_key(identity, input_ids=(2, 3), response_ids=(4, 5))
    assert (
        len(
            {
                cache_key(identity),
                cache_key(identity, input_ids=[2, 4]),
                cache_key(identity, response_ids=[4, 6]),
                cache_key(identity, top_k=64),
            }
        )
        == 4
    )


@pytest.mark.parametrize(
    "change",
    [
        {"input_ids": []},
        {"input_ids": [True]},
        {"response_ids": [-1]},
        {"response_ids": [1.0]},
        {"top_k": 0},
        {"top_k": True},
    ],
)
def test_malformed_cache_inputs_fail(change):
    with pytest.raises(ValueError):
        cache_key(pinned_base_teacher_identity(), **change)


def test_locator_moves_do_not_change_scientific_or_cache_identity(tmp_path):
    identity = identity_from_manifest(manifest())
    before = identity.to_mapping(), identity.sha256, teacher_science_bindings(identity), cache_key(identity)
    for name in ("persistent", "node-local"):
        root = tmp_path / name
        root.mkdir()
        (root / "dense-manifest.json").write_text(json.dumps(manifest()))
        locator = TeacherCheckpointLocator.from_mapping({"checkpoint_root": str(root)})
        assert validate_checkpoint_locator(locator, allowed_root=tmp_path) == root
    assert before == (
        identity.to_mapping(),
        identity.sha256,
        teacher_science_bindings(identity),
        cache_key(identity),
    )


@pytest.mark.parametrize(
    "value",
    ["relative/path", "/tmp/../elsewhere", "/tmp//name", "/tmp/name/", "//tmp/name", "/tmp/line\nname"],
)
def test_locator_syntax_is_separate_and_strict(value):
    with pytest.raises(ValueError):
        TeacherCheckpointLocator(value)


def test_locator_rejects_unknown_fields_and_unapproved_or_symbolic_storage(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        TeacherCheckpointLocator.from_mapping({"checkpoint_root": str(tmp_path), "sha256": "a" * 64})
    approved = tmp_path / "approved"
    external = tmp_path / "external"
    approved.mkdir()
    external.mkdir()
    (external / "dense-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        validate_checkpoint_locator(TeacherCheckpointLocator(str(external)), allowed_root=approved)
    link = approved / "link"
    link.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        validate_checkpoint_locator(TeacherCheckpointLocator(str(link)), allowed_root=approved)
    with pytest.raises(ValueError, match="dense manifest"):
        validate_checkpoint_locator(TeacherCheckpointLocator(str(approved)), allowed_root=approved)
    (approved / "dense-manifest.json").symlink_to(external / "dense-manifest.json")
    with pytest.raises(ValueError, match="dense manifest"):
        validate_checkpoint_locator(TeacherCheckpointLocator(str(approved)), allowed_root=approved)
