#!/usr/bin/env python3
"""Qualify one frozen prepared student with the unchanged original scientific gates.

This additive adapter changes checkpoint loading and records evidence. The
original validation scorer, anti-shortcut suite, parser and verifier own all
scientific outcomes. It never trains, reselects a checkpoint, or accepts full G0.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

SCHEMA = "quest-sdsc-student-focus-qualification-report-v1"
INPUT_SCHEMA = "quest-sdsc-student-focus-qualification-inputs-v1"
TASK = "qwen3-v2-student-focus-qualification-v1"
FLAGS = dict.fromkeys(
    (
        "student_accepted",
        "g0_passed",
        "pilot_passed",
        "factorial_ready",
        "formal_initial_accepted",
        "execution_class_certified",
    ),
    False,
)
GENERATION = dict(
    max_new_tokens=256,
    do_sample=False,
    use_cache=False,
    explicit_attention_mask=False,
    explicit_autocast=False,
    precision="native_bfloat16",
    truncation=False,
    seed=42,
    validation_max_model_input_length=1536,
    anti_shortcut_max_model_input_length=2244,
)
RAW_NAMES = ("qualification-prompts.jsonl", "qualification-records.jsonl")


def sibling(name):
    spec = importlib.util.spec_from_file_location("_qualify_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


util = sibling("sdsc_student_initial_probe")
require, canonical, digest = util.require, util.canonical, util.digest
file_sha, document, atomic, safe, confined = (
    util.file_sha,
    util.document,
    util.atomic,
    util.safe,
    util.confined,
)


# Exact immutable helper reuse is limited to original scoring/runtime primitives.
# Candidate, parent completion, checkpoint identity and loading stay V5-owned.
FROZEN_QUALIFICATION_HELPER_SHA256 = {
    "tools/sdsc_student_branch_qualify_worker.py": (
        "35cd93ef1a36b7f53f4c887efe283bba5c15248c9cf4a836c1bcc74ab696515f"
    ),
}
for _name, _sha in FROZEN_QUALIFICATION_HELPER_SHA256.items():
    require(
        file_sha(Path(__file__).with_name(Path(_name).name)) == _sha,
        "frozen original qualification helper differs",
    )
_original = sibling("sdsc_student_branch_qualify_worker")
validate_config = _original.validate_config
runtime = _original.runtime
prepare_prompts = _original.prepare_prompts
recording_generate = _original.recording_generate
validation_summary = _original.validation_summary
score_original = _original.score_original
raw_artifacts = _original.raw_artifacts

COMPLETION_NAMES = ("fit_plan", "verified_status", "publication")
MAX_COMPLETION_BYTES = 4 * 1024**2


def validate_completion_files(records, work, candidate):
    """Bind flat V5 evidence after the controller's fresh full admission check."""
    from posttrain_circuits.experiments.protocols.student_focus_qualification import (
        validate_parent_completion_evidence,
    )

    require(
        isinstance(records, dict) and set(records) == set(COMPLETION_NAMES),
        "completion evidence inventory differs",
    )
    raw, documents, summary = {}, {}, {}
    total = 0
    for name in COMPLETION_NAMES:
        record = records[name]
        require(
            isinstance(record, dict)
            and type(record.get("size")) is int
            and 0 < record["size"] <= MAX_COMPLETION_BYTES,
            "completion file exceeds bound",
        )
        total += record["size"]
        require(total <= MAX_COMPLETION_BYTES, "completion evidence exceeds total bound")
        path = file_record(record, work)
        raw[name] = path.read_bytes()
        require(
            len(raw[name]) == record["size"] and digest(raw[name]) == record["sha256"],
            "completion bytes changed after staging",
        )
        documents[name] = json.loads(raw[name])
        require(
            isinstance(documents[name], dict) and canonical(documents[name]) == raw[name],
            "completion evidence must be canonical JSON without LF",
        )
        summary[name] = {key: record[key] for key in ("size", "sha256")}
    validate_parent_completion_evidence(
        candidate,
        fit_plan=documents["fit_plan"],
        status=documents["verified_status"],
        publication_raw=raw["publication"],
    )
    return summary


def file_record(record, work, *, expected_sha=None, expected_size=None):
    require(
        isinstance(record, dict) and set(record) == {"path", "size", "sha256"}, "file record fields differ"
    )
    require(type(record["size"]) is int and record["size"] > 0, "input file size differs")
    require(
        isinstance(record["sha256"], str) and re.fullmatch("[a-f0-9]{64}", record["sha256"]),
        "invalid input hash",
    )
    return util.verified(record, work, expected_sha or record["sha256"], expected_size or record["size"])


def staged_science(inputs, output):
    work = safe(output).parent.resolve()
    require(
        set(inputs)
        == {
            "schema",
            "job_id",
            "run_id",
            "source_code_sha256",
            "plan_sha256",
            "science_root",
            "hf_home",
            "dataset_root",
            "original_science_files",
            "original_config",
            "preparation_protocol",
            "qualification_protocol",
            "prepared_checkpoint",
            "preparation_evidence",
        },
        "unexpected qualification input fields",
    )
    require(inputs["schema"] == INPUT_SCHEMA, "qualification input schema differs")
    require(
        isinstance(inputs["job_id"], str)
        and re.fullmatch("[0-9]+", inputs["job_id"])
        and inputs["job_id"] == os.environ.get("SLURM_JOB_ID"),
        "job allocation differs",
    )
    require(
        isinstance(inputs["run_id"], str) and re.fullmatch("[A-Za-z0-9-]+", inputs["run_id"]),
        "run identity differs",
    )
    for key in ("source_code_sha256", "plan_sha256"):
        require(
            isinstance(inputs[key], str) and re.fullmatch("[a-f0-9]{64}", inputs[key]), "invalid input SHA"
        )
    science = confined(inputs["science_root"], work, directory=True)
    require(Path.cwd().resolve() == science.resolve(), "worker cwd must equal verified science root")
    require(
        not any(k == "posttrain_circuits" or k.startswith("posttrain_circuits.") for k in sys.modules),
        "science imported before source verification",
    )
    originals = inputs["original_science_files"]
    require(isinstance(originals, dict) and len(originals) == 49, "original scientific inventory differs")
    for name, expected in originals.items():
        require(
            isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "unsafe scientific path",
        )
        require(file_sha(confined(science / name, work)) == expected, "original scientific bytes changed")
    cache = confined(inputs["hf_home"], work, directory=True)
    require(str(cache) == os.environ.get("HF_HOME"), "offline cache differs")
    sys.path.insert(0, str(science / "src"))
    return work


def validate_preparation_evidence(evidence, work, candidate, parent):
    """Bind one accepted V5 fit and its independent full-response audit."""
    from posttrain_circuits.experiments.protocols import student_focus_preparation as preparation
    from posttrain_circuits.experiments.protocols.student_focus_qualification import (
        PARENT_WINDOW_INPUT_TOKENS,
        validate_prepared_initial,
    )

    candidate = validate_prepared_initial(candidate, require_bound=True)
    require(
        isinstance(evidence, dict)
        and set(evidence)
        == {"fit_job_id", "publication_sha256", "report", "independent_audit", "completion"},
        "fit evidence fields differ",
    )
    require(
        evidence["fit_job_id"] == candidate["fit_job_id"]
        and evidence["publication_sha256"] == candidate["fit_publication_sha256"],
        "fit origin differs",
    )
    completion = validate_completion_files(evidence["completion"], work, candidate)
    report_path = file_record(evidence["report"], work, expected_sha=candidate["fit_report_sha256"])
    audit_path = file_record(
        evidence["independent_audit"], work, expected_sha=candidate["independent_raw_audit_sha256"]
    )
    report, audit = document(report_path), document(audit_path)
    require(
        audit_path.read_bytes() == canonical(audit) + b"\n", "parent audit must be canonical JSON plus LF"
    )
    require(
        report["schema"] == "quest-sdsc-student-focus-report-v1"
        and report["mode"] == "fit"
        and report["job_id"] == candidate["fit_job_id"]
        and report["passed"] is True
        and report["execution_complete"] is True
        and report["preparation_complete"] is True
        and report["optimizer_steps"] == 32
        and report["consumed_fit_training_views"] == 2048
        and report["fit_base_examples"] == report["development_base_examples"] == 256
        and report["fit_training_views"] == 2048
        and report["development_views"] == 1536
        and report["training_input_tokens"] == 1665064
        and report["world_size"] == 2
        and report["global_batch_size"] == 64
        and report["initial_checkpoint_sha256"] == util.INITIAL_SHA
        and report["learning_rate"] == 5e-5
        and report["protocol_sha256"] == parent.protocol_sha256
        and report["protocol_artifact_sha256"] == parent.artifact_sha256
        and report["dataset_sha256"] == preparation.DATASET_MANIFEST_SHA256
        and report["source_kind"] == "symbolic_canonical_focus_preparation"
        and report["teacher_data_required"] is False
        and "arm" not in report
        and "diagnostic_complete" not in report
        and all(report[key] is False for key in FLAGS),
        "fit execution evidence differs",
    )
    envelope = report.get("data_audit", {}).get("token_envelope", {})
    windows = envelope.get("window_input_tokens")
    require(
        isinstance(windows, list)
        and len(windows) == 32
        and all(type(value) is int and 0 < value <= 64 * 1536 for value in windows)
        and windows == list(PARENT_WINDOW_INPUT_TOKENS)
        and sum(windows) == 1665064
        and canonical(envelope)
        == canonical(
            dict(
                fit_rows=2048,
                optimizer_windows=32,
                window_input_tokens=windows,
                total_input_tokens=1665064,
                token_budget=2000000,
            )
        ),
        "fit full-population token evidence differs",
    )
    selected = preparation.select_checkpoint(report["development"])
    require(
        selected is not None
        and canonical(selected)
        == canonical(report["selected_checkpoint"])
        == canonical(audit["selected_checkpoint"])
        and selected["step"] == candidate["checkpoint_step"]
        and selected["checkpoint_sha256"] == candidate["checkpoint_sha256"],
        "frozen prepared selection differs",
    )
    require(
        audit["schema"] == "quest-sdsc-student-focus-raw-audit-v1"
        and audit["raw_replay_passed"] is True
        and audit["preparation_complete"] is True
        and audit["prompts_replayed"] == 1536
        and audit["responses_replayed"] == 18432
        and audit["gpu_numerics_independently_recomputed"] is False
        and audit["execution_acceptance_claim"] is False
        and canonical(audit["development"]) == canonical(report["development"])
        and audit["job_id"] == candidate["fit_job_id"]
        and audit["mode"] == "fit"
        and audit["publication_sha256"] == candidate["fit_publication_sha256"]
        and audit["plan_sha256"] == report["plan_sha256"] == candidate["fit_plan_sha256"]
        and audit["result_sha256"] == candidate["fit_report_sha256"]
        and audit["auditor_sha256"] == parent.science_file_sha256["tools/sdsc_student_focus_audit.py"]
        and audit["protocol_sha256"] == parent.protocol_sha256
        and audit["protocol_artifact_sha256"] == parent.artifact_sha256
        and audit["dataset_sha256"] == preparation.DATASET_MANIFEST_SHA256
        and audit["implementation_commit"] == parent.implementation_commit
        and audit["acceptance_commit"] == parent.acceptance_commit
        and all(audit[key] is False for key in FLAGS),
        "fit independent audit differs",
    )
    require(
        canonical(audit.get("training_evidence"))
        == canonical(
            dict(
                training_order_reconstructed=True,
                training_rows_reconstructed=2048,
                consumed_training_rows=2048,
                optimizer_windows_replayed=32,
                full_population_input_tokens=1665064,
                training_input_tokens=1665064,
                checkpoint_cumulative_input_tokens={
                    str(step): sum(windows[:step]) for step in preparation.CHECKPOINT_STEPS
                },
                gpu_training_independently_recomputed=False,
            )
        ),
        "independent complete-fit training replay differs",
    )
    isolation = report.get("data_audit", {}).get("isolation", {})
    publication_path = file_record(evidence["completion"]["publication"], work)
    publication = document(publication_path)
    isolation_records = [row for row in publication["files"] if row["path"] == "prepare-data-isolation.json"]
    # The frozen V5 producer uses util.atomic: canonical JSON followed by one LF.
    # Bind both the original bytes' publication record and the independently
    # replayed audit, rather than silently substituting a different JSON encoding.
    isolation_raw = canonical(isolation) + b"\n"
    require(
        len(isolation_records) == 1
        and isolation_records[0]["size"] == len(isolation_raw)
        and isolation_records[0]["sha256"] == digest(isolation_raw),
        "published retained-isolation bytes differ",
    )
    require(
        report.get("data_audit", {}).get("passed") is True
        and isolation.get("passed") is True
        and isolation.get("scientific_acceptance") is False
        and isolation.get("formal_generated_responses_or_scores_read") is False
        and isolation.get("dataset_manifest_sha256") == preparation.DATASET_MANIFEST_SHA256
        and canonical(audit.get("isolation_evidence"))
        == canonical(
            dict(
                producer_evidence_sha256=isolation_records[0]["sha256"],
                dataset_manifest_sha256=preparation.DATASET_MANIFEST_SHA256,
                producer_isolation_consistency_verified=True,
                historical_isolation_independently_recomputed=False,
            )
        ),
        "independent retained-isolation binding differs",
    )
    checkpoints = report["checkpoints"]
    require(
        isinstance(checkpoints, list)
        and [row["step"] for row in checkpoints] == list(preparation.CHECKPOINT_STEPS),
        "complete parent checkpoint schedule required",
    )
    for checkpoint, development in zip(checkpoints, report["development"], strict=True):
        require(checkpoint["sha256"] == development["checkpoint_sha256"], "parent checkpoint summary differs")
    matches = [row for row in checkpoints if row["step"] == candidate["checkpoint_step"]]
    require(len(matches) == 1, "selected parent checkpoint is missing or duplicate")
    checkpoint = matches[0]
    require(
        checkpoint["sha256"] == candidate["checkpoint_sha256"]
        and checkpoint["path"] == candidate["checkpoint_path"]
        and checkpoint["size"] == candidate["checkpoint_size"]
        and checkpoint["scope"] == "student_focus_preparation_model_only",
        "fit selected artifact differs",
    )
    reloads = checkpoint["reload_by_rank"]
    require(
        isinstance(reloads, list)
        and all(type(row.get("rank")) is int for row in reloads)
        and [row["rank"] for row in reloads] == [0, 1],
        "both selected checkpoint rank reloads required",
    )
    for row in reloads:
        require(
            row["model_state_sha256"] == candidate["master_model_state_sha256"]
            and row["exact_saved_master_reload"]
            == dict(all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True)
            and isinstance(row["root_export_logits"], list)
            and len(row["root_export_logits"]) == 2
            and all(
                x["bitwise_equal"] is True
                and x["max_abs_error"] == x["rms_error"] == x["argmax_mismatches"] == 0
                for x in row["root_export_logits"]
            )
            and canonical(row.get("inference_envelope_probe"))
            == canonical(
                dict(
                    passed=True,
                    input_length=2454,
                    token_id=17,
                    all_logits_finite=True,
                    use_cache=False,
                    native_bfloat16=True,
                    scope="synthetic_focus_context_no_development_content",
                )
            ),
            "selected parent master/parity evidence differs",
        )
    return dict(
        completion=completion,
        fit_job_id=candidate["fit_job_id"],
        publication_sha256=candidate["fit_publication_sha256"],
        report_sha256=file_sha(report_path),
        independent_audit_sha256=file_sha(audit_path),
        selected_checkpoint=selected,
        checkpoint_path=checkpoint["path"],
        checkpoint_size=checkpoint["size"],
        master_model_state_sha256=candidate["master_model_state_sha256"],
    )


def load_inputs(inputs, work):
    import yaml

    from posttrain_circuits.artifacts.hashing import sha256_value
    from posttrain_circuits.artifacts.runs import require_git_output
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import build_anti_shortcut_suite
    from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.rendering import RESPONSE_FORMAT_INSTRUCTIONS
    from posttrain_circuits.datasets.proofgraph.splits import build_split
    from posttrain_circuits.experiments.protocols import student_focus_preparation as preparation
    from posttrain_circuits.experiments.protocols import student_focus_qualification as qualification

    science = Path(inputs["science_root"])
    metadata = science / (".git" if (science / ".git/HEAD").is_file() else ".opd-git")
    binding = qualification.resolve_student_focus_qualification_protocol(science, git_dir=metadata)
    require(
        qualification.FROZEN_QUALIFICATION_HELPER_SHA256 == FROZEN_QUALIFICATION_HELPER_SHA256,
        "accepted pure-helper inventory differs",
    )
    candidate = qualification.validate_prepared_initial(
        binding.payload["prepared_initial"], require_bound=True
    )
    parent = preparation.resolve_student_focus_preparation_protocol(
        science, git_dir=metadata, expected_head=binding.head
    )
    for name, resolved, relative in (
        ("qualification_protocol", binding, qualification.PROTOCOL_PATH),
        ("preparation_protocol", parent, preparation.PROTOCOL_PATH),
    ):
        row = inputs[name]
        require(
            isinstance(row, dict) and set(row) == {"path", "size", "sha256", "protocol_sha256"},
            "protocol file record differs",
        )
        path = file_record(
            {key: row[key] for key in ("path", "size", "sha256")}, work, expected_sha=resolved.artifact_sha256
        )
        require(
            path == science / relative and row["protocol_sha256"] == resolved.protocol_sha256,
            "accepted protocol binding differs",
        )
    for name in (
        "sdsc_student_focus_qualify_worker.py",
        "sdsc_student_branch_qualify_worker.py",
        "sdsc_student_initial_probe.py",
        "sdsc_student_lr_probe.py",
    ):
        require(
            file_sha(Path(__file__).with_name(name)) == binding.science_file_sha256["tools/" + name],
            "executing helper differs from accepted science: " + name,
        )
    config_path = file_record(
        inputs["original_config"], work, expected_sha=util.CONFIG_SHA, expected_size=7923
    )
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    require(
        digest(RESPONSE_FORMAT_INSTRUCTIONS.encode()) == util.INSTRUCTION_SHA, "original instruction differs"
    )
    checkpoint = file_record(
        inputs["prepared_checkpoint"],
        work,
        expected_sha=candidate["checkpoint_sha256"],
        expected_size=candidate["checkpoint_size"],
    )
    fit = validate_preparation_evidence(inputs["preparation_evidence"], work, candidate, parent)
    dataset = confined(inputs["dataset_root"], work, directory=True)
    manifest = confined(dataset / "manifest.json", work)
    require(file_sha(manifest) == util.FAMILY_SHA, "original family manifest differs")
    require(
        file_sha(confined(dataset / "validation/examples.jsonl", work)) == util.VALIDATION_SHA,
        "original validation file differs",
    )
    family = load_dataset_family(dataset)
    validation = family.examples("validation")[:128]
    iid = build_split(ProofGraphTask(), "iid_test", 128, 42, config["task"])
    require(
        [asdict(x) for x in iid] == [asdict(x) for x in family.examples("iid_test")[:128]],
        "original generated IID differs from frozen family",
    )
    cases = build_anti_shortcut_suite(iid, seed=42, distractor_ood_count=32)
    require(len(validation) == len(iid) == 128 and len(cases) == 640, "qualification population incomplete")
    require(
        sha256_value([asdict(x) for x in validation]) == binding.payload["base"]["examples_sha256"]
        and sha256_value([asdict(x) for x in iid]) == binding.payload["anti_shortcut"]["iid_examples_sha256"],
        "fixed qualification population changed",
    )
    suite_sha = sha256_value(
        [
            dict(
                source_example_id=x.source_example_id,
                transformation=x.transformation,
                semantic_hash=x.semantic_hash,
                prompt=x.prompt,
            )
            for x in cases
        ]
    )
    require(suite_sha == binding.payload["anti_shortcut"]["suite_sha256"], "original anti suite changed")
    sources = dict(
        head=binding.head,
        implementation_commit=binding.implementation_commit,
        acceptance_commit=binding.acceptance_commit,
        science_file_sha256=binding.science_file_sha256,
        prereg_commit=require_git_output(["log", "-n", "1", "--format=%H", "--", "prereg/qwen3_v2.yaml"]),
    )
    require(re.fullmatch("[a-f0-9]{40}", sources["prereg_commit"]), "preregistration commit missing")
    identity = dict(
        prepared_initial=candidate,
        protocol_sha256=binding.protocol_sha256,
        protocol_artifact_sha256=binding.artifact_sha256,
        preparation_protocol_sha256=parent.protocol_sha256,
        preparation_protocol_artifact_sha256=parent.artifact_sha256,
        fit_evidence=fit,
        scientific_binding=sources,
        dataset_manifest_sha256=util.FAMILY_SHA,
        dataset_family_semantic_sha256=family.manifest["sha256"],
        validation_file_sha256=util.VALIDATION_SHA,
        validation_examples_sha256=binding.payload["base"]["examples_sha256"],
        iid_examples_sha256=binding.payload["anti_shortcut"]["iid_examples_sha256"],
        suite_sha256=suite_sha,
        original_instruction_sha256=util.INSTRUCTION_SHA,
    )
    return config, checkpoint, validation, iid, cases, identity


def validate_dense_payload(saved, identity):
    from posttrain_circuits.experiments.protocols import student_focus_preparation as preparation

    require(
        isinstance(saved, dict)
        and set(saved)
        == {
            "format",
            "mode",
            "model",
            "global_step",
            "parent_checkpoint_sha256",
            "dataset_sha256",
            "protocol_sha256",
            "protocol_artifact_sha256",
            "learning_rate",
            "scope",
            *FLAGS,
        },
        "prepared checkpoint fields differ",
    )
    require(
        saved["format"] == "student_focus_preparation_dense_v5"
        and saved["mode"] == "fit"
        and type(saved["global_step"]) is int
        and saved["global_step"] == identity["prepared_initial"]["checkpoint_step"]
        and saved["parent_checkpoint_sha256"] == util.INITIAL_SHA
        and saved["dataset_sha256"] == preparation.DATASET_MANIFEST_SHA256
        and saved["protocol_sha256"] == identity["preparation_protocol_sha256"]
        and saved["protocol_artifact_sha256"] == identity["preparation_protocol_artifact_sha256"]
        and saved["learning_rate"] == 5e-5
        and saved["scope"] == "student_focus_preparation_model_only"
        and all(saved[key] is False for key in FLAGS),
        "frozen selected focus checkpoint provenance differs",
    )


def load_prepared(model_config, checkpoint, identity, *, device):
    """Restore exact CPU FP32 masters before the single native-BF16 evaluation copy."""
    import torch

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.models.loading import load_model_and_tokenizer

    require(
        not torch.is_autocast_enabled("cuda") and not torch.is_autocast_enabled("cpu"), "autocast prohibited"
    )
    from posttrain_circuits.experiments.protocols.student_focus_qualification import (
        validate_prepared_initial,
    )

    candidate = validate_prepared_initial(identity["prepared_initial"], require_bound=True)
    checkpoint_sha = candidate["checkpoint_sha256"]
    require(
        file_sha(checkpoint) == checkpoint_sha and checkpoint.stat().st_size == candidate["checkpoint_size"],
        "prepared checkpoint bytes changed before load",
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_dense_payload(saved, identity)
    config_before = canonical(model_config)
    loaded = load_model_and_tokenizer(model_config, for_training=False)
    require(
        loaded.resolved_model_commit == util.REVISION
        and loaded.resolved_tokenizer_commit == util.REVISION
        and loaded.tokenizer_hash == util.TOKENIZER_SHA
        and loaded.chat_template_sha256 == util.TEMPLATE_SHA
        and loaded.prompt_protocol == "qwen3_non_thinking_v1",
        "model/tokenizer identity differs",
    )
    exact = sibling("sdsc_student_lr_probe").strict_load_export(loaded.model, saved["model"])
    require(exact["key_count"] == 311, "prepared dense state tensor count differs")
    master_sha = torch_state_hash(saved["model"])
    require(
        master_sha == candidate["master_model_state_sha256"], "saved master hash differs from both fit ranks"
    )
    del saved
    gc.collect()
    model = loaded.model.to(device=device, dtype=torch.bfloat16).eval()
    require(
        all(p.dtype == torch.bfloat16 and not p.requires_grad for p in model.parameters())
        and not model.is_gradient_checkpointing
        and model.config.use_cache is False,
        "native frozen inference state differs",
    )
    require(canonical(model_config) == config_before, "loader mutated original model configuration")
    evidence = dict(
        prepared_checkpoint_sha256=checkpoint_sha,
        exact_saved_master_reload=exact,
        master_model_state_sha256=master_sha,
        forward_parameter_dtype="torch.bfloat16",
        forward_model_state_sha256=torch_state_hash(model.state_dict()),
        all_parameters_frozen=True,
        gradient_checkpointing=False,
        use_cache=False,
    )
    return loaded, evidence


def execute(inputs, output, work):
    import torch

    from posttrain_circuits.artifacts.checkpoints import torch_state_hash
    from posttrain_circuits.core.seeding import seed_everything

    config, checkpoint, validation, iid, cases, identity = load_inputs(inputs, work)
    checkpoint_sha = identity["prepared_initial"]["checkpoint_sha256"]
    gpu = runtime()
    seed_everything(42)
    report = dict(
        schema=SCHEMA,
        task=TASK,
        passed=False,
        execution_complete=False,
        qualification_passed=False,
        base_gate_passed=False,
        anti_shortcut_passed=False,
        **FLAGS,
        **{k: inputs[k] for k in ("job_id", "run_id", "source_code_sha256", "plan_sha256")},
        **identity,
        prepared_checkpoint_sha256=checkpoint_sha,
        generation=GENERATION,
        gpu=gpu,
        raw_record_count=0,
        expected_record_count=896,
        validation=None,
        anti_shortcut=None,
        scope=(
            "fixed selected focus-prepared base and anti-shortcut only; "
            "remaining calibration/cohort/circuit/G0 gates required"
        ),
    )
    atomic(output / "qualification-report.json", report)
    loaded, loading = load_prepared(config["model"], checkpoint, identity, device=torch.device("cuda", 0))
    prompts, examples = prepare_prompts(
        validation,
        iid,
        cases,
        loaded.tokenizer,
        config["model"],
        max_positions=loaded.model.config.max_position_embeddings,
    )
    with (output / RAW_NAMES[0]).open("xb") as stream:
        for row in prompts:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    envelope = {}
    for row in prompts:
        name = row["transformation"] or row["cohort"]
        item = envelope.setdefault(name, dict(num_examples=0, max_prefix_tokens=0, max_full_tokens=0))
        item["num_examples"] += 1
        item["max_prefix_tokens"] = max(item["max_prefix_tokens"], len(row["prompt_ids"]))
        item["max_full_tokens"] = item["max_prefix_tokens"] + 256
    report.update(
        loading=loading,
        tokenizer_fingerprint=loaded.tokenizer_hash,
        chat_template_sha256=loaded.chat_template_sha256,
        prompt_protocol=loaded.prompt_protocol,
        model_revision=loaded.resolved_model_commit,
        prompt_count=len(prompts),
        token_envelope=envelope,
    )
    atomic(output / "qualification-report.json", report)
    started = time.monotonic()
    try:
        with (output / RAW_NAMES[1]).open("xb") as stream:

            def captured(row):
                stream.write(canonical(row) + b"\n")
                stream.flush()
                report["raw_record_count"] += 1
                if report["raw_record_count"] % 16 == 0:
                    os.fsync(stream.fileno())
                    atomic(output / "qualification-report.json", report)

            base, anti, records = score_original(
                loaded.model,
                loaded.tokenizer,
                validation,
                iid,
                cases,
                prompts,
                examples,
                config["model"],
                checkpoint_sha=checkpoint_sha,
                code_commit=identity["scientific_binding"]["head"],
                prereg_commit=identity["scientific_binding"]["prereg_commit"],
                on_record=captured,
            )
            os.fsync(stream.fileno())
    finally:
        report.update(raw_artifacts=raw_artifacts(output))
        atomic(output / "qualification-report.json", report)
    unchanged = torch_state_hash(loaded.model.state_dict()) == loading["forward_model_state_sha256"]
    require(unchanged, "qualification mutated prepared model parameters")
    require(report["raw_record_count"] == len(records) == 896, "qualification raw population incomplete")
    require(
        anti["dataset_hash"] == identity["iid_examples_sha256"]
        and anti["suite_hash"] == identity["suite_sha256"],
        "actual anti evaluation population differs",
    )
    report.update(
        execution_complete=True,
        qualification_passed=base["passed"] and anti["passed"],
        passed=base["passed"] and anti["passed"],
        base_gate_passed=base["passed"],
        anti_shortcut_passed=anti["passed"],
        validation=base,
        anti_shortcut=anti,
        parameters_unchanged=unchanged,
        generation_seconds=time.monotonic() - started,
        raw_artifacts=raw_artifacts(output),
    )
    atomic(output / "qualification-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = safe(args.output_dir)
    output.mkdir(exist_ok=True)
    try:
        inputs = document(args.inputs_json)
        work = staged_science(inputs, output)
        result = execute(inputs, output, work)
        return 0 if result["passed"] else 2
    except Exception as error:
        path = output / "qualification-report.json"
        try:
            report = document(path) if path.is_file() else {"schema": SCHEMA}
        except (ValueError, OSError):
            report = {"schema": SCHEMA}
        report.update(
            passed=False,
            execution_complete=False,
            qualification_passed=False,
            **FLAGS,
            error=f"{type(error).__name__}: {error}",
            raw_artifacts=raw_artifacts(output),
        )
        atomic(path, report)
        print(json.dumps({"error": report["error"], "passed": False}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
