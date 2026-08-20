from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain_circuits.circuits.pilot_scope import PILOT_CELLS, resolve_pilot_circuit_scope
from posttrain_circuits.cli.finalize_g0 import (
    _require_formal_binding,
    _teacher_readiness_formal_binding,
)
from posttrain_circuits.cli.finalize_pilot import (
    _require_cell_chain,
    _require_circuit_slot,
    _require_dynamics_slot,
    _terminal_paths,
)
from posttrain_circuits.cli.finalize_pilot_training import _terminal_tasks
from posttrain_circuits.cli.gpu_preflight import cgroup_memory_snapshot, validate_memory_headroom
from posttrain_circuits.core import provenance
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.readiness import require_formal_prerequisite_binding
from posttrain_circuits.training.factorial_trainer import TrainerConfig
from posttrain_circuits.training.grpo_backend import GrpoSettings, GrpoTokenBudgetCallback
from posttrain_circuits.training.token_budget import TOKEN_BUDGET_UNIT, TokenBudgetState

ROOT = Path(__file__).parents[2]


@pytest.mark.unit
def test_qwen3_four_gpu_jobs_request_explicit_node_memory() -> None:
    for relative in (
        "scripts/slurm/qwen3_gpu_preflight.slurm",
        "scripts/slurm/g0_qwen.slurm",
        "scripts/slurm/pilot_qwen_core.slurm",
        "scripts/slurm/pilot_resume.slurm",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "#SBATCH --mem=192G" in source, relative


@pytest.mark.unit
def test_preregistration_is_not_a_global_core_v2_default() -> None:
    assert not hasattr(provenance, "PREREG_PATH")
    assert not hasattr(provenance, "PREREG_VERSION")
    assert "prereg/core_v2.yaml" not in inspect.getsource(provenance)


@pytest.mark.unit
def test_qwen3_formal_artifact_binding_rejects_dirty_source(monkeypatch, tmp_path: Path) -> None:
    prereg = tmp_path / "qwen3_v2.yaml"
    prereg.write_text("version: qwen3_v2\n", encoding="utf-8")
    monkeypatch.setattr(
        provenance,
        "resolve_preregistration",
        lambda _config: provenance.PreregistrationBinding(
            path=prereg,
            version="qwen3_v2",
            git_commit="a" * 40,
            sha256=sha256_file(prereg),
            dirty=False,
        ),
    )
    monkeypatch.setattr(
        provenance,
        "require_git_output",
        lambda args: " M src/posttrain_circuits/example.py"
        if args[:2] == ["status", "--porcelain"]
        else "b" * 40,
    )
    with pytest.raises(RuntimeError, match="dirty source checkout"):
        provenance.formal_artifact_binding(
            {
                "protocol_track": "qwen3_v2",
                "model": {"prompt_protocol": {}},
                "teacher": {},
            }
        )


@pytest.mark.unit
def test_trainer_config_requires_a_real_token_budget() -> None:
    config = TrainerConfig(max_steps=10, token_budget=128)
    assert config.token_budget == 128
    with pytest.raises(ValueError, match="token_budget"):
        TrainerConfig(max_steps=10, token_budget=0)


@pytest.mark.unit
def test_pilot_circuit_scope_is_full_and_stage_explicit() -> None:
    initial = (ROOT / "scripts/slurm/pilot_initial_circuits.slurm").read_text(encoding="utf-8")
    final = (ROOT / "scripts/slurm/pilot_final_circuits.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3" in initial
    assert "#SBATCH --array=0-31" in final
    scope = (ROOT / "src/posttrain_circuits/circuits/pilot_scope.py").read_text(encoding="utf-8")
    assert "offline_hard" in scope
    assert "offline_verified_replay" in scope
    assert '"process": "first_rule_selection"' in scope
    assert '"final_answer": "final_answer"' in scope
    assert "resolve_pilot_circuit_scope" in initial
    assert "resolve_pilot_circuit_scope" in final
    assert '--stage "${stage}"' in initial
    assert '--stage "${stage}"' in final


@pytest.mark.unit
def test_pilot_supervisor_does_not_treat_empty_accounting_as_success() -> None:
    source = (ROOT / "scripts/production/submit_pilot.sh").read_text(encoding="utf-8")
    helper = (ROOT / "scripts/production/slurm_supervision.sh").read_text(encoding="utf-8")
    assert "SLURM_ACCOUNTING_RETRIES" in helper
    assert "UNKNOWN" in helper
    assert "require_no_competing_opd_gpu_job" in source


@pytest.mark.unit
def test_qwen3_v2_scope_is_exactly_eight_by_two_by_two() -> None:
    config = compose_config(["pilot=qwen3_v2_core", "model=qwen3_v2_1p7b", "teacher=qwen3_v2_teacher_8b"])
    scope = resolve_pilot_circuit_scope(config)
    assert tuple(scope["cells"]) == PILOT_CELLS
    assert scope["initial_count"] == 4
    assert scope["final_count"] == 32
    assert {(row["cell"], row["stage_label"], row["cohort"]) for row in scope["final_matrix"]} == {
        (cell, stage, cohort)
        for cell in PILOT_CELLS
        for stage in ("process", "final_answer")
        for cohort in ("base_capable", "challenge")
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile",
    ("production=qwen3_v2_primary", "g0=qwen3_v2_eap_separation", "pilot=qwen3_v2_core"),
)
def test_qwen3_v2_profiles_have_no_legacy_scientific_fallback(profile: str) -> None:
    config = compose_config([profile])
    serialized = json.dumps(config, sort_keys=True)
    assert config["model"]["model_name_or_path"] == "Qwen/Qwen3-1.7B"
    assert config["teacher"]["model_name_or_path"] == "Qwen/Qwen3-8B"
    assert config["prereg_path"] == "prereg/qwen3_v2.yaml"
    assert config["prereg_version"] == "qwen3_v2"
    assert config["output_root"] == "outputs/qwen3-v2"
    for forbidden in ("Qwen2.5", "qwen25", "outputs/qwen3-v1", "prereg/qwen3_v1.yaml"):
        assert forbidden not in serialized


@pytest.mark.unit
def test_cgroup_memory_gate_rejects_implicit_or_insufficient_memory() -> None:
    gib = 1024**3
    with pytest.raises(RuntimeError, match="finite Slurm cgroup"):
        validate_memory_headroom(
            {"limit_bytes": None, "current_bytes": 1, "peak_bytes": 1},
            requested_gib=192,
            minimum_headroom_gib=32,
            minimum_headroom_fraction=0.2,
            observed_process_peak_bytes=1,
        )
    failed = validate_memory_headroom(
        {"limit_bytes": 192 * gib, "current_bytes": 170 * gib, "peak_bytes": 170 * gib},
        requested_gib=192,
        minimum_headroom_gib=32,
        minimum_headroom_fraction=0.2,
        observed_process_peak_bytes=170 * gib,
    )
    assert failed["passed"] is False
    passed = validate_memory_headroom(
        {"limit_bytes": 192 * gib, "current_bytes": 100 * gib, "peak_bytes": 100 * gib},
        requested_gib=192,
        minimum_headroom_gib=32,
        minimum_headroom_fraction=0.2,
        observed_process_peak_bytes=100 * gib,
    )
    assert passed["passed"] is True


@pytest.mark.unit
@pytest.mark.parametrize("version", (1, 2))
def test_cgroup_memory_snapshot_supports_quest_v1_and_v2(tmp_path: Path, version: int) -> None:
    proc = tmp_path / "proc-self-cgroup"
    root = tmp_path / "cgroup"
    relative = Path("slurm/uid_8959/job_123")
    if version == 2:
        proc.write_text(f"0::/{relative}\n", encoding="utf-8")
        controller = root / relative
        names = ("memory.max", "memory.current", "memory.peak")
    else:
        proc.write_text(f"5:memory:/{relative}\n", encoding="utf-8")
        controller = root / "memory" / relative
        names = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
    controller.mkdir(parents=True)
    for name, value in zip(names, (192 * 1024**3, 80 * 1024**3, 100 * 1024**3), strict=True):
        (controller / name).write_text(str(value), encoding="utf-8")
    assert cgroup_memory_snapshot(proc_cgroup=proc, cgroup_root=root) == {
        "limit_bytes": 192 * 1024**3,
        "current_bytes": 80 * 1024**3,
        "peak_bytes": 100 * 1024**3,
    }


@pytest.mark.unit
def test_cgroup_v1_unlimited_sentinel_is_not_a_finite_limit(tmp_path: Path) -> None:
    proc = tmp_path / "proc-self-cgroup"
    root = tmp_path / "cgroup"
    relative = Path("slurm/job_123")
    proc.write_text(f"5:memory:/{relative}\n", encoding="utf-8")
    controller = root / "memory" / relative
    controller.mkdir(parents=True)
    (controller / "memory.limit_in_bytes").write_text(str(2**63 - 4096), encoding="utf-8")
    (controller / "memory.usage_in_bytes").write_text("1", encoding="utf-8")
    (controller / "memory.max_usage_in_bytes").write_text("1", encoding="utf-8")
    snapshot = cgroup_memory_snapshot(proc_cgroup=proc, cgroup_root=root)
    with pytest.raises(RuntimeError, match="finite Slurm cgroup"):
        validate_memory_headroom(
            snapshot,
            requested_gib=192,
            minimum_headroom_gib=32,
            minimum_headroom_fraction=0.2,
            observed_process_peak_bytes=1,
        )


@pytest.mark.unit
def test_token_budget_is_distributed_exact_and_resume_safe(monkeypatch) -> None:
    monkeypatch.setattr(
        "posttrain_circuits.training.token_budget.distributed_token_sum", lambda value: value * 4
    )
    state = TokenBudgetState(100)
    assert state.reserve_optimizer_update(20) == (True, 80)
    assert state.reserve_optimizer_update(6) == (False, 24)
    assert state.consumed == 80
    restored = TokenBudgetState(100)
    restored.load_state_dict(state.state_dict())
    assert restored.consumed == 80
    with pytest.raises(ValueError, match="differs"):
        TokenBudgetState(101).load_state_dict(state.state_dict())


@pytest.mark.unit
def test_grpo_budget_reduces_rank_deltas_and_never_resets(monkeypatch) -> None:
    monkeypatch.setattr(
        "posttrain_circuits.training.grpo_backend.distributed_token_sum", lambda value: value * 4
    )
    callback = GrpoTokenBudgetCallback(
        GrpoSettings(
            token_budget=200,
            reserved_tokens_per_update=80,
            initial_token_budget_consumed=40,
        )
    )
    control = SimpleNamespace(should_training_stop=False)
    callback.processed_tokens[0] = 10
    callback.on_step_end(None, SimpleNamespace(global_step=1), control)
    assert callback.consumed == 80
    callback.processed_tokens[0] = 10
    callback.on_step_end(None, SimpleNamespace(global_step=2), control)
    assert callback.consumed == 120
    callback.processed_tokens[0] = 10
    callback.on_step_end(None, SimpleNamespace(global_step=3), control)
    assert callback.consumed == 160
    assert control.should_training_stop is True
    assert callback.state_dict()["unit"] == TOKEN_BUDGET_UNIT


@pytest.mark.unit
def test_formal_binding_rejects_cross_prereg_and_revision() -> None:
    expected = {
        key: value
        for key, value in {
            "protocol_track": "qwen3_v2",
            "artifact_namespace": "qwen3-v2",
            "model_revision": "student",
            "teacher_revision": "teacher",
            "tokenizer_revision": "tokenizer",
            "tokenizer_fingerprint": "fingerprint",
            "chat_template_sha256": "chat",
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "code_commit": "code",
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_version": "qwen3_v2",
            "prereg_commit": "prereg-commit",
            "prereg_sha256": "prereg-hash",
        }.items()
    }
    _require_formal_binding(dict(expected), expected, name="fixture")
    for key, stale_value in (
        ("prereg_version", "qwen3_v1"),
        ("prereg_path", "prereg/core_v2.yaml"),
        ("teacher_revision", "qwen25"),
    ):
        stale = dict(expected)
        stale[key] = stale_value
        with pytest.raises(ValueError, match="formal binding mismatch"):
            _require_formal_binding(stale, expected, name="fixture")
        with pytest.raises(RuntimeError, match="formal binding mismatch"):
            require_formal_prerequisite_binding({"bindings": stale}, expected, name="readiness", nested=True)


@pytest.mark.unit
def test_teacher_readiness_keeps_distinct_student_and_teacher_tokenizer_revisions() -> None:
    expected = {
        "model_revision": "student-model",
        "teacher_revision": "teacher-model",
        "tokenizer_revision": "student-tokenizer",
    }
    artifact = {
        "bindings": {
            "student_model_revision": "student-model",
            "teacher_model_revision": "teacher-model",
            "student_tokenizer_revision": "student-tokenizer",
            "teacher_tokenizer_revision": "teacher-tokenizer",
            "tokenizer_revision": "teacher-tokenizer",
        }
    }
    normalized = _teacher_readiness_formal_binding(artifact)
    assert {key: normalized[key] for key in expected} == expected
    assert normalized["teacher_tokenizer_revision"] == "teacher-tokenizer"


def _write_cell_chain(tmp_path: Path) -> tuple[Path, dict[str, object], str]:
    root = tmp_path / "cell"
    root.mkdir(parents=True)
    metrics = root / "metrics.jsonl"
    metrics.write_text('{"step": 1, "validation_accuracy": 0.5}\n', encoding="utf-8")
    checkpoint = root / "checkpoints" / "step.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    resolved = root / "resolved_config.yaml"
    resolved.write_text("protocol_track: qwen3_v2\n", encoding="utf-8")
    terminal_hash = "t" * 64
    dataset_hashes = {
        "initial_checkpoint": "i" * 64,
        "prerequisite_probe_cohorts": "p" * 64,
        "validation_manifest": "v" * 64,
    }
    expected: dict[str, object] = {
        "protocol_track": "qwen3_v2",
        "artifact_namespace": "qwen3-v2",
        "model_revision": "student",
        "teacher_revision": "teacher",
        "tokenizer_revision": "tokenizer",
        "tokenizer_fingerprint": "fingerprint",
        "chat_template_sha256": "chat",
        "prompt_protocol": "qwen3_non_thinking_v1",
        "enable_thinking": False,
        "code_commit": "code",
        "prereg_path": "prereg/qwen3_v2.yaml",
        "prereg_version": "qwen3_v2",
        "prereg_commit": "prereg-commit",
        "prereg_sha256": "prereg-hash",
    }
    binding: dict[str, object] = {
        "cell": "offline_hard",
        "seed": 42,
        "resolved_config_sha256": sha256_file(resolved),
        "metrics_sha256": sha256_file(metrics),
        "final_checkpoint_sha256": sha256_file(checkpoint),
        "slurm_terminal_evidence_sha256": terminal_hash,
        "slurm": {"job_id_raw": "123_0", "state": "COMPLETED", "exit_code": "0:0"},
        "token_budget": 100,
        "token_budget_consumed": 80,
        "token_budget_unit": TOKEN_BUDGET_UNIT,
        "training_stop_reason": "max_steps_safety_limit",
        "dataset_hashes_sha256": sha256_value(dataset_hashes),
        "probe_manifest_hashes": ["p" * 64],
        "initial_checkpoint_sha256": "i" * 64,
        "state_source_artifact_sha256": "s" * 64,
    }
    binding["sha256"] = sha256_value(binding)
    manifest: dict[str, object] = {
        "experiment_cell": "offline_hard",
        "seed": 42,
        **{key: value for key, value in expected.items() if key != "code_commit"},
        "git_commit": expected["code_commit"],
        "prereg_git_commit": expected["prereg_commit"],
        "protocol_teacher_revision": expected["teacher_revision"],
        "slurm_terminal_evidence_sha256": terminal_hash,
        "final_checkpoint_path": str(checkpoint),
        "final_checkpoint_sha256": sha256_file(checkpoint),
        "metrics_sha256": sha256_file(metrics),
        "token_budget": 100,
        "token_budget_consumed": 80,
        "token_budget_unit": TOKEN_BUDGET_UNIT,
        "training_stop_reason": "max_steps_safety_limit",
        "dataset_hashes": dataset_hashes,
        "rollout_bank_hash": "s" * 64,
        "pilot_terminal_binding": binding,
    }
    manifest["sha256"] = sha256_value(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, expected, terminal_hash


@pytest.mark.unit
def test_pilot_cell_chain_rejects_metrics_and_checkpoint_tamper(tmp_path: Path) -> None:
    root, expected, terminal_hash = _write_cell_chain(tmp_path)
    _require_cell_chain(
        root=root,
        cell="offline_hard",
        expected=expected,
        terminal_hash=terminal_hash,
        training_job_id="123",
    )
    (root / "metrics.jsonl").write_text('{"step": 1, "validation_accuracy": 0.9}\n')
    with pytest.raises(ValueError, match="changed after binding"):
        _require_cell_chain(
            root=root,
            cell="offline_hard",
            expected=expected,
            terminal_hash=terminal_hash,
            training_job_id="123",
        )
    root, expected, terminal_hash = _write_cell_chain(tmp_path / "second")
    Path(str(json.loads((root / "manifest.json").read_text())["final_checkpoint_path"])).write_bytes(
        b"tampered"
    )
    with pytest.raises(ValueError, match="changed after binding"):
        _require_cell_chain(
            root=root,
            cell="offline_hard",
            expected=expected,
            terminal_hash=terminal_hash,
            training_job_id="123",
        )


@pytest.mark.unit
def test_pilot_cell_chain_rejects_cross_cell_checkpoint_path(tmp_path: Path) -> None:
    root, expected, terminal_hash = _write_cell_chain(tmp_path)
    foreign = tmp_path / "other-cell" / "checkpoints" / "step.pt"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"foreign-checkpoint")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest["pilot_terminal_binding"]
    digest = sha256_file(foreign)
    binding["final_checkpoint_sha256"] = digest
    binding["sha256"] = sha256_value({key: value for key, value in binding.items() if key != "sha256"})
    manifest["final_checkpoint_path"] = str(foreign)
    manifest["final_checkpoint_sha256"] = digest
    manifest["sha256"] = sha256_value({key: value for key, value in manifest.items() if key != "sha256"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its cell directory"):
        _require_cell_chain(
            root=root,
            cell="offline_hard",
            expected=expected,
            terminal_hash=terminal_hash,
            training_job_id="123",
        )


@pytest.mark.unit
def test_pilot_cell_chain_rejects_rehashed_wrong_slurm_job(tmp_path: Path) -> None:
    root, expected, terminal_hash = _write_cell_chain(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest["pilot_terminal_binding"]
    binding["slurm"]["job_id_raw"] = "999_0"
    binding["sha256"] = sha256_value({key: value for key, value in binding.items() if key != "sha256"})
    manifest["sha256"] = sha256_value({key: value for key, value in manifest.items() if key != "sha256"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal binding mismatch"):
        _require_cell_chain(
            root=root,
            cell="offline_hard",
            expected=expected,
            terminal_hash=terminal_hash,
            training_job_id="123",
        )


@pytest.mark.unit
def test_terminal_evidence_rejects_rows_from_a_different_job(tmp_path: Path) -> None:
    terminal = tmp_path / "training.txt"
    terminal.write_text(
        "".join(f"999_{index}|COMPLETED|0:0\n" for index in range(len(PILOT_CELLS))),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly the eight pilot tasks"):
        _terminal_tasks(terminal, "123")

    stages = ("training", "initial_circuits", "final_circuits", "local_fork", "resume", "dynamics")
    job_ids = tmp_path / "job-ids.txt"
    job_ids.write_text("".join(f"{stage}=123\n" for stage in stages), encoding="utf-8")
    for stage in stages:
        (tmp_path / f"terminal-{stage}.txt").write_text("999|COMPLETED|0:0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete success evidence"):
        _terminal_paths(tmp_path, job_ids)


def _slot_artifact(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["sha256"] = sha256_value(result)
    return result


@pytest.mark.unit
def test_circuit_and_dynamics_slots_reject_rehashed_wrong_stage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = sha256_file(checkpoint)
    circuit_path = tmp_path / "circuit.json"
    circuit = _slot_artifact(
        {
            "checkpoint_sha256": digest,
            "checkpoint_path": str(checkpoint),
            "probe_cohort": "challenge",
            "probe_stage": "final_answer",
        }
    )
    exact = _slot_artifact(
        {
            "checkpoint_sha256": digest,
            "probe_cohort": "challenge",
            "probe_stage": "first_rule_selection",
            "artifacts": {"circuit_artifact": str(circuit_path)},
        }
    )
    with pytest.raises(ValueError, match="wrong pilot matrix slot"):
        _require_circuit_slot(
            circuit=circuit,
            exact=exact,
            circuit_path=circuit_path,
            expected_checkpoint_sha256=digest,
            cohort="challenge",
            probe_stage="final_answer",
        )
    dynamics = _slot_artifact(
        {
            "probe_cohort": "challenge",
            "probe_stage": "first_rule_selection",
            "transitions": [{"source_checkpoint_sha256": "a" * 64, "target_checkpoint_sha256": digest}],
        }
    )
    with pytest.raises(ValueError, match="wrong pilot matrix slot"):
        _require_dynamics_slot(
            artifact=dynamics,
            expected_initial_sha256="a" * 64,
            expected_final_sha256=digest,
            cohort="challenge",
            probe_stage="final_answer",
        )


@pytest.mark.unit
def test_supervisor_retries_transient_and_unknown_accounting(tmp_path: Path) -> None:
    terminal = tmp_path / "terminal.txt"
    script = f"""
set -euo pipefail
source scripts/production/slurm_supervision.sh
sq=0
sa=0
squeue() {{ sq=$((sq+1)); if [[ $sq -eq 1 ]]; then return 1; fi; return 0; }}
sacct() {{
  sa=$((sa+1))
  if [[ $sa -eq 1 ]]; then return 0; fi
  if [[ $sa -eq 2 ]]; then printf '123|UNKNOWN|0:0\\n'; return 0; fi
  printf '123|COMPLETED|0:0\\n'
}}
export SLURM_QUERY_RETRIES=2 SLURM_QUERY_INITIAL_BACKOFF_SECONDS=0
export SLURM_ACCOUNTING_RETRIES=3 SLURM_ACCOUNTING_INITIAL_BACKOFF_SECONDS=0
wait_for_slurm_terminal 123 {terminal}
"""
    subprocess.run(["bash", "-c", script], cwd=ROOT, check=True)
    assert terminal.read_text(encoding="utf-8").strip() == "123|COMPLETED|0:0"
