"""Production go/no-go readiness report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.probe_cohorts import validate_probe_cohort_manifest
from posttrain_circuits.core.config import is_production_scale
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.scientific_versions import (
    require_core_v2_artifact,
    scientific_compatibility_fields,
)


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    evidence: str
    required: bool = True


@dataclass
class ReadinessReport:
    checks: list[ReadinessCheck]
    created_at: str
    bindings: dict[str, str] | None = None

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks if check.required)

    def write(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 2,
            **scientific_compatibility_fields(),
            "ready": self.ready,
            "created_at": self.created_at,
            "checks": [asdict(check) for check in self.checks],
            "bindings": self.bindings or {},
        }
        payload["sha256"] = sha256_value(payload)
        atomic_write_json(root / "readiness.json", payload)
        lines = ["# Production readiness", "", f"Overall: **{'GO' if self.ready else 'NO-GO'}**", ""]
        lines.extend(
            f"- [{'x' if check.passed else ' '}] {check.name}: {check.evidence}" for check in self.checks
        )
        (root / "readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_readiness_report(
    evidence: dict[str, tuple[bool, str]],
    *,
    bindings: dict[str, str] | None = None,
) -> ReadinessReport:
    required_names = (
        "base_task_accuracy_nontrivial",
        "pilot_improves_accuracy",
        "verifier_deterministic",
        "fixed_bank_mixed_rewards",
        "teacher_topk_mass_acceptable",
        "hf_circuit_logit_parity",
        "eap_ig_beats_random",
        "exact_patching_distinguishes_groups",
        "attribution_bootstrap_stable",
        "checkpoint_resume_verified",
        "split_leakage_absent",
        "anti_shortcut_gap",
        "probe_cohorts_frozen",
        "teacher_correctness",
        "label_leakage",
    )
    checks = [
        ReadinessCheck(name, *evidence.get(name, (False, "no evidence supplied"))) for name in required_names
    ]
    return ReadinessReport(checks, utc_now(), bindings)


def validate_anti_shortcut_report(
    path: Path,
    *,
    max_shortcut_gap: float,
    expected_model_checkpoint_hash: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = payload.pop("sha256", None)
    if expected_hash != sha256_value(payload):
        raise ValueError("anti-shortcut report hash mismatch")
    payload["sha256"] = expected_hash
    require_core_v2_artifact(payload)
    if str(payload.get("model_checkpoint_hash")) != expected_model_checkpoint_hash:
        raise RuntimeError(
            "anti-shortcut evidence is for a different initial student checkpoint: "
            f"expected={expected_model_checkpoint_hash}, "
            f"observed={payload.get('model_checkpoint_hash')}"
        )
    observed = float(payload["shortcut_gap"])
    capability_failures = []
    if float(payload.get("iid_accuracy", 0.0)) < float(payload.get("minimum_iid_accuracy", 1.0)):
        capability_failures.append("IID accuracy")
    if float(payload.get("transformed_accuracy", 0.0)) < float(
        payload.get("minimum_transformed_accuracy", 1.0)
    ):
        capability_failures.append("transformed accuracy")
    per_minimum = float(payload.get("minimum_per_transformation_accuracy", 1.0))
    per_values = payload.get("transformation_accuracy", {})
    if not isinstance(per_values, dict) or any(float(value) < per_minimum for value in per_values.values()):
        capability_failures.append("per-transformation accuracy")
    if observed > max_shortcut_gap or capability_failures or payload.get("passed") is not True:
        raise RuntimeError(
            "anti-shortcut gate failed: "
            f"shortcut_gap={observed:.6f}, max_shortcut_gap={max_shortcut_gap:.6f}, "
            f"capability_failures={capability_failures}"
        )
    return payload


def validate_readiness_report(
    path: Path,
    *,
    expected_initial_checkpoint_hash: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = payload.pop("sha256", None)
    if expected_hash != sha256_value(payload):
        raise ValueError("readiness report hash mismatch")
    payload["sha256"] = expected_hash
    require_core_v2_artifact(payload)
    checks = payload.get("checks", [])
    names = {str(check.get("name")) for check in checks if isinstance(check, dict)}
    required = {check.name for check in build_readiness_report({}).checks}
    if names != required:
        raise RuntimeError(f"readiness report check set mismatch: missing={sorted(required - names)}")
    failures = [
        str(check["name"])
        for check in checks
        if check.get("required", True) and check.get("passed") is not True
    ]
    if payload.get("ready") is not True or failures:
        raise RuntimeError(f"production readiness is NO-GO: failed={failures}")
    bindings = payload.get("bindings", {})
    if str(bindings.get("initial_checkpoint_hash")) != expected_initial_checkpoint_hash:
        raise RuntimeError("readiness evidence is bound to a different initial checkpoint")
    for name in ("dataset_hash", "suite_hash", "code_commit", "prereg_commit"):
        if not str(bindings.get(name, "")).strip() or bindings.get(name) == "unavailable":
            raise RuntimeError(f"readiness evidence lacks binding {name}")
    return payload


def require_factorial_prerequisites(config: dict[str, Any]) -> dict[str, Any]:
    """Refuse a production factorial cell until both pre-training gates pass."""

    safety = config["production_safety"]
    checkpoint_hash = str(safety.get("initial_checkpoint_hash", config["model"]["model_revision"]))
    evidence: dict[str, Any] = {}
    if is_production_scale(config):
        readiness = validate_readiness_report(
            Path(str(safety["readiness_report"])),
            expected_initial_checkpoint_hash=checkpoint_hash,
        )
        evidence["full_readiness"] = {
            "report_hash": readiness["sha256"],
            "bindings": readiness["bindings"],
        }
    if bool(safety.get("require_anti_shortcut_gate", True)):
        report = validate_anti_shortcut_report(
            Path(str(config["anti_shortcut"]["report_path"])),
            max_shortcut_gap=float(config["anti_shortcut"]["max_shortcut_gap"]),
            expected_model_checkpoint_hash=checkpoint_hash,
        )
        evidence["anti_shortcut"] = {
            "report_hash": report["sha256"],
            "shortcut_gap": report["shortcut_gap"],
            "threshold": config["anti_shortcut"]["max_shortcut_gap"],
        }
        if "full_readiness" in evidence:
            bindings = evidence["full_readiness"]["bindings"]
            expected = {
                "dataset_hash": bindings["dataset_hash"],
                "suite_hash": bindings["suite_hash"],
                "code_commit": bindings["code_commit"],
                "prereg_commit": bindings["prereg_commit"],
            }
            observed_bindings = {key: str(report.get(key, "")) for key in expected}
            if observed_bindings != expected:
                raise RuntimeError(
                    "anti-shortcut evidence bindings differ from full readiness: "
                    f"expected={expected}, observed={observed_bindings}"
                )
    if bool(safety.get("require_frozen_probe_cohorts", True)):
        probes = validate_probe_cohort_manifest(Path(str(safety["probe_cohort_manifest"])))
        if probes["initial_student_checkpoint_hash"] != checkpoint_hash:
            raise RuntimeError(
                "probe cohorts are pinned to a different initial student checkpoint: "
                f"expected={checkpoint_hash}, observed={probes['initial_student_checkpoint_hash']}"
            )
        if "full_readiness" in evidence:
            bindings = evidence["full_readiness"]["bindings"]
            if probes.get("git_commit") != bindings["code_commit"]:
                raise RuntimeError("probe cohorts were frozen under a different code commit")
            if probes.get("prereg_commit") != bindings["prereg_commit"]:
                raise RuntimeError("probe cohorts were frozen under a different prereg commit")
            if (
                probes.get("construction_phase") != "before_confirmatory_training"
                or probes.get("training_ancestry") != []
            ):
                raise RuntimeError("probe cohorts are not demonstrably frozen before training")
        evidence["probe_cohorts"] = {
            "manifest_hash": probes["sha256"],
            "initial_student_checkpoint_hash": probes["initial_student_checkpoint_hash"],
        }
    return evidence
