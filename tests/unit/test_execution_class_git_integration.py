from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from posttrain_circuits.artifacts.protocol_amendments import (
    CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
    ProtocolAmendmentError,
    resolve_accepted_execution_class_amendment,
    validate_execution_class_lineage_commit,
)
from posttrain_circuits.artifacts.execution_safety_certification import (
    CERTIFICATION_RELATIVE_PATH,
    DESCRIPTOR_RELATIVE_PATH,
    IMPLEMENTATION_FILE_PATHS,
    SUCCESSOR_AMENDMENT_RELATIVE_PATH,
    build_execution_safety_descriptor,
    certification_core_sha256,
)
from posttrain_circuits.scheduler_adapter.gpu_preflight_request import (
    _accepted_lineage_git_commit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = Path("/scr/del6500/OPD/tmp")


class ExecutionClassGitIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="execution-class-git-", dir=SCRATCH_ROOT
        )
        self.repository = Path(self.temporary.name) / "repository"
        self._run(
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            str(PROJECT_ROOT),
            str(self.repository),
            cwd=Path(self.temporary.name),
        )
        for relative in IMPLEMENTATION_FILE_PATHS:
            self._copy(relative)
        self._copy(str(CERTIFICATION_RELATIVE_PATH))
        self._copy(str(SUCCESSOR_AMENDMENT_RELATIVE_PATH))
        self._copy(str(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH))
        self._materialize_proposed_documents()
        self.implementation_commit = self._commit(
            "test: execution-class implementation"
        )
        intermediate = self.repository / "notes" / "pre-acceptance-review.txt"
        intermediate.parent.mkdir(parents=True)
        intermediate.write_text("review preparation only\n", encoding="utf-8")
        self.pre_acceptance_commit = self._commit(
            "test: non-safety review preparation"
        )
        self._accept_review()
        self.acceptance_commit = self._commit(
            "test: accept execution-class certification"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, *command: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            command,
            cwd=str(cwd or self.repository),
            check=True,
            capture_output=True,
            text=True,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "HOME": "/nonexistent",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
        )
        return result.stdout.strip()

    def _copy(self, relative: str) -> None:
        source = PROJECT_ROOT / relative
        destination = self.repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _write_json(self, relative: Path, payload: object) -> bytes:
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        path = self.repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def _write_yaml(self, relative: Path, payload: object) -> bytes:
        raw = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
        path = self.repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def _materialize_proposed_documents(self) -> None:
        descriptor = build_execution_safety_descriptor(self.repository)
        descriptor_raw = self._write_json(DESCRIPTOR_RELATIVE_PATH, descriptor)
        certificate = yaml.safe_load(
            (PROJECT_ROOT / CERTIFICATION_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        descriptor_sha256 = hashlib.sha256(descriptor_raw).hexdigest()
        fingerprint = descriptor["fingerprint_sha256"]
        certificate["execution_class"]["descriptor_sha256"] = descriptor_sha256
        certificate["execution_class"]["fingerprint_sha256"] = fingerprint
        certificate["legacy_evidence_condensation"][
            "attested_execution_fingerprint_sha256"
        ] = fingerprint
        implementation_files = descriptor["subject"]["implementation_files"]
        runtime = descriptor["subject"]["fixed_runtime"]["gpu_preflight"]
        for row in certificate["legacy_evidence_condensation"]["evidence"].values():
            row["certified_preflight_handler_sha256"] = implementation_files[
                "scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py"
            ]
            row["certified_runtime_dependency_lock_sha256"] = (
                implementation_files[
                    "deployments/qwen3_v2_gpu_preflight/dependency-lock.json"
                ]
            )
            row["certified_runtime_executable_sha256"] = runtime[
                "runtime_executable_sha256"
            ]
            row["certified_training_contract_sha256"] = runtime[
                "training_contract_sha256"
            ]
        certificate["review"] = {
            "status": "proposed",
            "reviewed_implementation_commit": None,
            "reviewer": None,
            "reviewed_at_utc": None,
            "rationale": None,
        }
        self._write_yaml(CERTIFICATION_RELATIVE_PATH, certificate)
        amendment = yaml.safe_load(
            (PROJECT_ROOT / SUCCESSOR_AMENDMENT_RELATIVE_PATH).read_text(
                encoding="utf-8"
            )
        )
        amendment["execution_safety_certification"].update(
            {
                "descriptor_sha256": descriptor_sha256,
                "certification_core_sha256": certification_core_sha256(
                    certificate
                ),
                "fingerprint_sha256": fingerprint,
            }
        )
        amendment["review"] = copy.deepcopy(certificate["review"])
        self._write_yaml(SUCCESSOR_AMENDMENT_RELATIVE_PATH, amendment)
        science = yaml.safe_load(
            (
                self.repository / CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH
            ).read_text(encoding="utf-8")
        )
        science["execution_class"].update(
            {
                "descriptor_sha256": descriptor_sha256,
                "fingerprint_sha256": fingerprint,
                "certification_core_sha256": certification_core_sha256(
                    certificate
                ),
            }
        )
        science["review"] = copy.deepcopy(certificate["review"])
        self._write_yaml(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH, science)

    def _accept_review(self) -> None:
        amendment_path = self.repository / SUCCESSOR_AMENDMENT_RELATIVE_PATH
        certification_path = self.repository / CERTIFICATION_RELATIVE_PATH
        science_path = self.repository / CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH
        amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
        certificate = yaml.safe_load(certification_path.read_text(encoding="utf-8"))
        science = yaml.safe_load(science_path.read_text(encoding="utf-8"))
        review = {
            "status": "accepted",
            "reviewed_implementation_commit": self.implementation_commit,
            "reviewer": "temporary-git-independent-reviewer",
            "reviewed_at_utc": "2026-09-06T23:30:00Z",
            "rationale": "Accepted the exact reusable execution-safety class.",
        }
        amendment["review"] = copy.deepcopy(review)
        certificate["review"] = copy.deepcopy(review)
        science["review"] = copy.deepcopy(review)
        self._write_yaml(SUCCESSOR_AMENDMENT_RELATIVE_PATH, amendment)
        self._write_yaml(CERTIFICATION_RELATIVE_PATH, certificate)
        self._write_yaml(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH, science)

    def _commit(self, message: str) -> str:
        self._run("git", "add", "--all")
        self._run(
            "git",
            "-c",
            "user.name=OPD Integration Test",
            "-c",
            "user.email=opd-integration-test@invalid.example",
            "commit",
            "--quiet",
            "-m",
            message,
        )
        return self._run("git", "rev-parse", "HEAD")

    def _resolve(self):  # type: ignore[no-untyped-def]
        head = self._run("git", "rev-parse", "HEAD")
        return resolve_accepted_execution_class_amendment(
            code_root=self.repository,
            configured_path=str(SUCCESSOR_AMENDMENT_RELATIVE_PATH),
            expected_head=head,
        )

    def test_real_git_joint_acceptance_resolves(self) -> None:
        binding = self._resolve()
        self.assertEqual(binding.git_commit, self.acceptance_commit)
        self.assertEqual(
            binding.reviewed_implementation_commit, self.implementation_commit
        )
        self.assertEqual(binding.certification_binding.review_status, "accepted")
        self.assertEqual(
            _accepted_lineage_git_commit(self.repository), self.acceptance_commit
        )
        self.assertNotEqual(
            self._run("git", "rev-parse", f"{self.acceptance_commit}^"),
            self.implementation_commit,
        )

    def test_noncritical_merge_history_reuses_certification(self) -> None:
        base_branch = self._run("git", "branch", "--show-current")
        self._run("git", "switch", "--quiet", "-c", "noncritical-side")
        path = self.repository / "notes" / "new-scientific-experiment.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new seed and output identity\n", encoding="utf-8")
        noncritical_commit = self._commit("test: add noncritical experiment")
        self._run("git", "switch", "--quiet", base_branch)
        main_path = self.repository / "notes" / "new-analysis.txt"
        main_path.parent.mkdir(parents=True, exist_ok=True)
        main_path.write_text("non-execution analysis\n", encoding="utf-8")
        self._commit("test: add noncritical analysis")
        self._run(
            "git",
            "-c",
            "user.name=OPD Integration Test",
            "-c",
            "user.email=opd-integration-test@invalid.example",
            "merge",
            "--quiet",
            "--no-ff",
            "-m",
            "test: merge noncritical science",
            "noncritical-side",
        )
        merge_commit = self._run("git", "rev-parse", "HEAD")
        binding = self._resolve()
        self.assertEqual(binding.git_commit, self.acceptance_commit)
        self.assertEqual(_accepted_lineage_git_commit(self.repository), merge_commit)
        validate_execution_class_lineage_commit(
            code_root=self.repository,
            candidate_commit=noncritical_commit,
            current_binding=binding,
            expected_head=merge_commit,
            role="temporary request commit",
        )

    def test_seed_change_reuses_execution_class(self) -> None:
        science = self.repository / "configs" / "g0" / "qwen3_v2_eap_separation.yaml"
        raw = science.read_text(encoding="utf-8")
        self.assertIn("seed: 42", raw)
        science.write_text(raw.replace("seed: 42", "seed: 43"), encoding="utf-8")
        new_experiment_commit = self._commit("test: change scientific seed")
        binding = self._resolve()
        validate_execution_class_lineage_commit(
            code_root=self.repository,
            candidate_commit=new_experiment_commit,
            current_binding=binding,
            expected_head=new_experiment_commit,
            role="new experiment request commit",
        )

    def test_critical_delta_is_rejected_by_recomputed_descriptor(self) -> None:
        critical = (
            self.repository
            / "src/posttrain_circuits/learning/supervision/losses.py"
        )
        critical.write_text(
            critical.read_text(encoding="utf-8") + "# unsafe test delta\n",
            encoding="utf-8",
        )
        self._commit("test: mutate critical execution surface")
        with self.assertRaisesRegex(
            ProtocolAmendmentError, "descriptor does not match the current checkout"
        ):
            self._resolve()

    def test_reviewed_implementation_blob_must_match_descriptor(self) -> None:
        base_commit = self._run(
            "git", "rev-parse", f"{self.implementation_commit}^"
        )
        self._run("git", "switch", "--quiet", "-c", "mismatched-implementation", base_commit)
        for relative in IMPLEMENTATION_FILE_PATHS:
            self._copy(relative)
        self._copy(str(CERTIFICATION_RELATIVE_PATH))
        self._copy(str(SUCCESSOR_AMENDMENT_RELATIVE_PATH))
        self._copy(str(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH))
        self._materialize_proposed_documents()
        critical = (
            self.repository
            / "src/posttrain_circuits/learning/supervision/losses.py"
        )
        critical.write_text(
            critical.read_text(encoding="utf-8") + "# mismatch in reviewed commit\n",
            encoding="utf-8",
        )
        self.implementation_commit = self._commit(
            "test: mismatched execution-class implementation"
        )
        self._copy("src/posttrain_circuits/learning/supervision/losses.py")
        self._accept_review()
        self.acceptance_commit = self._commit(
            "test: invalid acceptance restoring critical blob"
        )
        with self.assertRaisesRegex(
            ProtocolAmendmentError,
            "reviewed implementation safety-critical blob differs from descriptor",
        ):
            self._resolve()

    def test_compatibility_provenance_is_accepted_lineage(self) -> None:
        first = self.repository / "notes" / "request.txt"
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_text("request commit\n", encoding="utf-8")
        request_commit = self._commit("test: request commit")
        second = self.repository / "notes" / "execution.txt"
        second.write_text("execution commit\n", encoding="utf-8")
        execution_commit = self._commit("test: execution commit")
        binding = self._resolve()
        validate_execution_class_lineage_commit(
            code_root=self.repository,
            candidate_commit=request_commit,
            current_binding=binding,
            expected_head=execution_commit,
            role="temporary request commit",
        )
        with self.assertRaisesRegex(
            ProtocolAmendmentError, "predates or is disconnected"
        ):
            validate_execution_class_lineage_commit(
                code_root=self.repository,
                candidate_commit=self.pre_acceptance_commit,
                current_binding=binding,
                expected_head=execution_commit,
                role="temporary request commit",
            )
        with self.assertRaisesRegex(ProtocolAmendmentError, "is not a Git commit"):
            validate_execution_class_lineage_commit(
                code_root=self.repository,
                candidate_commit="not-a-commit",
                current_binding=binding,
                expected_head=execution_commit,
                role="temporary request commit",
            )

    def test_acceptance_commit_rejects_non_review_change(self) -> None:
        self._run(
            "git",
            "switch",
            "--quiet",
            "-c",
            "invalid-acceptance",
            self.pre_acceptance_commit,
        )
        self._accept_review()
        extra = self.repository / "notes" / "smuggled-change.txt"
        extra.write_text("not review metadata\n", encoding="utf-8")
        self._commit("test: invalid mixed acceptance")
        with self.assertRaisesRegex(
            ProtocolAmendmentError, "acceptance commit contains non-review changes"
        ):
            self._resolve()

    def test_certification_artifact_cannot_change_after_acceptance(self) -> None:
        certificate = self.repository / CERTIFICATION_RELATIVE_PATH
        certificate.write_text(
            certificate.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        self._commit("test: mutate accepted certification bytes")
        with self.assertRaisesRegex(
            ProtocolAmendmentError, "review artifacts must change together exactly once"
        ):
            self._resolve()


if __name__ == "__main__":
    unittest.main()
