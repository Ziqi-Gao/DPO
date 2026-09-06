from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from posttrain_circuits.artifacts.protocol_amendments import (
    AMENDMENT_RELATIVE_PATH,
    load_protocol_amendment_bytes,
)
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
    GATE_NAMES,
    PROFILE_NAME,
    WORKFLOW_ID,
)
from posttrain_circuits.scheduler_adapter.registry import require_handler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDLER = PROJECT_ROOT / "scripts" / "server_scheduler" / "qwen3-v2-g0-handler.py"
PREPARE = PROJECT_ROOT / "scripts" / "server_scheduler" / "prepare-qwen3-v2-g0-runtime.py"


class Qwen3V2G0HandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("opd_g0_handler", HANDLER)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)
        prepare_spec = importlib.util.spec_from_file_location(
            "opd_g0_prepare_runtime", PREPARE
        )
        assert prepare_spec is not None and prepare_spec.loader is not None
        cls.prepare_module = importlib.util.module_from_spec(prepare_spec)
        prepare_spec.loader.exec_module(cls.prepare_module)

    def _accepted_amendment(self, implementation_commit: str) -> tuple[bytes, bytes]:
        proposed = (PROJECT_ROOT / AMENDMENT_RELATIVE_PATH).read_bytes()
        self.assertEqual(
            hashlib.sha256(proposed).hexdigest(),
            self.module.PROPOSED_AMENDMENT_SHA256,
        )
        proposed_review = (
            b"review:\n"
            b"  status: proposed\n"
            b"  reviewed_implementation_commit: null\n"
            b"  reviewer: null\n"
            b"  reviewed_at_utc: null\n"
            b"  rationale: null\n"
        )
        accepted_review = (
            b"review:\n"
            b"  status: accepted\n"
            + f"  reviewed_implementation_commit: {implementation_commit}\n".encode()
            + b"  reviewer: independent-reviewer\n"
            + b"  reviewed_at_utc: '2026-09-05T12:00:00Z'\n"
            + b"  rationale: Reviewed implementation accepted.\n"
        )
        self.assertIn(proposed_review, proposed)
        return proposed, proposed.replace(proposed_review, accepted_review)

    def test_stage_plan_is_dynamic_and_foreground(self) -> None:
        root = Path("/scr/del6500/OPD/tmp/test-qwen3-v2-g0/qwen3-v2")
        for gpu_count, threads in ((1, 24), (2, 12), (3, 8), (4, 6)):
            stages = self.module._stage_plan(
                root,
                initial_checkpoint_sha256="a" * 64,
                gpu_count=gpu_count,
            )
            self.assertEqual(len(stages), 19)
            self.assertEqual(stages[0].name, "build_splits")
            self.assertEqual(stages[-1].name, "finalize_g0")
            finalizer = stages[-1]
            self.assertNotIn("--compatibility", finalizer.argv)
            final_compatibility_index = finalizer.argv.index(
                "--final-compatibility"
            ) + 1
            process_compatibility_index = finalizer.argv.index(
                "--process-compatibility"
            ) + 1
            self.assertEqual(
                finalizer.argv[final_compatibility_index],
                str(
                    root
                    / "circuits"
                    / "final_answer"
                    / "mib_raw"
                    / "compatibility.json"
                ),
            )
            self.assertEqual(
                finalizer.argv[process_compatibility_index],
                str(
                    root
                    / "circuits"
                    / "first_rule_selection"
                    / "mib_raw"
                    / "compatibility.json"
                ),
            )
            self.assertEqual(
                [stage.name for stage in stages if stage.distributed],
                ["calibration_sft", "resume_a", "resume_b"],
            )
            compare = next(
                stage for stage in stages if stage.name == "compare_distributed_resume"
            )
            world_size_index = compare.argv.index("--world-size") + 1
            self.assertEqual(compare.argv[world_size_index], str(gpu_count))
            score = next(
                stage for stage in stages if stage.name == "score_probe_candidates"
            )
            score_world_size_index = score.argv.index("--world-size") + 1
            self.assertEqual(score.argv[score_world_size_index], str(gpu_count))
            self.assertEqual(self.module.THREADS_PER_RANK[gpu_count], threads)

    def test_checkpoint_placeholders_are_not_erased_before_resume_runs_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".g0-placeholders-", dir=PROJECT_ROOT) as raw:
            root = Path(raw) / "qwen3-v2"
            calibration = root / "calibration"
            calibration.mkdir(parents=True)
            (calibration / "manifest.json").write_text("{}", encoding="utf-8")
            placeholders = tuple(
                str(root / name / "FINAL_CHECKPOINT")
                for name in ("calibration", "resume-a", "resume-b")
            )
            stages = (
                self.module.Stage("probe", "probe", placeholders),
            )

            def resolved(run_root: Path) -> Path:
                return run_root / "checkpoints" / "step-final.pt"

            with mock.patch.object(
                self.module,
                "_resolve_final_checkpoint",
                side_effect=resolved,
            ):
                after_calibration = self.module._replace_checkpoint_placeholders(
                    stages,
                    root,
                )
                self.assertEqual(
                    after_calibration[0].argv,
                    (
                        str(resolved(calibration)),
                        placeholders[1],
                        placeholders[2],
                    ),
                )
                for name in ("resume-a", "resume-b"):
                    run_root = root / name
                    run_root.mkdir()
                    (run_root / "manifest.json").write_text("{}", encoding="utf-8")
                after_resumes = self.module._replace_checkpoint_placeholders(
                    after_calibration,
                    root,
                )
            self.assertEqual(
                after_resumes[0].argv,
                tuple(
                    str(resolved(root / name))
                    for name in ("calibration", "resume-a", "resume-b")
                ),
            )

    def test_final_checkpoint_resolver_rejects_manifest_and_path_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".g0-checkpoint-", dir=PROJECT_ROOT) as raw:
            run_root = Path(raw) / "calibration"
            checkpoints = run_root / "checkpoints"
            checkpoints.mkdir(parents=True)
            checkpoint = checkpoints / "step-00000020.pt"
            checkpoint.write_bytes(b"checkpoint")

            def write_manifest(path: Path, digest: str) -> None:
                payload = {
                    "final_checkpoint_path": str(path),
                    "final_checkpoint_sha256": digest,
                }
                payload["sha256"] = self.module._sha256_value(payload)
                (run_root / "manifest.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            write_manifest(checkpoint, digest)
            self.assertEqual(self.module._resolve_final_checkpoint(run_root), checkpoint)

            payload = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
            payload["sha256"] = "0" * 64
            (run_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(self.module.G0Error, "manifest SHA-256"):
                self.module._resolve_final_checkpoint(run_root)

            outside = Path(raw) / "outside.pt"
            outside.write_bytes(b"checkpoint")
            write_manifest(outside, digest)
            with self.assertRaisesRegex(self.module.G0Error, "binding"):
                self.module._resolve_final_checkpoint(run_root)

            linked = checkpoints / "linked.pt"
            linked.symlink_to(checkpoint.name)
            write_manifest(linked, digest)
            with self.assertRaisesRegex(self.module.G0Error, "binding"):
                self.module._resolve_final_checkpoint(run_root)

    def test_handler_has_no_host_gpu_selection_or_scheduler_control(self) -> None:
        source = HANDLER.read_text(encoding="utf-8").lower()
        for forbidden in (
            "nvidia-smi",
            "sacct",
            "sbatch",
            "scontrol",
            "squeue",
            "serverscheduler submit",
            "serverscheduler dispatch",
            "serverscheduler serve",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("server_scheduler_memory_max_bytes", source)
        self.assertNotIn("server_scheduler_runtime_unit", source)

    def test_fixed_environment_preserves_scheduler_visibility(self) -> None:
        handler = require_handler("qwen3_v2_g0")
        profile = handler.profile(PROFILE_NAME)
        self.assertEqual(profile.process_count, 0)
        self.assertEqual(profile.cpu_cores_min, 24)
        self.assertEqual(profile.memory_mib_min, 196608)
        self.assertEqual(profile.gpu_count, 0)
        self.assertEqual(profile.gpu_count_policy, "scheduler")
        self.assertEqual(profile.scheduler_gpu_counts, (1, 2, 3, 4))
        self.assertEqual(dict(handler.fixed_environment), self.module.FIXED_ENVIRONMENT)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.module.FIXED_ENVIRONMENT)
        self.assertNotIn("PYTHONDONTWRITEBYTECODE", self.module.FIXED_ENVIRONMENT)
        self.assertEqual(self.module.FIXED_ENVIRONMENT["NCCL_P2P_DISABLE"], "1")
        self.assertEqual(
            self.module.FIXED_ENVIRONMENT["MIB_REPOSITORY"],
            "/scr/del6500/OPD/vendor/MIB-circuit-track-v1",
        )

    def test_allocation_environment_cross_checks_all_world_sizes(self) -> None:
        for gpu_count, threads in ((1, 24), (2, 12), (3, 8), (4, 6)):
            visible = ",".join(f"GPU-{index}" for index in range(gpu_count))
            environment = {
                **{key: str(threads) for key in self.module.THREAD_KEYS},
                "CUDA_VISIBLE_DEVICES": visible,
            }
            before = dict(environment)
            self.assertEqual(
                self.module._validate_allocation_environment(gpu_count, environment),
                tuple(visible.split(",")),
            )
            self.assertEqual(environment, before)
        invalid = {
            **{key: "8" for key in self.module.THREAD_KEYS},
            "CUDA_VISIBLE_DEVICES": "GPU-0,GPU-1",
        }
        with self.assertRaisesRegex(self.module.G0Error, "visibility"):
            self.module._validate_allocation_environment(3, invalid)
        invalid["CUDA_VISIBLE_DEVICES"] = "GPU-0,GPU-1,GPU-2"
        invalid["OMP_NUM_THREADS"] = "7"
        with self.assertRaisesRegex(self.module.G0Error, "thread"):
            self.module._validate_allocation_environment(3, invalid)

    def test_bytecode_isolation_is_programmatic_pid_unique_and_private(self) -> None:
        old_dont_write = sys.dont_write_bytecode
        old_prefix = sys.pycache_prefix
        old_module_prefix = self.module._PROCESS_PYCACHE_PREFIX
        created_prefix: str | None = None
        self.module._PROCESS_PYCACHE_PREFIX = None
        try:
            self.module._configure_bytecode_isolation()
            created_prefix = self.module._PROCESS_PYCACHE_PREFIX
            expected_start = (
                "/scr/del6500/OPD/tmp/.qwen3-v2-g0-pycache-disabled-"
                f"{self.module.os.getpid()}-"
            )
            self.assertTrue(sys.dont_write_bytecode)
            self.assertIsNotNone(self.module._PROCESS_PYCACHE_PREFIX)
            self.assertTrue(self.module._PROCESS_PYCACHE_PREFIX.startswith(expected_start))
            self.assertEqual(sys.pycache_prefix, self.module._PROCESS_PYCACHE_PREFIX)
            metadata = Path(sys.pycache_prefix).stat()
            self.assertEqual(metadata.st_uid, self.module.os.geteuid())
            self.assertEqual(metadata.st_mode & 0o777, 0o700)
            self.assertEqual(list(Path(sys.pycache_prefix).iterdir()), [])
        finally:
            sys.dont_write_bytecode = old_dont_write
            sys.pycache_prefix = old_prefix
            self.module._PROCESS_PYCACHE_PREFIX = old_module_prefix
            if created_prefix is not None:
                self.module.os.rmdir(created_prefix)

    def test_supervisor_bootstraps_lineage_before_installing_source(self) -> None:
        invocation = mock.Mock()
        events: list[str] = []
        binding = self.module.BootstrapLineage(
            amendment_sha256="1" * 64,
            amendment_git_commit="2" * 40,
            reviewed_implementation_commit="3" * 40,
            request_git_commit="4" * 40,
            preflight_git_commits=("5" * 40,) * 4,
        )

        def bootstrap(**_: object) -> object:
            events.append("bootstrap")
            return binding

        def install() -> None:
            events.append("install-source")

        def shared(*_: object, **__: object) -> object:
            events.append("shared-validator")
            raise self.module.G0Error("stop after ordering probe")

        with (
            mock.patch.object(self.module, "_configure_bytecode_isolation"),
            mock.patch.object(self.module, "_parse_outer", return_value=invocation),
            mock.patch.object(self.module, "_validate_environment"),
            mock.patch.object(self.module, "_require_clean_git", return_value="6" * 40),
            mock.patch.object(
                self.module,
                "_read_inputs",
                return_value=(
                    {"resolved_config_sha256": {}},
                    b"prereg",
                    b"amendment",
                    {},
                ),
            ),
            mock.patch.object(
                self.module,
                "_bootstrap_accepted_lineage",
                side_effect=bootstrap,
            ),
            mock.patch.object(self.module, "_install_source_path", side_effect=install),
            mock.patch.object(
                self.module,
                "_validate_config_and_preflight",
                side_effect=shared,
            ),
        ):
            with self.assertRaisesRegex(self.module.G0Error, "ordering probe"):
                self.module._supervise([])
        self.assertEqual(events, ["bootstrap", "install-source", "shared-validator"])

    def test_handler_rejects_untracked_files_hidden_by_ignore_rules(self) -> None:
        raw = (
            b".codex/config.toml\0"
            b"src/pkg/__pycache__/safe.cpython-312.pyc\0"
            b"src/pkg/ignored.py\0"
        )
        with mock.patch.object(self.module, "_git_bytes", return_value=raw):
            self.assertEqual(
                self.module._unsafe_untracked_paths(),
                ("src/pkg/ignored.py",),
            )

    def test_scientific_child_bootstraps_before_installing_source(self) -> None:
        code_commit = "a" * 40
        request_commit = "b" * 40
        preflight_commit = "c" * 40
        amendment_sha256 = "d" * 64
        events: list[str] = []
        binding = self.module.BootstrapLineage(
            amendment_sha256=amendment_sha256,
            amendment_git_commit="e" * 40,
            reviewed_implementation_commit="f" * 40,
            request_git_commit=request_commit,
            preflight_git_commits=(preflight_commit,) * 4,
        )
        imported = mock.Mock()
        imported.main = mock.Mock()

        def bootstrap(**_: object) -> object:
            events.append("bootstrap")
            return binding

        def install() -> None:
            events.append("install-source")

        def import_module(_: str) -> object:
            events.append("project-import")
            return imported

        argv = [
            "build_splits",
            "--code-commit",
            code_commit,
            "--request-git-commit",
            request_commit,
            "--preflight-git-commit",
            preflight_commit,
            "--amendment-sha256",
            amendment_sha256,
            "--reviewed-implementation-commit",
            "f" * 40,
            "--gpu-count",
            "3",
            "--allocation-sha256",
            "1" * 64,
            "--",
            "scientific-argument",
        ]
        with (
            mock.patch.object(self.module, "_configure_bytecode_isolation"),
            mock.patch.object(
                self.module,
                "_require_clean_git",
                return_value=code_commit,
            ),
            mock.patch.object(
                self.module,
                "_bootstrap_accepted_lineage",
                side_effect=bootstrap,
            ),
            mock.patch.object(self.module, "_install_source_path", side_effect=install),
            mock.patch.object(
                self.module.importlib,
                "import_module",
                side_effect=import_module,
            ),
        ):
            self.assertEqual(self.module._scientific_cli(argv), 0)
        self.assertEqual(events, ["bootstrap", "install-source", "project-import"])
        imported.main.assert_called_once_with(["scientific-argument"])

    def test_finalizer_receives_handler_owned_allocation_context(self) -> None:
        code_commit = "a" * 40
        request_commit = "b" * 40
        preflight_commit = "c" * 40
        implementation_commit = "d" * 40
        amendment_sha256 = "e" * 64
        allocation_sha256 = "f" * 64
        binding = self.module.BootstrapLineage(
            amendment_sha256=amendment_sha256,
            amendment_git_commit="1" * 40,
            reviewed_implementation_commit=implementation_commit,
            request_git_commit=request_commit,
            preflight_git_commits=(preflight_commit,) * 4,
        )
        imported = mock.Mock()
        imported.main = mock.Mock()
        argv = [
            "finalize_g0",
            "--code-commit",
            code_commit,
            "--request-git-commit",
            request_commit,
            "--preflight-git-commit",
            preflight_commit,
            "--amendment-sha256",
            amendment_sha256,
            "--reviewed-implementation-commit",
            implementation_commit,
            "--gpu-count",
            "3",
            "--allocation-sha256",
            allocation_sha256,
            "--",
            "scientific-argument",
        ]
        with (
            mock.patch.object(self.module, "_configure_bytecode_isolation"),
            mock.patch.object(self.module, "_require_clean_git", return_value=code_commit),
            mock.patch.object(
                self.module, "_bootstrap_accepted_lineage", return_value=binding
            ),
            mock.patch.object(self.module, "_install_source_path"),
            mock.patch.object(self.module.importlib, "import_module", return_value=imported),
        ):
            self.assertEqual(self.module._scientific_cli(argv), 0)
        call = imported.main.call_args
        self.assertEqual(call.args, (["scientific-argument"],))
        context = call.kwargs["scientific_context"]
        self.assertEqual(context.world_size, 3)
        self.assertEqual(context.allocation_sha256, allocation_sha256)
        self.assertEqual(context.request_git_commit, request_commit)
        self.assertEqual(context.gpu_preflight_git_commit, preflight_commit)

    def test_finalizer_direct_or_tampered_context_fails_closed(self) -> None:
        from posttrain_circuits.cli.finalize_g0 import (
            ScientificInvocationContext,
            main as finalize_g0,
        )

        with self.assertRaisesRegex(RuntimeError, "handler-owned"):
            finalize_g0([])
        tampered = ScientificInvocationContext(
            allocation_sha256="f" * 64,
            code_commit="a" * 40,
            gpu_preflight_git_commit="b" * 40,
            protocol_amendment_sha256="c" * 64,
            request_git_commit="d" * 40,
            reviewed_implementation_commit="e" * 40,
            world_size=5,
        )
        with self.assertRaisesRegex(ValueError, "world_size"):
            finalize_g0([], scientific_context=tampered)

    def test_bootstrap_rejects_source_change_even_when_later_reverted(self) -> None:
        implementation = "a" * 40
        acceptance = "b" * 40
        tamper = "c" * 40
        reverted = "d" * 40
        proposed, accepted = self._accepted_amendment(implementation)
        amendment_by_commit = {
            implementation: proposed,
            acceptance: accepted,
            tamper: accepted,
            reverted: accepted,
        }
        changed_by_edge = {
            (implementation, acceptance): frozenset(
                {str(self.module.AMENDMENT_RELATIVE_PATH)}
            ),
            (acceptance, tamper): frozenset(
                {"src/posttrain_circuits/artifacts/protocol_amendments.py"}
            ),
            (tamper, reverted): frozenset(
                {"src/posttrain_circuits/artifacts/protocol_amendments.py"}
            ),
        }
        with (
            mock.patch.object(
                self.module,
                "_git_amendment_bytes",
                side_effect=lambda commit: amendment_by_commit[commit],
            ),
            mock.patch.object(
                self.module,
                "_linear_commit_path",
                return_value=(acceptance, tamper, reverted),
            ),
            mock.patch.object(
                self.module,
                "_changed_paths",
                side_effect=lambda parent, commit: changed_by_edge[(parent, commit)],
            ),
        ):
            with self.assertRaisesRegex(
                self.module.G0Error,
                "non-handoff implementation change",
            ):
                self.module._validate_accepted_commit_chain(
                    implementation=implementation,
                    endpoint=reverted,
                    accepted_raw=accepted,
                    role="test lineage",
                )

    def test_bootstrap_accepts_one_review_transition_then_handoff(self) -> None:
        implementation = "a" * 40
        acceptance = "b" * 40
        handoff = "c" * 40
        proposed, accepted = self._accepted_amendment(implementation)
        with (
            mock.patch.object(
                self.module,
                "_git_amendment_bytes",
                side_effect=lambda commit: (
                    proposed if commit == implementation else accepted
                ),
            ),
            mock.patch.object(
                self.module,
                "_linear_commit_path",
                return_value=(acceptance, handoff),
            ),
            mock.patch.object(
                self.module,
                "_changed_paths",
                side_effect=(
                    frozenset({str(self.module.AMENDMENT_RELATIVE_PATH)}),
                    frozenset({self.module.HANDOFF_RELATIVE_PATH}),
                ),
            ),
        ):
            self.assertEqual(
                self.module._validate_accepted_commit_chain(
                    implementation=implementation,
                    endpoint=handoff,
                    accepted_raw=accepted,
                    role="test lineage",
                ),
                acceptance,
            )

    def test_bootstrap_rejects_accepted_scientific_term_tampering(self) -> None:
        implementation = "a" * 40
        proposed, accepted = self._accepted_amendment(implementation)
        tampered = accepted.replace(b"  token_budget: 2000000\n", b"  token_budget: 2000001\n")
        self.assertNotEqual(tampered, accepted)
        with mock.patch.object(
            self.module,
            "_git_amendment_bytes",
            return_value=proposed,
        ):
            with self.assertRaisesRegex(
                self.module.G0Error,
                "changed reviewed scientific terms",
            ):
                self.module._validate_accepted_commit_chain(
                    implementation=implementation,
                    endpoint="b" * 40,
                    accepted_raw=tampered,
                    role="test lineage",
                )

    def test_bootstrap_rejects_merge_lineage(self) -> None:
        implementation = "a" * 40
        merge = "b" * 40
        with (
            mock.patch.object(self.module, "_is_ancestor", return_value=True),
            mock.patch.object(
                self.module,
                "_commit_parents",
                return_value=(implementation, "c" * 40),
            ),
        ):
            with self.assertRaisesRegex(self.module.G0Error, "single-parent chain"):
                self.module._linear_commit_path(
                    implementation,
                    merge,
                    role="test lineage",
                )

    def test_deployment_hashes_bind_handler_lock_and_manifest(self) -> None:
        deployment = require_handler("qwen3_v2_g0").deployment
        for path, expected in (
            (deployment.implementation, deployment.implementation_sha256),
            (deployment.dependency_lock, deployment.dependency_lock_sha256),
            (deployment.package_manifest, deployment.package_manifest_sha256),
        ):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        deployment.validate_identity()

    def test_completion_uses_exact_registered_semantic_gates(self) -> None:
        invocation = self.module.Invocation(
            workflow_id=WORKFLOW_ID,
            plan_sha256="1" * 64,
            unit_id="g0",
            run_id="2" * 64,
            job_id="opd-" + "3" * 32,
            attempt=1,
            execution_profile=PROFILE_NAME,
            gpu_count=3,
            manifest_sha256="4" * 64,
            allocation_sha256="5" * 64,
            content_handles=tuple(
                self.module.ContentHandle(name, str(index) * 64, index + 10)
                for index, name in enumerate(self.module.EXPECTED_INPUT_NAMES, start=1)
            ),
            output_descriptor=30,
        )
        payload = self.module._completion(
            invocation,
            started_at="2026-09-04T04:00:00Z",
            completed_at="2026-09-04T05:00:00Z",
        )
        self.assertEqual(tuple(sorted(payload["scientific_validation"])), GATE_NAMES)
        self.assertEqual(payload["execution"]["execution_profile"], PROFILE_NAME)

    def test_runtime_preparation_is_explicit_and_pins_sources(self) -> None:
        source = PREPARE.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument(\n        "--execute"', source)
        self.assertIn("b759df34433c9e31043ba9e02908ce0bf20e894f", source)
        self.assertIn("submodule", source)
        self.assertIn('"submodule.EAP-IG.url",\n            EAP_URL', source)
        self.assertIn('"--checkout"', source)
        self.assertIn("repository / 'EAP-IG' / 'src'", source)
        self.assertIn('str(python),\n            "-B",\n            "-I"', source)
        self.assertLess(
            source.index("_offline_check_script(mib_stage)"),
            source.index('_require_clean_git_tree(mib_stage / "EAP-IG")'),
        )
        self.assertLess(
            source.index('_require_clean_git_tree(mib_stage / "EAP-IG")'),
            source.index("_require_clean_git_tree(mib_stage)"),
        )
        self.assertNotIn('f"submodule.EAP-IG.url={EAP_URL}"', source)
        self.assertNotIn("torch.cuda", source)

    def test_runtime_preparation_rejects_dirty_source_tree(self) -> None:
        clean = mock.Mock(stdout="")
        with mock.patch.object(
            self.prepare_module.subprocess,
            "run",
            return_value=clean,
        ) as run:
            self.prepare_module._require_clean_git_tree(Path("/fixed/mib"))
        run.assert_called_once_with(
            (
                "/usr/bin/git",
                "-C",
                "/fixed/mib",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ),
            check=True,
            capture_output=True,
            text=True,
        )

        dirty = mock.Mock(stdout="?? src/eap/__pycache__/graph.pyc\n")
        with mock.patch.object(
            self.prepare_module.subprocess,
            "run",
            return_value=dirty,
        ):
            with self.assertRaisesRegex(RuntimeError, "offline verification dirtied"):
                self.prepare_module._require_clean_git_tree(Path("/fixed/mib/EAP-IG"))

    def test_circuit_runner_uses_pinned_submodule_src_layout(self) -> None:
        from posttrain_circuits.causal_circuits.model.runner import _eap_source_root

        repository = Path("/scr/del6500/OPD/tmp/test-mib-source-root")
        expected = repository / "EAP-IG" / "src"
        with mock.patch.object(
            Path,
            "is_dir",
            autospec=True,
            side_effect=lambda candidate: candidate == expected / "eap",
        ):
            self.assertEqual(_eap_source_root(repository), expected)

        with mock.patch.object(Path, "is_dir", autospec=True, return_value=False):
            with self.assertRaisesRegex(RuntimeError, "EAP-IG submodule is unavailable"):
                _eap_source_root(repository)

    def test_accelerate_config_requires_dynamic_process_count(self) -> None:
        path = PROJECT_ROOT / "configs" / "accelerate" / "fsdp_server_scheduler.yaml"
        payload = path.read_text(encoding="utf-8")
        self.assertIn("distributed_type: FSDP", payload)
        self.assertNotIn("num_processes:", payload)
        self.assertIn(
            "fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer",
            payload,
        )
        self.assertIn("fsdp_use_orig_params: false", payload)
        source = HANDLER.read_text(encoding="utf-8")
        self.assertIn('"--num_processes",\n                str(gpu_count)', source)

    def test_distributed_launch_uses_actual_count_and_preserves_visibility(self) -> None:
        lineage = self.module.BootstrapLineage(
            amendment_sha256="1" * 64,
            amendment_git_commit="2" * 40,
            reviewed_implementation_commit="3" * 40,
            request_git_commit="4" * 40,
            preflight_git_commits=("5" * 40,) * 4,
        )
        stage = self.module.Stage(
            name="train",
            cli="train",
            argv=("scientific-argument",),
            distributed=True,
        )
        child_environment = {"CUDA_VISIBLE_DEVICES": "uuid-a,uuid-b,uuid-c"}
        with (
            mock.patch.object(
                self.module, "_child_environment", return_value=child_environment
            ),
            mock.patch.object(self.module.subprocess, "run") as run,
            mock.patch.object(self.module, "_log_phase"),
        ):
            self.module._run_stage(
                stage,
                script_path="/proc/123/fd/9",
                job_id="opd-test",
                bootstrap_lineage=lineage,
                code_commit="6" * 40,
                gpu_count=3,
                allocation_sha256="7" * 64,
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--num_processes") + 1], "3")
        self.assertLess(command.index("--num_processes"), command.index("/proc/123/fd/9"))
        self.assertIs(run.call_args.kwargs["env"], child_environment)
        self.assertEqual(
            run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"],
            "uuid-a,uuid-b,uuid-c",
        )

    def test_elastic_protocol_amendment_is_proposed_and_preserves_global_batch(self) -> None:
        amendment = load_protocol_amendment_bytes(
            (PROJECT_ROOT / AMENDMENT_RELATIVE_PATH).read_bytes()
        )
        self.assertEqual(amendment["review"]["status"], "proposed")
        batch = amendment["batch_token_invariants"]
        self.assertEqual(batch["global_logical_batch_size"], 64)
        self.assertEqual(batch["per_rank_samples_by_world_size"]["3"], [22, 21, 21])

    def test_registration_proposal_is_disabled_and_exact(self) -> None:
        path = (
            PROJECT_ROOT
            / "deployments"
            / "qwen3_v2_g0"
            / "registration-proposal-v2.toml"
        )
        proposal = tomllib.loads(path.read_text(encoding="utf-8"))
        self.assertIs(proposal["enabled"], False)
        self.assertEqual(
            proposal["integration"]["allowed_tasks"],
            ["repository_preflight", "qwen3_v2_g0", "qwen3_v2_gpu_preflight"],
        )
        g0 = next(task for task in proposal["tasks"] if task["name"] == "qwen3_v2_g0")
        profile = g0["execution_profiles"][0]
        self.assertEqual(profile["name"], PROFILE_NAME)
        self.assertEqual(profile["gpu_count_policy"], "scheduler")
        self.assertNotIn("gpu_count", profile)
        self.assertEqual(profile["memory_mib"], 196608)
        self.assertEqual(profile["estimated_runtime_seconds"], 86400.0)


if __name__ == "__main__":
    unittest.main()
