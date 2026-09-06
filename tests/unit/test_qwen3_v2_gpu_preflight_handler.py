from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from posttrain_circuits.scheduler_adapter.completion import AttemptCompletionDraft
from posttrain_circuits.scheduler_adapter.gpu_preflight_request import fixed_resolved_config
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import training_contract
from posttrain_circuits.scheduler_adapter.registry import require_handler


ROOT = Path(__file__).resolve().parents[2]
HANDLER = ROOT / "scripts" / "server_scheduler" / "qwen3-v2-gpu-preflight-handler.py"
PREPARE = ROOT / "scripts" / "server_scheduler" / "prepare-qwen3-v2-runtime.py"
PROPOSAL = (
    ROOT
    / "deployments"
    / "qwen3_v2_gpu_preflight"
    / "registration-proposal-v2.toml"
)
CODE_COMMIT = "a" * 40
PLAN_COMMIT = "b" * 40
EXECUTION_COMMIT = "c" * 40


def _load_handler():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("opd_gpu_preflight_handler", HANDLER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load GPU preflight handler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Qwen3V2GpuPreflightHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_handler()

    def test_handler_and_request_use_the_same_exact_scientific_config(self) -> None:
        self.assertEqual(
            self.module._fixed_config(code_commit=CODE_COMMIT),
            fixed_resolved_config(code_commit=CODE_COMMIT),
        )
        prereg = (ROOT / "prereg" / "qwen3_v2.yaml").read_bytes()
        self.assertEqual(
            self.module.PREREGISTRATION_SHA256,
            hashlib.sha256(prereg).hexdigest(),
        )
        self.assertEqual(self.module._training_contract(), training_contract())

    def test_student_gradient_checkpointing_is_explicitly_non_reentrant(self) -> None:
        calls: list[dict[str, bool]] = []

        class FakeModel:
            @staticmethod
            def gradient_checkpointing_enable(
                *, gradient_checkpointing_kwargs: dict[str, bool]
            ) -> None:
                calls.append(gradient_checkpointing_kwargs)

        self.module._enable_student_gradient_checkpointing(FakeModel())

        self.assertEqual(calls, [{"use_reentrant": False}])
        source = HANDLER.read_text(encoding="utf-8")
        self.assertNotIn("model.gradient_checkpointing_enable()", source)
        self.assertIn('gradient_checkpointing_mode="non_reentrant"', source)

    def test_registry_fixes_offline_environment_and_single_implementation(self) -> None:
        handler = require_handler("qwen3_v2_gpu_preflight")
        profile = handler.profile(self.module.PROFILE)
        self.assertEqual(profile.process_count, 0)
        self.assertEqual(profile.cpu_cores_min, 24)
        self.assertEqual(profile.cpu_cores_max, 24)
        self.assertEqual(profile.gpu_count, 0)
        self.assertEqual(profile.gpu_count_policy, "scheduler")
        self.assertEqual(profile.scheduler_gpu_counts, (1, 2, 3, 4))
        self.assertEqual(
            tuple(self.module._threads_per_rank(count) for count in (1, 2, 3, 4)),
            (24, 12, 8, 6),
        )
        self.assertEqual(dict(handler.fixed_environment), self.module.FIXED_ENVIRONMENT)
        self.assertEqual(
            {
                key: self.module.FIXED_ENVIRONMENT[key]
                for key in (
                    "NCCL_DEBUG",
                    "NCCL_DEBUG_SUBSYS",
                    "NCCL_P2P_DISABLE",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                    "TORCH_NCCL_DUMP_ON_TIMEOUT",
                    "TORCH_NCCL_TRACE_BUFFER_SIZE",
                )
            },
            {
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,NET,COLL",
                "NCCL_P2P_DISABLE": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
                "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
            },
        )
        self.assertEqual(handler.deployment.implementation, HANDLER)
        self.assertEqual(handler.deployment.runtime_flags, ("-I",))
        self.assertEqual(handler.output_names, ("gpu_preflight.json",))

    def test_disabled_registration_proposes_scheduler_managed_gpu_profile(self) -> None:
        with PROPOSAL.open("rb") as handle:
            proposal = tomllib.load(handle)
        self.assertIs(proposal["enabled"], False)
        task = next(
            item
            for item in proposal["tasks"]
            if item["name"] == "qwen3_v2_gpu_preflight"
        )
        self.assertEqual(len(task["execution_profiles"]), 1)
        profile = task["execution_profiles"][0]
        self.assertEqual(profile["name"], self.module.PROFILE)
        self.assertEqual(profile["kind"], "gpu")
        self.assertEqual(profile["gpu_count_policy"], "scheduler")
        self.assertNotIn("gpu_count", profile)
        self.assertEqual(profile["cpu_cores_min"], 24)
        self.assertEqual(profile["cpu_cores_max"], 24)
        self.assertEqual(profile["cpu_cores_preferred"], 24)

    def test_completion_draft_round_trips_through_adapter_schema(self) -> None:
        digest = "a" * 64
        inputs = {
            name: chr(ord("a") + index) * 64
            for index, name in enumerate(self.module.EXPECTED_INPUT_NAMES)
        }
        invocation = self.module.Invocation(
            workflow_id=f"{self.module.WORKFLOW_ID_PREFIX}{'a' * 32}",
            plan_sha256=digest,
            unit_id="gpu-preflight",
            run_id="b" * 64,
            job_id="opd-job",
            attempt=1,
            execution_profile=self.module.PROFILE,
            gpu_count=3,
            manifest_sha256="c" * 64,
            allocation_sha256="d" * 64,
            content_handles=tuple(
                self.module.ContentHandle(name, value, -1)
                for name, value in inputs.items()
            ),
            output_descriptor=-1,
        )
        payload = self.module._completion(
            invocation,
            started_at="2026-09-03T00:00:00Z",
            completed_at="2026-09-03T00:01:00Z",
        )
        self.assertEqual(AttemptCompletionDraft.from_payload(payload).to_payload(), payload)

    def test_handler_contains_no_host_gpu_selection_or_legacy_scheduler(self) -> None:
        source = HANDLER.read_text(encoding="utf-8").lower()
        for forbidden in ("nvidia-smi", "sbatch", "srun", "slurm", "screen", "tmux"):
            self.assertNotIn(forbidden, source)
        self.assertNotIn('os.environ["cuda_visible_devices"] =', source)
        self.assertNotIn("os.environ['cuda_visible_devices'] =", source)

    def test_scheduler_gpu_count_drives_threads_and_rejects_other_topologies(self) -> None:
        self.assertEqual(self.module.SUPPORTED_GPU_COUNTS, (1, 2, 3, 4))
        self.assertEqual(
            {
                count: self.module._threads_per_rank(count)
                for count in self.module.SUPPORTED_GPU_COUNTS
            },
            {1: 24, 2: 12, 3: 8, 4: 6},
        )
        for value in ("1", "2", "3", "4", 1, 2, 3, 4):
            with self.subTest(value=value):
                self.assertEqual(self.module._gpu_count(value), int(value))
        for value in (True, False, 0, 5, "0", "05", "5", "", None):
            with self.subTest(value=value), self.assertRaises(
                self.module.PreflightError
            ):
                self.module._gpu_count(value)
        valid_workflow_id = f"{self.module.WORKFLOW_ID_PREFIX}{'a' * 32}"
        self.assertEqual(
            self.module._preflight_workflow_id(valid_workflow_id),
            valid_workflow_id,
        )
        for value in (
            self.module.WORKFLOW_ID_PREFIX,
            f"{self.module.WORKFLOW_ID_PREFIX}2gpu",
            f"{self.module.WORKFLOW_ID_PREFIX}{'a' * 31}",
            f"{self.module.WORKFLOW_ID_PREFIX}{'A' * 32}",
        ):
            with self.subTest(workflow_id=value), self.assertRaises(
                self.module.PreflightError
            ):
                self.module._preflight_workflow_id(value)

    def test_handler_uses_self_contained_reviewed_handoff_lineage(self) -> None:
        accepted_raw = b"accepted-amendment\n"
        with (
            mock.patch.object(
                self.module,
                "_accepted_amendment",
                return_value=(accepted_raw, CODE_COMMIT),
            ),
            mock.patch.object(self.module, "_validate_review_chain") as review,
            mock.patch.object(
                self.module,
                "_linear_commit_steps",
                return_value=(
                    (
                        PLAN_COMMIT,
                        EXECUTION_COMMIT,
                        {str(self.module.HANDOFF_RELATIVE_PATH)},
                    ),
                ),
            ),
        ):
            self.module._validate_plan_execution_lineage(
                plan_commit=PLAN_COMMIT,
                execution_commit=EXECUTION_COMMIT,
            )
        review.assert_called_once_with(
            implementation_commit=CODE_COMMIT,
            candidate_commit=PLAN_COMMIT,
            accepted_raw=accepted_raw,
        )
        source = HANDLER.read_text(encoding="utf-8")
        self.assertNotIn("from posttrain_circuits.artifacts.protocol_amendments import", source)

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

    def test_hash_bound_bootstrap_rejects_source_change_even_if_reverted(self) -> None:
        proposed_raw = (
            ROOT / "prereg" / "amendments" / "qwen3_v2_g0_elastic_v1.yaml"
        ).read_bytes()
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
            + f"  reviewed_implementation_commit: {CODE_COMMIT}\n".encode()
            + b"  reviewer: independent-reviewer\n"
            + b'  reviewed_at_utc: "2026-09-05T07:00:00Z"\n'
            + b"  rationale: reviewed candidate implementation\n"
        )
        self.assertIn(proposed_review, proposed_raw)
        accepted_raw = proposed_raw.replace(proposed_review, accepted_review)
        with tempfile.TemporaryDirectory(dir="/scr/del6500/OPD/tmp") as directory:
            root = Path(directory)
            amendment = root / self.module.AMENDMENT_RELATIVE_PATH
            amendment.parent.mkdir(parents=True)
            amendment.write_bytes(accepted_raw)
            with (
                mock.patch.object(self.module, "SOURCE_ROOT", root),
                mock.patch.object(
                    self.module,
                    "_git",
                    return_value=EXECUTION_COMMIT,
                ),
                mock.patch.object(
                    self.module,
                    "_git_blob",
                    side_effect=lambda commit, _path: (
                        proposed_raw if commit == CODE_COMMIT else accepted_raw
                    ),
                ),
                mock.patch.object(self.module, "_is_ancestor", return_value=True),
            ):
                raw, implementation = self.module._accepted_amendment(
                    execution_commit=EXECUTION_COMMIT
                )
        self.assertEqual(raw, accepted_raw)
        self.assertEqual(implementation, CODE_COMMIT)

        changed_source = "src/posttrain_circuits/artifacts/protocol_amendments.py"
        steps = (
            (
                CODE_COMMIT,
                "d" * 40,
                {
                    str(self.module.AMENDMENT_RELATIVE_PATH),
                    str(self.module.HANDOFF_RELATIVE_PATH),
                },
            ),
            ("d" * 40, "e" * 40, {changed_source}),
            ("e" * 40, PLAN_COMMIT, {changed_source}),
        )
        with (
            mock.patch.object(self.module, "_linear_commit_steps", return_value=steps),
            mock.patch.object(
                self.module,
                "_git_blob",
                side_effect=lambda commit, _path: (
                    proposed_raw if commit == CODE_COMMIT else accepted_raw
                ),
            ),
            self.assertRaisesRegex(self.module.PreflightError, "implementation changes"),
        ):
            self.module._validate_review_chain(
                implementation_commit=CODE_COMMIT,
                candidate_commit=PLAN_COMMIT,
                accepted_raw=accepted_raw,
            )

    def test_hash_bound_bootstrap_rejects_merge_lineage(self) -> None:
        merge_commit = "d" * 40
        other_parent = "e" * 40

        def git_bytes(*arguments: str) -> bytes:
            if arguments[0] == "rev-list":
                return f"{merge_commit}\n".encode("ascii")
            if arguments == ("cat-file", "commit", merge_commit):
                return (
                    f"tree {'f' * 40}\n"
                    f"parent {CODE_COMMIT}\n"
                    f"parent {other_parent}\n\nmerge\n"
                ).encode("ascii")
            raise AssertionError(arguments)

        with (
            mock.patch.object(self.module, "_is_ancestor", return_value=True),
            mock.patch.object(
                self.module,
                "_git_bytes",
                side_effect=git_bytes,
            ),
            self.assertRaisesRegex(self.module.PreflightError, "linear commit chain"),
        ):
            self.module._linear_commit_steps(CODE_COMMIT, merge_commit)

    def test_real_git_lineage_accepts_handoff_and_rejects_source_revert(self) -> None:
        proposed_raw = (
            ROOT / "prereg" / "amendments" / "qwen3_v2_g0_elastic_v1.yaml"
        ).read_bytes()
        proposed_review = (
            b"review:\n"
            b"  status: proposed\n"
            b"  reviewed_implementation_commit: null\n"
            b"  reviewer: null\n"
            b"  reviewed_at_utc: null\n"
            b"  rationale: null\n"
        )
        git_environment = {
            "GIT_AUTHOR_EMAIL": "candidate-d-test@example.invalid",
            "GIT_AUTHOR_NAME": "Candidate D Test",
            "GIT_COMMITTER_EMAIL": "candidate-d-test@example.invalid",
            "GIT_COMMITTER_NAME": "Candidate D Test",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
        }

        with tempfile.TemporaryDirectory(dir="/scr/del6500/OPD/tmp") as directory:
            root = Path(directory)

            def git(*arguments: str) -> str:
                result = subprocess.run(
                    ("/usr/bin/git", "-C", str(root), *arguments),
                    check=True,
                    capture_output=True,
                    text=True,
                    env=git_environment,
                )
                return result.stdout.strip()

            def commit(message: str) -> str:
                git("add", "--all")
                git("commit", "--quiet", "-m", message)
                return git("rev-parse", "HEAD")

            git("init", "--quiet")
            amendment = root / self.module.AMENDMENT_RELATIVE_PATH
            handoff = root / self.module.HANDOFF_RELATIVE_PATH
            validator = (
                root
                / "src"
                / "posttrain_circuits"
                / "artifacts"
                / "protocol_amendments.py"
            )
            amendment.parent.mkdir(parents=True)
            handoff.parent.mkdir(parents=True)
            validator.parent.mkdir(parents=True)
            amendment.write_bytes(proposed_raw)
            handoff.write_text("implementation\n", encoding="utf-8")
            validator.write_text("TRUSTED = True\n", encoding="utf-8")
            implementation = commit("implementation")

            accepted_review = (
                b"review:\n"
                b"  status: accepted\n"
                + f"  reviewed_implementation_commit: {implementation}\n".encode()
                + b"  reviewer: independent-reviewer\n"
                + b'  reviewed_at_utc: "2026-09-05T07:00:00Z"\n'
                + b"  rationale: reviewed candidate implementation\n"
            )
            amendment.write_bytes(proposed_raw.replace(proposed_review, accepted_review))
            handoff.write_text("accepted\n", encoding="utf-8")
            plan_commit = commit("accept implementation")
            handoff.write_text("request recorded\n", encoding="utf-8")
            execution_commit = commit("record request")

            with mock.patch.object(self.module, "SOURCE_ROOT", root):
                self.module._validate_plan_execution_lineage(
                    plan_commit=plan_commit,
                    execution_commit=execution_commit,
                )

            validator.write_text("TRUSTED = False\n", encoding="utf-8")
            commit("change validator")
            validator.write_text("TRUSTED = True\n", encoding="utf-8")
            reverted_commit = commit("restore validator")
            with (
                mock.patch.object(self.module, "SOURCE_ROOT", root),
                self.assertRaisesRegex(self.module.PreflightError, "post-plan"),
            ):
                self.module._validate_plan_execution_lineage(
                    plan_commit=plan_commit,
                    execution_commit=reverted_commit,
                )

    def test_real_git_lineage_rejects_merge_with_range_hidden_parent(self) -> None:
        proposed_raw = (
            ROOT / "prereg" / "amendments" / "qwen3_v2_g0_elastic_v1.yaml"
        ).read_bytes()
        proposed_review = (
            b"review:\n"
            b"  status: proposed\n"
            b"  reviewed_implementation_commit: null\n"
            b"  reviewer: null\n"
            b"  reviewed_at_utc: null\n"
            b"  rationale: null\n"
        )
        git_environment = {
            "GIT_AUTHOR_EMAIL": "candidate-d-test@example.invalid",
            "GIT_AUTHOR_NAME": "Candidate D Test",
            "GIT_COMMITTER_EMAIL": "candidate-d-test@example.invalid",
            "GIT_COMMITTER_NAME": "Candidate D Test",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
        }

        with tempfile.TemporaryDirectory(dir="/scr/del6500/OPD/tmp") as directory:
            root = Path(directory)

            def git(*arguments: str) -> str:
                result = subprocess.run(
                    ("/usr/bin/git", "-C", str(root), *arguments),
                    check=True,
                    capture_output=True,
                    text=True,
                    env=git_environment,
                )
                return result.stdout.strip()

            def commit(message: str) -> str:
                git("add", "--all")
                git("commit", "--quiet", "-m", message)
                return git("rev-parse", "HEAD")

            git("init", "--quiet", "--initial-branch=main")
            amendment = root / self.module.AMENDMENT_RELATIVE_PATH
            handoff = root / self.module.HANDOFF_RELATIVE_PATH
            validator = root / "src" / "reviewed.py"
            amendment.parent.mkdir(parents=True)
            handoff.parent.mkdir(parents=True)
            validator.parent.mkdir(parents=True)
            handoff.write_text("main\n" + "padding\n" * 8 + "side\n", encoding="utf-8")
            commit("base")

            amendment.write_bytes(proposed_raw)
            validator.write_text("TRUSTED = True\n", encoding="utf-8")
            implementation = commit("implementation")
            git("branch", "side", "HEAD^")

            accepted_review = (
                b"review:\n"
                b"  status: accepted\n"
                + f"  reviewed_implementation_commit: {implementation}\n".encode()
                + b"  reviewer: independent-reviewer\n"
                + b'  reviewed_at_utc: "2026-09-05T07:00:00Z"\n'
                + b"  rationale: reviewed candidate implementation\n"
            )
            amendment.write_bytes(proposed_raw.replace(proposed_review, accepted_review))
            plan_commit = commit("accept implementation")
            handoff.write_text(
                "execution\n" + "padding\n" * 8 + "side\n",
                encoding="utf-8",
            )
            commit("record execution")

            git("checkout", "--quiet", "side")
            handoff.write_text(
                "main\n" + "padding\n" * 8 + "side branch\n",
                encoding="utf-8",
            )
            commit("side handoff")
            git("checkout", "--quiet", "main")
            git("merge", "--quiet", "--no-ff", "side", "-m", "merge hidden parent")
            merge_commit = git("rev-parse", "HEAD")

            with (
                mock.patch.object(self.module, "SOURCE_ROOT", root),
                self.assertRaisesRegex(self.module.PreflightError, "linear commit chain"),
            ):
                self.module._validate_plan_execution_lineage(
                    plan_commit=plan_commit,
                    execution_commit=merge_commit,
                )

    def test_handler_rejects_invalid_or_unreviewed_plan_lineage(self) -> None:
        with self.assertRaisesRegex(self.module.PreflightError, "not a Git identity"):
            self.module._validate_plan_execution_lineage(
                plan_commit="not-a-commit",
                execution_commit=EXECUTION_COMMIT,
            )
        with (
            mock.patch.object(
                self.module,
                "_accepted_amendment",
                return_value=(b"accepted\n", CODE_COMMIT),
            ),
            mock.patch.object(self.module, "_validate_review_chain"),
            mock.patch.object(
                self.module,
                "_linear_commit_steps",
                return_value=(
                    (
                        PLAN_COMMIT,
                        EXECUTION_COMMIT,
                        {"src/posttrain_circuits/cli/train.py"},
                    ),
                ),
            ),
            self.assertRaisesRegex(self.module.PreflightError, "post-plan"),
        ):
            self.module._validate_plan_execution_lineage(
                plan_commit=PLAN_COMMIT,
                execution_commit=EXECUTION_COMMIT,
            )

    def test_handler_disables_source_bytecode_cache(self) -> None:
        self.assertTrue(self.module.sys.dont_write_bytecode)
        self.assertEqual(self.module.sys.pycache_prefix, self.module.BYTECODE_CACHE_PREFIX)
        prefix = Path(self.module.BYTECODE_CACHE_PREFIX)
        self.assertRegex(prefix.name, r"^\.opd-gpu-preflight-pycache-[a-z0-9_]+$")
        self.assertTrue(prefix.is_dir())
        self.assertEqual(prefix.stat().st_mode & 0o777, 0o700)
        self.assertEqual(list(prefix.iterdir()), [])

    def test_handler_separates_gloo_control_from_nccl_data_plane(self) -> None:
        source = HANDLER.read_text(encoding="utf-8")
        self.assertIn('dist.init_process_group(\n        "gloo"', source)
        self.assertIn('backend="nccl"', source)
        self.assertIn("process_group=runtime.data_group", source)
        self.assertIn("use_orig_params=False", source)
        self.assertNotIn("use_orig_params=True", source)
        self.assertIn("async_op=True", source)
        self.assertIn(
            "timeout=timedelta(seconds=NCCL_PROBE_TIMEOUT_SECONDS)", source
        )
        self.assertIn("group=runtime.control_group", source)
        self.assertIn("group=runtime.data_group", source)
        self.assertNotIn("dist.barrier()", source)
        self.assertIn('"--run-path",', source)

    def test_exact_optimizer_window_schedules_cover_64_global_slots(self) -> None:
        expected = {
            1: ((4,) * 16,),
            2: ((4,) * 8, (4,) * 8),
            3: ((4, 4, 4, 4, 4, 2), (4, 4, 4, 4, 4, 1), (4, 4, 4, 4, 4, 1)),
            4: ((4,) * 4, (4,) * 4, (4,) * 4, (4,) * 4),
        }
        for gpu_count, rank_schedules in expected.items():
            observed_slots: list[int] = []
            for rank, rank_schedule in enumerate(rank_schedules):
                self.assertEqual(
                    self.module._microbatch_sizes(gpu_count, rank), rank_schedule
                )
                for microbatch_index in range(len(rank_schedule)):
                    observed_slots.extend(
                        self.module._global_slots_for_microbatch(
                            gpu_count, rank, microbatch_index
                        )
                    )
            self.assertEqual(sorted(observed_slots), list(range(64)))

    def test_training_probe_matches_production_shape(self) -> None:
        contract = self.module._training_contract()
        self.assertEqual(contract["global_logical_batch_size"], 64)
        self.assertEqual(contract["max_per_rank_microbatch_size"], 4)
        self.assertEqual(contract["max_model_input_length"], 1536)
        self.assertEqual(contract["gpu_memory_budget_mib"], 81920)
        self.assertEqual(
            contract["fsdp_requested_sharding_strategy"], "FULL_SHARD"
        )
        self.assertEqual(
            contract["fsdp_effective_sharding_strategy_by_world_size"],
            {
                "1": "NO_SHARD",
                "2": "FULL_SHARD",
                "3": "FULL_SHARD",
                "4": "FULL_SHARD",
            },
        )
        self.assertEqual(contract["fsdp_state_dict_type"], "FULL_STATE_DICT")
        self.assertEqual(
            contract["fsdp_auto_wrap_policy"], "TRANSFORMER_BASED_WRAP"
        )
        self.assertEqual(contract["fsdp_transformer_layer"], "Qwen3DecoderLayer")

    def test_effective_fsdp_strategy_reads_every_wrapper_and_handles_w1(self) -> None:
        class Qwen3DecoderLayer:
            pass

        class FakeCausalLM:
            pass

        class FakeFSDP:
            def __init__(self, strategy: str, module: object) -> None:
                self.sharding_strategy = SimpleNamespace(name=strategy)
                self.module = module
                self._module_tree: list[object] | None = None

            def modules(self):  # type: ignore[no-untyped-def]
                return iter(self._module_tree or (self, self.module))

        def fake_model(
            strategy: str,
            *,
            block_count: int = 2,
            wrapped_block_count: int | None = None,
            extra_wrappers: int = 0,
        ) -> FakeFSDP:
            if wrapped_block_count is None:
                wrapped_block_count = block_count
            blocks = [Qwen3DecoderLayer() for _ in range(block_count)]
            root_module = FakeCausalLM()
            root = FakeFSDP(strategy, root_module)
            child_wrappers = [
                FakeFSDP(strategy, block)
                for block in blocks[:wrapped_block_count]
            ]
            unrelated = [
                FakeFSDP(strategy, FakeCausalLM()) for _ in range(extra_wrappers)
            ]
            root._module_tree = [
                root,
                root_module,
                *child_wrappers,
                *unrelated,
                *blocks,
            ]
            return root

        expected = {
            1: "NO_SHARD",
            2: "FULL_SHARD",
            3: "FULL_SHARD",
            4: "FULL_SHARD",
        }
        for world_size, strategy in expected.items():
            with self.subTest(world_size=world_size):
                contract = self.module._validate_effective_fsdp_strategy(
                    fake_model(strategy),
                    fsdp_type=FakeFSDP,
                    world_size=world_size,
                )
                self.assertEqual(
                    contract,
                    {
                        "requested_fsdp_sharding_strategy": "FULL_SHARD",
                        "effective_fsdp_sharding_strategy": strategy,
                        "fsdp_wrapper_count": 3,
                    },
                )
                for wrapped_blocks, label in ((0, "root-only"), (1, "partial")):
                    with self.subTest(world_size=world_size, topology=label):
                        with self.assertRaisesRegex(
                            self.module.PreflightError,
                            "directly wrap every",
                        ):
                            self.module._validate_effective_fsdp_strategy(
                                fake_model(
                                    strategy,
                                    wrapped_block_count=wrapped_blocks,
                                ),
                                fsdp_type=FakeFSDP,
                                world_size=world_size,
                            )
        with self.assertRaisesRegex(self.module.PreflightError, "wrapper count differs"):
            self.module._validate_effective_fsdp_strategy(
                fake_model("FULL_SHARD", extra_wrappers=1),
                fsdp_type=FakeFSDP,
                world_size=3,
            )
        with self.assertRaisesRegex(self.module.PreflightError, "directly wrap every"):
            self.module._validate_effective_fsdp_strategy(
                fake_model(
                    "FULL_SHARD",
                    wrapped_block_count=1,
                    extra_wrappers=1,
                ),
                fsdp_type=FakeFSDP,
                world_size=3,
            )
        mixed = fake_model("FULL_SHARD")
        child_wrapper = next(
            module
            for module in mixed.modules()
            if isinstance(module, FakeFSDP) and module is not mixed
        )
        child_wrapper.sharding_strategy = SimpleNamespace(name="NO_SHARD")
        with self.assertRaisesRegex(self.module.PreflightError, "effective FSDP"):
            self.module._validate_effective_fsdp_strategy(
                mixed,
                fsdp_type=FakeFSDP,
                world_size=2,
            )
        with self.assertRaisesRegex(self.module.PreflightError, "no actual FSDP"):
            self.module._validate_effective_fsdp_strategy(
                SimpleNamespace(modules=lambda: iter(())),
                fsdp_type=FakeFSDP,
                world_size=1,
            )
        self.assertEqual(
            self.module._full_state_dict_options(1),
            {"offload_to_cpu": False, "rank0_only": False},
        )
        self.assertEqual(
            self.module._full_state_dict_options(2),
            {"offload_to_cpu": True, "rank0_only": True},
        )
        self.assertEqual(
            self.module._full_state_dict_options(2, loading=True),
            {"offload_to_cpu": True, "rank0_only": False},
        )

    def test_gpu_memory_envelope_rejects_invalid_or_overbudget_peaks(self) -> None:
        budget = self.module.GPU_MEMORY_BUDGET_MIB * 1024**2
        self.module._validate_gpu_memory_peaks(budget - 1, budget)
        for allocated, reserved in (
            (0, 1),
            (2, 1),
            (budget + 1, budget + 1),
            (budget, budget + 1),
            (True, 1),
        ):
            with self.subTest(allocated=allocated, reserved=reserved), self.assertRaisesRegex(
                self.module.PreflightError, "memory envelope"
            ):
                self.module._validate_gpu_memory_peaks(allocated, reserved)

    def test_handler_logs_each_student_training_boundary(self) -> None:
        source = HANDLER.read_text(encoding="utf-8")
        for phase in (
            "student_fsdp_ready",
            "student_forward_started",
            "student_forward_completed",
            "student_backward_started",
            "student_backward_completed",
            "optimizer_step_started",
            "optimizer_step_completed",
        ):
            with self.subTest(phase=phase):
                self.assertIn(f'"{phase}"', source)

    def test_nccl_version_normalization(self) -> None:
        class TupleNccl:
            @staticmethod
            def version() -> tuple[int, int, int]:
                return (2, 27, 3)

        class IntegerNccl:
            @staticmethod
            def version() -> int:
                return 22703

        tuple_torch = type("Torch", (), {"cuda": type("Cuda", (), {"nccl": TupleNccl})})
        integer_torch = type(
            "Torch", (), {"cuda": type("Cuda", (), {"nccl": IntegerNccl})}
        )
        self.assertEqual(self.module._nccl_runtime_version(tuple_torch), "2.27.3")
        self.assertEqual(self.module._nccl_runtime_version(integer_torch), "2.27.3")

    def test_distributed_initialization_routes_probe_to_bounded_nccl_group(self) -> None:
        class FakeTensor:
            def __init__(self, value: float) -> None:
                self.value = value

            def item(self) -> float:
                return self.value

        class FakeWork:
            def wait(self, *, timeout: object) -> bool:
                calls.append(("wait", timeout))
                return True

        for gpu_count in self.module.SUPPORTED_GPU_COUNTS:
            with self.subTest(gpu_count=gpu_count):
                calls: list[tuple[object, ...]] = []
                expected_sum = self.module._nccl_expected_sum(gpu_count)
                torch = ModuleType("torch")
                dist = ModuleType("torch.distributed")
                torch.__version__ = "2.8.0+cu128"
                torch.version = SimpleNamespace(cuda="12.8")
                torch.device = lambda kind, index: (kind, index)
                torch.tensor = lambda value, device: FakeTensor(value)
                torch.cuda = SimpleNamespace(
                    device_count=lambda: gpu_count,
                    get_device_properties=lambda _device: SimpleNamespace(
                        major=12,
                        minor=0,
                        name=self.module.GPU_MODEL,
                        total_memory=98_000 * 1024**2,
                    ),
                    nccl=SimpleNamespace(version=lambda: (2, 27, 3)),
                    set_device=lambda index: calls.append(("set_device", index)),
                )
                control_group = object()
                data_group = object()
                dist.group = SimpleNamespace(WORLD=control_group)
                dist.init_process_group = lambda backend, timeout: calls.append(
                    ("init_process_group", backend, timeout)
                )

                def new_group(
                    *, ranks: list[int], backend: str, timeout: object
                ) -> object:
                    calls.append(("new_group", tuple(ranks), backend, timeout))
                    return data_group

                def all_reduce(
                    tensor: FakeTensor, *, group: object, async_op: bool
                ) -> FakeWork:
                    tensor.value = expected_sum
                    calls.append(("all_reduce", group, async_op))
                    return FakeWork()

                dist.new_group = new_group
                dist.all_reduce = all_reduce
                torch.distributed = dist
                environment = {
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        f"GPU-{rank}" for rank in range(gpu_count)
                    ),
                    "LOCAL_RANK": "0",
                    "RANK": "0",
                    "WORLD_SIZE": str(gpu_count),
                }
                with mock.patch.dict(
                    sys.modules, {"torch": torch, "torch.distributed": dist}
                ), mock.patch.dict(
                    "os.environ", environment, clear=True
                ), mock.patch.object(
                    self.module, "_cuda_pci_bus_id", return_value="0000:48:00.0"
                ):
                    runtime = self.module._initialize_distributed(gpu_count)

                self.assertIs(runtime.control_group, control_group)
                self.assertIs(runtime.data_group, data_group)
                self.assertEqual(
                    runtime.nccl_diagnostic["observed_sum"], expected_sum
                )
                self.assertEqual(calls[1][0:2], ("init_process_group", "gloo"))
                self.assertEqual(
                    calls[2][0:3],
                    ("new_group", tuple(range(gpu_count)), "nccl"),
                )
                self.assertEqual(calls[3], ("all_reduce", data_group, True))
                self.assertEqual(
                    calls[4][1].total_seconds(),
                    self.module.NCCL_PROBE_TIMEOUT_SECONDS,
                )

    def test_spawn_main_path_reopens_the_supervisor_held_script(self) -> None:
        expected = f"/proc/{os.getpid()}/fd/7"
        main_module = sys.modules["__main__"]
        with mock.patch.object(
            self.module, "_script_parent_path", return_value=expected
        ), mock.patch.object(main_module, "__file__", "/proc/self/fd/7", create=True):
            self.assertEqual(self.module._spawn_safe_script_path(), expected)
            self.assertEqual(main_module.__file__, expected)

    def test_runtime_preparation_requires_explicit_execute(self) -> None:
        source = PREPARE.read_text(encoding="utf-8")
        self.assertIn('"PyYAML==6.0.3"', source)
        self.assertIn("version('PyYAML') == '6.0.3'", source)
        result = subprocess.run(
            (sys.executable, str(PREPARE)),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--execute", result.stderr)


if __name__ == "__main__":
    unittest.main()
