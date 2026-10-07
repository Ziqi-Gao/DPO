"""CPU-only checks for namespace construction, immutable resume inputs and comparison replay."""

import ast
import copy
import importlib.util
import json
import math
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/sdsc_resume.py"
SPEC = importlib.util.spec_from_file_location("sdsc_resume_tested", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(worker.canonical(value) + b"\n")
    return worker.file_identity(path)["sha256"]


def real_comparison_function():
    """Load the original pure comparator without importing Torch or running models."""
    source = ROOT / "src/posttrain_circuits/cli/compare_distributed_resume.py"
    tree = ast.parse(source.read_text())
    names = {
        "_logical_workspace_path",
        "_require_fixed_tolerance",
        "_metric_step",
        "_objective_loss",
        "_last_metric",
        "_build_comparison_payload",
    }
    constants = {
        "FIXED_DISTRIBUTED_RESUME_TOLERANCE",
        "_SUPPORTED_WORLD_SIZES",
        "_CORE_RUNTIME_STATE_NAMES",
        "_RESUME_SOURCE_FIELDS",
    }
    selected = [
        n
        for n in tree.body
        if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
        or (isinstance(n, ast.FunctionDef) and n.name in names)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
        )
    ]
    namespace = {
        "Path": Path,
        "math": math,
        "json": json,
        "Mapping": Mapping,
        "sha256_value": lambda value: worker.sha(worker.canonical(value)),
        "REQUESTED_FSDP_SHARDING_STRATEGY": "FULL_SHARD",
        "effective_fsdp_sharding_strategy": lambda world: "FULL_SHARD" if world > 1 else "NO_SHARD",
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.workspace = self.work / "workspace/qwen3-v2"
        self.workspace.mkdir(parents=True)
        self.source = Path("/scratch/zgao12/job_100/original/outputs/qwen3-v2")
        self.tools = self.work / "code/tools"
        self.tools.mkdir(parents=True)
        (self.tools / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
        self.python = self.work / "runtime/bin/python3.12"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"CPU fixture executable")
        self.python.chmod(0o755)
        self.runtime = self.work / "runtime/bin/singularity"
        self.runtime.write_bytes(b"CPU fixture container runtime")
        self.runtime.chmod(0o755)
        self.image = self.work / "fixed.sif"
        self.image.write_bytes(b"CPU fixture image")
        self.namespace = Path("/opd-pipeline/fixture")
        binds = {
            "science": {"host": str(self.work / "science"), "guest": str(self.namespace / "science")},
            "tools": {"host": str(self.tools), "guest": str(self.namespace / "tools")},
            "hf_home": {"host": str(self.work / "huggingface"), "guest": str(self.namespace / "huggingface")},
        }
        for pair in binds.values():
            Path(pair["host"]).mkdir(exist_ok=True)
        self.report = dict(
            task="qwen3-v2-g0-calibration",
            passed=True,
            exit_code=0,
            training_executed=True,
            job_id="100",
            run_id="reference",
            code_sha256="a" * 64,
            science_git_head=worker.SCIENCE_HEAD,
            bundle_sha256="b" * 64,
            provenance_manifest_sha256="c" * 64,
            training_artifact_root=str(self.source / "canonical_sft"),
            plan={"training_overrides": ["g0=qwen3_v2_eap_separation", "seed=42"]},
            **worker.FALSE_CLAIMS,
        )
        self.report_path = self.work / "proofs/reference.json"
        report_hash = write(self.report_path, self.report)
        self.records = [{"path": "g0-calibration.json", **worker.file_identity(self.report_path)}]
        for name in (
            "initial_checkpoint.pt",
            "dataset/manifest.json",
            "dataset/train/examples.jsonl",
            "teacher_demos/manifest.json",
            "canonical_sft/manifest.json",
            "canonical_sft/resolved_config.yaml",
            worker.STEP20,
            worker.STEP20.removesuffix(".pt") + ".accelerate/state.bin",
        ):
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"immutable CPU fixture")
            self.records.append({"path": "artifacts/" + name, **worker.file_identity(path)})
        self.receipt = {
            **self.report,
            "persisted": True,
            "persistent_read_back_verified": True,
            "files": self.records,
        }
        receipt_path = self.work / "proofs/receipt.json"
        self.request = dict(
            schema=worker.SCHEMA,
            target={"run_id": "resume", "code_sha256": "d" * 64, "job_id": "200"},
            work_dir=str(self.work),
            workspace=str(self.workspace),
            science_root=binds["science"]["host"],
            hf_home=binds["hf_home"]["host"],
            python=str(self.python),
            python_sha256=worker.file_identity(self.python)["sha256"],
            reference_report=str(self.report_path),
            reference_report_sha256=report_hash,
            publication_receipt=str(receipt_path),
            publication_receipt_sha256=write(receipt_path, self.receipt),
            expected_science_head=worker.SCIENCE_HEAD,
            expected_protocol_sha256="e" * 64,
            container={
                "runtime": str(self.runtime),
                "image": str(self.image),
                "size": self.image.stat().st_size,
                "mtime_ns": self.image.stat().st_mtime_ns,
            },
            namespace_root=str(self.namespace),
            readonly_binds=binds,
        )

    def validate(self):
        with patch.object(worker, "__file__", str(self.tools / SCRIPT.name)):
            return worker.validate_request(self.request)

    def test_request_and_exact_inventory_preserve_reference_bytes(self):
        self.validate()
        before = {
            str(p.relative_to(self.workspace)): p.read_bytes()
            for p in self.workspace.rglob("*")
            if p.is_file()
        }
        report, source, inventory = worker.input_inventory(self.request)
        self.assertEqual(report, self.report)
        self.assertEqual(source, self.source)
        self.assertEqual(set(inventory), set(before))
        self.assertEqual(
            before,
            {
                str(p.relative_to(self.workspace)): p.read_bytes()
                for p in self.workspace.rglob("*")
                if p.is_file()
            },
        )

    def test_corrupt_missing_unlisted_and_symlink_inputs_refused(self):
        path = self.workspace / worker.STEP20
        original = path.read_bytes()
        path.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "input SHA"):
            worker.input_inventory(self.request)
        path.write_bytes(original)
        extra = self.workspace / "canonical_sft/extra.pt"
        extra.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            worker.input_inventory(self.request)
        extra.unlink()
        path.unlink()
        path.symlink_to(self.report_path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.input_inventory(self.request)

    def test_changed_image_and_runtime_refused_before_execution(self):
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image changed"):
            self.validate()
        self.request["container"]["size"] = self.image.stat().st_size
        self.request["container"]["mtime_ns"] = self.image.stat().st_mtime_ns
        self.python.write_bytes(b"replaced interpreter")
        with self.assertRaisesRegex(ValueError, "Python executable changed"):
            self.validate()

    def test_bind_destinations_and_path_characters_are_fixed(self):
        self.request["readonly_binds"]["tools"]["guest"] = "/etc"
        with self.assertRaisesRegex(ValueError, "guest destination"):
            self.validate()
        for value in ("/tmp/a:b", "/tmp/a,b", "/tmp/a\nb", "/tmp/../etc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.absolute(value)

    def test_namespace_uses_host_python_and_exact_old_checkpoint_locations(self):
        command, binds = worker.build_namespace_command(
            self.request, self.source, self.work / "plan.json", "a" * 64
        )
        self.assertIn("--nv", command)
        self.assertIn("--cleanenv", command)
        self.assertIn("--no-home", command)
        self.assertIn(str(self.python), command)
        self.assertIn({"host": str(self.workspace), "guest": str(self.source), "access": "rw"}, binds)
        self.assertIn(
            {
                "host": str(self.workspace / "canonical_sft"),
                "guest": str(self.source / "canonical_sft"),
                "access": "ro",
            },
            binds,
        )
        train = worker.training_commands(self.request, self.report, self.source)
        self.assertEqual(len(train), 2)
        for name, argv in zip(("resume-a", "resume-b"), train, strict=True):
            self.assertEqual(argv[argv.index("--resume") + 1], str(self.source / worker.STEP20))
            self.assertEqual(argv[argv.index("--output") + 1], str(self.source / name))
            self.assertEqual(argv[argv.index("--num_processes") + 1], "2")
        self.assertNotIn("calibration/checkpoints", " ".join(train[0]))

    def test_comma_cuda_visibility_is_preserved_without_env_grammar_or_ambient_injection(self):
        original = dict(
            SLURM_JOB_ID="200",
            SLURM_CPUS_PER_TASK="24",
            SLURM_MEM_PER_NODE="196608",
            CUDA_VISIBLE_DEVICES="GPU-a,GPU-b",
            PYTHONPATH="/untrusted",
            LD_PRELOAD="/untrusted.so",
            SINGULARITY_BIND="/bad:/etc",
            OPENBLAS_NUM_THREADS="64",
            NUMEXPR_NUM_THREADS="64",
        )
        environment = worker.container_environment(self.request, original)
        self.assertEqual(environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertNotIn("SINGULARITY_BIND", environment)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(environment["SINGULARITYENV_" + name], "12")
        original["SLURM_JOB_ID"] = "201"
        with self.assertRaisesRegex(ValueError, "job differs"):
            worker.container_environment(self.request, original)

    def test_real_child_receives_fixed_numeric_threads_and_preserves_cuda_assignment(self):
        original = {
            "SLURM_JOB_ID": "200",
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEM_PER_NODE": "196608",
            "OPENBLAS_NUM_THREADS": "64",
            "NUMEXPR_NUM_THREADS": "64",
            "PYTHONPATH": "/untrusted",
        }
        environment = worker.training_environment(
            self.request, self.work / "science", self.work / "huggingface", "GPU-a,GPU-b", original
        )
        names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        code = "import json,os,sys; print(json.dumps({k:os.environ[k] for k in sys.argv[1:]}))"
        log = self.work / "numeric-environment.log"
        worker.run_foreground(
            [worker.sys.executable, "-I", "-B", "-c", code, *names, "CUDA_VISIBLE_DEVICES"],
            cwd=self.work,
            environment=environment,
            log=log,
        )
        observed = json.loads(log.read_text())
        self.assertEqual(observed, {**dict.fromkeys(names, "12"), "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"})
        self.assertEqual(environment["PYTHONPATH"], str(self.work / "science/src"))

    def test_published_plan_hash_includes_actual_newline(self):
        path = self.work / "plan.json"
        plan = {"test": "exact bytes"}
        worker.publish(path, plan)
        self.assertEqual(worker.document(path, worker.sha(worker.canonical(plan) + b"\n")), plan)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            worker.document(path, worker.sha(worker.canonical(plan)))

    def comparison_fixture(self):
        original = real_comparison_function()
        source_hash = worker.file_identity(self.workspace / worker.STEP20)["sha256"]
        source_binding = {name: None for name in original["_RESUME_SOURCE_FIELDS"]}
        source_binding.update(
            checkpoint={
                "logical_path": worker.STEP20,
                "sha256": source_hash,
                "format": "accelerate_fsdp_full_export_v1",
            },
            token_budget={"stop_reason": None},
            world_size=2,
        )
        source_binding["sha256"] = worker.sha(
            worker.canonical({k: v for k, v in source_binding.items() if k != "sha256"})
        )
        runs = {}
        for name in ("canonical_sft", "resume-a", "resume-b"):
            root = self.workspace / name
            metrics = root / "metrics.jsonl"
            write(metrics, {"step": 80, "canonical_sft_loss": 0.25})
            payload = dict(
                world_size=2,
                global_step=80,
                format="accelerate_fsdp_full_export_v1",
                optimizer_state_key_type="parameter_id",
                policy_version=0,
                online_rollout_round=0,
                token_budget={"consumed": 1999999},
                resume_ancestry=[] if name == "canonical_sft" else ["sha256:" + source_hash],
                update_norm_baseline_checkpoint_sha256=source_hash,
            )
            binding = {
                "runtime_state_hashes": {key: "a" * 64 for key in original["_CORE_RUNTIME_STATE_NAMES"]},
                "fsdp_sharding_contract": {
                    "requested_fsdp_sharding_strategy": "FULL_SHARD",
                    "effective_fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_wrapper_count": 29,
                },
            }
            binding["sha256"] = worker.sha(worker.canonical(binding))
            runs[name] = SimpleNamespace(
                root=root,
                metrics_path=metrics,
                checkpoint_payload=payload,
                manifest={"training_stop_reason": "token_budget_exhausted"},
                content_binding=Mock(return_value=binding),
            )
        validator = Mock(side_effect=lambda path, **kwargs: runs[path.name])
        api = SimpleNamespace(
            validate_factorial_run_artifacts=validator,
            _validate_resume_source_artifacts=Mock(return_value=source_binding),
            _build_comparison_payload=original["_build_comparison_payload"],
            FIXED_DISTRIBUTED_RESUME_TOLERANCE=1e-6,
        )
        return api, runs

    def test_comparison_reuses_real_exact_comparator_and_strict_validator_contract(self):
        api, runs = self.comparison_fixture()
        config = {"output_root": str(self.source), "trainer": {"max_steps": 120}}
        formal = {"code_commit": worker.SCIENCE_HEAD}
        with patch.object(worker.importlib, "import_module", return_value=api):
            comparison = worker.compare_runs(self.workspace, config, formal)
            self.assertTrue(comparison["passed"])
            self.assertTrue(all(comparison["checks"].values()))
            calls = api.validate_factorial_run_artifacts.call_args_list
            self.assertEqual([call.args[0].name for call in calls], ["canonical_sft", "resume-a", "resume-b"])
            self.assertEqual(
                calls[1].kwargs["expected_update_norm_baseline_checkpoint_path"],
                self.workspace / worker.STEP20,
            )
            runs["resume-b"].content_binding.return_value["runtime_state_hashes"]["optimizer"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "exact resume comparison failed"):
                worker.compare_runs(self.workspace, config, formal)

    def test_terminal_step20_stops_before_replica_validation(self):
        api, _ = self.comparison_fixture()
        api._validate_resume_source_artifacts.return_value["token_budget"]["stop_reason"] = "budget"
        with (
            patch.object(worker.importlib, "import_module", return_value=api),
            self.assertRaisesRegex(ValueError, "terminal"),
        ):
            worker.compare_runs(self.workspace, {}, {"code_commit": worker.SCIENCE_HEAD})
        self.assertEqual(api.validate_factorial_run_artifacts.call_count, 1)

    def test_report_replay_rejects_rehashed_comparison_forgery_and_world_change(self):
        api, _ = self.comparison_fixture()
        config = {"output_root": str(self.source), "trainer": {"max_steps": 120}}
        formal = {"code_commit": worker.SCIENCE_HEAD}
        with patch.object(worker.importlib, "import_module", return_value=api):
            comparison = worker.compare_runs(self.workspace, config, formal)
            report = dict(
                schema=worker.REPORT_SCHEMA,
                world_size=2,
                passed=True,
                exit_code=0,
                source_workspace=str(self.source),
                runtime_config_sha256=worker.sha(worker.canonical(config)),
                comparison=comparison,
                **worker.FALSE_CLAIMS,
            )
            report["sha256"] = worker.sha(worker.canonical(report))
            path = self.workspace / "distributed_resume.json"
            write(path, report)
            self.assertEqual(
                worker.validate_resume_report(
                    path, config=config, expected_world_size=2, expected_formal_binding=formal
                ),
                comparison,
            )
            with self.assertRaisesRegex(ValueError, "world size"):
                worker.validate_resume_report(
                    path, config=config, expected_world_size=4, expected_formal_binding=formal
                )
            changed = copy.deepcopy(report)
            changed["comparison"]["global_step"] = 120
            changed["sha256"] = worker.sha(
                worker.canonical({k: v for k, v in changed.items() if k != "sha256"})
            )
            write(path, changed)
            with self.assertRaisesRegex(ValueError, "revalidated artifacts"):
                worker.validate_resume_report(
                    path, config=config, expected_world_size=2, expected_formal_binding=formal
                )

            for name, value in (("passed", 1), ("global_step", float(comparison["global_step"]))):
                changed = copy.deepcopy(report)
                changed["comparison"][name] = value
                changed["sha256"] = worker.sha(
                    worker.canonical({k: v for k, v in changed.items() if k != "sha256"})
                )
                write(path, changed)
                with (
                    self.subTest(type_confusion=name),
                    self.assertRaisesRegex(ValueError, "revalidated artifacts"),
                ):
                    worker.validate_resume_report(
                        path, config=config, expected_world_size=2, expected_formal_binding=formal
                    )
            for name, value in (("world_size", 2.0), ("exit_code", False)):
                changed = copy.deepcopy(report)
                changed[name] = value
                changed["sha256"] = worker.sha(
                    worker.canonical({k: v for k, v in changed.items() if k != "sha256"})
                )
                write(path, changed)
                with self.subTest(type_confusion=name), self.assertRaises(ValueError):
                    worker.validate_resume_report(
                        path, config=config, expected_world_size=2, expected_formal_binding=formal
                    )

    def test_foreground_child_inherits_batch_group_and_failure_is_not_success(self):
        process = Mock()
        process.wait.return_value = 2
        process.poll.return_value = 2
        with (
            patch.object(worker.subprocess, "Popen", return_value=process) as launch,
            self.assertRaisesRegex(ValueError, "failed"),
        ):
            worker.run_foreground(["fixture"], cwd=self.work, environment={}, log=self.work / "failure.log")
        self.assertFalse(launch.call_args.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
