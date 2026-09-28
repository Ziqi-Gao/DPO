"""Bounded teacher-adaptation preflight transport/publication; no SSH or GPU."""

import ast
import base64
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
TASK = "qwen3-v2-teacher-adapt"
REPORT = "teacher-adapt.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


client_fixture = module("adapt_client_fixture", Path(__file__).with_name("test_cli.py"))
remote_fixture = module("adapt_remote_fixture", Path(__file__).with_name("test_remote.py"))
cli, remote = client_fixture.cli, remote_fixture.remote


def dataset_fixture():
    return {
        "teacher_fit": {"count": 256, "examples_sha256": "d" * 64},
        "teacher_dev": {"count": 32, "examples_sha256": "e" * 64},
    }


def checkpoint_fixture(output, identity):
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in (
        "merged/model.safetensors",
        "merged/config.json",
        "merged/tokenizer.json",
        "merged/tokenizer_config.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    ):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (name + " fixture bytes").encode()
        path.write_bytes(raw)
        files.append(dict(path=name, size=len(raw), sha256=remote.digest(raw)))
    metrics_raw = b"".join(
        remote.canonical(dict(step=step, samples=32 * step, tokens=320 * step)) + b"\n"
        for step in range(1, 9)
    )
    (output / "train-metrics.jsonl").write_bytes(metrics_raw)
    metrics_record = dict(
        path="artifacts/train-metrics.jsonl", size=len(metrics_raw), sha256=remote.digest(metrics_raw)
    )
    payload = dict(
        artifact_kind="adapted_dense_teacher",
        base_revision="b968826d9c46dd6066d109eabc6255188de91218",
        adaptation_plan_sha256=remote.digest(remote.canonical(remote.teacher_adapt_preflight_plan())),
        formal_teacher_accepted=False,
        files=files,
        training_provenance=dict(
            run_id=identity.get("run_id"),
            job_id=identity.get("job_id"),
            code_sha256=identity.get("code_sha256"),
            dataset_manifest_sha256=remote.digest(remote.canonical(dataset_fixture())),
            train_metrics_sha256=metrics_record["sha256"],
            optimizer_steps=8,
            consumed_tokens=2560,
        ),
    )
    raw = remote.canonical(dict(payload, sha256=remote.digest(remote.canonical(payload)))) + b"\n"
    (output / "checkpoint-manifest.json").write_bytes(raw)
    return (
        remote.digest(raw),
        [dict(record, path="artifacts/" + record["path"]) for record in files]
        + [
            dict(path="artifacts/checkpoint-manifest.json", size=len(raw), sha256=remote.digest(raw)),
            metrics_record,
        ],
        remote.digest(remote.canonical(payload)),
    )


def report_fixture(identity, checkpoint_hash, adapted_sha):
    plan = remote.teacher_adapt_preflight_plan()
    return dict(
        identity,
        passed=True,
        exit_code=0,
        exploratory=True,
        accepted_science=False,
        full_teacher_ready=False,
        g0_passed=False,
        execution_class_certified=False,
        resumable=False,
        artifact_kind="teacher_adaptation_preflight",
        adaptation_preflight=True,
        readiness_artifact_produced=False,
        student_training_started=False,
        training_started=True,
        checks={
            name: True
            for name in (
                "isolated_data",
                "finite_loss_and_gradients",
                "eight_real_optimizer_steps",
                "nonzero_adapter_update",
                "frozen_base_before_merge",
                "complete_module_coverage",
                "dense_export_reload",
                "merged_dev_evaluated",
                "cgroup_headroom",
            )
        },
        adaptation_plan=plan,
        adaptation_plan_sha256=remote.digest(remote.canonical(plan)),
        optimizer_steps=8,
        consumed_tokens=2560,
        adaptation_dataset=dataset_fixture(),
        checkpoint_manifest_sha256=checkpoint_hash,
        adapted_teacher_sha256=adapted_sha,
    )


class AdaptClientTests(unittest.TestCase):
    def setUp(self):
        self.fixture = client_fixture.ClientTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.record()

    def arguments(self, dry_run=False):
        args = self.fixture.preflight_args(dry_run)
        args.task, args.gpus, args.time = TASK, 1, "01:00:00"
        return args

    def test_dry_run_exact_bounded_resources_and_publication_grace(self):
        with patch.object(cli, "remote") as call, redirect_stdout(io.StringIO()):
            cli.submit_command(self.arguments(True))
        call.assert_not_called()
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        argv = plan["sbatch_argv"]
        for value in (
            "--gpus=h100:1",
            "--cpus-per-task=24",
            "--mem=192G",
            "--time=01:00:00",
            "--signal=B:TERM@300",
        ):
            self.assertIn(value, argv)
        index = next(index for index, arg in enumerate(argv) if arg.endswith("/sdsc_teacher_adapt_job.sh"))
        self.assertEqual(len(argv[index + 1 :]), 7)
        self.assertFalse(plan["authorized"])

    def test_future_four_gpu_fit_and_out_of_bounds_resources_are_not_enabled(self):
        for key, value in (("gpus", 4), ("cpus", 4), ("mem_gib", 16), ("time", "01:00:01")):
            args = self.arguments()
            setattr(args, key, value)
            with self.subTest(key=key), patch.object(cli, "remote") as call, self.assertRaises(cli.UserError):
                cli.submit_command(args)
            call.assert_not_called()

    def test_fetch_accepts_only_small_diagnostics_and_rejects_weights(self):
        intent_id = "a" * 32
        cli.write_json(
            cli.state_root() / "submissions" / (intent_id + ".json"),
            {"request": {"task": TASK}, "receipt": {"job_id": "12345"}},
        )
        raw = b"small diagnostic\n"
        files = [
            dict(path=name, data_b64=base64.b64encode(raw).decode(), size=len(raw), sha256=cli.sha(raw))
            for name in sorted(cli.TEACHER_ADAPT_SMALL_RESULTS)
        ]
        with patch.object(cli, "remote", return_value={"files": files}), redirect_stdout(io.StringIO()):
            cli.job_command(cli.parser().parse_args(["fetch", "12345"]))
        saved = list((cli.state_root() / "fetched/12345").glob("fetch-*"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            {path.name for path in saved[0].iterdir()},
            cli.TEACHER_ADAPT_SMALL_RESULTS | {"fetch-manifest.json"},
        )
        for name in ("merged/model.safetensors", "model.safetensors", "optimizer.pt"):
            with (
                self.subTest(name=name),
                patch.object(cli, "remote", return_value={"files": [dict(files[0], path=name)]}),
                self.assertRaisesRegex(cli.UserError, "Unexpected remote fetch"),
            ):
                cli.job_command(cli.parser().parse_args(["fetch", "12345"]))


class AdaptRemoteTests(unittest.TestCase):
    def setUp(self):
        self.fixture = remote_fixture.RemoteBoundaryTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.enable_preflight()
        self.request = self.fixture.request
        self.request.update(task=TASK)
        self.request["resources"].update(gpus=1, time="01:00:00")
        remote.upload(
            self.request,
            self.fixture.archive(
                {
                    "tools/sdsc_teacher_adapt_job.sh": b"#!/bin/bash\nexit 0\n",
                    "tools/sdsc_teacher_adapt.py": b"# fixed teacher adaptation preflight\n",
                    "tools/sdsc_teacher_prepare.py": b"# allocation/runtime guards\n",
                    "tools/sdsc_training_preflight.py": b"# pinned cache guards\n",
                    "tools/sdsc_teacher_probe.py": b"# durable observation helper\n",
                    "tools/sdsc_teacher_capability.py": b"# frozen metrics helper\n",
                    "src/posttrain_circuits/learning/teacher/adaptation.py": b"# preflight implementation\n",
                }
            ),
        )

    def submit(self):
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(
                remote,
                "run",
                return_value={
                    "returncode": 0,
                    "stdout": "12345;expanse\n",
                    "stderr": "",
                },
            ) as run,
        ):
            receipt = remote.submit(self.request)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--signal=B:TERM@300", run.call_args.args[0])
        return receipt

    def publication(self, receipt, **changes):
        identity = {key: receipt[key] for key in ("run_id", "job_id", "code_sha256", "task", "hf_home")}
        root = Path(receipt["result_dir"])
        checkpoint_hash, files, adapted_sha = checkpoint_fixture(root / "artifacts", identity)
        report = report_fixture(identity, checkpoint_hash, adapted_sha)
        report.update(changes)
        raw = remote.canonical(report)
        (root / REPORT).write_bytes(raw)
        remote.atomic_json(
            root / "receipt.json",
            dict(
                identity,
                passed=True,
                persisted=True,
                persistent_read_back_verified=True,
                files=[*files, dict(path=REPORT, size=len(raw), sha256=remote.digest(raw))],
            ),
        )

    def test_diagnostic_result_cannot_claim_formal_teacher_or_training_readiness(self):
        receipt = self.submit()
        self.publication(receipt)
        self.assertTrue(remote.verify_result(receipt)["verified"])
        for key, value in (
            ("accepted_science", True),
            ("full_teacher_ready", True),
            ("g0_passed", True),
            ("resumable", True),
            ("artifact_kind", "teacher_readiness"),
            ("adaptation_preflight", False),
            ("readiness_artifact_produced", True),
            ("readiness_artifact_produced", None),
            ("student_training_started", True),
            ("student_training_started", None),
            ("training_started", False),
            ("checks", {}),
            ("checks", {"reload_logits": False}),
            ("checkpoint_manifest_sha256", "c" * 64),
            ("adapted_teacher_sha256", "c" * 64),
            ("optimizer_steps", 7),
            ("optimizer_steps", 8.0),
            ("consumed_tokens", 2561),
            ("consumed_tokens", True),
            ("adaptation_dataset", {}),
            ("adaptation_dataset", {"teacher_fit": {"count": 256}}),
            ("adaptation_plan_sha256", "c" * 64),
            ("adaptation_plan", dict(remote.teacher_adapt_preflight_plan(), train_examples=128)),
        ):
            with self.subTest(key=key):
                self.publication(receipt, **{key: value})
                self.assertFalse(remote.verify_result(receipt)["verified"])

    def test_fetch_never_downloads_model_or_optimizer(self):
        receipt = self.submit()
        self.publication(receipt)
        self.request["job_id"] = receipt["job_id"]
        root = Path(receipt["result_dir"])
        artifacts = root / "artifacts"
        (artifacts / "model.safetensors").write_bytes(b"not fetched")
        (artifacts / "optimizer.pt").write_bytes(b"not fetched either")
        result = remote.fetch(self.request)
        self.assertEqual(
            {entry["path"] for entry in result["files"]},
            {REPORT, "receipt.json", "checkpoint-manifest.json", "train-metrics.jsonl"},
        )
        self.assertEqual(cli.TEACHER_ADAPT_SMALL_RESULTS, remote.TEACHER_ADAPT_SMALL_RESULTS)

    def test_oversized_optional_development_json_does_not_block_small_evidence(self):
        receipt = self.submit()
        self.publication(receipt)
        self.request["job_id"] = receipt["job_id"]
        root = Path(receipt["result_dir"])
        path = root / "artifacts/dev-capability.json"
        path.write_bytes(b"x" * (remote.MAX_FETCH_FILE + 1))
        result = remote.fetch(self.request)
        self.assertEqual(
            result["skipped"],
            [
                {
                    "path": "dev-capability.json",
                    "size": remote.MAX_FETCH_FILE + 1,
                    "reason": "optional_file_exceeds_1MiB_limit",
                }
            ],
        )
        self.assertTrue(
            {REPORT, "receipt.json", "train-metrics.jsonl"} <= {row["path"] for row in result["files"]}
        )
        self.assertNotIn("dev-capability.json", {row["path"] for row in result["files"]})
        path.unlink()
        path.symlink_to(root / REPORT)
        with self.assertRaises(ValueError):
            remote.fetch(self.request)

    def test_changed_resources_fail_before_claim(self):
        self.request["resources"]["gpus"] = 4
        with patch.object(remote, "run") as run, self.assertRaises(ValueError):
            remote.submit(self.request)
        run.assert_not_called()
        self.assertFalse((self.fixture.root / "run-claims").exists())

    def test_checkpoint_publication_record_and_manifest_are_required(self):
        receipt = self.submit()
        self.publication(receipt)
        root = Path(receipt["result_dir"])
        record = json.loads((root / "receipt.json").read_text())
        record["files"] = [
            entry for entry in record["files"] if entry["path"] != "artifacts/merged/model.safetensors"
        ]
        remote.atomic_json(root / "receipt.json", record)
        self.assertFalse(remote.verify_result(receipt)["verified"])

    def test_metrics_bytes_and_training_provenance_are_bound(self):
        receipt = self.submit()
        root = Path(receipt["result_dir"])
        for mutation in ("bytes", "receipt", "provenance"):
            with self.subTest(mutation=mutation):
                self.publication(receipt)
                if mutation == "bytes":
                    (root / "artifacts/train-metrics.jsonl").write_text("changed actual training metrics")
                elif mutation == "receipt":
                    record = json.loads((root / "receipt.json").read_text())
                    record["files"] = [
                        row for row in record["files"] if row["path"] != "artifacts/train-metrics.jsonl"
                    ]
                    remote.atomic_json(root / "receipt.json", record)
                else:
                    path = root / "artifacts/checkpoint-manifest.json"
                    manifest = json.loads(path.read_text())
                    del manifest["training_provenance"]
                    manifest["sha256"] = remote.digest(
                        remote.canonical({key: value for key, value in manifest.items() if key != "sha256"})
                    )
                    raw = remote.canonical(manifest)
                    path.write_bytes(raw)
                    report = json.loads((root / REPORT).read_text())
                    report.update(
                        checkpoint_manifest_sha256=remote.digest(raw),
                        adapted_teacher_sha256=manifest["sha256"],
                    )
                    report_raw = remote.canonical(report)
                    (root / REPORT).write_bytes(report_raw)
                    record = json.loads((root / "receipt.json").read_text())
                    for row in record["files"]:
                        if row["path"] == "artifacts/checkpoint-manifest.json":
                            row.update(size=len(raw), sha256=remote.digest(raw))
                        elif row["path"] == REPORT:
                            row.update(size=len(report_raw), sha256=remote.digest(report_raw))
                    remote.atomic_json(root / "receipt.json", record)
                self.assertFalse(remote.verify_result(receipt)["verified"])
        self.publication(receipt)
        (root / "artifacts/checkpoint-manifest.json").write_text("{}")
        self.assertFalse(remote.verify_result(receipt)["verified"])


def wrapper_functions():
    path = REPO / "tools/sdsc_teacher_adapt_job.sh"
    source = path.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(definitions, str(path), "exec"), namespace)
    return namespace


class AdaptWrapperTests(unittest.TestCase):
    def test_adaptation_wrapper_preserves_term_tail_after_log_scope_unwinds(self):
        fixture = module("adapt_pipe_fixture", Path(__file__).with_name("test_teacher_probe_wrapper.py"))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(fixture, "SCRIPT", REPO / "tools/sdsc_teacher_adapt_job.sh"),
        ):
            fixture.test_interrupted_wrapper_preserves_shutdown_pipe_after_original_log_closes(
                Path(directory)
            )

    def test_adaptation_wrapper_still_bounds_native_hang_shutdown(self):
        fixture = module("adapt_native_fixture", Path(__file__).with_name("test_teacher_probe_wrapper.py"))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(fixture, "SCRIPT", REPO / "tools/sdsc_teacher_adapt_job.sh"),
        ):
            fixture.test_native_block_forces_kill_and_preserves_faulthandler_stderr(Path(directory))

    def test_persistence_records_merged_model_adapter_and_manifest_hashes(self):
        helpers = wrapper_functions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, persistent = root / "node", root / "persistent"
            output.mkdir()
            persistent.mkdir()
            identity = dict(task=TASK, job_id="12345", run_id="adapt-fixture", code_sha256="a" * 64)
            manifest_sha, files, adapted_sha = checkpoint_fixture(output, identity)
            report = report_fixture(identity, manifest_sha, adapted_sha)
            helpers.update(identity=identity, result_root=persistent, process=None)
            helpers["validate_report"](report, 0)
            with redirect_stdout(io.StringIO()):
                helpers["publish"](output, None, report, True)
            receipt = json.loads((persistent / "receipt.json").read_text())
            self.assertTrue(receipt["persistent_read_back_verified"])
            for entry in receipt["files"]:
                self.assertEqual(helpers["file_hash"](persistent / entry["path"]), entry["sha256"])
            self.assertEqual(len(receipt["files"]), len(files) + 1)

    def test_mutated_checkpoint_bytes_cannot_publish_success(self):
        helpers = wrapper_functions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, persistent = root / "node", root / "persistent"
            persistent.mkdir()
            manifest_sha, _, adapted_sha = checkpoint_fixture(output, {})
            report = report_fixture({}, manifest_sha, adapted_sha)
            helpers.update(identity={}, result_root=persistent, process=None)
            model = output / "merged/model.safetensors"
            model.write_bytes(b"x" * model.stat().st_size)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "content differs"):
                helpers["publish"](output, None, report, True)
            self.assertFalse((persistent / "receipt.json").exists())

    def test_metrics_mutation_and_provenance_mismatch_cannot_publish_success(self):
        helpers = wrapper_functions()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            manifest_sha, _, adapted_sha = checkpoint_fixture(output, {})
            report = report_fixture({}, manifest_sha, adapted_sha)
            for changed in (
                dict(report, consumed_tokens=2561),
                dict(report, adaptation_dataset={}),
                dict(report, job_id="wrong-job"),
            ):
                with self.assertRaisesRegex(ValueError, "training provenance|training data"):
                    helpers["checkpoint_records"](output, changed)
            (output / "train-metrics.jsonl").write_text("changed actual training metrics")
            with self.assertRaisesRegex(ValueError, "training provenance"):
                helpers["checkpoint_records"](output, report)

    def test_control_plan_matches_scientific_preflight_and_rejects_reduced_checks(self):
        from posttrain_circuits.learning.teacher.adaptation import PREFLIGHT

        helpers = wrapper_functions()
        self.assertEqual(remote.teacher_adapt_preflight_plan(), PREFLIGHT)
        self.assertEqual(helpers["teacher_adapt_preflight_plan"](), PREFLIGHT)
        identity = dict(task=TASK, job_id="12345", run_id="adapt-fixture", code_sha256="a" * 64)
        helpers.update(identity=identity)
        report = report_fixture(identity, "b" * 64, "c" * 64)
        helpers["validate_report"](report, 0)
        for changed in (
            dict(report, checks={"isolated_data": True}),
            dict(report, optimizer_steps=0),
            dict(report, adaptation_plan=dict(PREFLIGHT, global_batch_size=16)),
            dict(report, adaptation_plan_sha256="d" * 64),
            dict(report, readiness_artifact_produced=True),
            dict(report, readiness_artifact_produced=None),
            dict(report, student_training_started=True),
            dict(report, student_training_started=None),
            dict(report, training_started=False),
        ):
            with self.assertRaises(ValueError):
                helpers["validate_report"](changed, 0)

    def test_failure_publication_needs_no_checkpoint_or_progress(self):
        helpers = wrapper_functions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helpers.update(identity={}, result_root=root, process=None)
            with redirect_stdout(io.StringIO()):
                helpers["publish"](None, None, {"passed": False}, False)
            receipt = json.loads((root / "receipt.json").read_text())
            self.assertFalse(receipt["passed"])
            self.assertEqual([entry["path"] for entry in receipt["files"]], [REPORT])


if __name__ == "__main__":
    unittest.main()
