"""Offline boundary tests: no SSH, scheduler, model or GPU calls."""

import argparse
import base64
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = load("sdsc_cli")
remote = load("sdsc_remote")


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "OPD"
        self.root.mkdir()
        self.patcher = patch.object(cli, "ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def inventory(self, paths):
        with patch.object(cli, "git_read", return_value=b"\0".join(p.encode() for p in paths) + b"\0"):
            return cli.inventory(self.root)

    def make_file(self, path, value=b"hello\n"):
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(value)
        return full

    def record(self, state="deployed"):
        manifest = {"run_id": "run-test", "code_sha256": "a" * 64}
        cli.write_json(cli.state_root() / "runs/run-test.json", {"state": state, "manifest": manifest})
        return manifest

    def submit_args(self, dry_run=False):
        values = [
            "submit",
            "run-test",
            "--task",
            "gpu-smoke",
            "--account",
            "nwu181",
            "--partition",
            "nairr-gpu-shared",
            "--qos",
            "nairr-gpu-shared-normal",
            "--gpu-type",
            "h100",
            "--gpus",
            "1",
            "--cpus",
            "4",
            "--mem-gib",
            "16",
            "--time",
            "00:05:00",
            "--dry-run" if dry_run else "--authorize",
        ]
        if not dry_run:
            values += [
                "--python",
                "/confirmed/bin/python",
                "--result-root",
                "/confirmed/results",
                "--storage-confirmed",
            ]
        return cli.parser().parse_args(values)

    def container_args(self, args, inspected=True):
        identity = {
            "runtime": "/discovered/bin/singularity",
            "image": "/shared/image ;$(not-executed).sif",
            "python": "/usr/bin/python",
            "size": 123456,
            "mtime_ns": 1234567890,
        }
        for key in ("runtime", "image", "python"):
            setattr(args, "container_" + key, identity[key])
        if inspected:
            metadata = {"supported_python": True, "packages": {"torch": "2.7.0", "numpy": "1.26.4"}}
            cli.write_json(
                cli.state_root() / "check.json",
                {
                    "connected": True,
                    "inventory": {
                        "candidate_container": identity,
                        "container_inspection": {
                            "returncode": 0,
                            "stdout": json.dumps(metadata),
                            "stderr": "",
                        },
                    },
                },
            )
        return identity

    def test_working_bytes_and_new_files_define_identity(self):
        self.make_file("src/existing.py", b"uncommitted = 2\n")
        self.make_file("tools/new.py", b"new = True\n")
        files, contents, excluded = self.inventory(["src/existing.py", "tools/new.py"])
        self.assertFalse(excluded)
        self.assertEqual(contents["src/existing.py"], b"uncommitted = 2\n")
        before = cli.sha(cli.canonical(files))
        self.make_file("src/existing.py", b"uncommitted = 3\n")
        after = self.inventory(["src/existing.py", "tools/new.py"])[0]
        self.assertNotEqual(before, cli.sha(cli.canonical(after)))

    def test_secrets_excluded_before_open_and_scientific_token_code_retained(self):
        paths = [
            "configs/auth.json",
            "configs/.env",
            ".ssh/id_rsa",
            ".codex/auth.json",
            "tools/token.json",
            ".venv/bin/python",
            "src/learning/token_budget.py",
        ]
        self.make_file(paths[-1])
        original = cli.safe_source_bytes
        seen = []

        def reading(root, path):
            seen.append(path)
            return original(root, path)

        with patch.object(cli, "safe_source_bytes", side_effect=reading):
            files, _, excluded = self.inventory(paths)
        self.assertEqual(seen, [paths[-1]])
        self.assertEqual(len(files), 1)
        self.assertEqual(len(excluded), 6)

    def test_symlinks_binary_large_files_are_not_snapshotted(self):
        outside = Path(self.temp.name) / "external.py"
        outside.write_text("outside")
        (self.root / "tools").mkdir()
        (self.root / "tools/link.py").symlink_to(outside)
        self.make_file("tools/binary.py", b"\x00x")
        self.make_file("tools/large.py", b"x" * (cli.MAX_FILE + 1))
        files, _, excluded = self.inventory(["tools/link.py", "tools/binary.py", "tools/large.py"])
        self.assertEqual(files, [])
        self.assertEqual(len(excluded), 3)

    def test_archive_is_accepted_by_real_remote_upload_validator(self):
        self.make_file("tools/new.py", b"current changes\n")
        self.make_file(".env.example", b"EXAMPLE_ROOT=/example\n")
        files, contents, _ = self.inventory(["tools/new.py", ".env.example"])
        with patch.object(cli, "git_read", return_value=b"123456\n"):
            manifest = cli.make_manifest(files)
        archive = cli.build_archive(manifest, contents)
        destination = Path(self.temp.name) / "remote"
        request = {
            "root": str(destination),
            "run_id": manifest["run_id"],
            "code_sha256": manifest["code_sha256"],
        }
        with patch.object(remote, "ROOT", destination):
            result = remote.upload(request, io.BytesIO(archive))
            with self.assertRaises(ValueError):
                remote.upload(request, io.BytesIO(archive))
        staged = Path(result["release"]) / "source/tools/new.py"
        self.assertEqual(staged.read_bytes(), contents["tools/new.py"])

    def test_changed_preview_never_connects(self):
        self.make_file("tools/new.py", b"initial")
        with (
            patch.object(cli, "git_read", side_effect=[b"tools/new.py\0", b"head"]),
            redirect_stdout(io.StringIO()),
        ):
            cli.sync_command(argparse.Namespace(dry_run=True))
        self.make_file("tools/new.py", b"edited")
        with (
            patch.object(cli, "git_read", return_value=b"tools/new.py\0"),
            patch.object(cli, "remote") as call,
            self.assertRaisesRegex(cli.UserError, "changed since preview"),
        ):
            cli.sync_command(argparse.Namespace(dry_run=False))
        call.assert_not_called()

    def test_submit_dry_run_without_guessed_runtime_is_local_and_blocked(self):
        self.record("preview")
        with (
            patch.object(cli, "remote") as call,
            patch.object(cli, "require_master") as check,
            redirect_stdout(io.StringIO()),
        ):
            cli.submit_command(self.submit_args(dry_run=True))
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        self.assertEqual(len(plan["blockers"]), 4)
        self.assertEqual(plan["resources"]["account"], "nwu181")
        self.assertEqual(plan["resources"]["qos"], "nairr-gpu-shared-normal")
        self.assertIn("--qos=nairr-gpu-shared-normal", plan["sbatch_argv"])
        self.assertFalse(plan["authorized"])
        call.assert_not_called()
        check.assert_not_called()

    def test_exact_resource_constraints(self):
        args = self.submit_args()
        for field, bad in [
            ("account", "expanse_nairr_gpu"),
            ("qos", "nairr-gpu-shared"),
            ("qos", "unverified-qos"),
            ("gpus", 4),
            ("cpus", 8),
            ("mem_gib", 192),
            ("time", "00:06:00"),
            ("time", "00:04:60"),
        ]:
            original = getattr(args, field)
            setattr(args, field, bad)
            with self.subTest(field=field), self.assertRaises(cli.UserError):
                cli.resources(args)
            setattr(args, field, original)

    def test_lost_receipt_blocks_all_subsequent_submissions_for_run(self):
        self.record()
        with (
            patch.object(cli, "require_master"),
            patch.object(cli, "remote", side_effect=cli.UserError("connection lost")) as send,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(cli.UserError, "UNKNOWN"):
                cli.submit_command(self.submit_args())
            with self.assertRaisesRegex(cli.UserError, "reconcile"):
                cli.submit_command(self.submit_args())
        self.assertEqual(send.call_count, 1)
        records = list((cli.state_root() / "submissions").glob("*.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(cli.read_json(records[0])["state"], "unknown")

    def test_unknown_intent_also_blocks_new_run_id(self):
        self.record()
        cli.write_json(
            cli.state_root() / "submissions" / ("b" * 32 + ".json"),
            {"state": "unknown", "request": {"run_id": "another-run"}},
        )
        with (
            patch.object(cli, "remote") as send,
            self.assertRaisesRegex(cli.UserError, "Unresolved submission"),
        ):
            cli.submit_command(self.submit_args())
        send.assert_not_called()

    def test_ssh_only_uses_existing_master_and_quotes_parameters(self):
        args = ["ssh", "-F", "/dev/null", "-S", "/socket", "-o", "ProxyCommand=false", "-o", "BatchMode=yes"]
        dangerous = "path with spaces;$(touch /tmp/should-not-exist)'"
        with (
            patch.object(cli, "require_master", return_value=args),
            patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run,
        ):
            cli.ssh_call(["python", "-c", "print(1)", dangerous])
        import shlex

        executed = run.call_args.args[0]
        self.assertEqual(executed[0], "ssh")
        self.assertEqual(executed[-2], cli.TARGET)
        self.assertEqual(shlex.split(executed[-1])[-1], dangerous)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_dead_master_stops_before_remote_command(self):
        base = ["ssh", "-F", "/dev/null", "-S", "/missing"]
        with (
            patch.object(cli, "ssh_base", return_value=base),
            patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 255, b"", b"missing")
            ) as run,
            self.assertRaisesRegex(cli.UserError, "authenticate manually"),
        ):
            cli.ssh_call(["python", "-c", "never"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][-3:], ["-O", "check", cli.TARGET])

    def bind_job(self):
        intent = {"request": {"run_id": "run-test"}, "receipt": {"job_id": "123", "run_id": "run-test"}}
        cli.write_json(cli.state_root() / "submissions" / ("a" * 32 + ".json"), intent)

    def test_fetch_refuses_symlink_into_source(self):
        self.bind_job()
        (cli.state_root() / "fetched").symlink_to(self.root, target_is_directory=True)
        with (
            patch.object(cli, "remote", return_value={"files": []}),
            self.assertRaisesRegex(cli.UserError, "symlink"),
        ):
            cli.job_command(argparse.Namespace(command="fetch", job_id="123"))
        self.assertFalse((self.root / "123").exists())

    def test_fetch_isolated_and_content_hash_checked(self):
        self.bind_job()
        data = b'{"passed": false}\n'
        entry = {
            "path": "result.json",
            "size": len(data),
            "sha256": cli.sha(data),
            "data_b64": base64.b64encode(data).decode(),
        }
        response = {"files": [entry]}
        with patch.object(cli, "remote", return_value=response), redirect_stdout(io.StringIO()):
            cli.job_command(argparse.Namespace(command="fetch", job_id="123"))
            cli.job_command(argparse.Namespace(command="fetch", job_id="123"))
        results = list((cli.state_root() / "fetched/123").glob("fetch-*/result.json"))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(path.read_bytes() == data for path in results))
        self.assertFalse((self.root / "result.json").exists())

    def test_cancel_requires_authorization_before_network(self):
        with patch.object(cli, "remote") as send, self.assertRaisesRegex(cli.UserError, "authorization"):
            cli.job_command(argparse.Namespace(command="cancel", job_id="123", authorize=False))
        send.assert_not_called()

    def test_state_write_refuses_any_symlink_ancestor(self):
        state = cli.state_root()
        (state / "submissions").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(cli.UserError, "symlink"):
            cli.write_json(state / "submissions/hidden/intent.json", {})
        self.assertFalse((self.root / "hidden").exists())

    def test_container_flags_are_atomic_before_any_connection(self):
        args = cli.parser().parse_args(["check", "--container-runtime", "/bin/singularity"])
        with patch.object(cli, "ssh_call") as ssh, self.assertRaisesRegex(cli.UserError, "together"):
            cli.check_command(args)
        ssh.assert_not_called()
        args.container_image = "/shared/existing.sif"
        args.container_python = "relative/python"
        with self.assertRaisesRegex(cli.UserError, "absolute"):
            cli.container_options(args)

    def test_check_passes_container_path_options_for_remote_metadata_only(self):
        args = cli.parser().parse_args(
            [
                "check",
                "--container-runtime",
                "/existing/singularity",
                "--container-image",
                "/existing/image.sif",
                "--container-python",
                "/usr/bin/python",
            ]
        )
        with (
            patch.object(cli, "ssh_base", return_value=["ssh", "-F", "/dev/null", "-S", "/fixture"]),
            patch.object(
                cli, "ssh_call", return_value=subprocess.CompletedProcess([], 0, b"/usr/bin/python3.11\n")
            ),
            patch.object(cli, "remote", return_value={}) as call,
            redirect_stdout(io.StringIO()),
        ):
            cli.check_command(args)
        self.assertEqual(
            call.call_args.args[0]["container"],
            {"runtime": "/existing/singularity", "image": "/existing/image.sif", "python": "/usr/bin/python"},
        )

    def test_container_dry_run_requires_matching_inspection_and_has_safe_exact_argv(self):
        self.record("preview")
        args = self.submit_args(dry_run=True)
        identity = self.container_args(args)
        with patch.object(cli, "remote") as call, redirect_stdout(io.StringIO()):
            cli.submit_command(args)
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        self.assertEqual(plan["container"], identity)
        self.assertEqual(json.loads(plan["sbatch_argv"][-1]), identity)
        self.assertEqual(len(plan["blockers"]), 4)
        call.assert_not_called()

    def test_uninspected_container_blocks_actual_submit_but_has_local_preview(self):
        self.record()
        args = self.submit_args()
        self.container_args(args, inspected=False)
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "container identity"):
            cli.submit_command(args)
        call.assert_not_called()
        args.authorize, args.dry_run = False, True
        with patch.object(cli, "remote") as call, redirect_stdout(io.StringIO()):
            cli.submit_command(args)
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        self.assertEqual(plan["container"]["size"], 0)
        self.assertEqual(len(plan["blockers"]), 1)
        call.assert_not_called()

    def test_stale_or_non_torch_container_check_is_not_reused(self):
        args = self.submit_args()
        identity = self.container_args(args)
        specification = cli.container_options(args)
        for case in ("path", "failed", "packages"):
            self.container_args(args)
            check = cli.read_json(cli.state_root() / "check.json")
            if case == "path":
                check["inventory"]["candidate_container"]["image"] = "/another/image.sif"
            elif case == "failed":
                check["inventory"]["container_inspection"]["returncode"] = 1
            else:
                check["inventory"]["container_inspection"]["stdout"] = (
                    '{"supported_python":true,"packages":{}}'
                )
            cli.write_json(cli.state_root() / "check.json", check)
            returned, blockers = cli.checked_container(specification)
            self.assertNotEqual(returned, identity)
            self.assertTrue(blockers)

    def preflight_args(self, dry_run=False):
        args = self.submit_args(dry_run=dry_run)
        args.task = "qwen3-v2-preflight"
        args.gpus, args.cpus, args.mem_gib, args.time = 2, 24, 192, "02:00:00"
        args.hf_home = str(cli.PREFLIGHT_STORAGE / "cache/huggingface")
        if not dry_run:
            args.result_root = str(cli.PREFLIGHT_STORAGE / "results")
        return args

    def test_preflight_dry_run_renders_fixed_worker_and_no_network(self):
        self.record("preview")
        args = self.preflight_args(dry_run=True)
        with patch.object(cli, "remote") as call, redirect_stdout(io.StringIO()):
            cli.submit_command(args)
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        self.assertIn("--gpus=h100:2", plan["sbatch_argv"])
        self.assertIn("--cpus-per-task=24", plan["sbatch_argv"])
        self.assertIn("--mem=192G", plan["sbatch_argv"])
        self.assertEqual(plan["sbatch_argv"][-1], args.hf_home)
        self.assertTrue(plan["worker_script"].endswith("sdsc_preflight_job.sh"))
        self.assertFalse(plan["authorized"])
        self.assertTrue(plan["blockers"])
        call.assert_not_called()

    def test_preflight_exact_resources_and_two_hour_ceiling(self):
        args = self.preflight_args()
        self.assertEqual(cli.resources(args)["gpus"], 2)
        for field, value in (
            ("gpus", 1),
            ("gpus", 3),
            ("cpus", 4),
            ("mem_gib", 16),
            ("time", "02:00:01"),
            ("time", "00:00:00"),
            ("time", "01:60:00"),
        ):
            original = getattr(args, field)
            setattr(args, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(cli.UserError):
                cli.resources(args)
            setattr(args, field, original)

    def test_preflight_rejects_container_and_unconfirmed_model_storage_before_network(self):
        self.record()
        for field, value in (
            ("hf_home", None),
            ("hf_home", "/home/zgao12/cache"),
            ("hf_home", str(cli.PREFLIGHT_STORAGE / "../escape")),
            ("result_root", cli.REMOTE_ROOT + "/smoke-results"),
        ):
            args = self.preflight_args()
            setattr(args, field, value)
            with patch.object(cli, "remote") as call, self.assertRaises(cli.UserError):
                cli.submit_command(args)
            call.assert_not_called()
        args = self.preflight_args()
        self.container_args(args)
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "container"):
            cli.submit_command(args)
        call.assert_not_called()

    def test_preflight_receipt_task_and_cache_mismatch_is_unknown_without_retry(self):
        self.record()
        args = self.preflight_args()

        def reply(payload):
            return dict(payload, job_id="23456", hf_home="/another/cache")

        with (
            patch.object(cli, "require_master"),
            patch.object(cli, "remote", side_effect=reply) as call,
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(cli.UserError, "UNKNOWN"),
        ):
            cli.submit_command(args)
        self.assertEqual(call.call_count, 1)
        intent = cli.read_json(next((cli.state_root() / "submissions").glob("*.json")))
        self.assertEqual(intent["state"], "unknown")
        self.assertEqual(intent["request"]["task"], "qwen3-v2-preflight")
        self.assertEqual(intent["request"]["hf_home"], args.hf_home)

    def test_preflight_has_no_implicit_authorization(self):
        self.record()
        args = self.preflight_args()
        args.authorize = False
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "authorization"):
            cli.submit_command(args)
        call.assert_not_called()

    def test_preflight_fetch_does_not_accept_checkpoint_or_smoke_result(self):
        intent = {
            "request": {"task": "qwen3-v2-preflight", "run_id": "run-test"},
            "receipt": {"job_id": "123", "run_id": "run-test"},
        }
        cli.write_json(cli.state_root() / "submissions" / ("a" * 32 + ".json"), intent)
        for name in ("result.json", "checkpoint.pt", "artifacts/rank-0.json"):
            with (
                patch.object(cli, "remote", return_value={"files": [{"path": name}]}),
                self.assertRaisesRegex(cli.UserError, "Unexpected remote fetch"),
            ):
                cli.job_command(argparse.Namespace(command="fetch", job_id="123"))

    def teacher_args(self, dry_run=False):
        args = self.preflight_args(dry_run)
        args.task, args.gpus, args.time = "qwen3-v2-teacher-prepare", 1, "04:00:00"
        args.provenance_manifest_sha256 = "b" * 64
        args.provenance_dir = cli.REMOTE_ROOT + "/provenance/" + args.provenance_manifest_sha256
        return args

    def test_teacher_dry_run_preserves_nine_arguments_and_never_connects(self):
        self.record("preview")
        args = self.teacher_args(True)
        with patch.object(cli, "remote") as call, redirect_stdout(io.StringIO()):
            cli.submit_command(args)
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        argv = plan["sbatch_argv"]
        script = next(i for i, value in enumerate(argv) if value.endswith("/sdsc_teacher_job.sh"))
        self.assertEqual(len(argv[script + 1 :]), 9)
        self.assertEqual(argv[-3:], [args.hf_home, args.provenance_dir, args.provenance_manifest_sha256])
        self.assertIn("--gpus=h100:1", argv)
        self.assertIn("--cpus-per-task=24", argv)
        self.assertIn("--mem=192G", argv)
        self.assertIn("--time=04:00:00", argv)
        self.assertFalse(plan["authorized"])
        call.assert_not_called()

    def test_teacher_exact_resource_and_provenance_constraints(self):
        self.record()
        for field, bad in (
            ("gpus", 2),
            ("cpus", 4),
            ("mem_gib", 16),
            ("time", "04:00:01"),
            ("provenance_dir", "/other/artifact"),
            ("provenance_manifest_sha256", None),
            ("provenance_manifest_sha256", "not-a-hash"),
        ):
            args = self.teacher_args()
            setattr(args, field, bad)
            with (
                self.subTest(field=field),
                patch.object(cli, "remote") as call,
                self.assertRaises(cli.UserError),
            ):
                cli.submit_command(args)
            call.assert_not_called()

    def test_teacher_container_and_home_results_are_rejected(self):
        self.record()
        args = self.teacher_args()
        self.container_args(args)
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "container"):
            cli.submit_command(args)
        call.assert_not_called()
        args = self.teacher_args()
        args.result_root = cli.REMOTE_ROOT + "/smoke-results"
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "result_root"):
            cli.submit_command(args)
        call.assert_not_called()

    def test_teacher_mismatched_provenance_receipt_is_unknown(self):
        self.record()
        args = self.teacher_args()
        with (
            patch.object(cli, "require_master"),
            redirect_stdout(io.StringIO()),
            patch.object(
                cli,
                "remote",
                side_effect=lambda payload: dict(
                    payload, job_id="67890", provenance_manifest_sha256="f" * 64
                ),
            ) as call,
            self.assertRaisesRegex(cli.UserError, "UNKNOWN"),
        ):
            cli.submit_command(args)
        self.assertEqual(call.call_count, 1)
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "Unresolved"):
            cli.submit_command(args)
        call.assert_not_called()

    def test_reconcile_cannot_change_teacher_provenance(self):
        identity = "a" * 32
        args = self.teacher_args()
        request = dict(
            run_id="run-test",
            code_sha256="a" * 64,
            intent_id=identity,
            task=args.task,
            hf_home=args.hf_home,
            provenance_dir=args.provenance_dir,
            provenance_manifest_sha256=args.provenance_manifest_sha256,
        )
        path = cli.state_root() / "submissions" / (identity + ".json")
        cli.write_json(path, {"state": "unknown", "request": request})
        with (
            patch.object(cli, "remote", return_value=dict(request, job_id="67890", provenance_dir="/wrong")),
            self.assertRaisesRegex(cli.UserError, "provenance"),
        ):
            cli.reconcile_command(argparse.Namespace(intent_id=identity))
        self.assertEqual(cli.read_json(path)["state"], "unknown")

    def test_provenance_flags_cannot_be_smuggled_into_preflight(self):
        self.record()
        args = self.preflight_args()
        args.provenance_dir = "/arbitrary"
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "only supported"):
            cli.submit_command(args)
        call.assert_not_called()

    def calibration_args(self, dry_run=False):
        args = self.teacher_args(dry_run)
        args.task, args.gpus, args.time = "qwen3-v2-g0-calibration", 2, "02:00:00"
        args.teacher_job_id, args.preflight_job_id = "101", "102"
        return args

    def test_calibration_dry_run_is_local_and_explicit_about_unverified_prerequisites(self):
        self.record()
        with patch.object(cli, "remote") as call, redirect_stdout(io.StringIO()):
            cli.submit_command(self.calibration_args(True))
        call.assert_not_called()
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        self.assertTrue(plan["prerequisites_sha256_is_placeholder"])
        self.assertIn("not checked", plan["prerequisites_verification"])
        argv = plan["sbatch_argv"]
        script = next(i for i, value in enumerate(argv) if value.endswith("/sdsc_calibration_job.sh"))
        self.assertEqual(len(argv[script + 1 :]), 11)
        self.assertEqual(argv[-1], "0" * 64)
        self.assertIn("--gpus=h100:2", argv)
        self.assertIn("--time=02:00:00", argv)

    def test_calibration_requires_exact_profile_and_distinct_real_upstreams(self):
        self.record()
        for field, value in (
            ("teacher_job_id", None),
            ("preflight_job_id", "101"),
            ("preflight_job_id", "12;command"),
            ("time", "02:00:01"),
            ("gpus", 4),
            ("provenance_dir", "/wrong"),
            ("hf_home", "/home/cache"),
            ("authorize", False),
        ):
            args = self.calibration_args()
            setattr(args, field, value)
            with (
                self.subTest(field=field),
                patch.object(cli, "remote") as call,
                self.assertRaises(cli.UserError),
            ):
                cli.submit_command(args)
            call.assert_not_called()

    def test_calibration_unknown_receipt_is_never_retried_and_validated_on_reconcile(self):
        self.record()

        def response(payload):
            return dict(
                payload,
                job_id="103",
                teacher_job_id="999",
                prerequisites_sha256="c" * 64,
                prerequisites_path=cli.REMOTE_ROOT
                + "/submissions/"
                + payload["intent_id"]
                + "/prerequisites.json",
            )

        with (
            patch.object(cli, "require_master"),
            patch.object(cli, "remote", side_effect=response) as call,
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(cli.UserError, "UNKNOWN"),
        ):
            cli.submit_command(self.calibration_args())
        self.assertEqual(call.call_count, 1)
        path = next((cli.state_root() / "submissions").glob("*.json"))
        intent = cli.read_json(path)
        self.assertEqual(intent["state"], "unknown")
        with patch.object(cli, "remote") as call, self.assertRaisesRegex(cli.UserError, "Unresolved"):
            cli.submit_command(self.calibration_args())
        call.assert_not_called()
        receipt = response(intent["request"])
        with (
            patch.object(cli, "remote", return_value=receipt),
            self.assertRaisesRegex(cli.UserError, "upstream"),
        ):
            cli.reconcile_command(argparse.Namespace(intent_id=path.stem))
        receipt["teacher_job_id"] = "101"
        with patch.object(cli, "remote", return_value=receipt), redirect_stdout(io.StringIO()):
            cli.reconcile_command(argparse.Namespace(intent_id=path.stem))
        self.assertEqual(cli.read_json(path)["state"], "submitted")

    def test_calibration_upstream_flags_are_forbidden_for_previous_tasks(self):
        self.record()
        for args in (self.submit_args(), self.preflight_args(), self.teacher_args()):
            args.teacher_job_id = "101"
            with (
                self.subTest(task=args.task),
                patch.object(cli, "remote") as call,
                self.assertRaisesRegex(cli.UserError, "only supported"),
            ):
                cli.submit_command(args)
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
