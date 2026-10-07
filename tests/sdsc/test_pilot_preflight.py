"""CPU-only guards and orchestration fixtures; never four-GPU evidence."""

import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/sdsc_pilot_preflight.py"
SPEC = importlib.util.spec_from_file_location("pilot_preflight", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class PilotPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opd-pilot-preflight-test-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)

    def allocation(self, rank=0):
        return dict(
            SLURM_JOB_ID="123",
            SLURM_CPUS_PER_TASK="24",
            SLURM_MEM_PER_NODE="196608",
            WORLD_SIZE="4",
            LOCAL_WORLD_SIZE="4",
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            CUDA_VISIBLE_DEVICES="GPU-first,GPU-second,GPU-third,GPU-fourth",
        )

    def cache(self):
        home = self.work / "huggingface"
        for name, revision in worker.MODELS.items():
            snapshot = home / "hub" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
            snapshot.mkdir(parents=True)
            for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
                (snapshot / filename).write_text("{}")
        return home

    def args(self):
        return [
            "--science-root",
            str(self.work / "science"),
            "--work-dir",
            str(self.work),
            "--output-dir",
            str(self.work / "output"),
            "--hf-home",
            str(self.work / "huggingface"),
            "--run-id",
            "pilot-candidate",
            "--code-sha256",
            "a" * 64,
        ]

    def rank_reports(self):
        return [
            dict(
                passed=True,
                allocation=worker.allocation_identity(self.allocation(rank)),
                local_global_slots=list(range(rank, 64, 4)),
                reserved_global_nonpadding_tokens=64 * 1536,
                full_state_optimizer_scheduler_rng_restore=True,
            )
            for rank in range(4)
        ]

    def test_import_is_stdlib_only_and_does_not_mutate_two_rank_worker(self):
        code = """
import importlib.abc, importlib.util, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {'torch','transformers','posttrain_circuits','trl'}:
            raise AssertionError('heavy import during bootstrap: ' + name)
sys.meta_path.insert(0, Guard())
spec = importlib.util.spec_from_file_location('pilot', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m.WORLD_SIZE == 4 and m.shared.WORLD_SIZE == 2
assert m.DEPENDENCIES['trl'] == '0.22.2' and 'trl' not in m.shared.DEPENDENCIES
print('stdlib-only')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.assertEqual(result.stdout.strip(), "stdlib-only")

    def test_all_twenty_pins_and_python_are_checked_before_model_import(self):
        lock = json.loads((ROOT / "deployments/qwen3_v2_g0/dependency-lock.json").read_text())
        expected = {row["name"]: row["version"] for row in lock["packages"]} | {"trl": "0.22.2"}
        self.assertEqual(worker.DEPENDENCIES, expected)
        self.assertEqual(len(expected), 20)
        self.assertEqual(worker.runtime_identity(expected.__getitem__, "3.12.13")["packages"], expected)
        for name in expected:
            wrong = {**expected, name: "wrong"}
            with self.subTest(package=name), self.assertRaisesRegex(ValueError, "dependencies differ"):
                worker.runtime_identity(wrong.__getitem__, "3.12.13")
        with self.assertRaisesRegex(ValueError, "Python 3.12.13"):
            worker.runtime_identity(expected.__getitem__, "3.12.3")

    def test_exact_four_rank_allocation_preserves_uuid_order(self):
        for rank in range(4):
            identity = worker.allocation_identity(self.allocation(rank))
            self.assertEqual(identity["threads_per_rank"], 6)
            self.assertEqual(identity["cuda_visible_devices"], self.allocation()["CUDA_VISIBLE_DEVICES"])
        for values in (
            {"WORLD_SIZE": "2"},
            {"LOCAL_WORLD_SIZE": "2"},
            {"LOCAL_RANK": "1"},
            {"RANK": "4"},
            {"SLURM_CPUS_PER_TASK": "12"},
            {"SLURM_MEM_PER_NODE": "98304"},
            {"CUDA_VISIBLE_DEVICES": "0,1"},
            {"CUDA_VISIBLE_DEVICES": "0,1,2,2"},
            {"CUDA_VISIBLE_DEVICES": "0,1,2, 3"},
            {"SLURM_JOB_ID": ""},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                worker.allocation_identity({**self.allocation(), **values})

    def test_cache_is_node_local_and_both_revisions_remain_offline(self):
        home = self.cache()
        with patch.object(worker, "local_paths", return_value={"fstype": "xfs"}) as mount:
            self.assertEqual(set(worker.staged_cache(self.work, home)), set(worker.MODELS))
            self.assertEqual(mount.call_count, 2)
        with self.assertRaisesRegex(ValueError, "staged under"):
            worker.staged_cache(self.work / "different", home)
        alias = self.work / "alias"
        alias.symlink_to(home, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.staged_cache(self.work, alias)
        next(home.glob("hub/*/snapshots/*/model.safetensors")).unlink()
        with patch.object(worker, "local_paths"), self.assertRaises(ValueError):
            worker.staged_cache(self.work, home)

    def test_science_guard_checks_real_checkout_root_head_and_cleanliness(self):
        root = self.work / "science"
        directory = root / "deployments/qwen3_v2_g0"
        directory.mkdir(parents=True)
        (directory / "dependency-lock.json").write_bytes(
            (ROOT / "deployments/qwen3_v2_g0/dependency-lock.json").read_bytes()
        )

        def result(value):
            return subprocess.CompletedProcess([], 0, value + "\n", "")

        with patch.object(
            worker.subprocess, "run", side_effect=[result(str(root)), result(worker.SCIENCE_HEAD), result("")]
        ):
            self.assertTrue(worker.science_identity(root)["clean"])
        for head, status in (
            ("b" * 40, ""),
            (worker.SCIENCE_HEAD, "!! hidden.py"),
            (worker.SCIENCE_HEAD, " M src/changed.py"),
        ):
            with (
                patch.object(
                    worker.subprocess, "run", side_effect=[result(str(root)), result(head), result(status)]
                ),
                self.assertRaises(ValueError),
            ):
                worker.science_identity(root)

    def test_cgroup_ancestor_and_aggregate_peak_enforce_192g_headroom(self):
        proc = self.work / "cgroup-member"
        root = self.work / "cgroup"
        leaf = root / "job/step/task"
        leaf.mkdir(parents=True)
        proc.write_text("0::/job/step/task\n")
        (leaf / "memory.max").write_text("max")
        job = root / "job"
        for name, value in (
            ("memory.max", worker.HOST_BYTES),
            ("memory.current", 50 * 1024**3),
            ("memory.peak", 100 * 1024**3),
        ):
            (job / name).write_text(str(value))
        result = worker.memory_envelope(proc, root)
        self.assertEqual(result["path"], str(job))
        self.assertGreaterEqual(result["minimum_headroom_bytes"], 32 * 1024**3)
        (job / "memory.peak").write_text(str(160 * 1024**3))
        with self.assertRaisesRegex(ValueError, "headroom"):
            worker.memory_envelope(proc, root)

    def test_complete_checkpoint_binds_model_optimizer_and_all_four_rank_states(self):
        directory = self.work / "checkpoint"
        directory.mkdir()
        names = worker.checkpoint_names(4)
        self.assertEqual(len(names), 6)
        for name in names:
            (directory / name).write_bytes(("fixture:" + name).encode())
        header = dict(
            kind="sdsc_h100_canary_checkpoint_v1",
            world_size=4,
            code_sha256="a" * 64,
            files=[worker.checkpoint_file(directory / name) for name in names],
        )
        (directory / "manifest.json").write_text(json.dumps(header))
        self.assertEqual(worker.verify_checkpoint(directory, "a" * 64, 4), header)
        for name in names:
            file = directory / name
            original = file.read_bytes()
            file.write_bytes(b"!" + original[1:])
            with (
                self.subTest(file=name),
                self.assertRaisesRegex(ValueError, "hash/size mismatch before load"),
            ):
                worker.verify_checkpoint(directory, "a" * 64, 4)
            file.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "world size mismatch"):
            worker.verify_checkpoint(directory, "a" * 64, 2)

    def test_state_digest_includes_non_tensor_optimizer_and_scheduler_values(self):
        torch = types.SimpleNamespace(is_tensor=lambda _value: False)
        state = {"state": {0: {"step": 1}}, "param_groups": [{"lr": 0.001, "params": [0]}]}
        digest = worker.state_digest(torch, state)
        self.assertEqual(digest, worker.state_digest(torch, dict(reversed(list(state.items())))))
        changed = copy.deepcopy(state)
        changed["param_groups"][0]["lr"] = 0.002
        self.assertNotEqual(digest, worker.state_digest(torch, changed))
        self.assertNotEqual(worker.state_digest(torch, [1]), worker.state_digest(torch, (1,)))
        self.assertNotEqual(
            worker.state_digest(torch, {"a": {"b": 1}, "c": 2}),
            worker.state_digest(torch, {"a": {"b": 1, "c": 2}}),
        )
        for invalid in (float("nan"), float("inf"), object()):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                worker.state_digest(torch, invalid)

    def test_four_rank_collective_propagates_remote_failures(self):
        dist = types.SimpleNamespace(
            all_gather_object=lambda output, value, **kwargs: output.__setitem__(
                slice(None), [value, None, "rank2: failed load", None]
            )
        )
        with self.assertRaisesRegex(ValueError, "rank2: failed load"):
            worker.collective_phase(dist, lambda: "local success")

    def test_rank_replay_rejects_missing_duplicate_shards_visibility_and_failed_restore(self):
        identity = worker.allocation_identity(self.allocation())
        reports = self.rank_reports()
        self.assertEqual(worker.validate_rank_reports(reports, identity), reports)
        self.assertEqual(
            sorted(slot for row in reports for slot in row["local_global_slots"]), list(range(64))
        )
        changes = [
            lambda rows: rows.pop(),
            lambda rows: rows[1].update(local_global_slots=rows[0]["local_global_slots"]),
            lambda rows: rows[2]["allocation"].update(cuda_visible_devices="3,2,1,0"),
            lambda rows: rows[3].update(full_state_optimizer_scheduler_rng_restore=False),
            lambda rows: rows[0].update(reserved_global_nonpadding_tokens=64 * 1535),
        ]
        for change in changes:
            wrong = copy.deepcopy(reports)
            change(wrong)
            with self.assertRaises(ValueError):
                worker.validate_rank_reports(wrong, identity)

    def bootstrap(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict(worker.os.environ, self.allocation()))
        stack.enter_context(patch.object(worker, "local_paths", return_value={"fstype": "xfs"}))
        stack.enter_context(
            patch.object(worker, "runtime_identity", return_value={"packages": worker.DEPENDENCIES})
        )
        stack.enter_context(
            patch.object(
                worker, "science_identity", return_value={"head": worker.SCIENCE_HEAD, "clean": True}
            )
        )
        stack.enter_context(patch.object(worker, "staged_cache", return_value={}))
        stack.enter_context(patch.object(worker, "memory_envelope", return_value={"passed": True}))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_success_report_has_all_identities_and_no_pilot_or_certificate_claim(self):
        reports = self.rank_reports()

        class Dist:
            def all_gather_object(self, output, value, **kwargs):
                output[:] = reports if isinstance(value, dict) else [None] * 4

            def monitored_barrier(self, **kwargs):
                pass

            def is_initialized(self):
                return True

            def destroy_process_group(self):
                pass

        def canary(args, identity, evidence):
            evidence.update(reports[0])
            return Dist()

        with self.bootstrap(), patch.object(worker, "run_canary", side_effect=canary):
            self.assertEqual(worker.main(self.args()), 0)
        report = json.loads((self.work / "output/preflight.json").read_text())
        self.assertEqual(report["task"], "qwen3-v2-pilot-preflight")
        self.assertEqual((report["world_size"], report["job_id"], report["exit_code"]), (4, "123", 0))
        self.assertEqual(report["run_id"], "pilot-candidate")
        self.assertEqual(report["code_sha256"], "a" * 64)
        self.assertTrue(report["passed"])
        for name in ("g0_passed", "pilot_passed", "execution_class_certified", "uses_blackwell_certificate"):
            self.assertFalse(report[name])
        with self.bootstrap(), patch.object(worker, "run_canary") as run, self.assertRaises(FileExistsError):
            worker.main(self.args())
        run.assert_not_called()

    def test_compute_failure_publishes_diagnostics_without_success_or_retry(self):
        with (
            self.bootstrap(),
            patch.object(worker, "run_canary", side_effect=ValueError("NCCL failed")),
            self.assertRaisesRegex(ValueError, "NCCL failed"),
        ):
            worker.main(self.args())
        report = json.loads((self.work / "output/rank-0-failure.json").read_text())
        self.assertFalse(report["passed"])
        self.assertFalse((self.work / "output/preflight.json").exists())
        self.assertIn("NCCL failed", report["error"])
        self.assertIn("TRL_GRPO_training", report["not_tested"])


if __name__ == "__main__":
    unittest.main()
