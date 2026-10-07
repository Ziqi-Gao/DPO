"""CPU fixtures for SDSC real-model preflight guards; these are not GPU evidence."""

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
SCRIPT = ROOT / "tools/sdsc_training_preflight.py"
SPEC = importlib.util.spec_from_file_location("sdsc_training_preflight", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opd-training-preflight-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def allocation(self):
        return {
            "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEM_PER_NODE": "196608",
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "CUDA_VISIBLE_DEVICES": "GPU-zero,GPU-one",
        }

    def test_import_never_imports_torch_or_scientific_code(self):
        code = """
import importlib.abc, importlib.util, sys
class BlockScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {'torch','transformers','posttrain_circuits'}:
            raise AssertionError('scientific import during bootstrap: ' + name)
sys.meta_path.insert(0, BlockScientificImports())
spec = importlib.util.spec_from_file_location('preflight', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('stdlib-only')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "stdlib-only")

    def test_runtime_pin_validated_without_importing_packages(self):
        identity = worker.runtime_identity(version=worker.DEPENDENCIES.__getitem__, python_version="3.12.13")
        self.assertEqual(identity["packages"]["torch"], "2.8.0+cu128")
        with self.assertRaises(ValueError):
            worker.runtime_identity(version=worker.DEPENDENCIES.__getitem__, python_version="3.12.3")
        with self.assertRaises(ValueError):
            worker.runtime_identity(
                version=lambda name: "2.7" if name == "torch" else worker.DEPENDENCIES[name],
                python_version="3.12.13",
            )

    def test_allocation_exact_two_gpu_contract_and_visibility_preservation(self):
        env = self.allocation()
        result = worker.allocation_identity(env)
        self.assertEqual(result["cuda_visible_devices"], env["CUDA_VISIBLE_DEVICES"])
        self.assertEqual(result["threads_per_rank"], 12)
        for change in (
            {"WORLD_SIZE": "4"},
            {"LOCAL_RANK": "1"},
            {"SLURM_CPUS_PER_TASK": "4"},
            {"SLURM_MEM_PER_NODE": "16384"},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {"CUDA_VISIBLE_DEVICES": "0,0"},
            {"SLURM_JOB_ID": ""},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.allocation_identity({**env, **change})

    def build_cache(self):
        home = self.root / "huggingface"
        for name, revision in worker.MODELS.items():
            snapshot = home / "hub" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
            snapshot.mkdir(parents=True)
            for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
                (snapshot / filename).write_text("{}")
        return home

    def test_cache_requires_both_exact_pinned_snapshots(self):
        home = self.build_cache()
        metadata = worker.pinned_cache(home)
        self.assertEqual(set(metadata), set(worker.MODELS))
        first = Path(next(iter(metadata.values()))["snapshot"])
        (first / "model.safetensors").unlink()
        with self.assertRaises(ValueError):
            worker.pinned_cache(home)

    def test_cache_refuses_external_weight_symlink(self):
        home = self.build_cache()
        snapshot = next((home / "hub").glob("*/snapshots/*"))
        weight = snapshot / "model.safetensors"
        weight.unlink()
        external = self.root / "external.safetensors"
        external.write_text("weight fixture")
        weight.symlink_to(external)
        with self.assertRaises(ValueError):
            worker.pinned_cache(home)

    def test_cache_accepts_internal_blob_symlink_and_validates_all_index_shards(self):
        home = self.build_cache()
        snapshot = next((home / "hub").glob("*/snapshots/*"))
        weight = snapshot / "model.safetensors"
        weight.unlink()
        blob = snapshot.parent.parent / "blobs" / "blob"
        blob.parent.mkdir()
        blob.write_text("weight fixture")
        weight.symlink_to(blob)
        self.assertEqual(len(worker.pinned_cache(home)), 2)
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"param": "missing.safetensors"}})
        )
        with self.assertRaises(ValueError):
            worker.pinned_cache(home)

    def memory_files(self, unified=True, peak=50 * 1024**3):
        proc = self.root / "proc-cgroup"
        root = self.root / "cgroup"
        if unified:
            proc.write_text("0::/slurm/job123\n")
            directory, names = root / "slurm/job123", ("memory.max", "memory.current", "memory.peak")
        else:
            proc.write_text("5:memory:/slurm/job123\n")
            directory = root / "memory/slurm/job123"
            names = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
        directory.mkdir(parents=True)
        for name, value in zip(names, (worker.HOST_BYTES, 1024**3, peak), strict=True):
            (directory / name).write_text(str(value))
        return proc, root, directory, names

    def test_finite_v2_memory_limit_and_required_headroom(self):
        proc, root, directory, names = self.memory_files()
        report = worker.memory_envelope(proc, root)
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["headroom_bytes"], report["minimum_headroom_bytes"])
        (directory / names[0]).write_text("max")
        with self.assertRaises(ValueError):
            worker.memory_envelope(proc, root)

    def test_v1_memory_and_insufficient_headroom(self):
        proc, root, directory, names = self.memory_files(unified=False)
        proc.write_text("0::/unified-without-memory-controller\n" + proc.read_text())
        self.assertTrue(worker.memory_envelope(proc, root)["passed"])
        (directory / names[2]).write_text(str(160 * 1024**3))
        with self.assertRaises(ValueError):
            worker.memory_envelope(proc, root)

    def test_unlimited_leaf_uses_ancestor_job_limit_and_peak(self):
        proc, root, directory, names = self.memory_files()
        (directory / names[0]).write_text("max")
        for name, value in zip(names, (worker.HOST_BYTES, 20 * 1024**3, 100 * 1024**3), strict=True):
            (directory.parent / name).write_text(str(value))
        report = worker.memory_envelope(proc, root)
        self.assertEqual(report["path"], str(directory.parent))
        self.assertEqual(report["peak_bytes"], 100 * 1024**3)
        self.assertIsNone(report["ancestors"][0]["limit_bytes"])
        (directory.parent / names[0]).write_text(str(128 * 1024**3))
        with self.assertRaisesRegex(ValueError, "192 GiB"):
            worker.memory_envelope(proc, root)

    def test_v1_unlimited_leaf_and_matching_limits_use_aggregate_peak(self):
        proc, root, directory, names = self.memory_files(unified=False)
        (directory / names[0]).write_text(str(9223372036854771712))
        for name, value in zip(names, (worker.HOST_BYTES, 20 * 1024**3, 100 * 1024**3), strict=True):
            (directory.parent / name).write_text(str(value))
        report = worker.memory_envelope(proc, root)
        self.assertEqual(report["peak_bytes"], 100 * 1024**3)
        (directory / names[0]).write_text(str(worker.HOST_BYTES))
        (directory.parent / names[2]).write_text(str(160 * 1024**3))
        with self.assertRaisesRegex(ValueError, "headroom"):
            worker.memory_envelope(proc, root)

    def test_local_workspace_is_not_shared_storage_or_outside_output(self):
        work = self.root / "scratch"
        work.mkdir()
        completed = subprocess.CompletedProcess([], 0, json.dumps({"filesystems": [{"fstype": "ext4"}]}))
        with patch.object(worker.subprocess, "run", return_value=completed):
            self.assertEqual(worker.local_paths(work, work / "output")["fstype"], "ext4")
            with self.assertRaises(ValueError):
                worker.local_paths(work, self.root / "outside")
        completed.stdout = json.dumps({"filesystems": [{"fstype": "lustre"}]})
        with patch.object(worker.subprocess, "run", return_value=completed), self.assertRaises(ValueError):
            worker.local_paths(work, work / "output")

    def test_synthetic_canary_realizes_exact_shape_and_distinct_slots(self):
        tokenizer = types.SimpleNamespace(encode=lambda value, **kwargs: [ord(c) for c in value])

        def formatter(raw, *_args):
            return types.SimpleNamespace(model_facing_prompt=raw)

        rows = worker.canary_rows(tokenizer, [0, 2, 4, 6], formatter, {})
        self.assertEqual([len(row) for row in rows], [1536] * 4)
        self.assertEqual(len({tuple(row) for row in rows}), 4)

    def test_changed_world_or_code_checkpoint_rejected_before_load(self):
        header = self.checkpoint_fixture()
        worker.checkpoint_header(header, "a" * 64, 2)
        for code, world in (("b" * 64, 2), ("a" * 64, 4)):
            with self.subTest(code=code, world=world), self.assertRaises(ValueError):
                worker.checkpoint_header(header, code, world)

    def checkpoint_fixture(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        for name in worker.checkpoint_names(2):
            (checkpoint / name).write_bytes(("fixture:" + name).encode())
        header = {
            "kind": "sdsc_h100_canary_checkpoint_v1",
            "world_size": 2,
            "code_sha256": "a" * 64,
            "files": [worker.checkpoint_file(checkpoint / name) for name in worker.checkpoint_names(2)],
        }
        (checkpoint / "manifest.json").write_text(json.dumps(header))
        return header

    def test_checkpoint_hashes_bind_every_file_and_detect_same_size_corruption(self):
        header = self.checkpoint_fixture()
        checkpoint = self.root / "checkpoint"
        self.assertEqual(worker.verify_checkpoint(checkpoint, "a" * 64, 2), header)
        for record in header["files"]:
            with self.subTest(file=record["path"]):
                path = checkpoint / record["path"]
                original = path.read_bytes()
                path.write_bytes(b"!" + original[1:])
                with self.assertRaisesRegex(ValueError, "hash/size mismatch before load"):
                    worker.verify_checkpoint(checkpoint, "a" * 64, 2)
                path.write_bytes(original)

    def test_checkpoint_rejects_missing_rank_manifest_and_file_symlinks(self):
        header = self.checkpoint_fixture()
        checkpoint = self.root / "checkpoint"
        with self.assertRaisesRegex(ValueError, "every rank runtime"):
            worker.checkpoint_header({**header, "files": header["files"][:-1]}, "a" * 64, 2)
        path = checkpoint / "rank-1-runtime.pt"
        original = checkpoint / "moved.pt"
        path.rename(original)
        path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.verify_checkpoint(checkpoint, "a" * 64, 2)

    def test_rank_local_failures_are_collected_before_advancing(self):
        class FakeDistributed:
            def all_gather_object(self, output, value, group=None):
                output[:] = [value, None]

        self.assertEqual(worker.collective_phase(FakeDistributed(), lambda: 7), 7)

        def failure():
            raise OSError("missing model")

        with self.assertRaisesRegex(ValueError, "missing model"):
            worker.collective_phase(FakeDistributed(), failure)

    def test_no_clobber_publication(self):
        path = self.root / "preflight.json"
        worker.publish_json(path, {"passed": False})
        with self.assertRaises(FileExistsError):
            worker.publish_json(path, {"passed": True})
        self.assertEqual(json.loads(path.read_text()), {"passed": False})

    def test_rank_zero_publishes_control_plane_identity_without_claiming_g0(self):
        work = self.root / "scratch"
        work.mkdir()
        output = work / "output"
        home = self.build_cache()

        class FakeDistributed:
            def all_gather_object(self, reports, evidence):
                reports[:] = [evidence, {**evidence, "allocation": {**evidence["allocation"], "rank": 1}}]

            def monitored_barrier(self, **kwargs):
                pass

            def destroy_process_group(self):
                pass

            def is_initialized(self):
                return True

        with (
            patch.dict(worker.os.environ, self.allocation()),
            patch.object(worker, "runtime_identity", return_value={"python": "fixture"}),
            patch.object(worker, "memory_envelope", return_value={"passed": True}),
            patch.object(worker, "local_paths", return_value={"fstype": "ext4"}),
            patch.object(worker, "run_canary", return_value=FakeDistributed()),
            patch.object(worker.sys, "stdout", new_callable=io.StringIO),
        ):
            exit_code = worker.main(
                [
                    "--work-dir",
                    str(work),
                    "--output-dir",
                    str(output),
                    "--hf-home",
                    str(home),
                    "--run-id",
                    "test-run",
                    "--code-sha256",
                    "a" * 64,
                ]
            )
        self.assertEqual(exit_code, 0)
        result = json.loads((output / "preflight.json").read_text())
        self.assertEqual(result["task"], "qwen3-v2-preflight")
        self.assertEqual((result["run_id"], result["job_id"], result["world_size"]), ("test-run", "123", 2))
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["g0_passed"])
        self.assertFalse(result["execution_class_certified"])
        self.assertFalse(result["uses_blackwell_certificate"])


if __name__ == "__main__":
    unittest.main()
