"""CPU-only inventory selection; immutable storage is never deleted or rewritten."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = load("sdsc_pipeline_inputs")


def record(size=1, name="file"):
    return {"storage": "/persistent/immutable/" + name, "size": size, "sha256": "a" * 64}


class PipelineInputTests(unittest.TestCase):
    def setUp(self):
        names = [
            "g0/canonical_sft/checkpoints/step-00000020.pt",
            "g0/canonical_sft/checkpoints/step-00000020.accelerate/random_states_3.pkl",
            "g0/canonical_sft/checkpoints/step-00000119.pt",
            "g0/resume-a/checkpoints/step-00000119.accelerate/optimizer.bin",
            "g0/resume-b/checkpoints/step-00000119.pt",
            "g0/teacher_demos/ledger.json",
            "g0/g0.json",
            "g0/initial_checkpoint.pt",
            "pilot/preflight4/checkpoint/model-full.pt",
            "pilot/preflight4/checkpoint/optimizer-full.pt",
            "pilot/preflight4/checkpoint/rank-3-runtime.pt",
            "pilot/preflight4/checkpoint/manifest.json",
            "pilot/preflight4/preflight.json",
            "pilot/preflight4/rank-3.json",
            "pilot/pilot_manifest.json",
            "pilot/pilot-input-admission.json",
            "pilot/inputs/teacher_demos/ledger.json",
            "pilot/inputs/teacher_demos/manifest.json",
            "pilot/inputs/teacher_demos/accepted_view.json",
            "pilot/inputs/teacher_scores/manifest.json",
            "pilot/inputs/teacher_scores/trajectories.arrow",
            "pilot/inputs/rollout_bank/manifest.json",
            "pilot/runs/canonical_sft/seed-42/checkpoints/step-00000020.pt",
            "pilot/runs/canonical_sft/seed-42/checkpoints/step-00000020.accelerate/random_states_3.pkl",
            "pilot/runs/canonical_sft/seed-42/checkpoints/step-00000119.pt",
            "pilot/runs/canonical_grpo/seed-42/checkpoints/final.pt",
            "pilot/runs/canonical_grpo/seed-42/native/checkpoint-37/optimizer.pt",
            "pilot/resume-a/checkpoints/step-00000119.pt",
            "pilot/resume-b/checkpoints/step-00000119.accelerate/random_states_3.pkl",
            "pilot/circuits/final/canonical_sft/process/challenge/circuit.json",
            "pilot/local_fork/bundle.pt",
            "pilot/local_fork/checkpoints/fork.pt",
            "pilot/dynamics/canonical_sft/process/challenge.json",
            "pilot/.sdsc-claim-teacher-shard-3",
            "pilot/.sdsc-claim-train-cell-0",
            "pilot/terminal-training.txt",
            "pilot/training_artifact_chain.json",
            "proofs/teacher-shard/submission.json",
            "proofs/preflight4/receipt.json",
        ]
        for index in range(16):
            names += [
                f"pilot/inputs/teacher_shards/{index:02d}/{name}"
                for name in ("manifest.json", "ledger.json", "accepted_view.json")
            ]
            names.append(f"pilot/sdsc-stage-teacher-shard-{index}.json")
        self.full = {name: record(number + 1, str(number)) for number, name in enumerate(names)}

    def test_all_g0_evidence_remains_for_every_stage_and_array_task(self):
        expected = {name for name in self.full if name.startswith("g0/")}
        for stage in selection.STAGES:
            indices = range(selection.ARRAY_COUNTS[stage]) if stage in selection.ARRAY_COUNTS else (None,)
            for index in indices:
                result = selection.select_inventory(self.full, stage, index)
                with self.subTest(stage=stage, index=index):
                    self.assertTrue(expected <= result.keys())

    def test_preflight_checkpoint_only_dropped_after_its_own_stage(self):
        checkpoint = {name for name in self.full if name.startswith("pilot/preflight4/checkpoint/")}
        for stage in selection.STAGES:
            result = selection.select_inventory(
                self.full, stage, 0 if stage in selection.ARRAY_COUNTS else None
            )
            with self.subTest(stage=stage):
                self.assertEqual(bool(checkpoint & result.keys()), stage in {"g0", "preflight4"})
                self.assertIn("pilot/preflight4/preflight.json", result)
                self.assertIn("pilot/preflight4/rank-3.json", result)
                self.assertIn("proofs/preflight4/receipt.json", result)

    def test_teacher_tasks_stage_only_own_shard_to_preserve_no_overwrite_guard(self):
        for index in range(16):
            result = selection.select_inventory(self.full, "teacher-shard", index)
            staged = {name for name in result if name.startswith("pilot/inputs/teacher_shards/")}
            self.assertEqual(
                staged,
                {
                    f"pilot/inputs/teacher_shards/{index:02d}/{name}"
                    for name in ("manifest.json", "ledger.json", "accepted_view.json")
                },
            )
            self.assertIn("pilot/.sdsc-claim-teacher-shard-3", result)
            self.assertEqual(
                len([name for name in result if name.startswith("pilot/sdsc-stage-teacher-shard-")]), 16
            )

    def test_merge_retains_every_accepted_and_rejected_attempt_ledger(self):
        result = selection.select_inventory(self.full, "pilot-inputs")
        self.assertEqual(
            len([name for name in result if name.startswith("pilot/inputs/teacher_shards/")]), 48
        )

    def test_training_and_later_stages_use_merged_inputs_not_shard_bytes(self):
        for stage in selection.AFTER_MERGE:
            result = selection.select_inventory(self.full, stage, "7" if stage == "train-cell" else "single")
            with self.subTest(stage=stage):
                self.assertFalse(any(name.startswith("pilot/inputs/teacher_shards/") for name in result))
                for name in self.full:
                    if name.startswith(
                        (
                            "pilot/inputs/teacher_demos/",
                            "pilot/inputs/teacher_scores/",
                            "proofs/",
                            "pilot/sdsc-stage-",
                            "pilot/.sdsc-claim-",
                        )
                    ):
                        self.assertIn(name, result)

    def test_no_guessing_final_checkpoint_or_native_dependency_from_step_number(self):
        protected = {
            name
            for name in self.full
            if name.startswith(("pilot/runs/", "pilot/resume-", "pilot/local_fork/"))
        }
        for stage in selection.STAGES:
            result = selection.select_inventory(
                self.full, stage, 0 if stage in selection.ARRAY_COUNTS else None
            )
            self.assertTrue(protected <= result.keys(), stage)

    def test_exact_component_boundaries_do_not_hide_unrelated_paths(self):
        names = (
            "pilot/preflight4/checkpoints/keep.pt",
            "pilot/preflight4/checkpoint-shadow/file",
            "pilot/preflight4/checkpoint",
            "pilot/inputs/teacher_shards-shadow/00/ledger.json",
            "pilot/inputs/teacher_shards",
            "proofs/pilot/preflight4/checkpoint/model-full.pt",
        )
        full = {name: record(name=str(index)) for index, name in enumerate(names)}
        self.assertEqual(selection.select_inventory(full, "finalize"), full)
        unknown = {"pilot/inputs/teacher_shards/unexpected/ledger.json": record()}
        self.assertEqual(selection.select_inventory(unknown, "teacher-shard", 0), unknown)

    def test_selection_does_not_mutate_or_rewrite_record_storage_or_metadata(self):
        before = json.dumps(self.full, sort_keys=True)
        selected = selection.select_inventory(self.full, "finalize")
        self.assertEqual(json.dumps(self.full, sort_keys=True), before)
        self.assertIsNot(selected, self.full)
        for name, value in selected.items():
            self.assertIs(value, self.full[name])
        self.assertGreater(len(self.full), len(selected))

    def test_exact_capacity_and_reserve_do_not_claim_to_bound_future_checkpoints(self):
        summary = selection.selection_summary(self.full, "train-cell", 0)
        selected = selection.select_inventory(self.full, "train-cell", 0)
        self.assertEqual(summary["selected_bytes"], sum(value["size"] for value in selected.values()))
        self.assertEqual(summary["full_inventory_bytes"], sum(value["size"] for value in self.full.values()))
        self.assertEqual(
            summary["excluded_bytes"], sum(value["bytes"] for value in summary["excluded_by_reason"].values())
        )
        self.assertEqual(
            summary["minimum_free_bytes_before_input_copy"], summary["selected_bytes"] + 256 * 1024**3
        )
        self.assertFalse(summary["future_output_upper_bound_established"])
        self.assertFalse(summary["durable_inventory_changed"])
        self.assertGreater(summary["retained_pilot_training_and_resume_bytes"], 0)

    def test_all_array_indices_fail_closed_and_scalar_has_no_hidden_array(self):
        for stage, count in selection.ARRAY_COUNTS.items():
            self.assertEqual(
                selection.select_inventory(self.full, stage, 0),
                selection.select_inventory(self.full, stage, "0"),
            )
            for value in (None, True, False, -1, count, "01", "1.0", "-1", "single", 0.0):
                with self.subTest(stage=stage, index=value), self.assertRaises(ValueError):
                    selection.select_inventory(self.full, stage, value)
        for value in (0, "0", False):
            with self.assertRaises(ValueError):
                selection.select_inventory(self.full, "finalize", value)
        with self.assertRaises(ValueError):
            selection.select_inventory(self.full, "arbitrary-command")

    def test_invalid_paths_rejected_even_when_they_look_like_excluded_files(self):
        for name in (
            "/pilot/file",
            "pilot/../secret",
            "pilot//file",
            "pilot/file/",
            "pilot/./file",
            "unknown/file",
            "pilot/preflight4/checkpoint/../../secret",
            "pilot/inputs/teacher_shards/00/../bad",
            "pilot/bad\x00file",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                selection.select_inventory({name: record()}, "finalize")

    def test_all_records_are_validated_including_excluded_files(self):
        name = "pilot/preflight4/checkpoint/model-full.pt"
        for key, value in (
            ("size", True),
            ("size", -1),
            ("size", "1"),
            ("sha256", "invalid"),
            ("storage", "relative/file"),
            ("storage", "/data/../secret"),
            ("storage", "//data/file"),
        ):
            bad = record()
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                selection.select_inventory({name: bad}, "finalize")

    def test_empty_success_log_and_extra_metadata_remain_byte_exact(self):
        original = {
            "pilot/stage-logs/prepare/silent.log": {
                **record(0),
                "sha256": hashlib.sha256(b"").hexdigest(),
                "original_receipt": {"job": "123"},
            }
        }
        self.assertEqual(selection.select_inventory(original, "finalize"), original)

    def test_real_storage_copy_uses_selected_original_paths_without_touching_durable_files(self):
        storage = load("sdsc_pipeline_storage")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            persistent = root / "persistent"
            persistent.mkdir()
            names = (
                "g0/resume-a/checkpoint.pt",
                "pilot/preflight4/checkpoint/model-full.pt",
                "pilot/preflight4/preflight.json",
                "pilot/inputs/teacher_shards/00/ledger.json",
                "pilot/inputs/teacher_demos/ledger.json",
                "pilot/sdsc-stage-teacher-shard-0.json",
            )
            full = {}
            for index, name in enumerate(names):
                target = persistent / str(index)
                target.write_bytes((name + "\n").encode())
                full[name] = {**storage.identity(target), "storage": str(target)}
            before = copy.deepcopy(full)
            node = root / "node"
            node.mkdir()
            roots = {role: node / role for role in ("g0", "pilot", "proofs")}
            selected = selection.select_inventory(full, "train-cell", 0)
            with (
                patch.object(storage, "PROJECT", persistent),
                patch.object(storage.shutil, "disk_usage", return_value=SimpleNamespace(free=2**50)),
            ):
                storage.stage_inventory(selected, roots)
            for name, value in full.items():
                self.assertEqual(
                    storage.identity(Path(value["storage"])), {key: value[key] for key in ("size", "sha256")}
                )
                self.assertEqual((node / name).exists(), name in selected)
                if name in selected:
                    self.assertEqual((node / name).read_bytes(), Path(value["storage"]).read_bytes())
            self.assertEqual(full, before)


if __name__ == "__main__":
    unittest.main()
