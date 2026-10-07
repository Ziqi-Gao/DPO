"""CPU-only pilot adapter regressions; no submission, model download or GPU use."""

import argparse
import copy
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sdsc_pilot_test", ROOT / "tools/sdsc_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(
            stage="prepare",
            science_root=ROOT,
            pilot_root=self.root / "qwen3-v2/pilot",
            python=Path(sys.executable),
            run_id="test-run",
            training_job_id="123",
            hf_home=self.root / "hf",
            mib_repository=self.root / "mib",
            code_sha256="a" * 64,
        )
        self.args.pilot_root.mkdir(parents=True)
        self.inputs = {
            name: str(self.root / "qwen3-v2" / name)
            for name in (
                "dataset",
                "anti_shortcut",
                "probes",
                "readiness",
                "initial_checkpoint",
                "scored_bank",
                "teacher_demos",
                "scientific_g0",
            )
        }
        self.inputs["initial_checkpoint_sha256"] = "b" * 64

    def test_real_composer_preserves_all_eight_original_pilot_configs(self):
        from posttrain_circuits.core.config import compose_config

        api = SimpleNamespace(compose=compose_config)
        for cell in pilot.CELLS:
            config = pilot.frozen_config(self.args, self.inputs, api, cell)
            self.assertEqual(config["task"]["num_examples"], 4096)
            self.assertEqual(config["trainer"]["token_budget"], 2_000_000)
            self.assertEqual(config["trainer"]["max_steps"], 120)
            self.assertEqual(config["trainer"]["batch_size"], 8 if cell == "canonical_grpo" else 4)
            self.assertEqual(config["trainer"]["gradient_accumulation_steps"], 4)
            self.assertNotIn("batch_partition_protocol", config["trainer"])
        fork = compose_config(
            pilot.common_overrides(self.args, self.inputs, "local_fork"), config_root=ROOT / "configs"
        )
        self.assertEqual(fork["experiment"]["name"], "local_fork")

    def test_real_parsers_accept_every_generated_scientific_command(self):
        class Parsed(Exception):
            pass

        original = argparse.ArgumentParser.parse_args

        def stop_after_parse(parser, *args, **kwargs):
            original(parser, *args, **kwargs)
            raise Parsed()

        checkpoints = {cell: self.root / (cell + ".pt") for cell in pilot.CELLS}
        observed = set()
        for stage in pilot.STAGES:
            self.args.stage = stage
            for index in range(8) if stage == "train-cell" else (0,):
                for step in pilot.stage_plan(self.args, self.inputs, index=index, checkpoints=checkpoints):
                    if step.module in observed:
                        continue
                    observed.add(step.module)
                    module = importlib.import_module("posttrain_circuits.cli." + step.module)
                    with (
                        self.subTest(module=step.module),
                        patch.object(argparse.ArgumentParser, "parse_args", stop_after_parse),
                        self.assertRaises(Parsed),
                    ):
                        module.main(list(step.arguments))
        self.assertGreaterEqual(len(observed), 12)

    def test_complete_circuit_matrix_and_fixed_four_rank_training(self):
        for stage, count in (("initial-circuits", 8), ("final-circuits", 64), ("dynamics", 32)):
            self.args.stage = stage
            plan = pilot.stage_plan(
                self.args, self.inputs, checkpoints={cell: "/fixed/" + cell for cell in pilot.CELLS}
            )
            self.assertEqual(len(plan), count)
            self.assertEqual(len({step.name for step in plan}), count)
        self.args.stage = "train-cell"
        for index, cell in enumerate(pilot.CELLS):
            (step,) = pilot.stage_plan(self.args, self.inputs, index=index)
            self.assertTrue(step.distributed)
            self.assertEqual(step.module, "run_grpo" if cell == "canonical_grpo" else "train")
            command = pilot.command(self.args, step)
            self.assertEqual(command[command.index("--num_processes") + 1], "4")
        for index in (-1, 8, True, "0"):
            with self.assertRaises(ValueError):
                pilot.stage_plan(self.args, self.inputs, index=index)

    def test_teacher_array_requires_all_sixteen_real_tasks(self):
        text = "".join(f"123_{index}|COMPLETED|0:0|00:01|1K|{200 + index}\n" for index in range(16))
        pilot.terminal_records(text, "123", count=16)
        for wrong in (
            text.replace("123_15", "123_16"),
            text.replace("COMPLETED", "FAILED", 1),
            text + "123_16.batch|COMPLETED|0:0\n",
            "123|COMPLETED|0:0\n",
            text + text[: text.index("\n") + 1],
        ):
            with self.subTest(wrong=wrong[:80]), self.assertRaises(ValueError):
                pilot.terminal_records(wrong, "123", count=16)

    def test_training_requires_array_not_scalar_or_failed_step(self):
        text = "".join(f"123_{index}|COMPLETED|0:0\n123_{index}.batch|COMPLETED|0:0\n" for index in range(8))
        pilot.terminal_records(text, "123", training=True)
        for wrong in ("123|COMPLETED|0:0\n", text + "124|COMPLETED|0:0\n", text.replace("0:0", "0:9", 1)):
            with self.assertRaises(ValueError):
                pilot.terminal_records(wrong, "123", training=True)

    def test_stop_at_first_failed_scientific_process_preserves_log(self):
        self.args.stage = "prepare"
        first = pilot.Step("failure", "not-used", ())
        second = pilot.Step("must-not-run", "not-used", ())
        marker = self.root / "second-ran"

        def command(_args, step):
            if step.name == "failure":
                return [sys.executable, "-c", "print('observed failure'); raise SystemExit(7)"]
            return [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('bad')"]

        with (
            patch.object(pilot, "command", command),
            self.assertRaisesRegex(ValueError, "scientific stage failed"),
        ):
            pilot.run_steps(self.args, (first, second))
        self.assertFalse(marker.exists())
        (log,) = self.args.pilot_root.rglob("*.log")
        self.assertIn("observed failure", log.read_text())
        (journal,) = log.parent.glob("*.json")
        self.assertEqual(json.loads(journal.read_text())["returncode"], 7)

    def test_environment_preserves_cuda_assignment_and_removes_fake_scheduler(self):
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a", "SERVER_SCHEDULER_GPU_COUNT": "2", "GIT_DIR": "/bad"},
        ):
            env = pilot.child_environment(self.args)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-b,GPU-a")
        self.assertNotIn("SERVER_SCHEDULER_GPU_COUNT", env)
        self.assertNotIn("GIT_DIR", env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_resume_rejects_rng_optimizer_cursor_token_and_loss_changes(self):
        base = {
            "runtime_state_hashes": {
                name: name for name in ("rng", "model", "optimizer", "state_source_by_rank", "token_budget")
            },
            "global_step": 120,
            "loss": 1.0,
            "optimizer_state_key_type": "PARAM_NAME",
            "launcher_sha256": "a" * 64,
            "stop_reason": "max_steps",
            "accelerator_rank_rng_sha256": {str(rank): str(rank) for rank in range(4)},
        }
        evidence = {
            label: {**copy.deepcopy(base), "root": "/distinct/" + label}
            for label in ("reference", "left", "right")
        }
        report = pilot.compare_resume_evidence(evidence, {"sha256": "b" * 64}, {})
        self.assertTrue(report["passed"])
        for key in base["runtime_state_hashes"]:
            changed = copy.deepcopy(evidence)
            changed["left"]["runtime_state_hashes"][key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.compare_resume_evidence(changed, {}, {})
        for value in (1.001, float("nan"), float("inf")):
            changed = copy.deepcopy(evidence)
            changed["right"]["loss"] = value
            with self.assertRaises(ValueError):
                pilot.compare_resume_evidence(changed, {}, {})

    def test_only_bank_binding_is_adapted_other_original_guards_still_run(self):
        bank = self.root / "bank.json"
        bank.write_text("{}")
        g0_path = self.root / "g0.json"
        g0 = {"passed": True}
        g0["sha256"] = pilot.sha(pilot.canonical(g0))
        g0_path.write_text(json.dumps(g0))
        proof = {
            "g0_path": str(g0_path),
            "g0_report_sha256": pilot.file_hash(g0_path),
            "population": {"prompt_count": 4096, "bank_files": {"manifest.json": pilot.file_hash(bank)}},
        }
        proof["sha256"] = pilot.sha(pilot.canonical(proof))
        proof_path = self.root / "proof.json"
        proof_path.write_text(json.dumps(proof))
        identity = {
            "path": str(proof_path),
            "sha256": pilot.file_hash(proof_path),
            "population": proof["population"],
        }
        calls = []

        def original_guard(g0, target, *, name):
            calls.append(name)
            if name == "probe manifest":
                raise ValueError("original probe gate")

        namespace = {"_require_g0_file_binding": original_guard}
        exec(
            "def original(g0, bank):\n"
            " _require_g0_file_binding(g0, bank, name='rollout bank')\n"
            " _require_g0_file_binding(g0, bank, name='probe manifest')\n",
            namespace,
        )
        original = namespace["original"]
        adapted = pilot.adapted_finalizer(original, bank, pilot.file_hash(bank), identity)
        with self.assertRaisesRegex(ValueError, "original probe gate"):
            adapted(g0, bank)
        self.assertEqual(calls, ["probe manifest"])
        self.assertIs(original.__globals__["_require_g0_file_binding"], original_guard)
        bank.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "bank changed"):
            adapted(g0, bank)

    def test_original_candidate_rng_and_full_ledger_are_partition_invariant(self):
        from posttrain_circuits.cli.build_teacher_demos import SmokeProofTeacher
        from posttrain_circuits.datasets.teacher_demos.store import write_teacher_demo_store
        from posttrain_circuits.learning.teacher.demo_generation import (
            TeacherDemoGenerationConfig,
            generate_teacher_demonstrations,
        )
        from posttrain_circuits.utils.smoke import build_smoke_examples
        from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

        tokenizer = build_tiny_tokenizer()
        examples = build_smoke_examples(4, seed=42)
        config = TeacherDemoGenerationConfig(
            teacher_id="fixture",
            teacher_revision="fixed",
            resolved_teacher_commit="fixed",
            sampling_request_seed=31415,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            candidates_per_prompt=8,
            max_prompt_tokens=4096,
            max_new_tokens=256,
        )
        whole = generate_teacher_demonstrations(examples, tokenizer, SmokeProofTeacher(), config)
        shards = [
            generate_teacher_demonstrations(examples[i : i + 1], tokenizer, SmokeProofTeacher(), config)
            for i in range(4)
        ]
        merged = [attempt for shard in shards for attempt in shard.attempts]
        self.assertEqual([asdict(row) for row in whole.attempts], [asdict(row) for row in merged])
        self.assertEqual(sum(row.accepted for row in merged), 4)
        self.assertEqual(len(merged), 32)
        kwargs = dict(
            ordered_prompt_ids=whole.ordered_prompt_ids,
            prompt_manifest_hash=whole.prompt_manifest_hash,
            tokenizer_hash=whole.tokenizer_hash,
            generation=asdict(config),
        )
        write_teacher_demo_store(self.root / "whole", whole.attempts, **kwargs)
        write_teacher_demo_store(self.root / "merged", merged, **kwargs)
        self.assertEqual(pilot.tree_hashes(self.root / "whole"), pilot.tree_hashes(self.root / "merged"))

    def proof_fixture(self):
        def write(name, payload):
            target = self.root / name
            target.write_text(json.dumps(payload))
            return {"path": str(target), "sha256": pilot.file_hash(target)}

        receipt = {
            "job_id": "123",
            "flow_id": "flow",
            "stage": "prepare",
            "run_id": "old-run",
            "code_sha256": "c" * 64,
        }
        identity = {**receipt, "passed": True}
        inner = {
            "passed": True,
            "stage": "prepare",
            "run_id": "old-run",
            "code_sha256": "c" * 64,
            "job_id": "123",
        }
        inner_path = self.args.pilot_root / "sdsc-stage-prepare-single.json"
        inner_path.write_text(json.dumps(inner))
        report = {
            **identity,
            "allocation_job_id": "123",
            "scientific_report": {
                "path": str(inner_path),
                "sha256": pilot.file_hash(inner_path),
                "value": inner,
            },
        }
        report_ref = write("report.json", report)
        publication = {
            **identity,
            "persisted": True,
            "persistent_read_back_verified": True,
            "result_sha256": report_ref["sha256"],
            "delta": {
                "pilot/sdsc-stage-prepare-single.json": {
                    "storage": "persistent/result.json",
                    "size": inner_path.stat().st_size,
                    "sha256": pilot.file_hash(inner_path),
                }
            },
        }
        submission_ref = write("submission.json", receipt)
        record = {
            "job_id": "123",
            "submission_receipt_path": submission_ref["path"],
            "submission_receipt_sha256": submission_ref["sha256"],
            "reports": [report_ref],
            "publications": [write("publication.json", publication)],
        }
        return record, {"flow_id": "flow"}, "123|COMPLETED|0:0|00:01|1K|123\n"

    def test_publication_binds_original_bytes_inner_report_and_actual_slurm_id(self):
        record, admission, raw = self.proof_fixture()
        pilot.stage_publications(self.args, admission, record, "prepare", raw)
        with self.assertRaisesRegex(ValueError, "JobIDRaw"):
            pilot.stage_publications(self.args, admission, record, "prepare", raw.replace("|123\n", "|999\n"))
        inner = self.args.pilot_root / "sdsc-stage-prepare-single.json"
        inner.write_text('{"passed":true}')
        with self.assertRaises(ValueError):
            pilot.stage_publications(self.args, admission, record, "prepare", raw)

    def test_publication_rejects_self_reported_pass_without_persistence_or_wrong_run(self):
        for key, replacement in (
            ("persisted", False),
            ("persistent_read_back_verified", False),
            ("run_id", "unrelated"),
            ("result_sha256", "0" * 64),
        ):
            record, admission, raw = self.proof_fixture()
            target = Path(record["publications"][0]["path"])
            payload = json.loads(target.read_text())
            payload[key] = replacement
            target.write_text(json.dumps(payload))
            record["publications"][0]["sha256"] = pilot.file_hash(target)
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.stage_publications(self.args, admission, record, "prepare", raw)

    def test_publication_rejects_duplicate_report_and_raw_byte_mutation(self):
        record, admission, raw = self.proof_fixture()
        record["reports"] *= 2
        with self.assertRaises(ValueError):
            pilot.stage_publications(self.args, admission, record, "prepare", raw)
        record, admission, raw = self.proof_fixture()
        target = Path(record["submission_receipt_path"])
        target.write_text(target.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "byte hash"):
            pilot.stage_publications(self.args, admission, record, "prepare", raw)

    def preflight_fixture(self):
        report = {
            "kind": "sdsc_four_h100_pilot_preflight_v1",
            "task": "qwen3-v2-pilot-preflight",
            "passed": True,
            "world_size": 4,
            "exit_code": 0,
            "job_id": "123",
            "run_id": "preflight-run",
            "code_sha256": "c" * 64,
            "ranks": [],
            "pilot_passed": False,
            "g0_passed": False,
            "execution_class_certified": False,
            "uses_blackwell_certificate": False,
        }
        saved = {name: "a" * 64 for name in ("model", "optimizer", "scheduler", "cpu_rng", "cuda_rng")}
        for index in range(4):
            report["ranks"].append(
                {
                    "passed": True,
                    "job_id": "123",
                    "run_id": "preflight-run",
                    "code_sha256": "c" * 64,
                    "runtime": {
                        "python": "3.12.13",
                        "packages": pilot.helper("sdsc_pilot_preflight").DEPENDENCIES,
                    },
                    "gpu": {
                        "name": "NVIDIA H100",
                        "total_memory_bytes": 80 * 1024**3,
                        "compute_capability": [9, 0],
                        "logical_device": index,
                    },
                    "peak_gpu_reserved_bytes": 40 * 1024**3,
                    "nccl_all_reduce": True,
                    "finite_gradients": True,
                    "parameter_update_nonzero": True,
                    "losses": [1.0, 1.1, 1.2, 1.3],
                    "fsdp": {
                        "requested_fsdp_sharding_strategy": "FULL_SHARD",
                        "effective_fsdp_sharding_strategy": "FULL_SHARD",
                        "fsdp_wrapper_count": 29,
                    },
                    "checkpoint": {
                        "all_files_verified_before_restore": True,
                        "files": [
                            {"path": name, "size": 1, "sha256": "a" * 64}
                            for name in (
                                "model-full.pt",
                                "optimizer-full.pt",
                                *(f"rank-{number}-runtime.pt" for number in range(4)),
                            )
                        ],
                    },
                    "local_parameter_sha256_before": "a" * 64,
                    "local_parameter_sha256_after": "b" * 64,
                    "initial_cgroup_memory": {
                        "passed": True,
                        "limit_bytes": 192 * 1024**3,
                        "current_bytes": 100 * 1024**3,
                        "peak_bytes": 100 * 1024**3,
                        "headroom_bytes": 92 * 1024**3,
                        "minimum_headroom_bytes": math.ceil(0.2 * 192 * 1024**3),
                    },
                    "cgroup_memory": {
                        "passed": True,
                        "limit_bytes": 192 * 1024**3,
                        "current_bytes": 100 * 1024**3,
                        "peak_bytes": 100 * 1024**3,
                        "headroom_bytes": 92 * 1024**3,
                        "minimum_headroom_bytes": math.ceil(0.2 * 192 * 1024**3),
                    },
                    "allocation": {
                        "rank": index,
                        "local_rank": index,
                        "world_size": 4,
                        "job_id": "123",
                        "threads_per_rank": 6,
                        "cuda_visible_devices": "0,1,2,3",
                    },
                    "local_global_slots": list(range(index, 64, 4)),
                    "reserved_global_nonpadding_tokens": 98304,
                    "full_state_optimizer_scheduler_rng_restore": True,
                    "state_restore_sha256": {
                        "all_exact": True,
                        "every_category_perturbed": True,
                        "saved": saved,
                        "restored": saved,
                        "perturbed": {key: "b" * 64 for key in saved},
                    },
                }
            )
        return report

    def test_four_card_preflight_rejects_old_two_card_or_fake_certification(self):
        report = self.preflight_fixture()
        pilot.validate_preflight_report(report)
        for key, value in (
            ("world_size", 2),
            ("task", "qwen3-v2-preflight"),
            ("kind", "old"),
            ("pilot_passed", True),
            ("execution_class_certified", True),
            ("exit_code", False),
        ):
            mutated = copy.deepcopy(report)
            mutated[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_preflight_report(mutated)

    def test_four_card_preflight_checks_all_real_rank_restore_categories(self):
        report = self.preflight_fixture()
        for rank in range(4):
            for category in ("model", "optimizer", "scheduler", "cpu_rng", "cuda_rng"):
                mutated = copy.deepcopy(report)
                mutated["ranks"][rank]["state_restore_sha256"]["restored"] = dict(
                    mutated["ranks"][rank]["state_restore_sha256"]["restored"]
                )
                mutated["ranks"][rank]["state_restore_sha256"]["restored"][category] = "f" * 64
                with self.subTest(rank=rank, category=category), self.assertRaises(ValueError):
                    pilot.validate_preflight_report(mutated)

    def test_actual_checkpoint_hasher_accepts_legacy_without_g0_only_rank_counters(self):
        import torch

        payload = {key: {} for key in pilot.LEGACY_CORE}
        payload.update(
            format="accelerate_fsdp_full_export_v1",
            model={"weight": torch.tensor([1.0])},
            accelerate_state_files={"random_states_0.pkl": "a" * 64},
            accelerate_state_sha256="b" * 64,
        )
        hashes = pilot.legacy_runtime_hashes(payload)
        self.assertEqual(set(hashes), pilot.LEGACY_CORE)
        self.assertNotIn("trainer_state_by_rank", hashes)
        for key in pilot.LEGACY_CORE:
            changed = dict(payload)
            del changed[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                pilot.legacy_runtime_hashes(changed)
        with self.assertRaises(ValueError):
            pilot.legacy_runtime_hashes({**payload, "trainer_state_by_rank": []})

    def test_every_native_accelerate_rank_rng_is_read_and_compared(self):
        import numpy as np
        import torch

        directory = self.root / "accelerate"
        directory.mkdir()
        payload = {"accelerate_state_dir": str(directory), "accelerate_state_files": {}}
        values = []
        for rank in range(4):
            value = {
                "step": 120,
                "random_state": (rank,),
                "numpy_random_seed": np.arange(1024),
                "torch_manual_seed": torch.tensor([rank], dtype=torch.uint8),
                "torch_cuda_manual_seed": [torch.tensor([rank], dtype=torch.uint8)],
            }
            values.append(value)
            target = directory / f"random_states_{rank}.pkl"
            torch.save(value, target)
            payload["accelerate_state_files"][target.name] = pilot.file_hash(target)
        before = pilot.native_rank_rng(payload)
        values[3]["numpy_random_seed"][500] += 1
        target = directory / "random_states_3.pkl"
        torch.save(values[3], target)
        with self.assertRaisesRegex(ValueError, "RNG bytes changed"):
            pilot.native_rank_rng(payload)
        payload["accelerate_state_files"][target.name] = pilot.file_hash(target)
        after = pilot.native_rank_rng(payload)
        self.assertEqual(set(before), set(after))
        self.assertNotEqual(before[target.name], after[target.name])
        self.assertTrue(all(before[name] == after[name] for name in before if name != target.name))

    def test_scientific_children_inherit_outer_process_group(self):
        command = [sys.executable, "-c", "import os; print(os.getpgrp())"]
        with patch.object(pilot, "command", return_value=command):
            result = pilot.run_steps(self.args, (pilot.Step("group", "unused", ()),))
        self.assertEqual(int(Path(result[0]["log"]).read_text()), os.getpgrp())

    def test_successful_silent_original_cli_keeps_valid_empty_log(self):
        with patch.object(pilot, "command", return_value=[sys.executable, "-c", "pass"]):
            result = pilot.run_steps(self.args, (pilot.Step("silent", "unused", ()),))
        self.assertEqual(Path(result[0]["log"]).read_bytes(), b"")
        self.assertEqual(result[0]["log_sha256"], pilot.sha(b""))
        self.assertEqual(result[0]["returncode"], 0)

    def test_array_caches_are_isolated_and_preparation_proof_does_not_write_shared_terminal(self):
        self.args.stage = "teacher-shard"
        caches = []
        for index in range(16):
            with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": str(index)}):
                caches.append(pilot.child_environment(self.args)["MPLCONFIGDIR"])
        self.assertEqual(len(set(caches)), 16)
        record, admission, raw = self.proof_fixture()
        terminal = self.root / "terminal.txt"
        terminal.write_text(raw)
        record.update(
            terminal_path=str(terminal),
            terminal_sha256=pilot.file_hash(terminal),
            status={
                "job_id": "123",
                "success": True,
                "state": "COMPLETED",
                "queue": {"returncode": 0, "stdout": ""},
                "accounting": {"returncode": 0, "stdout": raw},
                "result": {"verified": True},
            },
        )
        admission["stages"] = {"prepare": record}
        pilot.stage_evidence(self.args, admission, "prepare")
        self.assertFalse((self.args.pilot_root / "terminal-prepare.txt").exists())


if __name__ == "__main__":
    unittest.main()
