from __future__ import annotations

import hashlib
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

from posttrain_circuits.artifacts.protocol_amendments import (
    AMENDMENT_RELATIVE_PATH,
    load_protocol_amendment_bytes,
)
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import GATE_NAMES, PROFILE_NAME
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

    def test_stage_plan_is_exactly_two_gpu_and_foreground(self) -> None:
        root = Path("/scr/del6500/OPD/tmp/test-qwen3-v2-g0/qwen3-v2")
        stages = self.module._stage_plan(root, initial_checkpoint_sha256="a" * 64)
        self.assertEqual(len(stages), 19)
        self.assertEqual(stages[0].name, "build_splits")
        self.assertEqual(stages[-1].name, "finalize_g0")
        self.assertEqual(
            [stage.name for stage in stages if stage.distributed],
            ["calibration_sft", "resume_a", "resume_b"],
        )
        compare = next(stage for stage in stages if stage.name == "compare_distributed_resume")
        self.assertIn("2", compare.argv)
        self.assertEqual(self.module.GPU_COUNT, 2)
        self.assertEqual(self.module.THREADS_PER_RANK, 8)

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
        self.assertEqual(profile.process_count, 2)
        self.assertEqual(profile.cpu_cores_min, 16)
        self.assertEqual(profile.memory_mib_min, 196608)
        self.assertEqual(profile.gpu_count, 2)
        self.assertEqual(dict(handler.fixed_environment), self.module.FIXED_ENVIRONMENT)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.module.FIXED_ENVIRONMENT)
        self.assertEqual(self.module.FIXED_ENVIRONMENT["NCCL_P2P_DISABLE"], "1")
        self.assertEqual(
            self.module.FIXED_ENVIRONMENT["MIB_REPOSITORY"],
            "/scr/del6500/OPD/vendor/MIB-circuit-track-v1",
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
            workflow_id="qwen3-v2-g0-v1",
            plan_sha256="1" * 64,
            unit_id="g0",
            run_id="2" * 64,
            job_id="opd-" + "3" * 32,
            attempt=1,
            execution_profile=PROFILE_NAME,
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
        self.assertNotIn("torch.cuda", source)

    def test_two_gpu_accelerate_config_is_fixed(self) -> None:
        path = PROJECT_ROOT / "configs" / "accelerate" / "fsdp_2gpu_server_scheduler.yaml"
        payload = path.read_text(encoding="utf-8")
        self.assertIn("num_processes: 2", payload)
        self.assertIn("distributed_type: FSDP", payload)
        self.assertNotIn("num_processes: 4", payload)

    def test_two_gpu_protocol_amendment_is_proposed_and_preserves_global_batch(self) -> None:
        amendment = load_protocol_amendment_bytes(
            (PROJECT_ROOT / AMENDMENT_RELATIVE_PATH).read_bytes()
        )
        self.assertEqual(amendment["review"]["status"], "proposed")
        batch = amendment["batch_token_invariants"]
        self.assertEqual(batch["amended_gradient_accumulation_steps"], 8)
        self.assertEqual(batch["effective_global_batch_size"], 64)

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
        self.assertEqual(profile["gpu_count_policy"], "fixed")
        self.assertEqual(profile["gpu_count"], 2)
        self.assertEqual(profile["memory_mib"], 196608)
        self.assertEqual(profile["estimated_runtime_seconds"], 43200.0)


if __name__ == "__main__":
    unittest.main()
