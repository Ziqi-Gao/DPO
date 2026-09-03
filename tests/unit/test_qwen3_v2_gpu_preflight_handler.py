from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

from posttrain_circuits.scheduler_adapter.completion import AttemptCompletionDraft
from posttrain_circuits.scheduler_adapter.gpu_preflight_request import fixed_resolved_config
from posttrain_circuits.scheduler_adapter.registry import require_handler


ROOT = Path(__file__).resolve().parents[2]
HANDLER = ROOT / "scripts" / "server_scheduler" / "qwen3-v2-gpu-preflight-handler.py"
PREPARE = ROOT / "scripts" / "server_scheduler" / "prepare-qwen3-v2-runtime.py"


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
        self.assertEqual(self.module._fixed_config(), fixed_resolved_config())
        prereg = (ROOT / "prereg" / "qwen3_v2.yaml").read_bytes()
        self.assertEqual(
            self.module.PREREGISTRATION_SHA256,
            hashlib.sha256(prereg).hexdigest(),
        )

    def test_registry_fixes_offline_environment_and_single_implementation(self) -> None:
        handler = require_handler("qwen3_v2_gpu_preflight")
        self.assertEqual(dict(handler.fixed_environment), self.module.FIXED_ENVIRONMENT)
        self.assertEqual(handler.deployment.implementation, HANDLER)
        self.assertEqual(handler.deployment.runtime_flags, ("-I",))
        self.assertEqual(handler.output_names, ("gpu_preflight.json",))

    def test_completion_draft_round_trips_through_adapter_schema(self) -> None:
        digest = "a" * 64
        inputs = {
            name: chr(ord("a") + index) * 64
            for index, name in enumerate(self.module.EXPECTED_INPUT_NAMES)
        }
        invocation = self.module.Invocation(
            workflow_id="gpu-preflight",
            plan_sha256=digest,
            unit_id="gpu-preflight",
            run_id="b" * 64,
            job_id="opd-job",
            attempt=1,
            execution_profile=self.module.PROFILE,
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

    def test_runtime_preparation_requires_explicit_execute(self) -> None:
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
