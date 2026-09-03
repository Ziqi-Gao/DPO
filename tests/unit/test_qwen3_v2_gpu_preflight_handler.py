from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from posttrain_circuits.scheduler_adapter.completion import AttemptCompletionDraft
from posttrain_circuits.scheduler_adapter.gpu_preflight_request import fixed_resolved_config
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
        self.assertEqual(profile.process_count, 2)
        self.assertEqual(profile.cpu_cores_min, 16)
        self.assertEqual(profile.cpu_cores_max, 16)
        self.assertEqual(profile.gpu_count, 2)
        self.assertEqual(self.module.THREADS_PER_RANK, 8)
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

    def test_disabled_registration_proposes_exact_two_gpu_profile(self) -> None:
        with PROPOSAL.open("rb") as handle:
            proposal = tomllib.load(handle)
        self.assertIs(proposal["enabled"], False)
        task = next(
            item
            for item in proposal["tasks"]
            if item["name"] == "qwen3_v2_gpu_preflight"
        )
        self.assertEqual(
            task["execution_profiles"],
            [
                {
                    "cpu_cores_max": 16,
                    "cpu_cores_min": 16,
                    "cpu_cores_preferred": 16,
                    "cpu_scaling_efficiency": 0.0,
                    "estimated_runtime_seconds": 3600.0,
                    "gpu_count": 2,
                    "gpu_exclusivity": "required",
                    "gpu_memory_mib": 81920,
                    "gpu_models": [self.module.GPU_MODEL],
                    "gpu_utilization_pct": 95,
                    "kind": "gpu",
                    "memory_mib": 196608,
                    "name": self.module.PROFILE,
                    "resource_mode": "fixed",
                    "scheduling_goal": "latency",
                }
            ],
        )

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

    def test_optimizer_accepts_only_one_nonempty_flat_fsdp_shard(self) -> None:
        class FakeModel:
            def __init__(self, parameters: tuple[object, ...]) -> None:
                self._parameters = parameters

            def parameters(self):  # type: ignore[no-untyped-def]
                return iter(self._parameters)

        flat_parameter = SimpleNamespace(
            _is_flat_param=True,
            ndim=1,
            numel=lambda: 32,
        )
        model = FakeModel((flat_parameter,))
        self.assertEqual(
            self.module._flat_optimizer_parameters(model), (flat_parameter,)
        )

        invalid_parameters = (
            (),
            (flat_parameter, flat_parameter),
            (SimpleNamespace(_is_flat_param=False, ndim=1, numel=lambda: 32),),
            (SimpleNamespace(_is_flat_param=True, ndim=2, numel=lambda: 32),),
            (SimpleNamespace(_is_flat_param=True, ndim=1, numel=lambda: 0),),
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters), self.assertRaises(
                self.module.PreflightError
            ):
                self.module._flat_optimizer_parameters(FakeModel(parameters))

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
        calls: list[tuple[object, ...]] = []

        class FakeTensor:
            def __init__(self, value: float) -> None:
                self.value = value

            def item(self) -> float:
                return self.value

        class FakeWork:
            def wait(self, *, timeout: object) -> bool:
                calls.append(("wait", timeout))
                return True

        torch = ModuleType("torch")
        dist = ModuleType("torch.distributed")
        torch.__version__ = "2.8.0+cu128"
        torch.version = SimpleNamespace(cuda="12.8")
        torch.device = lambda kind, index: (kind, index)
        torch.tensor = lambda value, device: FakeTensor(value)
        torch.cuda = SimpleNamespace(
            device_count=lambda: 2,
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

        def new_group(*, ranks: list[int], backend: str, timeout: object) -> object:
            calls.append(("new_group", tuple(ranks), backend, timeout))
            return data_group

        def all_reduce(tensor: FakeTensor, *, group: object, async_op: bool) -> FakeWork:
            tensor.value = self.module.NCCL_EXPECTED_SUM
            calls.append(("all_reduce", group, async_op))
            return FakeWork()

        dist.new_group = new_group
        dist.all_reduce = all_reduce
        torch.distributed = dist
        environment = {
            "CUDA_VISIBLE_DEVICES": "GPU-0,GPU-1",
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "2",
        }
        with mock.patch.dict(
            sys.modules, {"torch": torch, "torch.distributed": dist}
        ), mock.patch.dict("os.environ", environment, clear=True), mock.patch.object(
            self.module, "_cuda_pci_bus_id", return_value="0000:48:00.0"
        ):
            runtime = self.module._initialize_distributed()

        self.assertIs(runtime.control_group, control_group)
        self.assertIs(runtime.data_group, data_group)
        self.assertEqual(
            runtime.nccl_diagnostic["observed_sum"], self.module.NCCL_EXPECTED_SUM
        )
        self.assertEqual(calls[1][0:2], ("init_process_group", "gloo"))
        self.assertEqual(calls[2][0:3], ("new_group", (0, 1), "nccl"))
        self.assertEqual(calls[3], ("all_reduce", data_group, True))
        self.assertEqual(
            calls[4][1].total_seconds(), self.module.NCCL_PROBE_TIMEOUT_SECONDS
        )

    def test_spawn_main_path_reopens_the_supervisor_held_script(self) -> None:
        expected = f"/proc/{os.getpid()}/fd/7"
        main_module = sys.modules["__main__"]
        with mock.patch.object(
            self.module, "_script_parent_path", return_value=expected
        ), mock.patch.object(main_module, "__file__", "/proc/self/fd/7"):
            self.assertEqual(self.module._spawn_safe_script_path(), expected)
            self.assertEqual(main_module.__file__, expected)

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
