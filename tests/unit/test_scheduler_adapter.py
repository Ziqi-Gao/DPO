from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from posttrain_circuits.artifacts import git_provenance
from posttrain_circuits.artifacts.completion import (
    ExecutionIdentity,
    ScientificCompletion,
    write_completion_marker,
)
from posttrain_circuits.artifacts.config_bindings import ConfigBinding, bind_config
from posttrain_circuits.artifacts.hashing import canonical_json, sha256_file, sha256_value
from posttrain_circuits.scheduler_adapter.completion import (
    AttemptCompletionDraft,
    execution_identity,
    publish_output_attempt,
    validate_unit_completion,
)
from posttrain_circuits.scheduler_adapter.content_store import (
    ATTEMPT_COMPLETION_NAME,
    ContentStore,
    tree_content_sha256,
)
from posttrain_circuits.scheduler_adapter.config_resolver import (
    ConfigBindingResolver as CasConfigBindingResolver,
)
from posttrain_circuits.scheduler_adapter.dispatch import (
    propagate_child_signal,
    run_foreground_child,
)
from posttrain_circuits.scheduler_adapter.environment import (
    THREAD_ENVIRONMENT_KEYS,
    RuntimeEnvelope,
    configure_thread_environment,
    validate_scheduler_environment,
)
from posttrain_circuits.scheduler_adapter.errors import (
    AdapterValidationError,
    DispatchError,
)
from posttrain_circuits.scheduler_adapter.manifest import (
    RunningManifest,
    hold_running_manifest,
    load_running_manifest,
)
from posttrain_circuits.scheduler_adapter.outbox import (
    build_outbox_request,
    prepare_outbox_request,
    validate_outbox_request,
)
from posttrain_circuits.scheduler_adapter.paths import (
    WorkflowLayout,
    validate_path_chain,
)
from posttrain_circuits.scheduler_adapter.plan_store import (
    publish_workflow_plan,
    require_published_workflow_plan,
    resolve_workflow_unit,
)
from posttrain_circuits.scheduler_adapter.registry import (
    CONFIG_HASH_CONTENT_INPUTS,
    HANDLER_REGISTRY,
    DeploymentContract,
    ExecutionProfileContract,
    HandlerSpec,
)
from posttrain_circuits.scheduler_adapter.secure_files import (
    HeldRegularFile,
    published_json_bytes,
)
from posttrain_circuits.scheduler_adapter.runtime import (
    build_child_environment,
    execute_validated_unit,
)
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json
from posttrain_circuits.workflows.contracts import (
    ContentIdentity,
    WorkflowPlan,
    WorkflowUnit,
    unit_identity_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def _digest(name: str) -> str:
    return hashlib.sha256(_content_bytes(name)).hexdigest()


def _content_bytes(name: str) -> bytes:
    return f"immutable fixture content: {name}\n".encode("utf-8")


def _config_fixture() -> tuple[
    ConfigBinding,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    resolved: dict[str, object] = {
        "model": {"name": "fixture"},
        "seed": 42,
    }
    input_hashes = {"dataset": _digest("dataset")}
    execution_context = {"launcher": "server-scheduler-fixture"}
    binding = bind_config(
        resolved,
        input_artifact_hashes=input_hashes,
        execution_context=execution_context,
    )
    scientific: dict[str, object] = {
        "schema_version": 3,
        "config": resolved,
        "input_artifact_hashes": input_hashes,
    }
    execution: dict[str, object] = {
        "schema_version": 3,
        "storage_locators": {},
        "execution_context": execution_context,
    }
    return binding, resolved, scientific, execution


def _semantic_validator(payload: dict[str, object], context: object) -> None:
    if payload.get("task") != getattr(context, "task"):
        raise ValueError("semantic task mismatch")


def _plan() -> WorkflowPlan:
    binding, _resolved, _scientific, _execution = _config_fixture()
    identities = {
        "config_binding_sha256": sha256_value(binding.as_dict()),
        "execution_config_sha256": binding.execution_config_sha256,
        "experiment_binding_sha256": _digest("experiment_binding_sha256"),
        "resolved_config_sha256": binding.resolved_config_sha256,
        "scientific_config_sha256": binding.scientific_config_sha256,
    }
    return WorkflowPlan(
        workflow_id="pilot",
        units=(
            WorkflowUnit(
                unit_id="cell",
                task="offline_hard",
                content_inputs=tuple(
                    ContentIdentity(name=name, sha256=digest, kind="file")
                    for name, digest in sorted(identities.items())
                ),
                dependencies=(),
                output_names=("result.json",),
            ),
        ),
    )


def _running_payload(root: Path, *, gpu: bool = False) -> dict[str, object]:
    allocation = {
        "cpu_cores": 2,
        "memory_mib": 4096,
        "gpu_count": 2 if gpu else 0,
        "gpu_memory_mib": 24000 if gpu else 0,
        "gpu_utilization_pct": 90 if gpu else 0,
        "exclusive_gpu": True,
    }
    return {
        "schema_version": 2,
        "job_id": "opd-job-1",
        "project": "OPD",
        "task": "offline_hard",
        "resources": None,
        "allocation": allocation,
        "parameters": {
            "workflow_id": "pilot",
            "plan_sha256": _plan().sha256(),
            "unit_id": "cell",
        },
        "priority": 0,
        "submitted_at": "2026-09-01T12:00:00Z",
        "state": "running",
        "updated_at": "2026-09-01T12:01:00Z",
        "requested_profile": None,
        "execution_profile": "gpu-reviewed" if gpu else "cpu-reviewed",
        "estimated_runtime_seconds": 60.125,
        "gpu_indices": [1, 3] if gpu else [],
        "gpu_uuids": ["GPU-aaaa", "GPU-bbbb"] if gpu else [],
        "gpu_pci_bus_ids": ["0000:41:00.0", "0000:81:00.0"] if gpu else [],
        "cpu_ids": [4, 5],
        "numa_node": None,
        "stdout_log": str(root / "job.stdout.log"),
        "stderr_log": str(root / "job.stderr.log"),
        "exit_code": None,
        "failure_reason": None,
    }


def _manifest(payload: dict[str, object]) -> RunningManifest:
    return RunningManifest.from_payload(payload, manifest_sha256="a" * 64)


def _environment(
    manifest: RunningManifest, manifest_path: Path, *, gpu: bool
) -> dict[str, str]:
    result = {
        "SERVER_SCHEDULER_JOB_ID": manifest.job_id,
        "SERVER_SCHEDULER_PROJECT": manifest.project,
        "SERVER_SCHEDULER_JOB_MANIFEST": str(manifest_path),
        "SERVER_SCHEDULER_LEASE_ID": "lease-opd-1",
        "SERVER_SCHEDULER_ATTEMPT": "1",
        "SERVER_SCHEDULER_EXECUTION_PROFILE": manifest.execution_profile,
        "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS": (
            f"{manifest.estimated_runtime_seconds:.6f}"
        ),
        "SERVER_SCHEDULER_CPU_CORES": str(manifest.allocation.cpu_cores),
        "SERVER_SCHEDULER_CPUSET": ",".join(str(value) for value in manifest.cpu_ids),
        "SERVER_SCHEDULER_MEMORY_MIB": str(manifest.allocation.memory_mib),
        "SERVER_SCHEDULER_GPU_COUNT": str(manifest.allocation.gpu_count),
        "SERVER_SCHEDULER_GPU_MEMORY_MIB": str(manifest.allocation.gpu_memory_mib),
        "SERVER_SCHEDULER_GPU_UTILIZATION_PCT": str(
            manifest.allocation.gpu_utilization_pct
        ),
        "SERVER_SCHEDULER_GPU_EXCLUSIVITY": "exclusive",
        "SERVER_SCHEDULER_GPU_INDICES": ",".join(
            str(value) for value in manifest.gpu_indices
        ),
        "SERVER_SCHEDULER_GPU_UUIDS": ",".join(manifest.gpu_uuids),
        "SERVER_SCHEDULER_GPU_PCI_BUS_IDS": ",".join(manifest.gpu_pci_bus_ids),
        "SERVER_SCHEDULER_STDOUT_LOG": manifest.stdout_log,
        "SERVER_SCHEDULER_STDERR_LOG": manifest.stderr_log,
    }
    if gpu:
        result["CUDA_VISIBLE_DEVICES"] = ",".join(manifest.gpu_uuids)
        result["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return result


class SchedulerAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".scheduler-adapter-test-", dir=PROJECT_ROOT
        )
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.data_root = self.root / "data"
        self.scratch_root = self.root / "scratch"
        for path in (self.code_root, self.data_root, self.scratch_root):
            path.mkdir()
        self.layout = WorkflowLayout(
            code_root=self.code_root,
            data_root=self.data_root,
            scratch_root=self.scratch_root,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self, payload: dict[str, object]) -> Path:
        path = self.root / "running.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _hold_manifest(
        self, payload: dict[str, object]
    ) -> tuple[RunningManifest, RuntimeEnvelope, HeldRegularFile]:
        manifest, held = hold_running_manifest(self._write_manifest(payload))
        return manifest, self._envelope(manifest), held

    def _publish_plan(self) -> tuple[WorkflowPlan, Path]:
        plan = _plan()
        path = publish_workflow_plan(plan, layout=self.layout)
        store = ContentStore(self.layout)
        binding, resolved, scientific, execution = _config_fixture()
        materialized = CasConfigBindingResolver(store).materialize(
            binding,
            resolved_config=resolved,
            scientific_config=scientific,
            execution_config=execution,
        )
        expected_configs = {
            identity.name: identity for identity in materialized.as_tuple()
        }
        observed_configs = {
            identity.name: identity
            for identity in plan.unit("cell").content_inputs
            if identity.name != "experiment_binding_sha256"
        }
        self.assertEqual(observed_configs, expected_configs)
        experiment = next(
            identity
            for identity in plan.unit("cell").content_inputs
            if identity.name == "experiment_binding_sha256"
        )
        store.publish_file(
            sha256=experiment.sha256,
            raw=_content_bytes("experiment_binding_sha256"),
        )
        return plan, path

    def _handler(self, manifest: RunningManifest) -> HandlerSpec:
        executable = self.scratch_root / "envs" / "reviewed" / "bin" / "python"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b"reviewed dependency-light python fixture\n")
        executable.chmod(0o750)
        implementation = self.code_root / "handlers" / "offline_hard.pyz"
        implementation.parent.mkdir(parents=True, exist_ok=True)
        implementation.write_text("# self-contained reviewed handler fixture\n", encoding="utf-8")
        dependency_lock = self.scratch_root / "envs" / "reviewed" / "dependency.lock"
        dependency_lock.write_text("fixture==1.0 --hash=sha256:reviewed\n", encoding="utf-8")
        package_manifest = self.code_root / "handlers" / "offline_hard.package.json"
        package_payload = {
            "dependency_lock_sha256": sha256_file(dependency_lock),
            "deployment_id": "offline-hard-fixture",
            "executable_sha256": sha256_file(executable),
            "implementation_sha256": sha256_file(implementation),
            "runtime_flags": ["-I", "-S"],
            "runtime_version": "Python-3.12.13-fixture",
            "schema_version": 1,
            "self_contained": True,
        }
        package_manifest.write_text(
            json.dumps(package_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        deployment = DeploymentContract(
            deployment_id="offline-hard-fixture",
            runtime_version="Python-3.12.13-fixture",
            runtime_flags=("-I", "-S"),
            executable=executable,
            executable_sha256=sha256_file(executable),
            implementation=implementation,
            implementation_sha256=sha256_file(implementation),
            dependency_lock=dependency_lock,
            dependency_lock_sha256=sha256_file(dependency_lock),
            package_manifest=package_manifest,
            package_manifest_sha256=sha256_file(package_manifest),
            deployment_identity_sha256="0" * 64,
        )
        deployment = replace(
            deployment,
            deployment_identity_sha256=sha256_value(deployment.identity_payload()),
        )
        if manifest.allocation.gpu_count:
            profile = ExecutionProfileContract(
                name=manifest.execution_profile,
                kind="gpu",
                process_count=2,
                cpu_cores_min=2,
                cpu_cores_max=2,
                memory_mib_min=4096,
                memory_mib_max=4096,
                gpu_count=2,
                gpu_memory_mib_min=24000,
                gpu_memory_mib_max=24000,
                gpu_utilization_pct_min=90,
                gpu_utilization_pct_max=90,
                exclusive_gpu=True,
                allowed_gpu_models=("Reviewed-GPU",),
            )
        else:
            profile = ExecutionProfileContract(
                name=manifest.execution_profile,
                kind="cpu",
                process_count=1,
                cpu_cores_min=2,
                cpu_cores_max=2,
                memory_mib_min=4096,
                memory_mib_max=4096,
                gpu_count=0,
                gpu_memory_mib_min=0,
                gpu_memory_mib_max=0,
                gpu_utilization_pct_min=0,
                gpu_utilization_pct_max=0,
                exclusive_gpu=True,
                allowed_gpu_models=(),
            )
        return HandlerSpec(
            task=manifest.task,
            deployment=deployment,
            profiles=MappingProxyType({profile.name: profile}),
            fixed_args=("--fixed-protocol", "v1"),
            fixed_environment=MappingProxyType({"HANDLER_PROTOCOL": "v1"}),
            cwd=self.code_root,
            output_names=("result.json",),
            required_gate_names=("domain_gate",),
            config_hash_bindings=CONFIG_HASH_CONTENT_INPUTS,
            semantic_validator_id="fixture-semantic-validator-v1",
            semantic_validator=_semantic_validator,
        )

    def _envelope(self, manifest: RunningManifest) -> RuntimeEnvelope:
        return RuntimeEnvelope(
            job_id=manifest.job_id,
            attempt=1,
            execution_profile=manifest.execution_profile,
            manifest_path=self.root / "running.json",
            manifest_sha256=manifest.manifest_sha256,
            allocation_sha256=sha256_value(manifest.allocation_payload()),
        )

    def _thread_environment(self) -> dict[str, str]:
        return {key: "2" for key in THREAD_ENVIRONMENT_KEYS}

    def _registry(self, handler: HandlerSpec) -> MappingProxyType:
        return MappingProxyType({handler.task: handler})

    def _write_completion(
        self,
        plan: WorkflowPlan,
        *,
        execution: ExecutionIdentity | None | object = ...,
        config_sha256: str | None = None,
        scientific_validation: dict[str, bool] | None = None,
    ) -> Path:
        plan_sha256 = plan.sha256()
        self.layout.prepare_attempt_parents(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id="cell",
        )
        self.layout.output_directory(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id="cell",
        ).mkdir()
        output = self.layout.output_path(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id="cell",
            output_name="result.json",
        )
        output.write_text('{"passed":true}\n', encoding="utf-8")
        marker_path = self.layout.completion_path(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id="cell",
        )
        if execution is ...:
            execution = execution_identity(self._envelope(_manifest(_running_payload(self.root))))
        assert execution is None or isinstance(execution, ExecutionIdentity)
        binding, _resolved, _scientific, _execution = _config_fixture()
        scientific_config = (
            binding.scientific_config_sha256
            if config_sha256 is None
            else config_sha256
        )
        marker = ScientificCompletion(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id="cell",
            task="offline_hard",
            run_id=unit_identity_sha256(
                workflow_id=plan.workflow_id,
                plan_sha256=plan_sha256,
                unit_id="cell",
            ),
            started_at="2026-09-01T12:00:00Z",
            completed_at="2026-09-01T12:01:00Z",
            scientific_config_sha256=scientific_config,
            execution_config_sha256=binding.execution_config_sha256,
            resolved_config_sha256=binding.resolved_config_sha256,
            input_hashes={
                identity.name: identity.sha256
                for identity in plan.unit("cell").content_inputs
            },
            output_files={str(output): sha256_file(output)},
            scientific_validation=(
                {"domain_gate": True}
                if scientific_validation is None
                else scientific_validation
            ),
            execution=execution,
        )
        write_completion_marker(
            marker_path, marker, approved_roots=(self.data_root, self.scratch_root)
        )
        return marker_path

    def _write_attempt_result(
        self,
        plan: WorkflowPlan,
        envelope: RuntimeEnvelope,
        *,
        extra_name: str | None = None,
    ) -> None:
        attempt_path = self.layout.attempt_directory(
            workflow_id=plan.workflow_id,
            plan_sha256=plan.sha256(),
            unit_id="cell",
            job_id=envelope.job_id,
            attempt=envelope.attempt,
        )
        (attempt_path / "result.json").write_text(
            '{"passed":true}\n', encoding="utf-8"
        )
        if extra_name is not None:
            (attempt_path / extra_name).write_text("unexpected\n", encoding="utf-8")
        binding, _resolved, _scientific, _execution = _config_fixture()
        draft = AttemptCompletionDraft(
            workflow_id=plan.workflow_id,
            plan_sha256=plan.sha256(),
            unit_id="cell",
            task="offline_hard",
            run_id=unit_identity_sha256(
                workflow_id=plan.workflow_id,
                plan_sha256=plan.sha256(),
                unit_id="cell",
            ),
            started_at="2026-09-01T12:00:00Z",
            completed_at="2026-09-01T12:01:00Z",
            scientific_config_sha256=binding.scientific_config_sha256,
            execution_config_sha256=binding.execution_config_sha256,
            resolved_config_sha256=binding.resolved_config_sha256,
            input_hashes={
                identity.name: identity.sha256
                for identity in plan.unit("cell").content_inputs
            },
            scientific_validation={"domain_gate": True},
            execution=execution_identity(envelope),
        )
        (attempt_path / ATTEMPT_COMPLETION_NAME).write_bytes(
            published_json_bytes(draft.to_payload())
        )

    def test_strict_json_rejects_duplicate_nan_utf8_symlink_and_nonregular(self):
        path = self.root / "strict.json"
        path.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaisesRegex(AdapterValidationError, "duplicate"):
            read_strict_json(path, context="fixture")
        path.write_text('{"a":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(AdapterValidationError, "non-finite"):
            read_strict_json(path, context="fixture")
        path.write_bytes(b'\xff')
        with self.assertRaisesRegex(AdapterValidationError, "UTF-8"):
            read_strict_json(path, context="fixture")
        target = self.root / "target.json"
        target.write_text("{}", encoding="utf-8")
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(AdapterValidationError):
            read_strict_json(path, context="fixture")
        with self.assertRaisesRegex(AdapterValidationError, "regular file"):
            read_strict_json(self.code_root, context="fixture")

        ancestor = self.root / "strict-real"
        ancestor.mkdir()
        (ancestor / "payload.json").write_text("{}", encoding="utf-8")
        linked_ancestor = self.root / "strict-linked"
        linked_ancestor.symlink_to(ancestor, target_is_directory=True)
        with self.assertRaises(AdapterValidationError):
            read_strict_json(linked_ancestor / "payload.json", context="fixture")

        fifo = self.root / "blocking.json"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(AdapterValidationError, "regular file"):
            # The no-follow O_PATH metadata check rejects the FIFO before any
            # read open, so this assertion must not wait for a writer.
            read_strict_json(fifo, context="fixture")

    def test_running_manifest_accepts_only_exact_protocol_v2_shape(self):
        payload = _running_payload(self.root)
        path = self._write_manifest(payload)
        manifest = load_running_manifest(path)
        self.assertEqual(manifest.project, "OPD")
        self.assertEqual(manifest.state, "running")
        self.assertEqual(manifest.manifest_sha256, hashlib.sha256(path.read_bytes()).hexdigest())

        for mutation in ("extra", "wrong-state", "extra-parameter"):
            invalid = json.loads(json.dumps(payload))
            if mutation == "extra":
                invalid["command"] = "/bin/sh"
            elif mutation == "wrong-state":
                invalid["state"] = "pending"
            else:
                invalid["parameters"]["env"] = {"CUDA_VISIBLE_DEVICES": "0"}
            with self.subTest(mutation=mutation), self.assertRaises(AdapterValidationError):
                RunningManifest.from_payload(invalid, manifest_sha256="a" * 64)

    def test_manifest_rejects_incoherent_cpu_and_gpu_allocations(self):
        cpu = _running_payload(self.root)
        cpu["cpu_ids"] = [4]
        with self.assertRaisesRegex(AdapterValidationError, "cpu_ids length"):
            _manifest(cpu)
        gpu = _running_payload(self.root, gpu=True)
        gpu["gpu_uuids"] = ["GPU-aaaa"]
        with self.assertRaisesRegex(AdapterValidationError, "UUID"):
            _manifest(gpu)

    def test_cpu_environment_cross_check_requires_empty_visibility(self):
        manifest = _manifest(_running_payload(self.root))
        manifest_path = self.root / "running.json"
        environment = _environment(manifest, manifest_path, gpu=False)
        envelope = validate_scheduler_environment(
            manifest, manifest_path=manifest_path, environ=environment
        )
        self.assertEqual(envelope.job_id, manifest.job_id)
        self.assertFalse(hasattr(envelope, "lease_id"))
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        with self.assertRaisesRegex(AdapterValidationError, "CPU-only"):
            validate_scheduler_environment(
                manifest, manifest_path=manifest_path, environ=environment
            )
        environment = _environment(manifest, manifest_path, gpu=False)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        with self.assertRaisesRegex(AdapterValidationError, "CPU-only"):
            validate_scheduler_environment(
                manifest, manifest_path=manifest_path, environ=environment
            )

    def test_gpu_environment_accepts_only_scheduler_uuid_order_without_mutation(self):
        manifest = _manifest(_running_payload(self.root, gpu=True))
        manifest_path = self.root / "running.json"
        environment = _environment(manifest, manifest_path, gpu=True)
        before = environment["CUDA_VISIBLE_DEVICES"]
        validate_scheduler_environment(
            manifest, manifest_path=manifest_path, environ=environment
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], before)
        environment["CUDA_VISIBLE_DEVICES"] = "1,3"
        with self.assertRaisesRegex(AdapterValidationError, "scheduler-provided GPU UUID"):
            validate_scheduler_environment(
                manifest, manifest_path=manifest_path, environ=environment
            )
        environment = _environment(manifest, manifest_path, gpu=True)
        environment["CUDA_VISIBLE_DEVICES"] = "GPU-bbbb,GPU-aaaa"
        with self.assertRaises(AdapterValidationError):
            validate_scheduler_environment(
                manifest, manifest_path=manifest_path, environ=environment
            )

    def test_thread_environment_is_explicit_per_rank(self):
        environment = {key: "999" for key in THREAD_ENVIRONMENT_KEYS}
        self.assertEqual(
            configure_thread_environment(
                cpu_cores=8, process_count=4, environ=environment
            ),
            2,
        )
        self.assertEqual({environment[key] for key in THREAD_ENVIRONMENT_KEYS}, {"2"})
        with self.assertRaisesRegex(AdapterValidationError, "divide equally"):
            configure_thread_environment(
                cpu_cores=7, process_count=4, environ=environment
            )

    def test_path_confinement_rejects_outside_and_symlink_ancestors(self):
        nested = self.data_root / "safe" / "file.json"
        nested.parent.mkdir()
        nested.write_text("{}", encoding="utf-8")
        self.assertEqual(
            validate_path_chain(
                nested, approved_roots=(self.data_root,), final_kind="file"
            ),
            nested,
        )
        with self.assertRaisesRegex(AdapterValidationError, "outside"):
            validate_path_chain(
                self.code_root / "outside.json",
                approved_roots=(self.data_root,),
                final_kind="file",
            )
        real = self.data_root / "real"
        real.mkdir()
        symlink = self.data_root / "linked"
        symlink.symlink_to(real, target_is_directory=True)
        via_symlink = symlink / "file.json"
        (real / "file.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(AdapterValidationError):
            validate_path_chain(
                via_symlink, approved_roots=(self.data_root,), final_kind="file"
            )

    def test_plan_publication_is_no_clobber_strict_and_required_by_resolver(self):
        plan, path = self._publish_plan()
        self.assertEqual(
            require_published_workflow_plan(plan, layout=self.layout), path
        )
        self.assertEqual(publish_workflow_plan(plan, layout=self.layout), path)
        manifest = _manifest(_running_payload(self.root))
        resolved = resolve_workflow_unit(manifest, layout=self.layout)
        self.assertEqual(resolved.plan, plan)
        self.assertEqual(resolved.unit.unit_id, "cell")

        # Identical JSON content with a different byte format is not accepted as
        # an artifact produced by the no-clobber publisher.
        path.write_text(json.dumps(plan.to_payload()), encoding="utf-8")
        with self.assertRaisesRegex(AdapterValidationError, "bytes differ"):
            require_published_workflow_plan(plan, layout=self.layout)

    def test_plan_resolver_binds_task_and_hash_from_exact_parameters(self):
        plan, _path = self._publish_plan()
        wrong_task = _running_payload(self.root)
        wrong_task["task"] = "canonical_grpo"
        with self.assertRaisesRegex(AdapterValidationError, "task does not match"):
            resolve_workflow_unit(_manifest(wrong_task), layout=self.layout)
        self.assertEqual(plan.workflow_id, "pilot")

    def test_content_store_rejects_missing_extra_tamper_and_symlink(self):
        store = ContentStore(self.layout)
        with self.assertRaises(AdapterValidationError):
            store.open(sha256="0" * 64, kind="file", name="missing")

        raw = _content_bytes("tamper")
        digest = hashlib.sha256(raw).hexdigest()
        file_path = store.publish_file(sha256=digest, raw=raw)
        file_path.write_bytes(b"tampered\n")
        with self.assertRaisesRegex(AdapterValidationError, "hash mismatch"):
            store.open(sha256=digest, kind="file", name="tampered")

        extra_files = {"a.bin": b"a\n", "b.json": b"b\n"}
        extra_digest = tree_content_sha256(extra_files)
        extra_path = store.publish_tree(sha256=extra_digest, files=extra_files)
        extra_path.chmod(0o750)
        (extra_path / "unexpected.bin").write_bytes(b"extra\n")
        with self.assertRaisesRegex(AdapterValidationError, "missing or extra"):
            store.open(sha256=extra_digest, kind="tree", name="extra_tree")

        linked_files = {"payload.bin": b"reviewed\n"}
        linked_digest = tree_content_sha256(linked_files)
        linked_path = store.publish_tree(sha256=linked_digest, files=linked_files)
        linked_path.chmod(0o750)
        payload_path = linked_path / "payload.bin"
        payload_path.unlink()
        payload_path.symlink_to(file_path)
        with self.assertRaisesRegex(AdapterValidationError, "regular file"):
            store.open(sha256=linked_digest, kind="tree", name="linked_tree")

    def test_config_binding_resolver_requires_canonical_strict_binding_bytes(self):
        store = ContentStore(self.layout)
        binding, resolved, scientific, execution = _config_fixture()
        CasConfigBindingResolver(store).materialize(
            binding,
            resolved_config=resolved,
            scientific_config=scientific,
            execution_config=execution,
        )
        experiment_digest = _digest("experiment_binding_sha256")
        store.publish_file(
            sha256=experiment_digest,
            raw=_content_bytes("experiment_binding_sha256"),
        )
        mutations = {
            "noncanonical": json.dumps(
                binding.as_dict(), indent=2, sort_keys=True
            ).encode("utf-8"),
            "extra-field": canonical_json(
                {**binding.as_dict(), "unexpected": True}
            ).encode("utf-8"),
        }
        for mutation, raw in mutations.items():
            with self.subTest(mutation=mutation):
                binding_digest = hashlib.sha256(raw).hexdigest()
                store.publish_file(sha256=binding_digest, raw=raw)
                base = _plan()
                unit = base.unit("cell")
                identities = tuple(
                    replace(identity, sha256=binding_digest)
                    if identity.name == "config_binding_sha256"
                    else identity
                    for identity in unit.content_inputs
                )
                plan = WorkflowPlan(
                    workflow_id=f"binding-{mutation}",
                    units=(replace(unit, content_inputs=identities),),
                )
                payload = _running_payload(self.root)
                payload["parameters"]["workflow_id"] = plan.workflow_id
                payload["parameters"]["plan_sha256"] = plan.sha256()
                manifest = _manifest(payload)
                with store.open_inputs(identities) as handles, self.assertRaises(
                    AdapterValidationError
                ):
                    CasConfigBindingResolver(store).resolve(
                        manifest=manifest,
                        plan=plan,
                        unit=plan.unit("cell"),
                        content_handles=handles.handles,
                    )

    def test_handler_registration_rejects_aliased_config_identities(self):
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        unit = _plan().unit("cell")
        resolved_digest = next(
            identity.sha256
            for identity in unit.content_inputs
            if identity.name == "resolved_config_sha256"
        )
        aliased = replace(
            unit,
            content_inputs=tuple(
                replace(identity, sha256=resolved_digest)
                if identity.name == "execution_config_sha256"
                else identity
                for identity in unit.content_inputs
            ),
        )
        with self.assertRaisesRegex(AdapterValidationError, "must not alias"):
            handler.validate_unit_contract(aliased)

    def test_scientific_completion_binds_deterministic_inputs_and_outputs(self):
        plan, _path = self._publish_plan()
        marker_path = self._write_completion(plan)
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        registry = self._registry(handler)
        payload = validate_unit_completion(
            plan,
            plan.unit("cell"),
            plan_sha256=plan.sha256(),
            layout=self.layout,
            handler_registry=registry,
        )
        self.assertEqual(payload["unit_id"], "cell")
        output = next(iter(payload["output_files"]))
        Path(output).write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(AdapterValidationError, "hash mismatch"):
            validate_unit_completion(
                plan,
                plan.unit("cell"),
                plan_sha256=plan.sha256(),
                layout=self.layout,
                handler_registry=registry,
            )
        self.assertNotIn("lease", marker_path.read_text(encoding="utf-8"))

    def test_scientific_completion_requires_execution_and_code_owned_semantics(self):
        plan, _path = self._publish_plan()
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        registry = self._registry(handler)
        self._write_completion(plan, execution=None)
        with self.assertRaisesRegex(AdapterValidationError, "non-empty execution"):
            validate_unit_completion(
                plan,
                plan.unit("cell"),
                plan_sha256=plan.sha256(),
                layout=self.layout,
                handler_registry=registry,
            )

    def test_new_completion_must_match_current_attempt_exactly(self):
        plan, _path = self._publish_plan()
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        registry = self._registry(handler)
        prior = replace(execution_identity(self._envelope(manifest)), attempt=7)
        self._write_completion(plan, execution=prior)
        # A prior hash-valid execution can be reused.
        validate_unit_completion(
            plan,
            plan.unit("cell"),
            plan_sha256=plan.sha256(),
            layout=self.layout,
            handler_registry=registry,
        )
        # It cannot be presented as the marker produced by this attempt.
        with self.assertRaisesRegex(AdapterValidationError, "current validated attempt"):
            validate_unit_completion(
                plan,
                plan.unit("cell"),
                plan_sha256=plan.sha256(),
                layout=self.layout,
                handler_registry=registry,
                expected_execution=execution_identity(self._envelope(manifest)),
            )

    def test_output_publication_rejects_staging_from_another_job(self):
        plan, _path = self._publish_plan()
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        envelope = self._envelope(manifest)
        store = ContentStore(self.layout)
        with store.create_output_attempt(
            workflow_id=plan.workflow_id,
            plan_sha256=plan.sha256(),
            unit_id="cell",
            job_id=envelope.job_id,
            attempt=envelope.attempt,
        ) as attempt:
            wrong_execution = replace(
                execution_identity(envelope),
                job_id="opd-another-job",
            )
            with self.assertRaisesRegex(
                AdapterValidationError,
                "differs from the workflow execution",
            ):
                publish_output_attempt(
                    plan,
                    plan.unit("cell"),
                    attempt,
                    plan_sha256=plan.sha256(),
                    layout=self.layout,
                    expected_execution=wrong_execution,
                    handler_registry=self._registry(handler),
                )

    def test_completion_config_relationships_are_code_owned(self):
        plan, _path = self._publish_plan()
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        registry = self._registry(handler)
        self._write_completion(plan, config_sha256=_digest("caller-selected"))
        with self.assertRaisesRegex(AdapterValidationError, "differs from"):
            validate_unit_completion(
                plan,
                plan.unit("cell"),
                plan_sha256=plan.sha256(),
                layout=self.layout,
                handler_registry=registry,
            )

    def test_completion_gate_names_are_code_owned(self):
        plan, _path = self._publish_plan()
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        registry = self._registry(handler)
        self._write_completion(
            plan,
            scientific_validation={"caller_claim": True, "domain_gate": True},
        )
        with self.assertRaisesRegex(AdapterValidationError, "gate names"):
            validate_unit_completion(
                plan,
                plan.unit("cell"),
                plan_sha256=plan.sha256(),
                layout=self.layout,
                handler_registry=registry,
            )

    def test_completion_semantic_validator_is_code_owned(self):
        plan, _path = self._publish_plan()
        manifest = _manifest(_running_payload(self.root))
        handler = replace(
            self._handler(manifest),
            semantic_validator_id="always-reject-v1",
            semantic_validator=mock.Mock(side_effect=ValueError("semantic rejection")),
        )
        self._write_completion(plan)
        with self.assertRaisesRegex(AdapterValidationError, "semantic validator"):
            validate_unit_completion(
                plan,
                plan.unit("cell"),
                plan_sha256=plan.sha256(),
                layout=self.layout,
                handler_registry=self._registry(handler),
            )

    def test_runtime_reuses_only_valid_completion_and_waits_for_child_outcome(self):
        plan, _path = self._publish_plan()
        manifest, envelope, running_manifest = self._hold_manifest(
            _running_payload(self.root)
        )
        handler = self._handler(manifest)
        registry = self._registry(handler)
        self._write_completion(plan)

        def forbidden_popen(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("valid completion must be reused before child launch")

        with running_manifest, handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    manifest,
                    envelope,
                    prepared,
                    running_manifest=running_manifest,
                    layout=self.layout,
                    handler_registry=registry,
                    environ=self._thread_environment(),
                    popen=forbidden_popen,
                ),
                0,
            )

    def test_runtime_rejects_zero_exit_without_scientific_completion(self):
        _plan_value, _path = self._publish_plan()
        manifest, envelope, running_manifest = self._hold_manifest(
            _running_payload(self.root)
        )
        handler = self._handler(manifest)
        registry = self._registry(handler)

        class Child:
            def poll(self) -> None:
                return None

            def send_signal(self, _signum: int) -> None:
                raise AssertionError("unexpected signal")

            def wait(self) -> int:
                return 0

        captured: dict[str, object] = {}

        def popen(*args: object, **kwargs: object) -> Child:
            captured["argv"] = args[0]
            captured.update(kwargs)
            return Child()

        held_descriptor = running_manifest.descriptor
        with running_manifest, handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared, self.assertRaises(AdapterValidationError):
            execute_validated_unit(
                manifest,
                envelope,
                prepared,
                running_manifest=running_manifest,
                layout=self.layout,
                handler_registry=registry,
                environ=self._thread_environment(),
                popen=popen,
            )
        self.assertNotIn("--running-manifest-handle", captured["argv"])
        self.assertNotIn(held_descriptor, captured["pass_fds"])

    def test_runtime_accepts_zero_exit_only_with_current_execution_identity(self):
        plan, _path = self._publish_plan()
        manifest, envelope, running_manifest = self._hold_manifest(
            _running_payload(self.root)
        )
        handler = self._handler(manifest)
        registry = self._registry(handler)

        test_case = self

        class Child:
            def poll(self) -> None:
                return None

            def send_signal(self, _signum: int) -> None:
                raise AssertionError("unexpected signal")

            def wait(self) -> int:
                test_case._write_attempt_result(plan, envelope)
                return 0

        with running_manifest, handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    manifest,
                    envelope,
                    prepared,
                    running_manifest=running_manifest,
                    layout=self.layout,
                    handler_registry=registry,
                    environ=self._thread_environment(),
                    popen=lambda *_args, **_kwargs: Child(),
                ),
                0,
            )

    def test_nonzero_child_exit_is_returned_without_local_retry(self):
        _plan_value, _path = self._publish_plan()
        manifest, envelope, running_manifest = self._hold_manifest(
            _running_payload(self.root, gpu=True)
        )
        handler = self._handler(manifest)
        registry = self._registry(handler)

        class Child:
            def poll(self) -> None:
                return None

            def send_signal(self, _signum: int) -> None:
                raise AssertionError("unexpected signal")

            def wait(self) -> int:
                return 17

        captured: dict[str, object] = {}

        def popen(*_args: object, **kwargs: object) -> Child:
            captured["argv"] = _args[0]
            captured.update(kwargs)
            return Child()

        environment = self._thread_environment()
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb",
                "SAFE_PROJECT_ENV": "must-not-pass",
                "SERVER_SCHEDULER_JOB_ID": manifest.job_id,
                "SERVER_SCHEDULER_LEASE_ID": "lease-must-not-reach-handler",
            }
        )
        held_descriptor = running_manifest.descriptor
        with running_manifest, handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    manifest,
                    envelope,
                    prepared,
                    running_manifest=running_manifest,
                    layout=self.layout,
                    handler_registry=registry,
                    environ=environment,
                    popen=popen,
                ),
                17,
            )
        self.assertEqual(running_manifest.descriptor, -1)
        with self.assertRaises(OSError):
            os.fstat(held_descriptor)
        argv = captured["argv"]
        self.assertIsInstance(argv, tuple)
        assert isinstance(argv, tuple)
        option_index = argv.index("--running-manifest-handle")
        self.assertEqual(argv[option_index + 1], f"/proc/self/fd/{held_descriptor}")
        pass_fds = captured["pass_fds"]
        self.assertIsInstance(pass_fds, tuple)
        assert isinstance(pass_fds, tuple)
        self.assertEqual(pass_fds.count(held_descriptor), 1)
        child_environment = captured["env"]
        self.assertIsInstance(child_environment, dict)
        assert isinstance(child_environment, dict)
        self.assertEqual(
            child_environment["CUDA_VISIBLE_DEVICES"], "GPU-aaaa,GPU-bbbb"
        )
        self.assertNotIn("SAFE_PROJECT_ENV", child_environment)
        self.assertEqual(child_environment["HANDLER_PROTOCOL"], "v1")
        self.assertEqual(child_environment["PYTHONNOUSERSITE"], "1")
        self.assertFalse(
            any(key.startswith("SERVER_SCHEDULER_") for key in child_environment)
        )

    def test_scheduler_retry_uses_a_fresh_attempt_and_never_reads_failed_staging(self):
        plan, _path = self._publish_plan()
        manifest, first, running_manifest = self._hold_manifest(
            _running_payload(self.root)
        )
        handler = self._handler(manifest)
        registry = self._registry(handler)
        test_case = self

        class FailedChild:
            def poll(self) -> None:
                return None

            def send_signal(self, _signum: int) -> None:
                raise AssertionError("unexpected signal")

            def wait(self) -> int:
                test_case._write_attempt_result(plan, first)
                return 17

        with running_manifest, handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    manifest,
                    first,
                    prepared,
                    running_manifest=running_manifest,
                    layout=self.layout,
                    handler_registry=registry,
                    environ=self._thread_environment(),
                    popen=lambda *_args, **_kwargs: FailedChild(),
                ),
                17,
            )
            failed_path = self.layout.attempt_directory(
                workflow_id=plan.workflow_id,
                plan_sha256=plan.sha256(),
                unit_id="cell",
                job_id=first.job_id,
                attempt=1,
            )
            (failed_path / "poison-from-failed-attempt").write_text(
                "must never be read\n", encoding="utf-8"
            )
            second = replace(first, attempt=2)

            class SuccessfulChild:
                def poll(self) -> None:
                    return None

                def send_signal(self, _signum: int) -> None:
                    raise AssertionError("unexpected signal")

                def wait(self) -> int:
                    test_case._write_attempt_result(plan, second)
                    return 0

            self.assertEqual(
                execute_validated_unit(
                    manifest,
                    second,
                    prepared,
                    running_manifest=running_manifest,
                    layout=self.layout,
                    handler_registry=registry,
                    environ=self._thread_environment(),
                    popen=lambda *_args, **_kwargs: SuccessfulChild(),
                ),
                0,
            )
        marker = json.loads(
            self.layout.completion_path(
                workflow_id=plan.workflow_id,
                plan_sha256=plan.sha256(),
                unit_id="cell",
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(marker["execution"]["attempt"], 2)
        self.assertTrue(failed_path.is_dir())

    def test_new_job_restarts_attempt_count_without_reusing_failed_staging(self):
        plan, _path = self._publish_plan()
        initial_payload = _running_payload(self.root)
        manifest, first, running_manifest = self._hold_manifest(initial_payload)
        handler = self._handler(manifest)
        registry = self._registry(handler)
        test_case = self

        class FailedChild:
            def poll(self) -> None:
                return None

            def send_signal(self, _signum: int) -> None:
                raise AssertionError("unexpected signal")

            def wait(self) -> int:
                test_case._write_attempt_result(plan, first)
                return 17

        with running_manifest, handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    manifest,
                    first,
                    prepared,
                    running_manifest=running_manifest,
                    layout=self.layout,
                    handler_registry=registry,
                    environ=self._thread_environment(),
                    popen=lambda *_args, **_kwargs: FailedChild(),
                ),
                17,
            )

        retry_payload = json.loads(json.dumps(initial_payload))
        retry_payload["job_id"] = "opd-retry-job-2"
        retry_manifest, retry, retry_running_manifest = self._hold_manifest(
            retry_payload
        )

        class SuccessfulChild:
            def poll(self) -> None:
                return None

            def send_signal(self, _signum: int) -> None:
                raise AssertionError("unexpected signal")

            def wait(self) -> int:
                test_case._write_attempt_result(plan, retry)
                return 0

        with retry_running_manifest, handler.prepare(
            retry_manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    retry_manifest,
                    retry,
                    prepared,
                    running_manifest=retry_running_manifest,
                    layout=self.layout,
                    handler_registry=registry,
                    environ=self._thread_environment(),
                    popen=lambda *_args, **_kwargs: SuccessfulChild(),
                ),
                0,
            )

        first_path = self.layout.attempt_directory(
            workflow_id=plan.workflow_id,
            plan_sha256=plan.sha256(),
            unit_id="cell",
            job_id=first.job_id,
            attempt=1,
        )
        retry_path = self.layout.attempt_directory(
            workflow_id=plan.workflow_id,
            plan_sha256=plan.sha256(),
            unit_id="cell",
            job_id=retry.job_id,
            attempt=1,
        )
        self.assertNotEqual(first_path, retry_path)
        self.assertTrue(first_path.is_dir())
        self.assertFalse(retry_path.exists())
        marker = json.loads(
            self.layout.completion_path(
                workflow_id=plan.workflow_id,
                plan_sha256=plan.sha256(),
                unit_id="cell",
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(marker["execution"]["job_id"], retry.job_id)
        self.assertEqual(marker["execution"]["attempt"], 1)

    def test_child_environment_rejects_loader_and_python_injection(self):
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        for forbidden in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
            source = self._thread_environment()
            source[forbidden] = "/untrusted"
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(
                AdapterValidationError, "injection"
            ):
                build_child_environment(source, handler=handler)

    def test_outbox_requires_published_plan_and_has_no_injection_surface(self):
        plan = _plan()
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        registry = self._registry(handler)
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.registry.HANDLER_REGISTRY", registry
        ):
            with self.assertRaises(AdapterValidationError):
                prepare_outbox_request(
                    plan,
                    unit_id="cell",
                    layout=self.layout,
                )
            publish_workflow_plan(plan, layout=self.layout)
            path = prepare_outbox_request(
                plan,
                unit_id="cell",
                layout=self.layout,
            )
        observed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            observed["parameters"],
            {
                "workflow_id": "pilot",
                "plan_sha256": plan.sha256(),
                "unit_id": "cell",
            },
        )
        self.assertFalse(
            {"command", "cwd", "env", "resources", "path", "lease_id"} & set(observed)
        )
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.registry.HANDLER_REGISTRY", registry
        ):
            retry_path = prepare_outbox_request(
                plan,
                unit_id="cell",
                layout=self.layout,
            )
        self.assertNotEqual(retry_path, path)
        retry = json.loads(retry_path.read_text(encoding="utf-8"))
        self.assertNotEqual(retry["job_id"], observed["job_id"])
        self.assertEqual(retry["parameters"], observed["parameters"])
        injected = dict(observed)
        injected["command"] = "/bin/sh"
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.registry.HANDLER_REGISTRY", registry
        ), self.assertRaises(AdapterValidationError):
            validate_outbox_request(injected)
        malformed_id = dict(observed)
        malformed_id["job_id"] = "opd-not-a-submission-id"
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.registry.HANDLER_REGISTRY", registry
        ), self.assertRaisesRegex(AdapterValidationError, "opaque OPD"):
            validate_outbox_request(malformed_id)

    def test_outbox_has_no_request_side_profile_or_resource_override(self):
        plan = _plan()
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        registry = self._registry(handler)
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.registry.HANDLER_REGISTRY", registry
        ):
            request = build_outbox_request(plan, unit_id="cell")
            self.assertNotIn("execution_profile", request)
            self.assertNotIn("resources", request)
            for field, value in (
                ("execution_profile", "cpu-reviewed"),
                ("resources", {"gpu_count": 1}),
            ):
                injected = dict(request)
                injected[field] = value
                with self.subTest(field=field), self.assertRaisesRegex(
                    AdapterValidationError, "must omit"
                ):
                    validate_outbox_request(injected)
        self.assertNotIn(
            "execution_profile", inspect.signature(build_outbox_request).parameters
        )
        self.assertNotIn(
            "execution_profile", inspect.signature(prepare_outbox_request).parameters
        )

    def test_running_manifest_rejects_request_side_resource_steering(self):
        for field, value in (
            ("requested_profile", "cpu-reviewed"),
            ("resources", _running_payload(self.root)["allocation"]),
        ):
            payload = _running_payload(self.root)
            payload[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                AdapterValidationError, "must omit"
            ):
                _manifest(payload)

    def test_production_registry_rejects_unmigrated_outbox_requests(self):
        plan = _plan()
        for function in (
            build_outbox_request,
            prepare_outbox_request,
            validate_outbox_request,
        ):
            self.assertNotIn("handler_registry", inspect.signature(function).parameters)
            self.assertFalse(
                any(
                    "reviewed" in name
                    for name in inspect.signature(function).parameters
                )
            )
        with self.assertRaisesRegex(DispatchError, "no reviewed OPD handler"):
            build_outbox_request(plan, unit_id="cell")

    def test_handler_rejects_path_lookup_shell_symlink_and_replacement(self):
        manifest = _manifest(_running_payload(self.root))
        valid = self._handler(manifest)

        def with_deployment(**changes: object) -> HandlerSpec:
            deployment = replace(valid.deployment, **changes)
            deployment = replace(
                deployment,
                deployment_identity_sha256=sha256_value(deployment.identity_payload()),
            )
            return replace(valid, deployment=deployment)

        for executable in (Path("/usr/bin/env"), Path("/bin/sh")):
            invalid = with_deployment(
                executable=executable,
                executable_sha256=sha256_file(executable),
            )
            with self.subTest(executable=executable), self.assertRaises(
                AdapterValidationError
            ):
                invalid.prepare(
                    manifest,
                    approved_code_root=self.code_root,
                    approved_runtime_root=self.scratch_root,
                )

        real_executable = valid.deployment.executable
        symlink = real_executable.with_name("python-link")
        symlink.symlink_to(real_executable)
        linked = with_deployment(executable=symlink)
        with self.assertRaises(AdapterValidationError):
            linked.prepare(
                manifest,
                approved_code_root=self.code_root,
                approved_runtime_root=self.scratch_root,
            )

        real_executable.write_bytes(b"replacement\n")
        real_executable.chmod(0o750)
        with self.assertRaisesRegex(AdapterValidationError, "hash differs"):
            valid.prepare(
                manifest,
                approved_code_root=self.code_root,
                approved_runtime_root=self.scratch_root,
            )

    def test_prepared_handler_rejects_executable_path_replacement_before_launch(self):
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        reviewed = handler.deployment.executable.read_bytes()
        with handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(
                prepared.executable_launch_path(),
                str(handler.deployment.executable),
            )
            replacement = handler.deployment.executable.with_name("replacement-python")
            replacement.write_bytes(b"unreviewed replacement\n")
            replacement.chmod(0o750)
            os.replace(replacement, handler.deployment.executable)
            self.assertEqual(Path(prepared.executable.proc_path).read_bytes(), reviewed)
            self.assertNotEqual(handler.deployment.executable.read_bytes(), reviewed)
            with self.assertRaisesRegex(AdapterValidationError, "reviewed inode"):
                prepared.executable_launch_path()

    def test_absolute_python_path_preserves_venv_identity(self):
        venv = self.scratch_root / "venv-identity"
        subprocess.run(
            (
                "/usr/bin/python3.12",
                "-m",
                "venv",
                "--without-pip",
                "--copies",
                str(venv),
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        python = venv / "bin" / "python"
        dist_info = (
            venv
            / "lib"
            / "python3.12"
            / "site-packages"
            / "accelerate-1.10.1.dist-info"
        )
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: accelerate\nVersion: 1.10.1\n",
            encoding="utf-8",
        )
        command = (
            str(python),
            "-I",
            "-c",
            (
                "import importlib.metadata,sys;"
                "print(sys.prefix);"
                "print(importlib.metadata.version('accelerate'))"
            ),
        )
        launched = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        prefix, accelerate = launched.stdout.splitlines()
        self.assertEqual(Path(prefix), venv)
        self.assertEqual(accelerate, "1.10.1")

    def test_handler_binds_dependency_lock_package_manifest_and_deployment_identity(self):
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        handler.deployment.dependency_lock.write_text("changed==2\n", encoding="utf-8")
        with self.assertRaisesRegex(AdapterValidationError, "hash differs"):
            handler.prepare(
                manifest,
                approved_code_root=self.code_root,
                approved_runtime_root=self.scratch_root,
            )
        handler = self._handler(manifest)
        invalid = replace(
            handler,
            deployment=replace(
                handler.deployment,
                deployment_identity_sha256="f" * 64,
            ),
        )
        with self.assertRaisesRegex(AdapterValidationError, "deployment identity"):
            invalid.prepare(
                manifest,
                approved_code_root=self.code_root,
                approved_runtime_root=self.scratch_root,
            )

        handler = self._handler(manifest)
        package_payload = json.loads(
            handler.deployment.package_manifest.read_text(encoding="utf-8")
        )
        package_payload["self_contained"] = False
        handler.deployment.package_manifest.write_text(
            json.dumps(package_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        deployment = replace(
            handler.deployment,
            package_manifest_sha256=sha256_file(
                handler.deployment.package_manifest
            ),
        )
        deployment = replace(
            deployment,
            deployment_identity_sha256=sha256_value(deployment.identity_payload()),
        )
        with self.assertRaisesRegex(AdapterValidationError, "complete deployment"):
            replace(handler, deployment=deployment).prepare(
                manifest,
                approved_code_root=self.code_root,
                approved_runtime_root=self.scratch_root,
            )

    def test_profile_contract_enforces_cpu_memory_gpu_and_model_identity(self):
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        wrong_memory = replace(
            manifest,
            allocation=replace(manifest.allocation, memory_mib=8192),
        )
        with self.assertRaisesRegex(AdapterValidationError, "memory"):
            handler.prepare(
                wrong_memory,
                approved_code_root=self.code_root,
                approved_runtime_root=self.scratch_root,
            )

        gpu_manifest = _manifest(_running_payload(self.root, gpu=True))
        gpu_handler = self._handler(gpu_manifest)
        # Protocol v2 has no trusted model-name field. Central admission applies
        # the registered allowlist and the scientific worker validates CUDA
        # device properties; a supplied observation is still fail-closed.
        with gpu_handler.prepare(
            gpu_manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared:
            self.assertEqual(prepared.profile.process_count, 2)
        with self.assertRaisesRegex(AdapterValidationError, "allowlist"):
            gpu_handler.prepare(
                gpu_manifest,
                approved_code_root=self.code_root,
                approved_runtime_root=self.scratch_root,
                observed_gpu_models=("Wrong-GPU", "Wrong-GPU"),
            )
        with gpu_handler.prepare(
            gpu_manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
            observed_gpu_models=("Reviewed-GPU", "Reviewed-GPU"),
        ) as prepared:
            self.assertEqual(prepared.profile.process_count, 2)

    def test_scheduler_managed_profile_derives_each_reviewed_process_count(self):
        profile = ExecutionProfileContract(
            name="gpu-elastic",
            kind="gpu",
            process_count=0,
            cpu_cores_min=24,
            cpu_cores_max=24,
            memory_mib_min=4096,
            memory_mib_max=4096,
            gpu_count=0,
            gpu_memory_mib_min=24000,
            gpu_memory_mib_max=24000,
            gpu_utilization_pct_min=90,
            gpu_utilization_pct_max=90,
            exclusive_gpu=True,
            allowed_gpu_models=("Reviewed-GPU",),
            gpu_count_policy="scheduler",
            scheduler_gpu_counts=(1, 2, 3, 4),
        )
        profile.validate_contract()
        for world_size in (1, 2, 3, 4):
            payload = _running_payload(self.root, gpu=True)
            payload["execution_profile"] = profile.name
            payload["allocation"]["cpu_cores"] = 24
            payload["allocation"]["gpu_count"] = world_size
            payload["cpu_ids"] = list(range(24))
            payload["gpu_indices"] = list(range(world_size))
            payload["gpu_uuids"] = [
                f"GPU-reviewed-{index}" for index in range(world_size)
            ]
            payload["gpu_pci_bus_ids"] = [
                f"0000:{index + 1:02x}:00.0" for index in range(world_size)
            ]
            manifest = _manifest(payload)
            with self.subTest(world_size=world_size):
                self.assertEqual(profile.process_count_for(manifest), world_size)
                profile.validate_allocation(
                    manifest,
                    observed_gpu_models=("Reviewed-GPU",) * world_size,
                )

        payload = _running_payload(self.root, gpu=True)
        payload["execution_profile"] = profile.name
        payload["allocation"]["cpu_cores"] = 24
        payload["allocation"]["gpu_count"] = 6
        payload["cpu_ids"] = list(range(24))
        payload["gpu_indices"] = list(range(6))
        payload["gpu_uuids"] = [f"GPU-reviewed-{index}" for index in range(6)]
        payload["gpu_pci_bus_ids"] = [
            f"0000:{index + 1:02x}:00.0" for index in range(6)
        ]
        with self.assertRaisesRegex(AdapterValidationError, "outside"):
            profile.validate_allocation(
                _manifest(payload),
                observed_gpu_models=("Reviewed-GPU",) * 6,
            )
    def test_handler_contract_is_immutable_and_cannot_override_adapter_fields(self):
        manifest = _manifest(_running_payload(self.root))
        handler = self._handler(manifest)
        invalid_specs = (
            replace(handler, profiles=dict(handler.profiles)),
            replace(handler, fixed_environment={"HANDLER_PROTOCOL": "v1"}),
            replace(handler, config_hash_bindings=dict(handler.config_hash_bindings)),
            replace(handler, fixed_args=("--job-id=caller-selected",)),
            replace(
                handler,
                fixed_environment=MappingProxyType({"OMP_NUM_THREADS": "999"}),
            ),
        )
        for invalid in invalid_specs:
            with self.subTest(invalid=invalid), self.assertRaises(
                AdapterValidationError
            ):
                invalid.validate_contract()

    def test_handler_argv_contains_only_fixed_prefix_and_validated_identities(self):
        plan, _path = self._publish_plan()
        manifest, envelope, running_manifest = self._hold_manifest(
            _running_payload(self.root)
        )
        handler = self._handler(manifest)
        store = ContentStore(self.layout)
        with running_manifest, handler.prepare(
            manifest,
            approved_code_root=self.code_root,
            approved_runtime_root=self.scratch_root,
        ) as prepared, store.open_inputs(
            plan.unit("cell").content_inputs
        ) as inputs, store.create_output_attempt(
            workflow_id=plan.workflow_id,
            plan_sha256=plan.sha256(),
            unit_id="cell",
            job_id=envelope.job_id,
            attempt=1,
        ) as attempt:
            argv = prepared.argv(
                manifest,
                envelope,
                running_manifest=running_manifest,
                content_handles=inputs.handles,
                output_attempt=attempt,
            )
            self.assertEqual(
                argv[:6],
                (
                    str(handler.deployment.executable),
                    "-I",
                    "-S",
                    prepared.implementation.proc_path,
                    "--fixed-protocol",
                    "v1",
                ),
            )
            self.assertEqual(argv.count("--content-handle"), 5)
            self.assertIn(unit_identity_sha256(
                workflow_id="pilot", plan_sha256=plan.sha256(), unit_id="cell"
            ), argv)
            self.assertIn(attempt.proc_path, argv)
            self.assertIn(ATTEMPT_COMPLETION_NAME, argv)
            self.assertNotIn("--gpu-count", argv)
            self.assertNotIn("--running-manifest-handle", argv)
            for handle in inputs.handles:
                self.assertIn(handle.proc_path, argv)
                self.assertIn(handle.sha256, argv)
            self.assertNotIn("lease-opd-1", argv)

    def test_foreground_dispatch_waits_and_forwards_only_to_exact_child(self):
        captured: dict[str, object] = {}
        installed: dict[signal.Signals, object] = {}

        class Child:
            def __init__(self) -> None:
                self.signals: list[int] = []

            def poll(self) -> None:
                return None

            def send_signal(self, signum: int) -> None:
                self.signals.append(signum)

            def wait(self) -> int:
                handler = installed[signal.SIGTERM]
                assert callable(handler)
                handler(signal.SIGTERM, None)
                return 143

        child = Child()

        def popen(command: tuple[str, ...], **kwargs: object) -> Child:
            captured["command"] = command
            captured.update(kwargs)
            return child

        def install(signum: signal.Signals, handler: object) -> object:
            if callable(handler):
                installed[signum] = handler
            return None

        with mock.patch("signal.getsignal", return_value=signal.SIG_DFL), mock.patch(
            "signal.signal", side_effect=install
        ):
            exit_code = run_foreground_child(
                ("/fixed/python", "/fixed/handler.py"),
                cwd=self.code_root,
                environ={"SAFE": "1"},
                popen=popen,
            )
        self.assertEqual(exit_code, 143)
        self.assertEqual(child.signals, [signal.SIGTERM])
        self.assertIs(captured["shell"], False)
        self.assertIs(captured["start_new_session"], False)
        self.assertEqual(captured["cwd"], self.code_root)
        self.assertEqual(captured["command"], ("/fixed/python", "/fixed/handler.py"))
        self.assertNotIn("executable", captured)

    def test_real_sigterm_is_forwarded_and_entrypoint_preserves_signal_status(self):
        ready = self.root / "signal-child-ready"
        child_code = (
            "import pathlib,sys,time;"
            "pathlib.Path(sys.argv[1]).write_text('ready');"
            "time.sleep(60)"
        )
        adapter_code = (
            "import pathlib,sys;"
            f"sys.path.insert(0,{str(SRC_ROOT)!r});"
            "from posttrain_circuits.scheduler_adapter.dispatch import "
            "propagate_child_signal,run_foreground_child;"
            "rc=run_foreground_child((sys.executable,'-c',sys.argv[1],sys.argv[2]),"
            "cwd=pathlib.Path(sys.argv[3]),environ={});"
            "raise SystemExit(propagate_child_signal(rc))"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                adapter_code,
                child_code,
                str(ready),
                str(self.code_root),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            for _attempt in range(500):
                if ready.exists():
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            if not ready.exists():
                process.terminate()
                stdout, stderr = process.communicate(timeout=5)
                self.fail(
                    "signal test child did not become ready: "
                    f"returncode={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
                )
            process.send_signal(signal.SIGTERM)
            _stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, -signal.SIGTERM, stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_foreground_dispatch_does_not_retry_launch_failure(self):
        calls = 0

        def fail(*_args: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise OSError("injected launch failure")

        with self.assertRaises(DispatchError):
            run_foreground_child(
                ("/fixed/python", "/fixed/handler.py"),
                cwd=self.code_root,
                environ={},
                popen=fail,
            )
        self.assertEqual(calls, 1)

    def test_foreground_dispatch_tolerates_child_exit_signal_race(self):
        installed: dict[signal.Signals, object] = {}

        class Child:
            def poll(self) -> None:
                return None

            def send_signal(self, _signum: int) -> None:
                raise ProcessLookupError("child exited during forwarding")

            def wait(self) -> int:
                handler = installed[signal.SIGTERM]
                assert callable(handler)
                handler(signal.SIGTERM, None)
                return 0

        def install(signum: signal.Signals, handler: object) -> object:
            if callable(handler):
                installed[signum] = handler
            return signal.SIG_DFL

        with mock.patch("signal.getsignal", return_value=signal.SIG_DFL), mock.patch(
            "signal.signal", side_effect=install
        ):
            self.assertEqual(
                run_foreground_child(
                    ("/fixed/python", "/fixed/handler.py"),
                    cwd=self.code_root,
                    environ={},
                    popen=lambda *_args, **_kwargs: Child(),
                ),
                0,
            )

    def test_signal_propagation_supports_nonforwarded_child_death_signals(self):
        with mock.patch("os.kill") as kill:
            self.assertEqual(propagate_child_signal(-signal.SIGKILL), 137)
        kill.assert_called_once_with(os.getpid(), signal.SIGKILL)
        with self.assertRaises(DispatchError):
            propagate_child_signal(-9999)

    def test_adapter_ast_contains_no_submission_or_local_scheduler_primitives(self):
        self.assertIsInstance(HANDLER_REGISTRY, MappingProxyType)
        self.assertEqual(
            tuple(HANDLER_REGISTRY),
            ("qwen3_v2_g0", "qwen3_v2_gpu_preflight", "repository_preflight"),
        )
        banned_modules = {"fcntl", "multiprocessing"}
        banned_calls = {
            ("os", "fork"),
            ("os", "system"),
            ("subprocess", "call"),
            ("subprocess", "check_call"),
            ("subprocess", "check_output"),
            ("subprocess", "run"),
        }
        banned_command_text = ("nvidia-smi", "sacct", "sbatch", "scontrol", "squeue")
        adapter_root = SRC_ROOT / "posttrain_circuits" / "scheduler_adapter"
        for source_path in sorted(adapter_root.glob("*.py")):
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(source_path))
            with self.subTest(source=source_path.name):
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        names = (
                            [alias.name for alias in node.names]
                            if isinstance(node, ast.Import)
                            else [node.module or ""]
                        )
                        self.assertFalse(
                            {name.split(".", 1)[0] for name in names} & banned_modules
                        )
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                    ):
                        self.assertNotIn(
                            (node.func.value.id, node.func.attr), banned_calls
                        )
                lowered = source.lower()
                self.assertFalse(
                    [token for token in banned_command_text if token in lowered]
                )

    def test_git_provenance_disables_mutable_git_configuration(self):
        completed = mock.Mock(stdout=b"abc123\n")
        with mock.patch.object(
            git_provenance.subprocess,
            "run",
            return_value=completed,
        ) as run:
            self.assertEqual(
                git_provenance.require_git_output(PROJECT_ROOT, ("rev-parse", "HEAD")),
                "abc123",
            )
        command = run.call_args.args[0]
        self.assertEqual(
            command[:6],
            (
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(PROJECT_ROOT),
                "rev-parse",
            ),
        )
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], "/dev/null")
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(environment["GIT_NO_REPLACE_OBJECTS"], "1")
        self.assertEqual(environment["GIT_OPTIONAL_LOCKS"], "0")

    def test_git_provenance_finds_files_hidden_by_git_excludes(self):
        with tempfile.TemporaryDirectory(
            prefix=".git-provenance-untracked-",
            dir=PROJECT_ROOT,
        ) as raw_root:
            repository = Path(raw_root)
            subprocess.run(
                ("/usr/bin/git", "-C", str(repository), "init", "-q"),
                check=True,
                capture_output=True,
            )
            (repository / ".git" / "info" / "exclude").write_text(
                "src/ignored.py\n",
                encoding="utf-8",
            )
            ignored = repository / "src" / "ignored.py"
            ignored.parent.mkdir()
            ignored.write_text("raise RuntimeError('must never import')\n", encoding="utf-8")
            self.assertEqual(
                git_provenance.unsafe_untracked_paths(repository),
                ("src/ignored.py",),
            )

    def test_entrypoint_imports_no_torch_or_workflow_before_validation(self):
        code = (
            "import json,sys;"
            f"sys.path.insert(0,{str(SRC_ROOT)!r});"
            "from posttrain_circuits.scheduler_adapter.entrypoint import main;"
            "rc=main(['--invalid']);"
            "print(json.dumps([rc,'torch' in sys.modules,"
            "'posttrain_circuits.workflows.contracts' in sys.modules,"
            "'posttrain_circuits.scheduler_adapter.runtime' in sys.modules]))"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout.strip()), [2, False, False, False])

    def test_valid_boundary_still_imports_no_workflow_when_handler_is_unmigrated(self):
        payload = _running_payload(self.root)
        path = self._write_manifest(payload)
        manifest = load_running_manifest(path)
        environment = os.environ.copy()
        environment.update(_environment(manifest, path, gpu=False))
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        code = (
            "import json,sys;"
            f"sys.path.insert(0,{str(SRC_ROOT)!r});"
            "from posttrain_circuits.scheduler_adapter.entrypoint import main;"
            "rc=main(['--job-manifest',sys.argv[1]]);"
            "print(json.dumps([rc,'torch' in sys.modules,"
            "'posttrain_circuits.workflows.contracts' in sys.modules,"
            "'posttrain_circuits.scheduler_adapter.runtime' in sys.modules]))"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(path)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(json.loads(result.stdout.strip()), [2, False, False, False])
        self.assertIn("no reviewed OPD handler", result.stderr)

    def test_entrypoint_script_uses_verified_absolute_python_312(self):
        script = PROJECT_ROOT / "scripts" / "server_scheduler" / "opd-entrypoint"
        self.assertEqual(
            script.read_text(encoding="utf-8").splitlines()[0],
            "#!/usr/bin/python3.12 -I",
        )
        self.assertTrue(os.access(script, os.X_OK))
        version = subprocess.run(
            ["/usr/bin/python3.12", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertTrue(version.stdout.startswith("Python 3.12."))
        injected_environment = os.environ.copy()
        injected_environment["PYTHONPATH"] = "/untrusted"
        rejected = subprocess.run(
            [str(script), "--invalid"],
            check=False,
            capture_output=True,
            text=True,
            env=injected_environment,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("rejected ambient injection", rejected.stderr)

    def test_entrypoint_isolates_project_bytecode_before_import(self):
        script = PROJECT_ROOT / "scripts" / "server_scheduler" / "opd-entrypoint"
        source = script.read_text(encoding="utf-8")
        dont_write_offset = source.index("sys.dont_write_bytecode = True")
        prefix_offset = source.index("sys.pycache_prefix = ")
        clean_checkout_offset = source.index("_require_clean_checkout_before_import()")
        project_path_offset = source.index("sys.path.insert(0, str(project_src))")
        project_import_offset = source.index(
            "from posttrain_circuits.scheduler_adapter.entrypoint import main"
        )
        self.assertLess(dont_write_offset, project_path_offset)
        self.assertLess(prefix_offset, project_path_offset)
        self.assertLess(clean_checkout_offset, project_path_offset)
        self.assertLess(project_path_offset, project_import_offset)

        code = (
            "import json,os,runpy,subprocess,sys,types;"
            "real_run=subprocess.run;"
            "subprocess.run=lambda command,*args,**kwargs: "
            "types.SimpleNamespace(stdout=b'',returncode=0) "
            "if command[0]=='/usr/bin/git' else real_run(command,*args,**kwargs);"
            "exit_code=None;"
            "\ntry:\n runpy.run_path(sys.argv[1],run_name='__main__')"
            "\nexcept SystemExit as exc:\n exit_code=exc.code"
            "\nprint(json.dumps({'exit_code':exit_code,"
            "'dont_write_bytecode':sys.dont_write_bytecode,"
            "'pycache_prefix':sys.pycache_prefix,"
            "'prefix_exists':os.path.exists(sys.pycache_prefix)}))"
        )
        environment = os.environ.copy()
        for key in (
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONHOME",
            "PYTHONPATH",
        ):
            environment.pop(key, None)
        result = subprocess.run(
            [
                "/usr/bin/python3.12",
                "-I",
                "-v",
                "-c",
                code,
                str(script),
                "--invalid",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        payload = json.loads(result.stdout.strip())
        self.assertEqual(payload["exit_code"], 2)
        self.assertIs(payload["dont_write_bytecode"], True)
        self.assertRegex(
            payload["pycache_prefix"],
            r"^/scr/del6500/OPD/tmp/\.opd-entrypoint-pycache-[a-z0-9_]+$",
        )
        self.assertIs(payload["prefix_exists"], False)
        production_src_root = Path("/home/del6500/projects/OPD/src")
        self.assertIn(
            f"code object from {production_src_root}/posttrain_circuits/",
            result.stderr,
        )
        self.assertNotIn(
            f"{SRC_ROOT}/posttrain_circuits/__pycache__",
            result.stderr,
        )
        self.assertNotIn(
            f"{SRC_ROOT}/posttrain_circuits/scheduler_adapter/__pycache__",
            result.stderr,
        )

    def test_entrypoint_rejects_tracked_source_change_before_import(self):
        script = PROJECT_ROOT / "scripts" / "server_scheduler" / "opd-entrypoint"
        code = (
            "import json,runpy,subprocess,sys,types;"
            "real_run=subprocess.run;"
            "subprocess.run=lambda command,*args,**kwargs: "
            "types.SimpleNamespace(stdout=b' M src/posttrain_circuits/scheduler_adapter/registry.py\\n',returncode=0) "
            "if command[0]=='/usr/bin/git' else real_run(command,*args,**kwargs);"
            "exit_code=None;"
            "\ntry:\n runpy.run_path(sys.argv[1],run_name='__main__')"
            "\nexcept SystemExit as exc:\n exit_code=exc.code"
            "\nprint(json.dumps([exit_code,"
            "'posttrain_circuits' in sys.modules]))"
        )
        environment = os.environ.copy()
        for key in ("LD_AUDIT", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
            environment.pop(key, None)
        result = subprocess.run(
            ["/usr/bin/python3.12", "-I", "-B", "-c", code, str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(json.loads(result.stdout.strip()), [2, False])
        self.assertIn("dirty or unverifiable checkout", result.stderr)

    def test_entrypoint_rejects_top_level_src_shadow_before_import(self):
        source = (
            PROJECT_ROOT / "scripts" / "server_scheduler" / "opd-entrypoint"
        ).read_text(encoding="utf-8")
        scratch_root = Path("/scr/del6500/OPD/tmp")
        for exclusion in ("none", "gitignore", "info-exclude"):
            with self.subTest(exclusion=exclusion), tempfile.TemporaryDirectory(
                prefix=".entrypoint-shadow-", dir=scratch_root
            ) as raw_repository:
                repository = Path(raw_repository)
                subprocess.run(
                    ("/usr/bin/git", "-C", str(repository), "init", "-q"),
                    check=True,
                    capture_output=True,
                )
                package = repository / "src" / "posttrain_circuits"
                adapter = package / "scheduler_adapter"
                adapter.mkdir(parents=True)
                (package / "__init__.py").write_text("", encoding="utf-8")
                (adapter / "__init__.py").write_text("", encoding="utf-8")
                (adapter / "entrypoint.py").write_text(
                    "import json\n\ndef main():\n    return 0\n", encoding="utf-8"
                )
                script = repository / "scripts" / "server_scheduler" / "opd-entrypoint"
                script.parent.mkdir(parents=True)
                script.write_text(
                    source.replace(
                        f'SOURCE_ROOT = "{PROJECT_ROOT}"',
                        f'SOURCE_ROOT = "{repository}"',
                        1,
                    ),
                    encoding="utf-8",
                )
                script.chmod(0o750)
                gitignore = repository / ".gitignore"
                if exclusion == "gitignore":
                    gitignore.write_text("/src/json.py\n", encoding="utf-8")
                subprocess.run(
                    ("/usr/bin/git", "-C", str(repository), "add", "--", "."),
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    (
                        "/usr/bin/git",
                        "-c",
                        "user.name=OPD fixture",
                        "-c",
                        "user.email=opd-fixture@example.invalid",
                        "-C",
                        str(repository),
                        "commit",
                        "-qm",
                        "fixture",
                    ),
                    check=True,
                    capture_output=True,
                )
                if exclusion == "info-exclude":
                    (repository / ".git" / "info" / "exclude").write_text(
                        "/src/json.py\n", encoding="utf-8"
                    )
                sentinel = repository / "shadow-imported"
                (repository / "src" / "json.py").write_text(
                    "from pathlib import Path\n"
                    f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
                    encoding="utf-8",
                )
                environment = os.environ.copy()
                for key in (
                    "LD_AUDIT",
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "PYTHONHOME",
                    "PYTHONPATH",
                ):
                    environment.pop(key, None)
                result = subprocess.run(
                    (str(script), "--invalid"),
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("dirty or unverifiable checkout", result.stderr)
                self.assertFalse(sentinel.exists())

    def test_production_outbox_path_is_fixed_and_not_request_selectable(self):
        self.assertEqual(
            WorkflowLayout.production().outbox_directory(),
            Path("/scr/del6500/OPD/scheduler/outbox"),
        )


if __name__ == "__main__":
    unittest.main()
