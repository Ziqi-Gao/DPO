from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.environment import (
    THREAD_ENVIRONMENT_KEYS,
    RuntimeEnvelope,
)
from posttrain_circuits.scheduler_adapter.manifest import hold_running_manifest
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.plan_store import publish_workflow_plan
from posttrain_circuits.scheduler_adapter.preflight_request import (
    UNIT_ID,
    build_repository_preflight_plan,
)
from posttrain_circuits.scheduler_adapter.registry import (
    FIXED_RUNTIME_ROOT,
    HANDLER_REGISTRY,
    require_handler,
)
from posttrain_circuits.scheduler_adapter.runtime import execute_validated_unit


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RepositoryPreflightAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".repository-preflight-adapter-test-",
            dir=PROJECT_ROOT,
        )
        self.root = Path(self._temporary.name)
        self.layout = WorkflowLayout(
            code_root=self.root / "code",
            data_root=self.root / "data",
            scratch_root=self.root / "scratch",
        )
        for path in (
            self.layout.code_root,
            self.layout.data_root,
            self.layout.scratch_root,
        ):
            path.mkdir()
        self.plan = build_repository_preflight_plan(layout=self.layout)
        publish_workflow_plan(self.plan, layout=self.layout)
        self.payload = {
            "allocation": {
                "cpu_cores": 1,
                "exclusive_gpu": False,
                "gpu_count": 0,
                "gpu_memory_mib": 0,
                "gpu_utilization_pct": 0,
                "memory_mib": 128,
            },
            "cpu_ids": [0],
            "estimated_runtime_seconds": 30.0,
            "execution_profile": "repository-preflight-cpu",
            "exit_code": None,
            "failure_reason": None,
            "gpu_indices": [],
            "gpu_pci_bus_ids": [],
            "gpu_uuids": [],
            "job_id": "opd-repository-preflight-test",
            "numa_node": None,
            "parameters": {
                "plan_sha256": self.plan.sha256(),
                "unit_id": UNIT_ID,
                "workflow_id": self.plan.workflow_id,
            },
            "priority": 0,
            "project": "OPD",
            "requested_profile": None,
            "resources": None,
            "schema_version": 2,
            "state": "running",
            "stderr_log": str(self.root / "stderr.log"),
            "stdout_log": str(self.root / "stdout.log"),
            "submitted_at": "2026-09-02T12:00:00Z",
            "task": "repository_preflight",
            "updated_at": "2026-09-02T12:00:01Z",
        }
        self.manifest_path = self.root / "running.json"
        self.manifest_path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.manifest, running_manifest = hold_running_manifest(self.manifest_path)
        running_manifest.close()
        self.envelope = RuntimeEnvelope(
            job_id=self.manifest.job_id,
            attempt=1,
            execution_profile=self.manifest.execution_profile,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest.manifest_sha256,
            allocation_sha256=sha256_value(self.manifest.allocation_payload()),
        )
        self.environment = {key: "1" for key in THREAD_ENVIRONMENT_KEYS}

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_real_handler_publishes_and_reuses_semantic_completion(self) -> None:
        handler = require_handler("repository_preflight")
        manifest, running_manifest = hold_running_manifest(self.manifest_path)
        self.assertEqual(manifest, self.manifest)
        with running_manifest, handler.prepare(
            self.manifest,
            approved_code_root=handler.deployment.implementation.parents[2],
            approved_runtime_root=FIXED_RUNTIME_ROOT,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    self.manifest,
                    self.envelope,
                    prepared,
                    running_manifest=running_manifest,
                    layout=self.layout,
                    handler_registry=HANDLER_REGISTRY,
                    environ=self.environment,
                    popen=subprocess.Popen,
                ),
                0,
            )

        completion_path = self.layout.completion_path(
            workflow_id=self.plan.workflow_id,
            plan_sha256=self.plan.sha256(),
            unit_id=UNIT_ID,
        )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        self.assertEqual(
            completion["scientific_validation"],
            {
                "config_binding": True,
                "no_gpu_required": True,
                "runtime_isolation": True,
            },
        )

        def must_not_launch(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("valid completion must be reused without relaunch")

        retry_payload = json.loads(json.dumps(self.payload))
        retry_payload["job_id"] = "opd-repository-preflight-retry"
        retry_manifest_path = self.root / "retry-running.json"
        retry_manifest_path.write_text(
            json.dumps(retry_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        retry_manifest, retry_running_manifest = hold_running_manifest(
            retry_manifest_path
        )
        retry_envelope = RuntimeEnvelope(
            job_id=retry_manifest.job_id,
            attempt=2,
            execution_profile=retry_manifest.execution_profile,
            manifest_path=retry_manifest_path,
            manifest_sha256=retry_manifest.manifest_sha256,
            allocation_sha256=sha256_value(retry_manifest.allocation_payload()),
        )
        with retry_running_manifest, handler.prepare(
            retry_manifest,
            approved_code_root=handler.deployment.implementation.parents[2],
            approved_runtime_root=FIXED_RUNTIME_ROOT,
        ) as prepared:
            self.assertEqual(
                execute_validated_unit(
                    retry_manifest,
                    retry_envelope,
                    prepared,
                    running_manifest=retry_running_manifest,
                    layout=self.layout,
                    handler_registry=HANDLER_REGISTRY,
                    environ=self.environment,
                    popen=must_not_launch,
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
