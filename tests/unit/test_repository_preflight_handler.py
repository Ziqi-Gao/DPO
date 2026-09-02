"""Stdlib-only contract tests for the standalone repository preflight handler."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HANDLER = ROOT / "scripts" / "server_scheduler" / "repository-preflight-handler.py"
COMPLETION_NAME = ".opd-scientific-completion.json"
REPORT_NAME = "preflight_report.json"
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_value(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class RepositoryPreflightHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".repository-preflight-test-",
            dir=ROOT,
        )
        self.root = Path(self._temporary.name)
        self.input_root = self.root / "inputs"
        self.input_root.mkdir()
        self.output_root = self.root / "attempt"
        self.output_root.mkdir()
        resolved = {"model": {"name": "fixture"}, "seed": 42}
        artifact_hashes: dict[str, str] = {}
        execution_context = {"entrypoint": "repository_preflight", "world_size": 1}
        scientific = {
            "config": resolved,
            "input_artifact_hashes": artifact_hashes,
            "schema_version": 3,
        }
        execution = {
            "execution_context": execution_context,
            "schema_version": 3,
            "storage_locators": {},
        }
        binding = {
            "execution_config_sha256": _sha256_value(execution),
            "execution_context": execution_context,
            "input_artifact_hashes": artifact_hashes,
            "resolved_config_sha256": _sha256_value(resolved),
            "schema_version": 3,
            "scientific_config_sha256": _sha256_value(scientific),
            "storage_locators": {},
        }
        payloads = {
            "config_binding_sha256": binding,
            "execution_config_sha256": execution,
            "resolved_config_sha256": resolved,
            "scientific_config_sha256": scientific,
        }
        self.paths: dict[str, Path] = {}
        self.hashes: dict[str, str] = {}
        for name, payload in payloads.items():
            raw = _canonical_json(payload).encode("utf-8")
            path = self.input_root / f"{name}.json"
            path.write_bytes(raw)
            self.paths[name] = path
            self.hashes[name] = hashlib.sha256(raw).hexdigest()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _environment(**updates: str) -> dict[str, str]:
        environment = {key: "1" for key in THREAD_KEYS}
        environment["PYTHONNOUSERSITE"] = "1"
        environment.update(updates)
        return environment

    def _invoke(
        self,
        *,
        environment: dict[str, str] | None = None,
        mutate_arguments=None,  # type: ignore[no-untyped-def]
    ) -> subprocess.CompletedProcess[str]:
        descriptors: list[int] = []
        try:
            handles: list[tuple[str, int]] = []
            for name in sorted(self.paths):
                descriptor = os.open(self.paths[name], os.O_RDONLY)
                descriptors.append(descriptor)
                handles.append((name, descriptor))
            output_descriptor = os.open(self.output_root, os.O_RDONLY | os.O_DIRECTORY)
            descriptors.append(output_descriptor)
            argv = [
                sys.executable,
                "-I",
                "-S",
                str(HANDLER),
                "--workflow-id",
                "repository-check",
                "--plan-sha256",
                _digest("plan"),
                "--unit-id",
                "repository-preflight",
                "--run-id",
                _digest("run"),
                "--job-id",
                "job-1",
                "--attempt",
                "1",
                "--execution-profile",
                "cpu-preflight",
                "--manifest-sha256",
                _digest("manifest"),
                "--allocation-sha256",
                _digest("allocation"),
            ]
            for name, descriptor in handles:
                argv.extend(
                    (
                        "--content-handle",
                        name,
                        "file",
                        self.hashes[name],
                        f"/proc/self/fd/{descriptor}",
                    )
                )
            argv.extend(
                (
                    "--output-attempt-handle",
                    f"/proc/self/fd/{output_descriptor}",
                    "--attempt-completion-name",
                    COMPLETION_NAME,
                )
            )
            if mutate_arguments is not None:
                mutate_arguments(argv)
            return subprocess.run(
                argv,
                cwd=ROOT,
                env=self._environment() if environment is None else environment,
                pass_fds=tuple(descriptors),
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def test_writes_path_free_report_and_idempotent_completion_draft(self) -> None:
        first = self._invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        report_path = self.output_root / REPORT_NAME
        completion_path = self.output_root / COMPLETION_NAME
        self.assertEqual(
            {path.name for path in self.output_root.iterdir()},
            {REPORT_NAME, COMPLETION_NAME},
        )
        report_raw = report_path.read_bytes()
        completion_raw = completion_path.read_bytes()
        report_stat = report_path.stat()
        completion_stat = completion_path.stat()
        report = json.loads(report_raw)
        completion = json.loads(completion_raw)
        self.assertEqual(
            report["scientific_validation"],
            {
                "config_binding": True,
                "no_gpu_required": True,
                "runtime_isolation": True,
            },
        )
        self.assertEqual(report["input_hashes"], self.hashes)
        self.assertEqual(completion["input_hashes"], self.hashes)
        completion_content = {
            key: value for key, value in completion.items() if key != "sha256"
        }
        self.assertEqual(completion["sha256"], _sha256_value(completion_content))
        self.assertNotIn("/proc/self/fd/", report_raw.decode("utf-8"))
        self.assertNotIn("/proc/self/fd/", completion_raw.decode("utf-8"))

        second = self._invoke()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(report_path.read_bytes(), report_raw)
        self.assertEqual(completion_path.read_bytes(), completion_raw)
        self.assertEqual(report_path.stat().st_ino, report_stat.st_ino)
        self.assertEqual(completion_path.stat().st_ino, completion_stat.st_ino)
        self.assertEqual(report_path.stat().st_mtime_ns, report_stat.st_mtime_ns)
        self.assertEqual(completion_path.stat().st_mtime_ns, completion_stat.st_mtime_ns)

    def test_rejects_environment_injection_without_outputs(self) -> None:
        injected_environments = (
            self._environment(PYTHONPATH="/untrusted"),
            self._environment(SERVER_SCHEDULER_LEASE_ID="must-not-reach-handler"),
            self._environment(CUDA_VISIBLE_DEVICES="GPU-unexpected"),
        )
        for environment in injected_environments:
            with self.subTest(environment=environment):
                result = self._invoke(environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(self.output_root.iterdir()), [])

    def test_rejects_bad_adapter_arguments_without_following_paths(self) -> None:
        def wrong_hash(argv: list[str]) -> None:
            index = argv.index("config_binding_sha256")
            argv[index + 2] = "0" * 64

        result = self._invoke(mutate_arguments=wrong_hash)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.output_root.iterdir()), [])

        def external_path(argv: list[str]) -> None:
            index = argv.index("--output-attempt-handle")
            argv[index + 1] = str(self.output_root)

        result = self._invoke(mutate_arguments=external_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.output_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
