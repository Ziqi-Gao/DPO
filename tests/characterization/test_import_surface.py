"""Characterize the dependency-light contract import surface."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


def test_artifact_contract_import_does_not_load_torch() -> None:
    script = """
import sys
import posttrain_circuits
import posttrain_circuits.artifacts.checkpoints
import posttrain_circuits.artifacts.completion
import posttrain_circuits.artifacts.config_bindings
assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_artifacts_are_a_dependency_leaf_at_module_import_time() -> None:
    forbidden = (
        "torch",
        "posttrain_circuits.methods",
        "posttrain_circuits.experiments",
        "posttrain_circuits.data",
        "posttrain_circuits.datasets",
        "posttrain_circuits.workflows",
        "posttrain_circuits.scheduler",
        "posttrain_circuits.cli",
        "posttrain_circuits.core",
    )
    violations: list[str] = []
    for path in Path("src/posttrain_circuits/artifacts").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            for module in modules:
                if module.startswith(forbidden):
                    violations.append(f"{path}:{node.lineno}:{module}")
    assert violations == []
