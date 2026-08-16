"""Pinned MIB/EAP-IG execution over fixed local discovery pairs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.exact_patching import (
    ExactPatchingBackend,
)
from posttrain_circuits.circuits.graph import CircuitScores
from posttrain_circuits.circuits.probes import CIRCUIT_PROBE_SCHEMA_VERSION
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json

MIB_REVISION = "b759df34433c9e31043ba9e02908ce0bf20e894f"
MIB_EAP_IG_METHOD = "EAP-IG-inputs"


def write_fixed_discovery_pairs(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("MIB discovery pair manifest cannot be empty")
    required = {
        "pair_id",
        "clean_prompt",
        "corrupt_prompt",
        "clean_target",
        "corrupt_target",
        "clean_input_ids",
        "corrupt_input_ids",
        "clean_target_ids",
        "corrupt_target_ids",
        "clean_metric_positions",
        "corrupt_metric_positions",
        "clean_intervention_positions",
        "corrupt_intervention_positions",
        "stage",
        "semantic_pair_hash",
        "tokenized_pair_hash",
        "semantic_manifest_hash",
        "tokenizer_hash",
    }
    identifiers = []
    normalized = []
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"discovery pair is missing {sorted(missing)}")
        item = {key: row[key] for key in sorted(required)}
        for key in ("pair_id", "clean_prompt", "corrupt_prompt", "clean_target", "corrupt_target"):
            item[key] = str(item[key])
        for key in (
            "clean_input_ids",
            "corrupt_input_ids",
            "clean_target_ids",
            "corrupt_target_ids",
            "clean_metric_positions",
            "corrupt_metric_positions",
            "clean_intervention_positions",
            "corrupt_intervention_positions",
        ):
            item[key] = [int(value) for value in item[key]]
        if len(item["clean_input_ids"]) != len(item["corrupt_input_ids"]):
            raise ValueError(f"discovery pair {item['pair_id']} is not token-shape matched")
        if len(item["clean_target_ids"]) != len(item["corrupt_target_ids"]):
            raise ValueError(f"discovery pair {item['pair_id']} targets are not shape matched")
        if item["stage"] != "final_answer" and {item["clean_target"], item["corrupt_target"]} <= {
            "0",
            "1",
        }:
            raise ValueError("process-stage MIB rows cannot use final-answer token IDs")
        identifiers.append(item["pair_id"])
        normalized.append(item)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("discovery pair IDs must be unique")
    payload = {
        "format_version": 2,
        "circuit_probe_schema_version": CIRCUIT_PROBE_SCHEMA_VERSION,
        "prereg_version": "core_v2",
        "pair_count": len(normalized),
        "pairs": normalized,
        "sha256": sha256_value(normalized),
        "metadata": metadata or {},
    }
    atomic_write_json(path, payload)
    return payload


class MibEapIgAdapter:
    backend_name = "mib-eap-ig"
    backend_revision = MIB_REVISION
    method = MIB_EAP_IG_METHOD

    def __init__(self, repository: Path | None = None) -> None:
        value = repository or (Path(os.environ["MIB_REPOSITORY"]) if "MIB_REPOSITORY" in os.environ else None)
        if value is None:
            raise RuntimeError(
                "MIB EAP-IG requires an external pinned checkout; "
                "set MIB_REPOSITORY. See third_party/README.md."
            )
        self.repository = value.resolve()
        actual = subprocess.check_output(
            [
                "git",
                "-C",
                str(self.repository),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            env={key: value for key, value in os.environ.items() if key not in {"GIT_DIR", "GIT_WORK_TREE"}},
        ).strip()
        if actual != MIB_REVISION:
            raise RuntimeError(f"MIB revision mismatch: expected {MIB_REVISION}, got {actual}")
        if not (self.repository / "run_attribution.py").is_file():
            raise RuntimeError("MIB checkout is missing run_attribution.py")

    @property
    def version(self) -> str:
        return f"{self.backend_name}@{self.backend_revision}"

    def command(
        self,
        *,
        model: str,
        model_revision: str,
        checkpoint: Path,
        checkpoint_sha256: str,
        level: str,
        steps: int,
        pair_manifest: Path,
        output_path: Path,
        bootstrap_replicates: int,
        seed: int,
        parity_tolerance: float,
    ) -> list[str]:
        if level not in {"node", "edge"}:
            raise ValueError("MIB level must be node or edge")
        if steps < 1:
            raise ValueError("MIB integrated-gradient steps must be positive")
        if bootstrap_replicates < 2:
            raise ValueError("MIB uncertainty requires at least two bootstrap replicates")
        return [
            sys.executable,
            "-m",
            "posttrain_circuits.circuits.mib_runner",
            "--repository",
            str(self.repository),
            "--expected-revision",
            MIB_REVISION,
            "--model",
            model,
            "--model-revision",
            model_revision,
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            checkpoint_sha256,
            "--pairs",
            str(pair_manifest),
            "--method",
            MIB_EAP_IG_METHOD,
            "--level",
            level,
            "--ig-steps",
            str(steps),
            "--bootstrap-replicates",
            str(bootstrap_replicates),
            "--seed",
            str(seed),
            "--parity-tolerance",
            str(parity_tolerance),
            "--output",
            str(output_path),
        ]

    @staticmethod
    def _score_mapping(
        rows: Any,
    ) -> dict[str, float]:
        if not isinstance(rows, dict):
            return {}
        output = {}
        for name, value in rows.items():
            if isinstance(value, int | float):
                output[str(name)] = float(value)
            elif isinstance(value, dict) and isinstance(
                value.get("score"),
                int | float,
            ):
                output[str(name)] = float(value["score"])
        return output

    @classmethod
    def _parse_scores(
        cls,
        payload: Any,
        *,
        level: str | None = None,
    ) -> CircuitScores:
        if isinstance(payload, dict):
            graph = payload.get("graph", payload)
            if not isinstance(graph, dict):
                graph = {}
            node_scores = cls._score_mapping(graph.get("nodes", {}))
            edge_scores = cls._score_mapping(graph.get("edges", {}))
            direct = cls._score_mapping(
                payload.get(
                    "scores",
                    payload.get("attributions", {}),
                )
            )
            if level == "node":
                scores = node_scores or direct
            elif level == "edge":
                scores = edge_scores or direct
            else:
                scores = edge_scores or node_scores or direct
            uncertainty = cls._score_mapping(payload.get("uncertainty", {}))
            if scores:
                return CircuitScores(
                    scores=scores,
                    uncertainty=uncertainty,
                    node_scores=node_scores,
                    edge_scores=edge_scores,
                    metadata={
                        key: payload[key]
                        for key in (
                            "backend",
                            "backend_revision",
                            "method",
                            "integrated_gradient_steps",
                            "level",
                            "pair_manifest_hash",
                            "pair_count",
                            "bootstrap_replicates",
                            "uncertainty_method",
                            "compatibility_hash",
                            "bootstrap_score_vectors",
                            "bootstrap_resample_indices",
                            "bootstrap_raw_graph_hashes",
                            "primary_raw_graph_hash",
                            "checkpoint_path",
                            "checkpoint_sha256",
                            "base_model_revision",
                            "tokenizer_hash",
                            "circuit_probe_schema_version",
                            "prereg_version",
                            "probe_stages",
                            "semantic_manifest_hash",
                            "semantic_pair_hashes",
                            "tokenized_pair_hashes",
                            "target_strings",
                            "target_token_ids",
                            "target_metric_positions",
                            "intervention_positions",
                        )
                        if key in payload
                    },
                )
        if isinstance(payload, list):
            scores = {}
            uncertainty = {}
            for row in payload:
                if not isinstance(row, dict):
                    continue
                name = row.get("component", row.get("name"))
                value = row.get("score", row.get("attribution"))
                if name is not None and isinstance(
                    value,
                    int | float,
                ):
                    scores[str(name)] = float(value)
                    if isinstance(row.get("uncertainty"), int | float):
                        uncertainty[str(name)] = float(row["uncertainty"])
            if scores:
                return CircuitScores(scores, uncertainty)
        raise ValueError("MIB output does not contain convertible node/edge scores")

    def run(
        self,
        *,
        model: str,
        model_revision: str,
        checkpoint: Path,
        checkpoint_sha256: str,
        level: str,
        steps: int,
        pair_manifest: Path,
        output_dir: Path,
        bootstrap_replicates: int,
        seed: int,
        parity_tolerance: float,
    ) -> CircuitScores:
        pair_payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
        expected_pair_hash = sha256_value(pair_payload.get("pairs", []))
        if pair_payload.get("sha256") != expected_pair_hash:
            raise ValueError("fixed discovery pair manifest hash does not match")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "mib_result.json"
        command = self.command(
            model=model,
            model_revision=model_revision,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            level=level,
            steps=steps,
            pair_manifest=pair_manifest,
            output_path=result_path,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            parity_tolerance=parity_tolerance,
        )
        subprocess.run(
            command,
            cwd=self.repository,
            check=True,
        )
        if not result_path.is_file():
            raise RuntimeError("MIB completed without mib_result.json")
        try:
            scores = self._parse_scores(
                json.loads(result_path.read_text(encoding="utf-8")),
                level=level,
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"MIB score artifact is invalid: {error}") from error
        missing_uncertainty = set(scores.scores) - set(scores.uncertainty)
        if missing_uncertainty:
            raise RuntimeError(
                "MIB result lacks bootstrap uncertainty for: " + ", ".join(sorted(missing_uncertainty)[:10])
            )
        return scores


class TinyExactScreenBackend(ExactPatchingBackend):
    """Named CPU exact-screen backend; not an EAP-IG substitute."""

    version = "tiny-exact-node-screen-v2"
