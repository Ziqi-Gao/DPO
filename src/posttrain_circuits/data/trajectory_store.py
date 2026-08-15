"""Sharded Parquet + safetensors trajectory store with ragged offsets."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.types import TrajectoryRecord


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("TrajectoryStore requires the 'train' extra: pip install -e '.[train]'") from error
    return pa, pq


def _concat_ragged(
    values: list[list[int]], dtype: torch.dtype = torch.long
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flattened: list[int] = []
    for value in values:
        flattened.extend(value)
        offsets.append(len(flattened))
    return torch.tensor(flattened, dtype=dtype), torch.tensor(offsets, dtype=torch.long)


class TrajectoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(
        self,
        records: list[TrajectoryRecord],
        *,
        behavior_policy: dict[str, Any],
        prompt_manifest_hash: str,
        sampling_configuration: dict[str, Any],
        verifier_version: str,
        teacher_version: str | None,
        top_k: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pa, pq = _require_pyarrow()
        if not records:
            raise ValueError("trajectory store cannot be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        for record in records:
            record.validate()

        token_tensors: dict[str, torch.Tensor] = {}
        for name in ("input_ids", "response_ids"):
            flattened, offsets = _concat_ragged([getattr(record, name) for record in records])
            token_tensors[name] = flattened
            token_tensors[f"{name}_offsets"] = offsets
        mask_values = [[int(value) for value in record.response_token_mask] for record in records]
        mask_tokens, mask_offsets = _concat_ragged(mask_values)
        token_tensors["response_token_mask"] = mask_tokens.to(torch.bool)
        token_tensors["response_token_mask_offsets"] = mask_offsets
        logprobs = [value for record in records for value in record.behavior_logprobs]
        logprob_offsets = [0]
        for record in records:
            logprob_offsets.append(logprob_offsets[-1] + len(record.behavior_logprobs))
        token_tensors["behavior_logprobs"] = torch.tensor(logprobs, dtype=torch.float32)
        token_tensors["behavior_logprobs_offsets"] = torch.tensor(logprob_offsets, dtype=torch.long)
        tokens_path = self.root / "tokens-00000.safetensors"
        save_file(token_tensors, tokens_path)

        teacher_tensors: dict[str, torch.Tensor] = {}
        scored = [record for record in records if record.teacher_topk_ids]
        if scored:
            ids = [position for record in records for position in record.teacher_topk_ids]
            logs = [position for record in records for position in record.teacher_topk_logprobs]
            teacher_tensors["topk_ids"] = torch.tensor(ids, dtype=torch.long)
            teacher_tensors["topk_logprobs"] = torch.tensor(logs, dtype=torch.float32)
            teacher_tensors["topk_mass"] = torch.tensor(
                [value for record in records for value in record.teacher_topk_mass],
                dtype=torch.float32,
            )
            teacher_tensors["entropy"] = torch.tensor(
                [value for record in records for value in record.teacher_entropy],
                dtype=torch.float32,
            )
            teacher_offsets = [0]
            for record in records:
                teacher_offsets.append(teacher_offsets[-1] + len(record.teacher_topk_ids))
            teacher_tensors["trajectory_offsets"] = torch.tensor(teacher_offsets, dtype=torch.long)
            save_file(teacher_tensors, self.root / "teacher-00000.safetensors")

        metadata_rows = []
        omitted = {
            "input_ids",
            "response_ids",
            "response_token_mask",
            "behavior_logprobs",
            "teacher_topk_ids",
            "teacher_topk_logprobs",
            "teacher_topk_mass",
            "teacher_entropy",
        }
        for record in records:
            row = {key: value for key, value in asdict(record).items() if key not in omitted}
            row["verification_trace"] = json.dumps(row["verification_trace"], sort_keys=True)
            metadata_rows.append(row)
        metadata_path = self.root / "metadata-00000.parquet"
        pq.write_table(pa.Table.from_pylist(metadata_rows), metadata_path)

        rewards = [float(record.verifier_reward or 0.0) for record in records]
        lengths = [len(record.response_ids) for record in records]
        manifest: dict[str, Any] = {
            "format_version": 1,
            "behavior_policy": behavior_policy,
            "prompt_manifest_hash": prompt_manifest_hash,
            "sampling_configuration": sampling_configuration,
            "verifier_version": verifier_version,
            "teacher_version": teacher_version,
            "top_k": top_k,
            "total_trajectories": len(records),
            "reward_distribution": {
                "mean": sum(rewards) / len(rewards),
                "positive": sum(x > 0 for x in rewards),
            },
            "length_distribution": {
                "minimum": min(lengths),
                "maximum": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
            "effective_supervised_tokens": sum(sum(record.response_token_mask) for record in records),
            "files": {
                metadata_path.name: sha256_file(metadata_path),
                tokens_path.name: sha256_file(tokens_path),
            },
            "created_at": utc_now(),
        }
        if extra_metadata:
            overlap = set(manifest) & set(extra_metadata)
            if overlap:
                raise ValueError(
                    f"extra trajectory-store metadata overwrites reserved keys: {sorted(overlap)}"
                )
            manifest.update(extra_metadata)
        teacher_path = self.root / "teacher-00000.safetensors"
        if teacher_path.exists():
            manifest["files"][teacher_path.name] = sha256_file(teacher_path)
        manifest["sha256"] = sha256_value(
            {key: value for key, value in manifest.items() if key != "created_at"}
        )
        atomic_write_json(self.root / "manifest.json", manifest)
        return manifest

    def check_integrity(self) -> dict[str, Any]:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            actual = sha256_file(self.root / name)
            if actual != expected:
                raise ValueError(f"hash mismatch for {name}: expected {expected}, got {actual}")
        expected_manifest_hash = sha256_value(
            {key: value for key, value in manifest.items() if key not in {"created_at", "sha256"}}
        )
        if manifest.get("sha256") != expected_manifest_hash:
            raise ValueError("trajectory-store manifest hash does not match its contents")
        load_file(self.root / "tokens-00000.safetensors")
        return manifest

    def read(self) -> list[TrajectoryRecord]:
        _, pq = _require_pyarrow()
        manifest = self.check_integrity()
        tokens = load_file(self.root / "tokens-00000.safetensors")
        teacher_path = self.root / "teacher-00000.safetensors"
        teacher = load_file(teacher_path) if teacher_path.exists() else None
        rows = pq.read_table(self.root / "metadata-00000.parquet").to_pylist()
        records: list[TrajectoryRecord] = []
        for index, row in enumerate(rows):

            def ragged(name: str, row_index: int = index) -> list[Any]:
                offsets = tokens[f"{name}_offsets"]
                start = int(offsets[row_index])
                end = int(offsets[row_index + 1])
                return tokens[name][start:end].tolist()

            trace = row.get("verification_trace")
            if isinstance(trace, str):
                row["verification_trace"] = json.loads(trace)
            elif trace is None:
                row["verification_trace"] = None
            row["input_ids"] = [int(value) for value in ragged("input_ids")]
            row["response_ids"] = [int(value) for value in ragged("response_ids")]
            row["response_token_mask"] = [bool(value) for value in ragged("response_token_mask")]
            row["behavior_logprobs"] = [float(value) for value in ragged("behavior_logprobs")]
            if teacher is not None:
                teacher_offsets = teacher["trajectory_offsets"]
                start = int(teacher_offsets[index])
                end = int(teacher_offsets[index + 1])
                row["teacher_topk_ids"] = teacher["topk_ids"][start:end].tolist()
                row["teacher_topk_logprobs"] = teacher["topk_logprobs"][start:end].tolist()
                row["teacher_topk_mass"] = teacher["topk_mass"][start:end].tolist()
                row["teacher_entropy"] = teacher["entropy"][start:end].tolist()
            else:
                row["teacher_topk_ids"] = []
                row["teacher_topk_logprobs"] = []
                row["teacher_topk_mass"] = []
                row["teacher_entropy"] = []
            record = TrajectoryRecord(**row)
            record.validate()
            records.append(record)
        if len(records) != int(manifest["total_trajectories"]):
            raise ValueError(
                f"metadata count {len(records)} != manifest total {manifest['total_trajectories']}"
            )
        return records
