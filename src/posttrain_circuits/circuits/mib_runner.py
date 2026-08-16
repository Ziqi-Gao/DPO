"""Execute pinned MIB attribution on an immutable local pair manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.models.loading import tokenizer_fingerprint


def load_checkpoint_into_hf_model(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> str:
    actual = sha256_file(checkpoint)
    if actual != expected_sha256:
        raise ValueError("checkpoint bytes changed before circuit discovery")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("circuit checkpoint is incomplete: expected a payload with model state")
    try:
        model.load_state_dict(payload["model"], strict=True)
    except RuntimeError as error:
        raise ValueError("circuit checkpoint architecture/state mismatch") from error
    return actual


def _collate(
    rows: list[tuple[str, str, dict[str, Any]]],
) -> tuple[list[str], list[str], tuple[dict[str, Any], ...]]:
    clean, corrupt, labels = zip(*rows, strict=True)
    return list(clean), list(corrupt), labels


class _FixedPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        indices: list[int],
    ) -> None:
        self.items = []
        for index in indices:
            row = rows[index]
            clean_tokens = tokenizer.encode(
                row["clean_prompt"],
                add_special_tokens=False,
            )
            corrupt_tokens = tokenizer.encode(
                row["corrupt_prompt"],
                add_special_tokens=False,
            )
            if clean_tokens != [int(value) for value in row["clean_input_ids"]]:
                raise ValueError(f"pair {row['pair_id']} clean token bytes changed")
            if corrupt_tokens != [int(value) for value in row["corrupt_input_ids"]]:
                raise ValueError(f"pair {row['pair_id']} corrupt token bytes changed")
            if len(clean_tokens) != len(corrupt_tokens):
                raise ValueError(f"pair {row['pair_id']} is not token-shape matched")
            clean_target = [int(value) for value in row["clean_target_ids"]]
            corrupt_target = [int(value) for value in row["corrupt_target_ids"]]
            clean_positions = [int(value) for value in row["clean_metric_positions"]]
            corrupt_positions = [int(value) for value in row["corrupt_metric_positions"]]
            if not clean_target or len(clean_target) != len(corrupt_target):
                raise ValueError(f"pair {row['pair_id']} target sequences are invalid")
            if len(clean_positions) != len(clean_target) or len(corrupt_positions) != len(corrupt_target):
                raise ValueError(f"pair {row['pair_id']} target positions are invalid")
            self.items.append(
                (
                    row["clean_prompt"],
                    row["corrupt_prompt"],
                    {
                        "pair_id": str(row["pair_id"]),
                        "stage": str(row["stage"]),
                        "clean_target_ids": clean_target,
                        "corrupt_target_ids": corrupt_target,
                        "clean_metric_positions": clean_positions,
                        "corrupt_metric_positions": corrupt_positions,
                        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
                    },
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[str, str, dict[str, Any]]:
        return self.items[index]


def _target_sequence_difference(
    circuit_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: Any,
    *,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    del clean_logits
    values = []
    for index, label in enumerate(labels):
        clean_ids = torch.tensor(label["clean_target_ids"], dtype=torch.long, device=circuit_logits.device)
        corrupt_ids = torch.tensor(
            label["corrupt_target_ids"], dtype=torch.long, device=circuit_logits.device
        )
        positions = torch.tensor(
            label["clean_metric_positions"], dtype=torch.long, device=circuit_logits.device
        )
        if label.get("padding_side") == "left":
            positions = positions + (circuit_logits.shape[1] - int(input_length[index]))
        if int(positions.min()) < 0 or int(positions.max()) >= circuit_logits.shape[1]:
            raise ValueError(f"MIB target positions are invalid for pair {label['pair_id']}")
        log_probs = circuit_logits[index].index_select(0, positions).log_softmax(dim=-1)
        clean = log_probs.gather(-1, clean_ids.view(-1, 1)).sum()
        corrupt = log_probs.gather(-1, corrupt_ids.view(-1, 1)).sum()
        values.append(clean - corrupt)
    result = torch.stack(values)
    if loss:
        result = -result
    return result.mean() if mean else result


def _extract_scores(
    graph: dict[str, Any],
    level: str,
) -> dict[str, float]:
    section = "edges" if level == "edge" else "nodes"
    scores = {}
    for name, row in graph.get(section, {}).items():
        if isinstance(row, dict) and isinstance(
            row.get("score"),
            int | float,
        ):
            scores[str(name)] = float(row["score"])
    if not scores:
        raise RuntimeError(f"MIB graph has no complete {section} score mapping")
    return scores


def _run_graph(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    indices: list[int],
    level: str,
    method: str,
    ig_steps: int,
    output_path: Path,
    graph_class: Any,
    attribute_edge: Any,
    attribute_node: Any,
) -> dict[str, Any]:
    dataset = _FixedPairDataset(rows, model.tokenizer, indices)
    dataloader = DataLoader(
        dataset,
        batch_size=min(20, len(dataset)),
        shuffle=False,
        collate_fn=_collate,
    )
    graph = graph_class.from_model(
        model,
        neuron_level=False,
        node_scores=level == "node",
    )
    metric = partial(_target_sequence_difference, mean=True, loss=True)
    if level == "edge":
        attribute_edge(
            model,
            graph,
            dataloader,
            metric,
            method,
            "patching",
            ig_steps=ig_steps,
            intervention_dataloader=dataloader,
        )
    else:
        attribute_node(
            model,
            graph,
            dataloader,
            metric,
            method,
            "patching",
            neuron=False,
            ig_steps=ig_steps,
            intervention_dataloader=dataloader,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_json(str(output_path))
    return json.loads(output_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pinned fixed-pair MIB EAP-IG runner")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--level", choices=("node", "edge"), required=True)
    parser.add_argument("--ig-steps", type=int, required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    actual_revision = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.repository),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        env={key: value for key, value in os.environ.items() if key not in {"GIT_DIR", "GIT_WORK_TREE"}},
    ).strip()
    if actual_revision != args.expected_revision:
        raise RuntimeError("MIB revision changed after adapter validation")
    eap_root = args.repository / "EAP-IG"
    if not (eap_root / "eap").is_dir():
        raise RuntimeError(
            "MIB EAP-IG submodule is unavailable; run "
            "'git submodule update --init --recursive' in the pinned checkout"
        )
    sys.path.insert(0, str(eap_root))
    sys.path.insert(0, str(args.repository))
    from eap.attribute import attribute
    from eap.attribute_node import attribute_node
    from eap.graph import Graph
    from transformer_lens import HookedTransformer
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from posttrain_circuits.circuits.model_adapter import (
        require_compatible_for_extraction,
        require_transformerlens_parity,
    )

    pair_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    if pair_payload.get("circuit_probe_schema_version") != "circuit-probe-v2-stage-sequence":
        raise ValueError("MIB requires the core-v2 stage-sequence probe schema")
    rows = pair_payload["pairs"]
    if pair_payload.get("sha256") != sha256_value(rows):
        raise ValueError("fixed pair manifest hash mismatch")
    if args.bootstrap_replicates < 2:
        raise ValueError("bootstrap uncertainty needs >=2 replicates")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    hf_model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    actual_checkpoint_hash = load_checkpoint_into_hf_model(
        hf_model,
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    hf_model = hf_model.to(device).eval()
    hf_model.config.use_cache = False
    model = HookedTransformer.from_pretrained(
        args.model,
        revision=args.model_revision,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        attn_implementation="eager",
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    ).eval()
    parity_ids = tokenizer.encode(
        rows[0]["clean_prompt"],
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    compatibility = require_transformerlens_parity(
        hf_model,
        model,
        parity_ids,
        tolerance=args.parity_tolerance,
        output_path=args.output.parent / "compatibility.json",
    )
    require_compatible_for_extraction(
        compatibility,
        require_transformerlens=True,
    )
    compatibility_hash = compatibility.sha256
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    raw_root = args.output.parent / "raw_graphs"
    primary_graph = _run_graph(
        model=model,
        rows=rows,
        indices=list(range(len(rows))),
        level=args.level,
        method=args.method,
        ig_steps=args.ig_steps,
        output_path=raw_root / "primary.json",
        graph_class=Graph,
        attribute_edge=attribute,
        attribute_node=attribute_node,
    )
    rng = random.Random(args.seed)
    bootstrap_scores = []
    bootstrap_resample_indices = []
    bootstrap_raw_graph_hashes = []
    for replicate in range(args.bootstrap_replicates):
        indices = [rng.randrange(len(rows)) for _ in range(len(rows))]
        bootstrap_resample_indices.append(indices)
        graph_path = raw_root / f"bootstrap-{replicate:04d}.json"
        graph = _run_graph(
            model=model,
            rows=rows,
            indices=indices,
            level=args.level,
            method=args.method,
            ig_steps=args.ig_steps,
            output_path=graph_path,
            graph_class=Graph,
            attribute_edge=attribute,
            attribute_node=attribute_node,
        )
        bootstrap_scores.append(_extract_scores(graph, args.level))
        bootstrap_raw_graph_hashes.append(sha256_file(graph_path))
    primary_scores = _extract_scores(primary_graph, args.level)
    uncertainty = {}
    for component in primary_scores:
        values = [scores[component] for scores in bootstrap_scores if component in scores]
        if len(values) != args.bootstrap_replicates:
            raise RuntimeError(f"bootstrap graph omitted component {component}")
        uncertainty[component] = statistics.stdev(values) / math.sqrt(len(values))
    atomic_write_json(
        args.output,
        {
            "backend": "mib-eap-ig",
            "backend_revision": args.expected_revision,
            "method": args.method,
            "integrated_gradient_steps": args.ig_steps,
            "level": args.level,
            "pair_manifest_hash": pair_payload["sha256"],
            "circuit_probe_schema_version": pair_payload["circuit_probe_schema_version"],
            "prereg_version": pair_payload.get("prereg_version"),
            "probe_stages": sorted({str(row["stage"]) for row in rows}),
            "semantic_manifest_hash": sha256_value(
                sorted({str(row["semantic_manifest_hash"]) for row in rows})
            ),
            "semantic_pair_hashes": [str(row["semantic_pair_hash"]) for row in rows],
            "tokenized_pair_hashes": [str(row["tokenized_pair_hash"]) for row in rows],
            "target_strings": [[str(row["clean_target"]), str(row["corrupt_target"])] for row in rows],
            "target_token_ids": [[row["clean_target_ids"], row["corrupt_target_ids"]] for row in rows],
            "target_metric_positions": [
                [row["clean_metric_positions"], row["corrupt_metric_positions"]] for row in rows
            ],
            "intervention_positions": [
                [row["clean_intervention_positions"], row["corrupt_intervention_positions"]] for row in rows
            ],
            "pair_count": len(rows),
            "bootstrap_replicates": args.bootstrap_replicates,
            "uncertainty_method": "prompt_bootstrap_standard_error",
            "compatibility_hash": compatibility_hash,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": actual_checkpoint_hash,
            "base_model_revision": args.model_revision,
            "tokenizer_hash": tokenizer_fingerprint(tokenizer),
            "graph": primary_graph,
            "primary_raw_graph_hash": sha256_file(raw_root / "primary.json"),
            "scores": primary_scores,
            "uncertainty": uncertainty,
            "bootstrap_score_vectors": bootstrap_scores,
            "bootstrap_resample_indices": bootstrap_resample_indices,
            "bootstrap_raw_graph_hashes": bootstrap_raw_graph_hashes,
        },
    )


if __name__ == "__main__":
    main()
