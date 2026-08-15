"""Execute pinned MIB attribution on an immutable local pair manifest."""

from __future__ import annotations

import argparse
import json
import math
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
    rows: list[tuple[str, str, tuple[int, int]]],
) -> tuple[list[str], list[str], tuple[tuple[int, int], ...]]:
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
            if len(clean_tokens) != len(corrupt_tokens):
                raise ValueError(f"pair {row['pair_id']} is not token-shape matched")
            clean_target = tokenizer.encode(
                row["clean_target"],
                add_special_tokens=False,
            )
            corrupt_target = tokenizer.encode(
                row["corrupt_target"],
                add_special_tokens=False,
            )
            if len(clean_target) != 1 or len(corrupt_target) != 1:
                raise ValueError(f"pair {row['pair_id']} targets must each be one token")
            self.items.append(
                (
                    row["clean_prompt"],
                    row["corrupt_prompt"],
                    (clean_target[0], corrupt_target[0]),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[str, str, tuple[int, int]]:
        return self.items[index]


def _logit_difference(
    circuit_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: Any,
    *,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    del clean_logits
    positions = input_length - 1
    batch = torch.arange(
        circuit_logits.shape[0],
        device=circuit_logits.device,
    )
    final_logits = circuit_logits[batch, positions]
    label_tensor = torch.as_tensor(
        labels,
        dtype=torch.long,
        device=final_logits.device,
    )
    selected = torch.gather(final_logits, -1, label_tensor)
    result = selected[:, 0] - selected[:, 1]
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
    metric = partial(_logit_difference, mean=True, loss=True)
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
            "pair_count": len(rows),
            "bootstrap_replicates": args.bootstrap_replicates,
            "uncertainty_method": "prompt_bootstrap_standard_error",
            "compatibility_hash": compatibility_hash,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": actual_checkpoint_hash,
            "base_model_revision": args.model_revision,
            "tokenizer_hash": sha256_value(tokenizer.get_vocab()),
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
