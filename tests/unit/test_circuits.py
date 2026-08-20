from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from posttrain_circuits.circuits.dynamics import (
    attribution_rank_stability,
    continuous_churn,
    locking_time_bootstrap,
    summarize_dynamics,
    thresholded_churn,
    weighted_overlap,
)
from posttrain_circuits.circuits.exact_patching import (
    ExactPatchingBackend,
    ExactTokenPair,
    component_metadata,
    normalize_circuit_scores,
)
from posttrain_circuits.circuits.faithfulness import (
    REQUIRED_SPARSITY_GRID,
    faithfulness_sparsity_curve,
    integrate_curve,
)
from posttrain_circuits.circuits.graph import (
    AblationSpec,
    CircuitArtifact,
    CircuitEvaluation,
    CircuitMask,
)
from posttrain_circuits.circuits.masks import (
    layer_matched_random_mask,
    top_mask,
)
from posttrain_circuits.circuits.mib_eap_ig import MibEapIgAdapter
from posttrain_circuits.circuits.model_adapter import (
    build_transformerlens_qwen_from_hf,
    check_hf_identity_compatibility,
    q_to_kv_head_mapping,
    require_compatible_for_extraction,
    require_transformerlens_parity,
)
from posttrain_circuits.circuits.plots import (
    write_attribution_patching_calibration,
)
from posttrain_circuits.circuits.tiny_eap_ig import TinyEapIgBackend
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_qwen3


@pytest.mark.unit
def test_genuine_hf_transformerlens_parity_and_extraction_gate(
    tiny_model,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    hf_model = tiny_model.eval()
    tokens = torch.tensor([[2, 4, 5]])
    identity = check_hf_identity_compatibility(
        hf_model,
        tokens,
    )
    assert identity.q_to_kv_mapping == (0, 0)
    assert identity.layer_count == len(hf_model.model.layers)
    assert identity.hook_positions == {
        "residual_stream": ("model.layers.0:output",),
        "mlp_output": ("model.layers.0.mlp:output",),
    }
    with pytest.raises(RuntimeError, match="no passing"):
        require_compatible_for_extraction(
            identity,
            require_transformerlens=True,
        )
    tl_model = build_transformerlens_qwen_from_hf(hf_model)
    report_path = tmp_path / "compatibility.json"
    report = require_transformerlens_parity(
        hf_model,
        tl_model,
        tokens,
        tolerance=1e-6,
        output_path=report_path,
    )
    require_compatible_for_extraction(
        report,
        require_transformerlens=True,
    )
    assert report.transformerlens_logit_error is not None
    assert report.transformerlens_logit_error < 1e-6
    assert report.transformerlens_layer_count == report.layer_count
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["hf_identity_max_error"] == 0.0
    assert written["transformerlens_logit_error"] < 1e-6
    with torch.no_grad():
        tl_model.unembed.W_U[0, 0] += 0.1
    with pytest.raises(RuntimeError, match="extraction refused"):
        require_transformerlens_parity(
            hf_model,
            tl_model,
            tokens,
            tolerance=1e-6,
        )


@pytest.mark.unit
def test_tiny_qwen3_hf_transformerlens_parity_head_dim_gqa_and_qk_hook_semantics() -> None:
    hf_model = build_tiny_qwen3(71).eval()
    assert hf_model.config.head_dim == 8
    assert hf_model.config.hidden_size // hf_model.config.num_attention_heads == 6
    tokens = torch.tensor([[2, 4, 5]])
    identity = check_hf_identity_compatibility(hf_model, tokens)
    assert identity.architecture == "Qwen3ForCausalLM"
    assert identity.q_to_kv_mapping == (0, 0, 1, 1)
    assert identity.hook_positions["query_projection"] == (
        "model.layers.0.self_attn.q_proj:output:pre_q_norm_pre_rope",
    )
    assert identity.hook_positions["key_projection"] == (
        "model.layers.0.self_attn.k_proj:output:pre_k_norm_pre_rope",
    )
    metadata = component_metadata(hf_model)
    assert metadata["layer.0.q_head.3"]["head_width"] == 8
    assert metadata["layer.0.q_head.3"]["semantic_position"] == (
        "q_projection_output_pre_q_norm_pre_rope:kv_head=1"
    )
    assert metadata["layer.0.k_head.1"]["semantic_position"] == ("k_projection_output_pre_k_norm_pre_rope")
    tl_model = build_transformerlens_qwen_from_hf(hf_model)
    assert tl_model.cfg.original_architecture == "Qwen3ForCausalLM"
    assert tl_model.cfg.d_head == 8
    assert tl_model.cfg.n_key_value_heads == 2
    assert tl_model.cfg.use_qk_norm is True
    report = require_transformerlens_parity(
        hf_model,
        tl_model,
        tokens,
        tolerance=1e-6,
    )
    assert report.transformerlens_parity_passed is True


@pytest.mark.unit
def test_gqa_mapping_validation() -> None:
    assert q_to_kv_head_mapping(4, 2) == (0, 0, 1, 1)
    with pytest.raises(ValueError):
        q_to_kv_head_mapping(3, 2)


@pytest.mark.unit
def test_exact_head_gqa_and_selected_path_patching(
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    model = tiny_model.eval()
    discovery = ExactTokenPair(
        "discovery",
        torch.tensor([[2, 4, 5, 6]]),
        torch.tensor([[2, 4, 8, 6]]),
    )
    validation = [
        ExactTokenPair(
            "heldout-1",
            torch.tensor([[2, 5, 6, 7]]),
            torch.tensor([[2, 5, 9, 7]]),
        ),
        ExactTokenPair(
            "heldout-2",
            torch.tensor([[2, 6, 7, 8]]),
            torch.tensor([[2, 6, 10, 8]]),
        ),
    ]
    backend = ExactPatchingBackend(
        discovery.clean_ids,
        discovery.corrupt_ids,
    )

    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, 10].mean() - logits[:, -1, 11].mean()

    scores = backend.score_all_components(model, metric).scores
    assert set(scores) == {
        "layer.0.attention_head.0.output",
        "layer.0.attention_head.1.output",
        "layer.0.q_head.0",
        "layer.0.q_head.1",
        "layer.0.k_head.0",
        "layer.0.v_head.0",
        "layer.0.mlp_out",
        "layer.0.resid_out",
    }
    metadata = component_metadata(model)
    assert metadata["layer.0.q_head.1"]["semantic_position"] == "q_projection_output_pre_rope:kv_head=0"
    path = backend.score_path(
        model,
        metric,
        sender="layer.0.attention_head.0.output",
        receiver="layer.0.mlp_out",
    )
    assert set(path.edge_scores) == {"layer.0.attention_head.0.output->layer.0.mlp_out"}
    assert math.isfinite(next(iter(path.edge_scores.values())))
    normalized, unsupported = normalize_circuit_scores(
        model,
        {
            "a0.h1->m0": 0.5,
            "a0.h1->a0.h1<k>": 0.25,
            "input->m0": 0.1,
        },
    )
    assert unsupported == ["input->m0"]
    assert normalized == {
        "layer.0.attention_head.1.output->layer.0.mlp_out": 0.5,
        "layer.0.attention_head.1.output->layer.0.k_head.0": 0.25,
    }
    path_mask = CircuitMask(
        ("layer.0.attention_head.0.output->layer.0.mlp_out",),
        0.5,
    )
    path_evaluation = backend.evaluate_mask(
        model,
        validation,
        path_mask,
        AblationSpec("counterfactual_replacement"),
        metric,
    )
    assert math.isfinite(path_evaluation.necessity)
    assert path_evaluation.sufficiency is not None
    mask = CircuitMask(
        ("layer.0.attention_head.0.output",),
        1 / len(scores),
    )
    evaluation = backend.evaluate_mask(
        model,
        validation,
        mask,
        AblationSpec("counterfactual_replacement"),
        metric,
    )
    assert math.isfinite(evaluation.necessity)
    assert evaluation.sufficiency is not None
    with pytest.raises(ValueError, match="held-out"):
        backend.evaluate_mask(
            model,
            [],
            mask,
            AblationSpec("zero"),
            metric,
        )


class _FastFaithfulnessBackend:
    def __init__(self, universe: tuple[str, ...]) -> None:
        self.universe = universe

    def activation_statistics(
        self,
        model: object,
        names: tuple[str, ...],
    ) -> dict[str, dict[str, float]]:
        del model
        assert names == self.universe
        return {
            name: {
                "activation_size": float(8 + index % 3),
                "activation_norm": float(1 + index / 10),
            }
            for index, name in enumerate(self.universe)
        }

    def evaluate_mask_per_pair(
        self,
        model: object,
        pairs: list[object],
        mask: CircuitMask,
        ablation: AblationSpec,
        metric: object,
    ) -> list[CircuitEvaluation]:
        del model, ablation, metric
        fraction = len(mask.components) / len(self.universe)
        return [
            CircuitEvaluation(
                clean_metric=1.0 + index * 0.1,
                corrupt_metric=0.0,
                patched_metric=1.0 - fraction,
                faithfulness=fraction,
                necessity=fraction,
                sufficiency=fraction,
            )
            for index, _ in enumerate(pairs)
        ]


@pytest.mark.unit
def test_faithfulness_grid_controls_bootstrap_and_calibration_plot(
    tmp_path: Path,
) -> None:
    scores = {f"layer.{index // 10}.component.{index}": float(20 - index) for index in range(20)}
    backend = _FastFaithfulnessBackend(tuple(scores))
    patching_scores = {name: value * 0.5 for name, value in scores.items()}
    result = faithfulness_sparsity_curve(
        backend,
        object(),
        scores,
        object(),
        list(REQUIRED_SPARSITY_GRID),
        [object(), object(), object()],
        patching_scores=patching_scores,
        random_repeats=3,
        bootstrap_samples=40,
    )
    assert result["sparsity_grid"] == list(REQUIRED_SPARSITY_GRID)
    assert len(result["random_controls"]) == 8
    assert all(len(controls) == 3 for controls in result["random_controls"])
    assert result["random_matching"] == [
        "layer",
        "activation_size",
        "activation_norm",
    ]
    assert result["cpr_ci"]["lower"] <= result["cpr"]
    assert result["cpr"] <= result["cpr_ci"]["upper"]
    assert result["cmd_ci"]["lower"] <= result["cmd_ci"]["upper"]
    assert result["attribution_patching_spearman"] == pytest.approx(1.0)
    paths = write_attribution_patching_calibration(
        result["calibration"],
        spearman=result["attribution_patching_spearman"],
        output_prefix=tmp_path / "calibration",
    )
    assert Path(paths["png"]).is_file()
    assert Path(paths["pdf"]).is_file()


@pytest.mark.unit
def test_mib_parser_preserves_full_graph_and_uncertainty() -> None:
    payload = {
        "backend": "mib-eap-ig",
        "backend_revision": "commit",
        "method": "EAP-IG-inputs",
        "pair_count": 4,
        "uncertainty_method": "prompt_bootstrap_standard_error",
        "graph": {
            "nodes": {
                "a0.h0": {"score": 0.2, "in_graph": True},
                "m0": {"score": -0.1, "in_graph": False},
            },
            "edges": {
                "a0.h0->m0": {"score": 0.3, "in_graph": True},
                "m0->logits": {"score": -0.4, "in_graph": True},
            },
        },
        "uncertainty": {
            "a0.h0->m0": 0.02,
            "m0->logits": 0.03,
        },
    }
    parsed = MibEapIgAdapter._parse_scores(payload, level="edge")
    assert parsed.scores == {
        "a0.h0->m0": 0.3,
        "m0->logits": -0.4,
    }
    assert len(parsed.node_scores) == 2
    assert len(parsed.edge_scores) == 2
    assert parsed.uncertainty["a0.h0->m0"] == 0.02
    assert parsed.metadata["method"] == "EAP-IG-inputs"


@pytest.mark.unit
def test_tiny_eap_ig_writes_base_and_sft_artifacts(
    tmp_path: Path,
) -> None:
    pairs = [
        ExactTokenPair(
            "p1",
            torch.tensor([[2, 4, 5]]),
            torch.tensor([[2, 4, 6]]),
        ),
        ExactTokenPair(
            "p2",
            torch.tensor([[2, 7, 8]]),
            torch.tensor([[2, 7, 9]]),
        ),
    ]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, 10].mean() - logits[:, -1, 11].mean()

    base = build_tiny_qwen(101).eval()
    sft = copy.deepcopy(base).train()
    optimizer = torch.optim.AdamW(sft.parameters(), lr=1e-3)
    loss = -sft(input_ids=pairs[0].clean_ids).logits[:, -1, 10].mean()
    loss.backward()
    optimizer.step()
    sft.eval()
    backend = TinyEapIgBackend(
        pairs,
        integrated_gradient_steps=2,
    )
    base_scores = backend.score_all_components(base, metric)
    sft_scores = backend.score_all_components(sft, metric)
    assert base_scores.uncertainty
    assert sft_scores.uncertainty
    assert base_scores.scores != sft_scores.scores
    for checkpoint, scores in (
        ("base", base_scores),
        ("tiny-sft", sft_scores),
    ):
        artifact = CircuitArtifact(
            run_id="tiny-smoke",
            checkpoint_id=checkpoint,
            task_manifest_hash="task",
            pair_manifest_hash="pairs",
            backend_version=backend.version,
            model_compatibility_hash="compatibility",
            node_or_edge_level="node",
            integrated_gradient_steps=2,
            ablation_baseline="counterfactual_replacement",
            scores=scores.scores,
            score_uncertainty=scores.uncertainty,
            node_scores=scores.node_scores,
            backend_name="tiny-hf-eap-ig",
            backend_revision="in-repository-v1",
            attribution_method=backend.method,
            discovery_pair_count=2,
            uncertainty_method="prompt_standard_error",
        )
        path = tmp_path / f"{checkpoint}.json"
        artifact.write(path)
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["scores"]
        assert written["score_uncertainty"]
        assert written["integrated_gradient_steps"] == 2


@pytest.mark.unit
def test_masks_curves_dynamics_and_serialization(
    tmp_path: Path,
) -> None:
    scores = {
        "layer.0.a": 3.0,
        "layer.0.b": 1.0,
        "layer.1.c": 2.0,
    }
    mask = top_mask(scores, 2 / 3)
    assert mask.components == ("layer.0.a", "layer.1.c")
    random_mask = layer_matched_random_mask(
        tuple(scores),
        mask,
        3,
    )
    assert len(random_mask.components) == len(mask.components)
    assert integrate_curve([(0.0, 0.0), (1.0, 1.0)]) == pytest.approx(0.5)
    assert weighted_overlap(scores, scores) == pytest.approx(1.0)
    assert attribution_rank_stability(
        scores,
        scores,
    ) == pytest.approx(1.0)
    assert continuous_churn(scores, scores) == pytest.approx(0.0)
    assert thresholded_churn(
        scores,
        {"layer.0.a": 3.0},
        threshold=1.5,
    ) == pytest.approx(0.5)
    series = [
        {"edge": 0.0},
        {"edge": 0.2},
        {"edge": 0.3},
    ]
    summary = summarize_dynamics(
        series,
        [0, 10, 20],
        activation_threshold=0.1,
        churn_tolerance=0.5,
    )
    assert summary["edge_lifecycle"]["edge"] == "newly_born"
    interval = locking_time_bootstrap(
        [series, series],
        [0, 10, 20],
        churn_tolerance=0.5,
        samples=40,
        seed=1,
    )
    assert interval["lower"] <= interval["estimate"]
    artifact = CircuitArtifact(
        "run",
        "step-1",
        "task",
        "pairs",
        "backend",
        "compat",
        "node",
        5,
        "counterfactual_replacement",
        scores,
        {},
    )
    path = tmp_path / "circuit.json"
    artifact.write(path)
    assert '"scores"' in path.read_text(encoding="utf-8")
