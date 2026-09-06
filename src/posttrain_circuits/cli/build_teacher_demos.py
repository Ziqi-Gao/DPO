"""Build an independent exact-verifier-gated teacher demonstration store."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from posttrain_circuits.artifacts.runs import formal_artifact_binding
from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_FIXTURE_UNAVAILABLE,
    TeacherCandidateOutput,
)
from posttrain_circuits.datasets.teacher_demos.store import write_teacher_demo_store
from posttrain_circuits.learning.teacher.demo_generation import (
    HfTeacherCandidateGenerator,
    TeacherCandidateGenerator,
    TeacherDemoGenerationConfig,
    generate_teacher_demonstrations,
)
from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer


class SmokeProofTeacher:
    """Deterministic CPU fixture standing in for the pinned production teacher generator."""

    def __init__(self) -> None:
        self.task = ProofGraphTask()

    def __call__(
        self,
        *,
        example: TaskExample,
        candidate_index: int,
        actual_sampling_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> TeacherCandidateOutput:
        del actual_sampling_seed, temperature, top_p, top_k, min_p
        if candidate_index == 0:
            response_text = self.task.canonical_target(example)
        else:
            response_text = f"<proof>\n\n</proof>\n<answer>{1 - example.label}</answer>"
        return TeacherCandidateOutput(
            response_text=response_text,
            response_ids=None,
            token_logprobs=None,
            logprob_status=LOGPROB_FIXTURE_UNAVAILABLE,
            finish_reason="oracle_fixture",
        )


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Build verified teacher demonstrations", argv)
    state_config = config["state_source"]
    output = args.output or Path(str(state_config["store_path"]))
    if not enforce_production_guard(
        config, dry_run=args.dry_run, confirm_production=args.confirm_production, output=output
    ):
        return
    teacher_config = config["teacher"]
    task_config = config["task"]
    family = load_dataset_family(
        Path(str(task_config.get("dataset_family_path", "")))
    )
    family_train = family.examples("train")
    count = int(task_config.get("num_examples", len(family_train)))
    if count < 1 or count > len(family_train):
        raise ValueError("task.num_examples is outside the frozen train family split")
    examples = family_train[:count]
    family_train_ids = {example.example_id for example in family_train}
    candidate_generator: TeacherCandidateGenerator
    if str(teacher_config.get("backend", "")).lower() == "huggingface":
        loaded_teacher = load_model_and_tokenizer(
            teacher_config,
            for_training=False,
        )
        tokenizer = loaded_teacher.tokenizer
        candidate_generator = HfTeacherCandidateGenerator(
            move_model_to_local_cuda(loaded_teacher.model),
            tokenizer,
            max_new_tokens=int(state_config["max_new_tokens"]),
            model_config=teacher_config,
        )
        teacher_id = loaded_teacher.model_id
        teacher_revision = loaded_teacher.requested_model_revision
        resolved_teacher_commit = loaded_teacher.resolved_model_commit
    else:
        tokenizer = build_tiny_tokenizer()
        candidate_generator = SmokeProofTeacher()
        teacher_id = str(teacher_config["teacher_id"])
        teacher_revision = str(teacher_config["teacher_revision"])
        resolved_teacher_commit = str(teacher_config["resolved_teacher_commit"])
    generation_config = TeacherDemoGenerationConfig(
        teacher_id=teacher_id,
        teacher_revision=teacher_revision,
        resolved_teacher_commit=resolved_teacher_commit,
        sampling_request_seed=int(teacher_config["generation_seed"]),
        temperature=float(teacher_config["temperature"]),
        top_p=float(teacher_config["top_p"]),
        top_k=int(teacher_config.get("top_k", 0)),
        min_p=float(teacher_config.get("min_p", 0.0)),
        candidates_per_prompt=int(state_config["num_candidates"]),
        max_prompt_tokens=int(state_config["max_prompt_tokens"]),
        max_new_tokens=int(state_config["max_new_tokens"]),
    )
    result = generate_teacher_demonstrations(
        examples,
        tokenizer,
        candidate_generator,
        generation_config,
        model_config=teacher_config,
    )
    unknown_prompt_ids = {attempt.prompt_id for attempt in result.attempts} - family_train_ids
    if unknown_prompt_ids:
        raise ValueError(
            "teacher-demo prompt IDs are outside the configured train family: "
            f"{sorted(unknown_prompt_ids)}"
        )
    manifest = write_teacher_demo_store(
        output,
        result.attempts,
        ordered_prompt_ids=result.ordered_prompt_ids,
        prompt_manifest_hash=result.prompt_manifest_hash,
        tokenizer_hash=result.tokenizer_hash,
        generation=asdict(result.config),
        protocol_bindings={
            **formal_artifact_binding(config),
            "dataset_family_sha256": str(family.manifest["sha256"]),
            "train_examples_file_sha256": str(
                family.boundary("train")["examples_file_sha256"]
            ),
        },
    )
    print_json({"output": str(output), "manifest": manifest})


if __name__ == "__main__":
    main()
