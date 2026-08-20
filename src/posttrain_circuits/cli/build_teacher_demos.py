"""Build an independent exact-verifier-gated teacher demonstration store."""

from __future__ import annotations

from pathlib import Path

from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.data.splits import build_split
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample
from posttrain_circuits.teacher.demo_generation import (
    HfTeacherCandidateGenerator,
    TeacherCandidateGenerator,
    TeacherDemoGenerationConfig,
    generate_teacher_demonstrations,
    write_teacher_demo_store,
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
        generation_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> str:
        del generation_seed, temperature, top_p, top_k, min_p
        if candidate_index == 0:
            return self.task.canonical_target(example)
        return f"<proof>\n\n</proof>\n<answer>{1 - example.label}</answer>"


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
    count = int(task_config.get("num_examples", 32))
    seed = int(task_config.get("seed", config["seed"]))
    examples = build_split(
        ProofGraphTask(),
        "train",
        count,
        seed,
        dict(task_config),
    )
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
        generation_seed=int(teacher_config["generation_seed"]),
        temperature=float(teacher_config["temperature"]),
        top_p=float(teacher_config["top_p"]),
        top_k=int(teacher_config.get("top_k", 0)),
        min_p=float(teacher_config.get("min_p", 0.0)),
        candidates_per_prompt=int(state_config["num_candidates"]),
    )
    result = generate_teacher_demonstrations(
        examples,
        tokenizer,
        candidate_generator,
        generation_config,
        model_config=teacher_config,
    )
    manifest = write_teacher_demo_store(output, result)
    print_json({"output": str(output), "manifest": manifest})


if __name__ == "__main__":
    main()
