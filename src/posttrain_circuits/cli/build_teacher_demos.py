"""Build an independent exact-verifier-gated teacher demonstration store."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, suppress
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterable

from posttrain_circuits.artifacts.runs import formal_artifact_binding
from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_FIXTURE_UNAVAILABLE,
    TeacherCandidateOutput,
    TeacherDemoAttempt,
)
from posttrain_circuits.datasets.teacher_demos.store import write_teacher_demo_store
from posttrain_circuits.learning.teacher.demo_generation import (
    HfTeacherCandidateGenerator,
    TeacherCandidateGenerator,
    TeacherDemoGenerationConfig,
    generate_teacher_demonstrations,
)
from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer


_DIAGNOSTICS_ROOT = Path("/scr/del6500/OPD/diagnostics/teacher_demos")
_DIAGNOSTIC_FILES = ("ledger.json", "accepted_view.json", "manifest.json")


def _bounded_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(str(value)[:80] for value in values)
    bounded = dict(counts.most_common(16))
    omitted = sum(counts.values()) - sum(bounded.values())
    if omitted:
        bounded["<other>"] = omitted
    return bounded


def _preserve_failure_diagnostics(output: Path) -> Path | None:
    """Copy the exact failed store outside the handler's disposable workspace."""

    with ExitStack() as stack:
        try:
            source_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        stack.callback(os.close, source_fd)
        sources = []
        for name in _DIAGNOSTIC_FILES:
            try:
                descriptor = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd
                )
            except FileNotFoundError:
                if name == "ledger.json":
                    return None
                raise
            source = stack.enter_context(os.fdopen(descriptor, "rb"))
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError(f"diagnostic source is not a regular file: {name}")
            sources.append((name, source))
        if _DIAGNOSTICS_ROOT.resolve() != _DIAGNOSTICS_ROOT:
            raise ValueError("diagnostic root must not contain symlinks")
        _DIAGNOSTICS_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="failure-", dir=_DIAGNOSTICS_ROOT))
        for name, source in sources:
            descriptor = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as target:
                shutil.copyfileobj(source, target)
        return destination


def _write_store_with_failure_diagnostics(
    output: Path, attempts: list[TeacherDemoAttempt], **kwargs: Any
) -> dict[str, Any]:
    diagnostics_eligible = False
    with suppress(OSError):
        diagnostics_eligible = not output.exists() or (output.is_dir() and not any(output.iterdir()))
    try:
        return write_teacher_demo_store(output, attempts, **kwargs)
    except Exception:
        # Diagnostics must neither change scientific acceptance nor replace its error.
        try:
            destination = _preserve_failure_diagnostics(output) if diagnostics_eligible else None
            if destination is not None:
                summary = {
                    "path": str(destination),
                    "failure_code_counts": _bounded_counts(
                        attempt.verification_trace.get("error_code") or "unspecified"
                        for attempt in attempts if not attempt.accepted
                    ),
                    "finish_reason_counts": _bounded_counts(attempt.finish_reason for attempt in attempts),
                }
                print("Teacher-demo failure diagnostics: " + json.dumps(summary, sort_keys=True), file=sys.stderr)
        except Exception as diagnostic_error:
            with suppress(Exception):
                print(
                    "Teacher-demo diagnostic preservation failed: "
                    + json.dumps(str(diagnostic_error)[:240]),
                    file=sys.stderr,
                )
        raise


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
    manifest = _write_store_with_failure_diagnostics(
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
