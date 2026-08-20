"""Small Hydra-style configuration composer used by all CLIs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

QWEN3_STUDENT = "Qwen/Qwen3-1.7B"
QWEN3_TEACHER = "Qwen/Qwen3-8B"
QWEN3_STUDENT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QWEN3_TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN3_CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_value(value: str) -> Any:
    return yaml.safe_load(value)


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override {path}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def _load_group(config_root: Path, group: str, name: str) -> dict[str, Any]:
    path = config_root / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"configuration group {group}={name} does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"configuration group {group}={name} must contain a mapping")
    return payload


def compose_config(
    overrides: list[str],
    *,
    config_root: Path = Path("configs"),
    root_name: str = "config.yaml",
) -> dict[str, Any]:
    root = yaml.safe_load((config_root / root_name).read_text(encoding="utf-8")) or {}
    defaults = root.pop("defaults", [])
    default_groups: dict[str, str] = {}
    for entry in defaults:
        if isinstance(entry, dict):
            default_groups.update({str(key): str(value) for key, value in entry.items()})

    explicit_groups: dict[str, str] = {}
    scalar_overrides: list[tuple[str, str]] = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, received {override!r}")
        key, value = override.split("=", 1)
        if "." not in key and (config_root / key).is_dir():
            explicit_groups[key] = value
        else:
            scalar_overrides.append((key, value))

    config: dict[str, Any] = {}
    for group, name in default_groups.items():
        config[group] = _load_group(config_root, group, name)
    config = _merge(config, root)

    experiment_name = explicit_groups.get("experiment", default_groups.get("experiment", "offline_hard"))
    config["experiment"] = _load_group(config_root, "experiment", experiment_name)

    for dependency in ("state_source", "supervision"):
        dependency_name = config["experiment"].get(dependency)
        if dependency_name is not None:
            config[dependency] = _load_group(config_root, dependency, str(dependency_name))

    for group, name in explicit_groups.items():
        if group == "experiment":
            continue
        payload = _load_group(config_root, group, name)
        if group in {"production", "pilot", "g0"}:
            selected_groups = payload.pop("groups", {})
            profile_overrides = payload.pop("overrides", {})
            if not isinstance(selected_groups, dict) or not isinstance(profile_overrides, dict):
                raise TypeError(f"{group} profile groups/overrides must be mappings")
            for selected_group, selected_name in selected_groups.items():
                config[str(selected_group)] = _load_group(
                    config_root, str(selected_group), str(selected_name)
                )
            config = _merge(config, profile_overrides)
            config[f"{group}_profile"] = name
            config[group] = payload
        else:
            config[group] = payload

    for key, value in scalar_overrides:
        _set_path(config, key, _parse_value(value))
    validate_config(config)
    return config


def validate_model_revision(model: dict[str, Any]) -> None:
    required = {
        "model_name_or_path",
        "model_revision",
        "tokenizer_name_or_path",
        "tokenizer_revision",
        "torch_dtype",
        "attn_implementation",
        "gradient_checkpointing",
        "use_cache",
        "trust_remote_code",
    }
    missing = sorted(required - model.keys())
    if missing:
        raise ValueError(f"model configuration is missing required keys: {missing}")
    allow_unpinned = bool(model.get("allow_unpinned_revision", False))
    for key in ("model_revision", "tokenizer_revision"):
        if str(model[key]).lower() in {"main", "master", "latest", ""} and not allow_unpinned:
            raise ValueError(
                f"official configuration rejects unpinned {key}={model[key]!r}; "
                "set allow_unpinned_revision=true explicitly to override"
            )
    model_id = str(model["model_name_or_path"])
    if model_id in {QWEN3_STUDENT, QWEN3_TEACHER}:
        expected_revision = QWEN3_STUDENT_REVISION if model_id == QWEN3_STUDENT else QWEN3_TEACHER_REVISION
        if model["model_revision"] != expected_revision or model["tokenizer_revision"] != expected_revision:
            raise ValueError(f"Qwen3 model/tokenizer revisions must both equal {expected_revision}")
        if model.get("tokenizer_name_or_path") != model_id:
            raise ValueError("Qwen3 tokenizer must come from the same pinned model repository")
        protocol = model.get("prompt_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("Qwen3 requires an explicit prompt_protocol mapping")
        expected_protocol = {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "messages": "single_user",
            "add_generation_prompt": True,
            "chat_template_sha256": QWEN3_CHAT_TEMPLATE_SHA256,
        }
        if {key: protocol.get(key) for key in expected_protocol} != expected_protocol:
            raise ValueError("Qwen3 prompt protocol differs from qwen3_non_thinking_v1")
        sampling = model.get("sampling_protocol")
        expected_sampling = {
            "name": "qwen3_non_thinking_sampling_v1",
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        }
        if (
            not isinstance(sampling, dict)
            or {key: sampling.get(key) for key in expected_sampling} != expected_sampling
        ):
            raise ValueError("Qwen3 sampling protocol is not fully pinned")
        if model.get("trust_remote_code") is not False:
            raise ValueError("Qwen3 controlled runs require trust_remote_code=false")
        if model.get("torch_dtype") != "bfloat16" or model.get("attn_implementation") != "sdpa":
            raise ValueError("Qwen3 controlled runs require BF16 and SDPA")
        expected_training = model_id == QWEN3_STUDENT
        if bool(model.get("gradient_checkpointing")) != expected_training:
            raise ValueError("Qwen3 student/teacher gradient-checkpointing role mismatch")
        if bool(model.get("use_cache")) == expected_training:
            raise ValueError("Qwen3 student must disable cache and teacher must enable cache")


def validate_config(config: dict[str, Any]) -> None:
    if "model" in config:
        validate_model_revision(config["model"])
    experiment = config.get("experiment", {})
    for dependency in ("state_source", "supervision"):
        expected = experiment.get(dependency)
        if expected is None:
            continue
        actual = config.get(dependency, {}).get("name")
        if actual != expected:
            raise ValueError(
                f"experiment {experiment.get('name')} requires {dependency}={expected}, "
                f"but resolved {dependency}.name={actual}"
            )
    if experiment.get("supervision") == "soft_teacher" and experiment.get("use_verifier_reward", False):
        raise ValueError("OPD/soft-teacher cells cannot enable verifier reward")
    if experiment.get("name", "").startswith("offline") and experiment.get("state_source") != "fixed_bank":
        raise ValueError("offline factorial cells must use the fixed common bank")
    if "production_profile" in config or "pilot_profile" in config:
        validate_production_training_config(config)
    if config.get("protocol_track") == "qwen3_v1":
        _validate_qwen3_track(config)


def _validate_qwen3_track(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    teacher = config.get("teacher", {})
    validate_model_revision(model)
    validate_model_revision(teacher)
    pair = (model.get("model_name_or_path"), teacher.get("model_name_or_path"))
    if pair != (QWEN3_STUDENT, QWEN3_TEACHER):
        raise ValueError(f"qwen3_v1 requires the exact registered model pair, observed={pair}")
    if str(config.get("prereg_path")) != "prereg/qwen3_v1.yaml":
        raise ValueError("qwen3_v1 must bind prereg/qwen3_v1.yaml")
    output_root = str(config.get("output_root", ""))
    if "qwen3-v1" not in Path(output_root).parts:
        raise ValueError("qwen3_v1 output_root must use the qwen3-v1 artifact namespace")
    bound_paths = {
        "state_source.store_path": config.get("state_source", {}).get("store_path"),
        "task.validation_split_path": config.get("task", {}).get("validation_split_path"),
        "anti_shortcut.report_path": config.get("anti_shortcut", {}).get("report_path"),
        "production_safety.readiness_report": config.get("production_safety", {}).get("readiness_report"),
        "production_safety.probe_cohort_manifest": config.get("production_safety", {}).get(
            "probe_cohort_manifest"
        ),
        "production_safety.initial_checkpoint_path": config.get("production_safety", {}).get(
            "initial_checkpoint_path"
        ),
    }
    wrong_paths = [
        name
        for name, value in bound_paths.items()
        if value is not None and str(value).strip() and "qwen3-v1" not in Path(str(value)).parts
    ]
    if wrong_paths:
        raise ValueError(f"qwen3_v1 artifact paths escape their namespace: {wrong_paths}")
    for section_name in ("state_source", "supervision"):
        section = config.get(section_name, {})
        for key, expected in (("temperature", 0.7), ("top_p", 0.8), ("top_k", 20), ("min_p", 0.0)):
            if key in section and section[key] != expected:
                raise ValueError(f"qwen3_v1 {section_name}.{key} must equal {expected}")


def validate_production_training_config(config: dict[str, Any]) -> None:
    """Reject production profiles that inherit any smoke/default execution semantics."""

    failures: list[str] = []
    model = str(config.get("model", {}).get("model_name_or_path", ""))
    teacher = str(config.get("teacher", {}).get("model_name_or_path", ""))
    task = str(config.get("task", {}).get("name", ""))
    trainer = config.get("trainer", {})
    profile = config.get("production", config.get("pilot", {}))
    if model.startswith("local/") or "tiny" in model.lower():
        failures.append("local/tiny model")
    if teacher.startswith("local/") or "tiny" in teacher.lower():
        failures.append("local/tiny teacher")
    allowed_model_teacher_pairs = {
        (
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        (QWEN3_STUDENT, QWEN3_TEACHER),
        (
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
        ),
    }
    if (model, teacher) not in allowed_model_teacher_pairs:
        failures.append(f"wrong production model/teacher pair: {model!r} / {teacher!r}")
    if task != "proofgraph":
        failures.append(f"task={task!r}")
    if str(trainer.get("backend")) != "accelerate":
        failures.append(f"trainer.backend={trainer.get('backend')!r}")
    if int(trainer.get("max_steps", 0)) <= int(
        config.get("production_safety", {}).get("max_smoke_steps", 100)
    ):
        failures.append("smoke-sized max_steps")
    if int(trainer.get("token_budget", 0)) <= int(
        config.get("production_safety", {}).get("max_smoke_tokens", 1_000_000)
    ):
        failures.append("smoke-sized token_budget")
    evaluation_every = int(trainer.get("evaluation_every", 0))
    if evaluation_every < 1 or evaluation_every > int(trainer.get("max_steps", 0)):
        failures.append("invalid formal evaluation interval")
    if not str(config.get("task", {}).get("validation_split_path", "")).strip():
        failures.append("missing frozen validation_split_path")
    if profile.get("full_parameter_training") is not True:
        failures.append("full_parameter_training is not true")
    if failures:
        raise ValueError("invalid production resolved config: " + ", ".join(failures))


def is_production_scale(config: dict[str, Any]) -> bool:
    model = str(config.get("model", {}).get("model_name_or_path", ""))
    teacher_config = config.get("teacher", {})
    teacher = str(
        teacher_config.get(
            "model_name_or_path",
            teacher_config.get("teacher_id", "local/unspecified"),
        )
    )
    max_steps = int(config.get("trainer", {}).get("max_steps", 0))
    token_budget = int(config.get("trainer", {}).get("token_budget", 0))
    return (
        not model.startswith("local/")
        or not teacher.startswith("local/")
        or max_steps > 100
        or token_budget > 1_000_000
    )
