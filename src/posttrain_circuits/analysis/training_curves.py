"""Budget accounting helpers."""

from __future__ import annotations


def nearest_budget_point(records: list[dict[str, float]], key: str, target: float) -> dict[str, float]:
    if not records or any(key not in record for record in records):
        raise ValueError(f"records must contain budget key {key!r}")
    return min(records, key=lambda record: abs(record[key] - target))


MATCHING_KEYS = {
    "generated_tokens": "response_tokens_generated",
    "supervised_tokens": "supervised_response_tokens",
    "validation_accuracy": "validation_accuracy",
    "output_kl": "output_kl_from_initial",
    "parameter_update_norm": "parameter_update_norm",
}
