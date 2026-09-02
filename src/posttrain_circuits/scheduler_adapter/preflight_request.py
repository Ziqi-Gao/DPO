"""Prepare the one reviewed repository-preflight request without submitting it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from posttrain_circuits.artifacts.config_bindings import bind_config
from posttrain_circuits.scheduler_adapter.config_resolver import ConfigBindingResolver
from posttrain_circuits.scheduler_adapter.content_store import ContentStore
from posttrain_circuits.scheduler_adapter.outbox import prepare_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.plan_store import publish_workflow_plan
from posttrain_circuits.scheduler_adapter.repository_preflight import (
    OUTPUT_NAME,
    PROFILE_NAME,
    TASK_NAME,
)
from posttrain_circuits.workflows.contracts import WorkflowPlan, WorkflowUnit


WORKFLOW_ID = "repository-preflight-v1"
UNIT_ID = "preflight"


@dataclass(frozen=True)
class PreparedPreflightRequest:
    workflow_id: str
    plan_sha256: str
    unit_id: str
    plan_path: Path
    outbox_path: Path

    def to_payload(self) -> dict[str, str]:
        return {
            "outbox_path": str(self.outbox_path),
            "plan_path": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "unit_id": self.unit_id,
            "workflow_id": self.workflow_id,
        }


def build_repository_preflight_plan(*, layout: WorkflowLayout) -> WorkflowPlan:
    """Materialize only the four adapter-required ConfigBinding identities."""

    resolved_config = {
        "pilot": {
            "access": "read-only",
            "task": TASK_NAME,
        }
    }
    execution_context = {
        "execution_profile": PROFILE_NAME,
        "scheduler_protocol": 2,
    }
    binding = bind_config(
        resolved_config,
        input_artifact_hashes={},
        execution_context=execution_context,
    )
    scientific_config = {
        "config": resolved_config,
        "input_artifact_hashes": {},
        "schema_version": 3,
    }
    execution_config = {
        "execution_context": execution_context,
        "schema_version": 3,
        "storage_locators": {},
    }
    identities = ConfigBindingResolver(ContentStore(layout)).materialize(
        binding,
        resolved_config=resolved_config,
        scientific_config=scientific_config,
        execution_config=execution_config,
    )
    return WorkflowPlan(
        workflow_id=WORKFLOW_ID,
        units=(
            WorkflowUnit(
                unit_id=UNIT_ID,
                task=TASK_NAME,
                content_inputs=identities.as_tuple(),
                dependencies=(),
                output_names=(OUTPUT_NAME,),
            ),
        ),
    )


def prepare_repository_preflight_request(
    *, layout: WorkflowLayout
) -> PreparedPreflightRequest:
    """Publish the immutable plan and formal outbox file; never submit or poll."""

    plan = build_repository_preflight_plan(layout=layout)
    plan_path = publish_workflow_plan(plan, layout=layout)
    outbox_path = prepare_outbox_request(
        plan,
        unit_id=UNIT_ID,
        layout=layout,
    )
    return PreparedPreflightRequest(
        workflow_id=plan.workflow_id,
        plan_sha256=plan.sha256(),
        unit_id=UNIT_ID,
        plan_path=plan_path,
        outbox_path=outbox_path,
    )


def main() -> int:
    receipt = prepare_repository_preflight_request(layout=WorkflowLayout.production())
    print(json.dumps(receipt.to_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PreparedPreflightRequest",
    "build_repository_preflight_plan",
    "prepare_repository_preflight_request",
]
