"""Scheduler-neutral OPD workflow contracts."""

from posttrain_circuits.workflows.catalog import (
    CANDIDATE_ENTRYPOINT_CATALOG,
    CandidateEntrypointTemplate,
    candidate_entrypoint,
    validate_candidate_catalog,
)
from posttrain_circuits.workflows.contracts import (
    CONTENT_KINDS,
    PROJECT,
    WORKFLOW_SCHEMA_VERSION,
    WORKFLOW_TASK_ALLOWLIST,
    WORKFLOW_TASK_REGISTRY,
    ContentIdentity,
    Dependency,
    WorkflowPlan,
    WorkflowUnit,
    canonical_json,
    completion_identity_sha256,
    output_identity_sha256,
    unit_identity_sha256,
)

__all__ = [
    "CANDIDATE_ENTRYPOINT_CATALOG",
    "CONTENT_KINDS",
    "CandidateEntrypointTemplate",
    "PROJECT",
    "WORKFLOW_SCHEMA_VERSION",
    "WORKFLOW_TASK_ALLOWLIST",
    "WORKFLOW_TASK_REGISTRY",
    "ContentIdentity",
    "Dependency",
    "WorkflowPlan",
    "WorkflowUnit",
    "canonical_json",
    "candidate_entrypoint",
    "completion_identity_sha256",
    "output_identity_sha256",
    "unit_identity_sha256",
    "validate_candidate_catalog",
]
