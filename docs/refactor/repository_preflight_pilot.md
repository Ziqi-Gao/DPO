# Repository preflight pilot

`repository_preflight` is the only task currently activated in OPD's code-owned
handler registry. It is a CPU-only, read-only scientific/adapter preflight. It
does not load a model, access the network, select hardware, submit a scheduler
job, or mutate project source.

## Fixed contract

- Parameters: `workflow_id`, `plan_sha256`, and `unit_id` only.
- Content inputs: the four ConfigBinding files required by the adapter:
  `config_binding_sha256`, `execution_config_sha256`,
  `resolved_config_sha256`, and `scientific_config_sha256`.
- Output: `preflight_report.json`.
- Completion gates: `config_binding`, `no_gpu_required`, and
  `runtime_isolation`.
- Runtime: `/usr/bin/python3.12` version 3.12.13 with `-I -S`; executable,
  standalone handler, dependency lock, and package manifest are bound by the
  deployment contract.
- Retry: each scheduler attempt receives a fresh staging directory; an already
  valid final ScientificCompletion is reused without launching the handler.

The formal request deliberately does not add a repository-snapshot content
identity. A clean committed checkout is an operator gate before request
preparation, not another scientific CAS layer.

## Resource declaration

The representative end-to-end adapter test executed the real handler, published
and validated its completion marker, then exercised completion reuse. On
2026-09-02, `/usr/bin/time -v` reported 0.18 seconds wall time, 98% CPU, and
23,552 KiB MaxRSS.

| Resource | Registered value |
| --- | ---: |
| Processes | 1 |
| CPU cores | 1 fixed |
| Host memory | 128 MiB |
| GPUs | 0 |
| Per-GPU memory | 0 MiB |
| Per-GPU utilization | 0% |
| GPU models | none |
| GPU exclusivity | `shareable` (not applicable to CPU work) |
| Initial runtime estimate | 30 seconds |
| CPU scaling efficiency | 0.0 |

The memory and runtime values include substantial headroom over the observed
test. The checked-in protocol-v2 proposal remains disabled:
`deployments/repository_preflight/registration-proposal-v2.toml`.

## Handoff sequence

After the implementation is committed and `git status --short` is empty, prepare
the project-owned workflow plan and formal outbox file with:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 \
  -m posttrain_circuits.scheduler_adapter.preflight_request
```

This command only writes immutable OPD plan/config content under
`/data/del6500/OPD` and one protocol-v2 request under
`/scr/del6500/OPD/scheduler/outbox`. It does not submit, poll, enable a
registration, or manage a service. Central review, registration installation,
enabling, and pilot execution remain separate approval gates.

Each invocation creates a fresh opaque `job_id` for one scheduler submission.
The scientific identity remains the immutable `workflow_id`, `plan_sha256`, and
`unit_id`; this permits an operational retry after an exhausted central job
without changing or re-hashing the scientific unit.
