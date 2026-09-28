"""Measured 192-GiB workload budget on Expanse's exclusive four-H100 partition.

The actual job cgroup may have a larger finite kernel limit. Never describe
that as a 192-GiB kernel cap or use its size to relax the workload peak gate.
Legacy shared-node and central scheduler memory checks remain unchanged.
"""

import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path

BUDGET = 192 * 1024**3
HEADROOM = max(32 * 1024**3, math.ceil(BUDGET * 0.20))
POLICY = "exclusive_node_workload_budget_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def memory_envelope(
    *, job_id=None, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup"), evidence_path=None
):
    """Preserve raw counters first; validate own job usage against the fixed budget."""
    job_id = os.environ.get("SLURM_JOB_ID", "") if job_id is None else job_id
    require(
        isinstance(job_id, str) and re.fullmatch(r"[1-9][0-9]*", job_id), "real memory job identity required"
    )
    raw = {
        "policy": POLICY,
        "job_id": job_id,
        "budget_bytes": BUDGET,
        "minimum_headroom_bytes": HEADROOM,
        "passed": False,
        "slurm_requested_memory_mib": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
        "ancestors": [],
    }
    try:
        locations = []
        raw["proc_cgroup"] = proc.read_text()
        for line in raw["proc_cgroup"].splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            require(relative.startswith("/") and ".." not in Path(relative).parts, "unsafe cgroup path")
            if hierarchy == "0" and not controllers:
                locations.append(
                    (root, root / relative.lstrip("/"), ("memory.max", "memory.current", "memory.peak"))
                )
            elif "memory" in controllers.split(","):
                locations.insert(
                    0,
                    (
                        root / "memory",
                        root / "memory" / relative.lstrip("/"),
                        ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes"),
                    ),
                )
        require(locations, "memory controller is unavailable")
        boundary, directory, names = locations[0]
        own_job = "job_" + job_id
        relative_parts = directory.relative_to(boundary).parts
        require(relative_parts.count(own_job) == 1, "memory hierarchy does not belong to this job")
        job_index = relative_parts.index(own_job)
        require(
            relative_parts[job_index + 1 : job_index + 2] == ("step_batch",),
            "expected own batch-step memory hierarchy",
        )
        job_path = boundary.joinpath(*relative_parts[: job_index + 1])
        step_path = job_path / "step_batch"
        raw.update(path=str(job_path), step_path=str(step_path))
        while True:
            row = {"path": str(directory), "limit_bytes": None}
            if (directory / names[0]).is_file():
                text = (directory / names[0]).read_text().strip()
                limit = None if text == "max" else int(text)
                require(limit is None or limit > 0, "invalid cgroup limit")
                row.update(raw_limit=text, limit_bytes=limit if limit is not None and limit < 2**60 else None)
                for name, key in zip(names[1:], ("current_bytes", "peak_bytes"), strict=True):
                    if (directory / name).is_file():
                        row[key] = int((directory / name).read_text().strip())
            raw["ancestors"].append(row)
            if directory == boundary:
                break
            require(boundary in directory.parents, "memory hierarchy escaped controller")
            directory = directory.parent
        own = [row for row in raw["ancestors"] if row["path"] in {str(job_path), str(step_path)}]
        require(
            len(own) == 2 and any(row["limit_bytes"] is not None for row in own),
            "own job/step lacks an attributable finite cgroup boundary",
        )
        finite = [row["limit_bytes"] for row in raw["ancestors"] if row["limit_bytes"] is not None]
        effective = min(finite)
        require(effective >= BUDGET, "actual job cgroup is smaller than the requested workload budget")
        require(
            all(
                type(row.get("current_bytes")) is int
                and type(row.get("peak_bytes")) is int
                and 0 < row["current_bytes"] <= row["peak_bytes"] <= effective
                for row in own
            ),
            "own job/step lacks consistent real current/peak memory",
        )
        current, peak = max(row["current_bytes"] for row in own), max(row["peak_bytes"] for row in own)
        raw.update(
            effective_limit_bytes=effective,
            current_bytes=current,
            peak_bytes=peak,
            headroom_bytes=BUDGET - peak,
            kernel_headroom_bytes=effective - peak,
            kernel_cap_equals_budget=effective == BUDGET,
            software_budget_only=effective > BUDGET,
        )
        require(BUDGET - peak >= HEADROOM, "192-GiB workload budget lacks its original 20%/32-GiB headroom")
        raw["passed"] = True
        return raw
    except (ValueError, OSError) as error:
        raw["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        if evidence_path is not None:
            path = Path(evidence_path)
            require(not path.is_symlink() and path.parent.is_dir(), "unsafe memory evidence destination")
            temporary = path.with_name("." + path.name + ".tmp")
            with temporary.open("x") as stream:
                json.dump(raw, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)


def local_environment(args, guards):
    """Same offline/local-storage rules, with a separate exclusive-node budget."""
    work, output, home = map(guards.real_path, (args.work_dir, args.output_dir, args.hf_home))
    require(work.is_dir() and work.stat().st_uid == os.getuid(), "work directory is missing or not owned")
    require(work != Path.home().resolve() and Path.home().resolve() not in work.parents, "work is in HOME")
    require(
        work in output.parents and work in home.parents and home.is_dir(), "output/cache must be node-local"
    )
    require(
        args.science_root not in output.parents and output != args.science_root, "output would modify source"
    )
    mount = json.loads(
        subprocess.run(
            ["findmnt", "--json", "--target", str(work), "--output", "TARGET,SOURCE,FSTYPE"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    )["filesystems"][0]
    require(mount["fstype"] in {"ext4", "xfs", "ext3", "ext2", "btrfs"}, "work is not node-local disk")
    # Metadata-only validation runs before the numerical worker creates its
    # outputs. Preserve raw cgroup evidence even when the first guard fails.
    output.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "validate")
    require(rank in {"validate", "0", "1", "2", "3"}, "unexpected memory evidence rank")
    memory = memory_envelope(evidence_path=output / ("memory-environment-" + rank + ".json"))
    cache = pinned_cache(home)
    os.environ.update(
        HF_HOME=str(home),
        HF_HUB_CACHE=str(home / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="6",
        MKL_NUM_THREADS="6",
        OPENBLAS_NUM_THREADS="6",
        NUMEXPR_NUM_THREADS="6",
    )
    return {"node_local_mount": mount, "cgroup_memory": memory, "pinned_cache": cache}


def pinned_cache(home):
    spec = importlib.util.spec_from_file_location(
        "_exclusive_cache", Path(__file__).with_name("sdsc_training_preflight.py")
    )
    cache_guards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache_guards)
    return cache_guards.pinned_cache(home)
