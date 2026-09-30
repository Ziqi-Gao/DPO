#!/usr/bin/env python3
"""Measured, fail-closed host-memory guard for the reviewed student successor.

No cgroup limits, counters or cache state are changed. A failed check retains
its raw hierarchy evidence, including aggregate job peaks and memory.stat.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

GIB = 1024**3
V6_PROTOCOL = "prereg/amendments/qwen3_adapted_student_calibration_v6.json"


class MemoryEnvelopeError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def expected_bytes(protocol_path):
    if protocol_path == V6_PROTOCOL:
        return 384 * GIB
    if re.fullmatch(r"prereg/amendments/qwen3_adapted_student_calibration_v[1-5]\.json", protocol_path):
        return 192 * GIB
    raise ValueError("unrecognized student memory protocol")


def _counters(path):
    if not path.is_file():
        return None
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        if key in result:
            raise ValueError("duplicate memory counter")
        result[key] = int(value)
    return result


def memory_envelope(*, expected_bytes, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    evidence = {"expected_limit_bytes": expected_bytes, "observed_at_unix": time.time(),
                "ancestors": [], "passed": False}
    try:
        def require(condition, message):
            if not condition:
                raise ValueError(message)
        require(type(expected_bytes) is int and expected_bytes in (192 * GIB, 384 * GIB),
                "unreviewed student memory envelope")
        locations = []
        for line in proc.read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            require(relative.startswith("/") and ".." not in Path(relative).parts, "invalid cgroup path")
            if hierarchy == "0" and not controllers:
                locations.append((root, root / relative.lstrip("/"),
                                  ("memory.max", "memory.current", "memory.peak"), 2))
            elif "memory" in controllers.split(","):
                locations.insert(0, (root / "memory", root / "memory" / relative.lstrip("/"),
                                     ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes"), 1))
        require(locations, "no readable memory cgroup")
        boundary, directory, names, version = locations[0]
        evidence["cgroup_version"] = version
        levels = evidence["ancestors"]
        while True:
            level = {"path": str(directory), "limit_bytes": None}
            levels.append(level)
            if (directory / names[0]).is_file():
                raw = (directory / names[0]).read_text().strip()
                limit = None if raw == "max" else int(raw)
                level["raw_limit"] = raw
                require(limit is None or limit > 0, "invalid cgroup memory limit")
                level["limit_bytes"] = limit if limit is not None and limit < 2**60 else None
                for name, key in zip(names[1:], ("current_bytes", "peak_bytes"), strict=True):
                    if (directory / name).is_file():
                        level[key] = int((directory / name).read_text().strip())
                level["memory_stat"] = _counters(directory / "memory.stat")
                level["memory_events"] = _counters(directory / "memory.events")
                level["memory_events_local"] = _counters(directory / "memory.events.local")
                level["memory_oom_control"] = _counters(directory / "memory.oom_control")
                failcnt = directory / "memory.failcnt"
                level["memory_failcnt"] = int(failcnt.read_text().strip()) if failcnt.is_file() else None
            if directory == boundary:
                break
            require(boundary in directory.parents, "memory cgroup escaped controller root")
            directory = directory.parent
        finite = [level for level in levels if level["limit_bytes"] is not None]
        require(finite, "cgroup must expose a finite student memory limit")
        limit = min(level["limit_bytes"] for level in finite)
        limiting = [level for level in finite if level["limit_bytes"] == limit]
        require(all("current_bytes" in level and "peak_bytes" in level for level in limiting),
                "limiting memory cgroup must expose current and peak usage")
        selected = max(limiting, key=lambda level: level["peak_bytes"])
        current, peak = selected["current_bytes"], selected["peak_bytes"]
        required = max(32 * GIB, math.ceil(limit * 0.20))
        evidence.update(path=selected["path"], limit_bytes=limit, current_bytes=current,
                        peak_bytes=peak, headroom_bytes=limit - peak, minimum_headroom_bytes=required,
                        memory_stat=selected.get("memory_stat"), memory_events=selected.get("memory_events"),
                        memory_events_local=selected.get("memory_events_local"),
                        memory_oom_control=selected.get("memory_oom_control"),
                        memory_failcnt=selected.get("memory_failcnt"))
        require(limit == expected_bytes and all(0 < item["current_bytes"] <= item["peak_bytes"] <= limit for item in limiting),
                "cgroup limit or measured peak differs from reviewed student memory envelope")
        require(limit - peak >= required,
                f"{limit // GIB} GiB cgroup lacks 32 GiB and 20% headroom: peak={peak}, current={current}, required={required}")
        evidence["passed"] = True
        return evidence
    except (ValueError, OSError) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise MemoryEnvelopeError(str(error), evidence) from error


def record_stage(report, stage, output_dir, *, expected_bytes):
    """Publish one immutable small measurement before propagating its failure."""
    if re.fullmatch(r"[a-z][a-z0-9_-]*", stage) is None:
        raise ValueError("invalid memory stage name")
    stages = report.setdefault("memory_stages", {})
    if stage in stages:
        raise ValueError("memory stage already observed")
    error = None
    try:
        evidence = memory_envelope(expected_bytes=expected_bytes)
    except MemoryEnvelopeError as caught:
        evidence, error = caught.evidence, caught
    stages[stage] = evidence
    path = Path(output_dir) / ("memory-" + stage + ".json")
    raw = (json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    if error is not None:
        raise error
    return evidence
