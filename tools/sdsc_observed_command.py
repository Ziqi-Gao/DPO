"""Observe one caller-authorized argv, preserving bounded raw failure evidence.

This library does not authorize commands, submit jobs, reconcile acknowledgements,
or retry. The caller owns its command allowlist and action gates. It is not wired
into any accepted controller. Killing a timed-out local client does not cancel a
request it may already have sent to a scheduler.

The timeout bounds the capture loop; uninterruptible OS operations can defeat a
wall-clock guarantee. Cleanup attempts are bounded and report an unreaped child
honestly. Only this invocation's direct child can be killed, never a process group.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import math
import os
import selectors
import subprocess
import time
from pathlib import Path

MAX_OUTPUT_BYTES = 16 * 1024**2
READ_BYTES = 65536
REAP_SECONDS = 2.0
DRAIN_SECONDS = 0.5


def _utc_now():
    return dt.datetime.now(dt.UTC).isoformat()


def _utc_readonly_query(argv):
    name = Path(argv[0]).name
    return name in {"sacct", "squeue", "sinfo", "sstat"} or (
        name == "scontrol" and len(argv) >= 2 and argv[1] in {"show", "ping"}
    )


def run_observed(argv, *, timeout, max_output_bytes=1024**2, utc_query=False):
    """Run once, returning JSON-safe evidence with base64 stdout and stderr.

    ``max_output_bytes`` is the combined retained-byte limit for both streams.
    Observed sizes/hashes describe bytes actually read; they do not claim bytes
    still in a pipe or never produced. Output-limit and timeout outcomes cannot
    have ``output_complete=True``, even if cleanup subsequently reaches EOF.
    ``utc_query`` only supports the enumerated Slurm read-query command forms;
    it does not turn the caller-owned command allowlist into an authorization.
    """
    if (
        not isinstance(argv, list | tuple)
        or not argv
        or any(not isinstance(value, str) or not value or "\0" in value for value in argv)
    ):
        raise ValueError("argv must be a nonempty sequence of nonempty strings")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be positive and finite")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise ValueError("invalid combined output limit")
    if type(utc_query) is not bool or (utc_query and not _utc_readonly_query(argv)):
        raise ValueError("UTC override is restricted to explicit read queries")
    argv = list(argv)
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("SBATCH_", "SQUEUE_", "SACCT_"))
    }
    if utc_query:
        environment["TZ"] = "UTC"
    started_at, started = _utc_now(), time.monotonic()
    deadline = started + timeout
    streams = {
        name: {"data": bytearray(), "observed": 0, "hash": hashlib.sha256(), "eof": False}
        for name in ("stdout", "stderr")
    }
    result = {
        "schema": "opd-observed-command-v1",
        "argv": argv,
        "started_at": started_at,
        "timeout_seconds": timeout,
        "max_output_bytes": max_output_bytes,
        "stdin": "DEVNULL",
        "removed_environment_prefixes": ["SBATCH_", "SQUEUE_", "SACCT_"],
        "utc_query": utc_query,
        "spawned": False,
        "pid": None,
        "returncode": None,
        "timed_out": False,
        "output_limited": False,
        "termination_requested": False,
        "child_reaped": False,
        "error": None,
        "retry_attempted": False,
    }
    process = None
    selector = selectors.DefaultSelector()
    retained = 0

    def capture(wait):
        nonlocal retained
        for key, _ in selector.select(wait):
            row = streams[key.data]
            try:
                chunk = os.read(key.fd, READ_BYTES)
            except BlockingIOError:
                continue
            if not chunk:
                row["eof"] = True
                selector.unregister(key.fileobj)
                continue
            row["observed"] += len(chunk)
            row["hash"].update(chunk)
            take = min(len(chunk), max_output_bytes - retained)
            row["data"].extend(chunk[:take])
            retained += take
            if take < len(chunk):
                result["output_limited"] = True

    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        result.update(spawned=True, pid=process.pid)
        for name in streams:
            pipe = getattr(process, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result["timed_out"] = True
                break
            capture(min(remaining, 0.1))
            if result["output_limited"]:
                break
            if process.poll() is not None and not selector.get_map():
                break
    except OSError as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                    result["termination_requested"] = True
                except ProcessLookupError:
                    pass
                except OSError as error:
                    result["error"] = f"{type(error).__name__}: {error}"
            try:
                result["returncode"] = process.wait(timeout=REAP_SECONDS)
                result["child_reaped"] = True
            except subprocess.TimeoutExpired:
                result["error"] = result["error"] or "direct child was not reaped within cleanup bound"
            drain_deadline = time.monotonic() + DRAIN_SECONDS
            try:
                while selector.get_map() and time.monotonic() < drain_deadline:
                    capture(min(0.05, max(0.0, drain_deadline - time.monotonic())))
            except OSError as error:
                result["error"] = f"{type(error).__name__}: {error}"
            for name in streams:
                getattr(process, name).close()
        selector.close()
    result.update(ended_at=_utc_now(), elapsed_seconds=time.monotonic() - started)
    for name, row in streams.items():
        data = bytes(row["data"])
        result[name] = {
            "base64": base64.b64encode(data).decode("ascii"),
            "retained_bytes": len(data),
            "retained_sha256": hashlib.sha256(data).hexdigest(),
            "observed_bytes": row["observed"],
            "observed_sha256": row["hash"].hexdigest(),
            "eof": row["eof"],
            "truncated": len(data) != row["observed"],
        }
    result["output_complete"] = (
        result["child_reaped"]
        and not result["timed_out"]
        and not result["output_limited"]
        and result["error"] is None
        and all(row["eof"] for row in streams.values())
    )
    return result
