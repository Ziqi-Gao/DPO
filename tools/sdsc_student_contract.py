"""Fixed accepted-teacher transport for the separate student calibration path.

Standard library only. Actual immutable artifact pins are input provenance, not
an assertion that a base-model revision names learned weights. Large inputs are
read only on the compute node; login admission checks metadata and small proofs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath

TASKS = {"qwen3-v2-adapted-preflight": "preflight", "qwen3-v2-adapted-calibration": "calibration"}
RESULTS = {key: "adapted-" + value + ".json" for key, value in TASKS.items()}
PRODUCER_HEAD = "929fb14834852a7c91e6656c76fd1834e1b5007d"
QUALIFICATION_JOB_ID = "54496291"
FIT_JOB_ID = "54494742"
DENSE_SHA256 = "6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7"
ACCEPTED_SHA256 = "5d6952823441bde567cdf7f5fad8b4625c58ee7e82425aad76c10433d0ec5337"
INVENTORY_SHA256 = "8d53783b9fb1d386de5a0a291c2b28e225347168cc7cfed4c0c0aceaf01d9bed"
ACCEPTANCE_INVENTORY_SHA256 = INVENTORY_SHA256
PRODUCER_REPORT_SHA256 = "2b067e9e33e7a50ce0697f8defde697e714f49ddc2caed9769bbcc3c0d82b7ba"
PUBLICATION_SHA256 = "0c8bfc0b31bd965244a9d61a158d62d1b8edd26c1f09373083f756d66ee1162e"
MAX_SMALL = 4 * 1024**2
MAX_RESULTS = 160 * 1024**3
BINDINGS = (
    "student_protocol_sha256",
    "student_protocol_artifact_sha256",
    "adapted_teacher_sha256",
    "teacher_acceptance_sha256",
    "teacher_acceptance_inventory_sha256",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def helper(name):
    spec = importlib.util.spec_from_file_location("_student_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe(path):
    path = Path(path)
    require(
        path.is_absolute()
        and ".." not in path.parts
        and not any(ord(c) < 32 for c in str(path))
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "unsafe student path",
    )
    return path


def relative(value):
    path = PurePosixPath(value)
    require(
        isinstance(value, str)
        and value
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value
        and "\\" not in value
        and not any(ord(c) < 32 for c in value),
        "unsafe student artifact name",
    )
    return value


def stable(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def file_record(path, *, maximum=MAX_SMALL, contents=False):
    path = safe(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum, "student file size/type differs"
        )
        hasher, chunks, total = hashlib.sha256(), [], 0
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            total += len(chunk)
            require(total <= maximum, "student file grew past bound")
            hasher.update(chunk)
            if contents:
                chunks.append(chunk)
        require(
            stable(before) == stable(os.fstat(stream.fileno())) == stable(path.lstat())
            and total == before.st_size,
            "student file changed during hashing",
        )
    row = dict(path=path.name, size=total, sha256=hasher.hexdigest())
    return (row, b"".join(chunks)) if contents else row


def document(path, expected=None):
    row, raw = file_record(path, contents=True)
    require(expected is None or row["sha256"] == expected, "student document physical hash differs")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate student JSON field")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(value, dict), "student document is not an object")
    canonical(value)
    return value


def verify_teacher_acceptance(acceptance_dir, *, expected_inventory_sha256):
    root = safe(acceptance_dir)
    require(
        expected_inventory_sha256 == INVENTORY_SHA256
        and digest(canonical(ACCEPTANCE_FILES)) == INVENTORY_SHA256,
        "independent acceptance inventory pin differs",
    )
    for row in ACCEPTANCE_FILES:
        require(file_record(root / row["path"]) == row, "actual accepted teacher file differs")
    accepted = document(root / "accepted-teacher.json")
    require(
        accepted.get("sha256") == ACCEPTED_SHA256
        and digest(canonical({k: v for k, v in accepted.items() if k != "sha256"})) == ACCEPTED_SHA256
        and accepted.get("formal_teacher_accepted") is True
        and accepted.get("original_128_token_readiness_pass_claim") is False
        and accepted["teacher_identity"].get("teacher_checkpoint_sha256") == DENSE_SHA256,
        "independent teacher acceptance binding differs",
    )
    return dict(
        accepted_teacher=accepted,
        teacher_identity=accepted["teacher_identity"],
        accepted_teacher_sha256=ACCEPTED_SHA256,
        inventory_sha256=INVENTORY_SHA256,
    )


def resolved_protocol(root, expected_head):
    root = safe(root)
    require(
        not any(k == "posttrain_circuits" or k.startswith("posttrain_circuits.") for k in sys.modules),
        "science imported before verified student root",
    )
    sys.path.insert(0, str(root / "src"))
    from posttrain_circuits.artifacts.adapted_student_protocol import resolve_adapted_student_protocol

    resolved = resolve_adapted_student_protocol(root, expected_head=expected_head, require_accepted=True)
    require("torch" not in sys.modules, "metadata protocol check imported Torch")
    return dict(
        head=expected_head,
        review_status="accepted",
        protocol_path=resolved.path.relative_to(root).as_posix(),
        protocol_sha256=resolved.protocol_sha256,
        artifact_sha256=resolved.sha256,
        implementation_commit=resolved.reviewed_implementation_commit,
        acceptance_commit=resolved.git_commit,
        science_file_sha256=resolved.science_file_sha256,
    )


def resolve_from_bundle(artifact, manifest_sha256, code_sha256, expected_head):
    import platform

    require(platform.python_version() == "3.12.13", "student metadata runtime Python differs")
    require(
        digest(Path(sys.executable).read_bytes())
        == "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
        "runtime binary differs",
    )
    verifier = helper("sdsc_provenance")
    with tempfile.TemporaryDirectory(prefix="opd-student-protocol-") as directory:
        checkout = Path(directory) / "science"
        try:
            restored = verifier.verify(Path(artifact), manifest_sha256, code_sha256, checkout)
            require(restored["git_head"] == expected_head, "student restored HEAD differs")
            return resolved_protocol(checkout, expected_head)
        finally:
            if checkout.exists():
                for directory, _, _ in os.walk(checkout):
                    Path(directory).chmod(0o700)


def input_plans(qualification):
    require(
        qualification["receipt"].get("job_id") == QUALIFICATION_JOB_ID
        and qualification["receipt"].get("science_git_head") == PRODUCER_HEAD
        and qualification["receipt"].get("task") == "qwen3-v2-teacher-qualify"
        and qualification["result_dir"] == str(QUALIFICATION_ROOT),
        "accepted producer origin differs",
    )
    require(
        file_record(QUALIFICATION_ROOT / "teacher-qualify.json")["sha256"] == PRODUCER_REPORT_SHA256
        and file_record(QUALIFICATION_ROOT / "receipt.json")["sha256"] == PUBLICATION_SHA256,
        "qualification publication identity differs",
    )
    published = {x["path"]: x for x in qualification["publication_receipt"]["files"]}
    inputs = []
    for row in STUDENT_INPUT_FILES:
        expected = dict(path=row["source"], size=row["size"], sha256=row["sha256"])
        require(published.get(row["source"]) == expected, "accepted student input publication differs")
        inputs.append(
            dict(
                path=row["path"],
                source=str(QUALIFICATION_ROOT / row["source"]),
                size=row["size"],
                sha256=row["sha256"],
            )
        )
    for row in ACCEPTANCE_FILES:
        inputs.append(dict(row, path="acceptance/" + row["path"], source=str(ACCEPTANCE_ROOT / row["path"])))
    dataset = [dict(row, source=str(DATASET_ROOT / row["path"])) for row in DATASET_FILES]
    # Dataset bytes were independently regenerated during teacher isolation.
    # Reusing these fixed data files does not accept the failed historical teacher.
    for row in inputs + dataset:
        info = safe(row["source"]).lstat()
        require(
            stat.S_ISREG(info.st_mode) and info.st_size == row["size"], "persistent input metadata differs"
        )
        if row["size"] <= MAX_SMALL:
            require(
                file_record(row["source"])["sha256"] == row["sha256"], "small persistent input hash differs"
            )
    return inputs, dataset


def stage_inputs(rows, destination, *, maximum):
    destination = safe(destination)
    require(
        not destination.exists() and isinstance(rows, list) and rows, "fresh nonempty input staging required"
    )
    require(len({row["path"] for row in rows}) == len(rows), "duplicate staged input")
    require(sum(row["size"] for row in rows) <= maximum, "student input budget exceeded")
    destination.mkdir(mode=0o700, parents=True)
    for row in rows:
        source = safe(row["source"])
        target = safe(destination / relative(row["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        require(
            type(row["size"]) is int
            and 0 <= row["size"] <= maximum
            and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]),
            "invalid staged input record",
        )
        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming:
            before = os.fstat(incoming.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == row["size"], "input size/type changed")
            hasher, total = hashlib.sha256(), 0
            with target.open("xb") as outgoing:
                for chunk in iter(lambda: incoming.read(8 * 1024**2), b""):
                    total += len(chunk)
                    require(total <= row["size"], "input grew during staging")
                    hasher.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            require(
                stable(before) == stable(os.fstat(incoming.fileno())) == stable(source.lstat())
                and total == row["size"]
                and hasher.hexdigest() == row["sha256"],
                "staged input hash differs",
            )
        actual = file_record(target, maximum=maximum)
        require(
            actual["sha256"] == row["sha256"] and actual["size"] == row["size"],
            "node input read-back differs",
        )
        target.chmod(0o444)
    return dict(files=len(rows), bytes=sum(row["size"] for row in rows), read_back_verified=True)


def validate_report(report, proof):
    task = proof["task"]
    require(
        task in TASKS
        and report.get("task") == task
        and report.get("passed") is True
        and report.get("exit_code") == 0
        and report.get("world_size") == 2,
        "student stage did not pass",
    )
    for key in ("g0_passed", "pilot_passed", "factorial_ready", "execution_class_certified"):
        require(report.get(key) is False, "student calibration cannot claim later-stage acceptance")
    expected = dict(
        student_protocol_sha256=proof["protocol"]["protocol_sha256"],
        science_git_head=proof["protocol"]["head"],
        teacher_identity=proof["acceptance"]["teacher_identity"],
        accepted_teacher_sha256=ACCEPTED_SHA256,
        teacher_acceptance_inventory_sha256=INVENTORY_SHA256,
    )
    require(all(report.get(k) == v for k, v in expected.items()), "student report scientific binding differs")
    require(
        report.get("teacher_identity") == proof["acceptance"]["teacher_identity"],
        "student report teacher identity differs",
    )
    if TASKS[task] == "preflight":
        if "protocol_path" in proof["protocol"]:
            expected["student_protocol_path"] = proof["protocol"]["protocol_path"]
        helper("sdsc_adapted_training_preflight").validate_completed_report(report, expected)
    else:
        require(
            report.get("training_executed") is True
            and isinstance(report.get("training_artifacts"), dict)
            and report["training_artifacts"],
            "missing completed student training artifacts",
        )


def validate_publication(root, report, publication):
    """Check durable metadata on login; the allocation hashes every large byte."""
    root = safe(root)
    require(
        publication.get("persisted") is True
        and publication.get("passed") is True
        and publication.get("persistent_read_back_verified") is True,
        "student publication did not pass",
    )
    rows = publication.get("files")
    require(isinstance(rows, list) and 0 < len(rows) <= 10000, "invalid student publication inventory")
    records = {}
    total = 0
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"path", "size", "sha256"}, "invalid publication row")
        name = relative(row["path"])
        require(
            name not in records
            and type(row["size"]) is int
            and row["size"] >= 0
            and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]),
            "duplicate/invalid publication identity",
        )
        total += row["size"]
        require(total <= MAX_RESULTS, "student publication exceeds limit")
        info = safe(root / name).lstat()
        require(
            stat.S_ISREG(info.st_mode) and info.st_size == row["size"], "published artifact size/type differs"
        )
        if row["size"] <= MAX_SMALL:
            require(
                file_record(root / name)["sha256"] == row["sha256"], "small published artifact hash differs"
            )
        records[name] = row
    result_name = RESULTS[report["task"]]
    require(
        result_name in records and document(root / result_name) == report,
        "published report missing/different",
    )
    if (
        TASKS[report["task"]] == "preflight"
        and report.get("student_protocol_path")
        == "prereg/amendments/qwen3_adapted_student_calibration_v5.json"
    ):
        ranks = report.get("ranks")
        require(
            isinstance(ranks, list) and len(ranks) == 2 and isinstance(ranks[0], dict),
            "v5 publication requires both preflight rank reports",
        )
        initial = ranks[0].get("checkpoint_precision", {}).get("initial_checkpoint")
        require(
            isinstance(initial, dict)
            and set(initial) == {"path", "size", "sha256"}
            and initial.get("path") == "initial-canary.pt",
            "v5 publication lacks physical initial checkpoint identity",
        )
        checkpoint = ranks[0].get("checkpoint", {}).get("files")
        require(
            isinstance(checkpoint, list)
            and len(checkpoint) == 4
            and all(isinstance(row, dict) and set(row) == {"path", "size", "sha256"} for row in checkpoint)
            and [row["path"] for row in checkpoint]
            == ["model-full.pt", "optimizer-full.pt", "rank-0-runtime.pt", "rank-1-runtime.pt"],
            "v5 publication lacks exact full-state checkpoint inventory",
        )
        for prefix, row in [("artifacts/", initial), *[("artifacts/checkpoint/", row) for row in checkpoint]]:
            name = prefix + row["path"]
            require(
                type(row["size"]) is int
                and row["size"] > 0
                and isinstance(row["sha256"], str)
                and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
                and records.get(name) == {**row, "path": name},
                "v5 published precision checkpoint differs: " + name,
            )
    if TASKS[report["task"]] == "calibration":
        binding = report["training_artifacts"]
        require(
            records.get("artifacts/initial_checkpoint.pt", {}).get("sha256")
            == report["initial_checkpoint"]["sha256"],
            "published initial checkpoint differs",
        )
        for name in (
            "checkpoint",
            "manifest",
            "config_binding",
            "factorial_update_evidence",
            "metrics",
            "resolved_config",
        ):
            item = binding[name]
            path = "artifacts/" + relative(item["logical_path"])
            require(
                records.get(path, {}).get("sha256") == item["sha256"],
                "published training binding differs: " + name,
            )
        accelerator = binding["accelerator_state"]
        require(
            isinstance(accelerator["files"], dict) and accelerator["files"], "empty accelerator inventory"
        )
        for name, sha256 in accelerator["files"].items():
            path = "artifacts/" + relative(accelerator["logical_path"]) + "/" + relative(name)
            require(records.get(path, {}).get("sha256") == sha256, "published accelerator state differs")
        for row in STUDENT_INPUT_FILES:
            require(
                records.get("artifacts/teacher_demos/" + row["path"], {}).get("sha256") == row["sha256"],
                "published original teacher inputs differ",
            )
        for row in ACCEPTANCE_FILES:
            require(
                records.get("artifacts/teacher_demos/acceptance/" + row["path"], {}).get("sha256")
                == row["sha256"],
                "published teacher acceptance differs",
            )
    return records


ACCEPTANCE_ROOT = Path(
    "/expanse/lustre/projects/nwu181/zgao12/OPD/teacher-acceptance/54496291/4641d1f2c54e627e41d0b2f0e1363c48cada91643fb5c18e946eb19a62a16964"
)
QUALIFICATION_ROOT = Path(
    "/expanse/lustre/projects/nwu181/zgao12/OPD/control-results/20260928T061138Z-d30d5264771a-e6664e24/9d7f9e3394c84b6c901aa98713e0dbda"
)
DATASET_ROOT = Path(
    "/expanse/lustre/projects/nwu181/zgao12/OPD/control-results/20260918T021516Z-82d6fc51200a-2cf5f753/f98dca38ea9d4d9bbe4f3917e151a796/artifacts/dataset"
)
ACCEPTANCE_FILES = [
    {
        "path": "accepted-teacher.json",
        "sha256": "a38be3e22778e9a44fa4fca3a3d5250c161582fb30204aa9615da85adc37d2e7",
        "size": 1093,
    },
    {
        "path": "audit-bindings.json",
        "sha256": "e40e81d349ff5f083e560fdf8d0d11f1f0cc63e2abcc2466887d4abcc42cfb2d",
        "size": 2121,
    },
    {
        "path": "independent-attestation.json",
        "sha256": "5f4aad09c79b0bb5c0ddcad953ef3d3094a82796ff997b7cbeb8554dc5e30278",
        "size": 717,
    },
    {
        "path": "publication-review.json",
        "sha256": "b1d05abeec734627445f6d60c32b4829708d08885623496cf665a3e75b5d0b3b",
        "size": 4336,
    },
]
STUDENT_INPUT_FILES = [
    {
        "path": "acceptance-evidence.json",
        "sha256": "26ae126f121054c3445d5edbe81e44326320759a8a43941fd4d48e75e8a290c8",
        "size": 548805,
        "source": "artifacts/acceptance-evidence.json",
    },
    {
        "path": "producer-report.json",
        "sha256": "2b067e9e33e7a50ce0697f8defde697e714f49ddc2caed9769bbcc3c0d82b7ba",
        "size": 32223,
        "source": "teacher-qualify.json",
    },
    {
        "path": "teacher-store/accepted-view.json",
        "sha256": "1b49719cb8e20a5a148d6b9f40a7560624fac5eb343f1a980e7824d825e185ae",
        "size": 103400,
        "source": "artifacts/teacher-store/accepted-view.json",
    },
    {
        "path": "teacher-store/attempts.jsonl",
        "sha256": "42e57c6ff09a38b2570c25502428baef885e5d1ff78cbd18adf2f67852b47c60",
        "size": 20647754,
        "source": "artifacts/teacher-store/attempts.jsonl",
    },
    {
        "path": "teacher-store/manifest.json",
        "sha256": "8dc9928dae74340366b07fc385d0eeb046ddfd288859ef7ac5150fc0436c7a41",
        "size": 10369,
        "source": "artifacts/teacher-store/manifest.json",
    },
]
DATASET_FILES = [
    {
        "path": "circuit_discovery/examples.jsonl",
        "sha256": "17987a1b646a8cf6acf5a2e9511110cb88491ab61935cd4b00a46700a6bee45e",
        "size": 14971350,
    },
    {
        "path": "circuit_validation/examples.jsonl",
        "sha256": "4eaed9bc0c8d9b660b7f1adf47afbe9976454730000e4e20de0a33267bdda42b",
        "size": 15110148,
    },
    {
        "path": "iid_test/examples.jsonl",
        "sha256": "d299ad96aceaa4e9574c5cf97c55968ed99068ed14239cd3ce23db2dbc0a4645",
        "size": 75079744,
    },
    {
        "path": "manifest.json",
        "sha256": "bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189",
        "size": 1722,
    },
    {
        "path": "ood_depth_test/examples.jsonl",
        "sha256": "fbda8afda9f0c67f23af00f08d7c81d22143acd72628e886d7a273980dba7de3",
        "size": 90978330,
    },
    {
        "path": "ood_structure_test/examples.jsonl",
        "sha256": "9b6ef6c3e33b80df80d9fae4a59e9fe4bca953323d5f40ac8f04b4a1f50f3dee",
        "size": 75791784,
    },
    {
        "path": "train/examples.jsonl",
        "sha256": "377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b",
        "size": 750561394,
    },
    {
        "path": "validation/examples.jsonl",
        "sha256": "8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3",
        "size": 75081250,
    },
]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolve-protocol", nargs=4, metavar=("BUNDLE", "MANIFEST_SHA", "CODE_SHA", "HEAD"), required=True
    )
    args = parser.parse_args(argv)
    print(json.dumps(resolve_from_bundle(*args.resolve_protocol), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
