"""Run the real v2 receiver and CLI seams locally; never SSH or submit a job."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_provenance_v2 import ROOT, legacy_fixtures, load


@pytest.fixture(autouse=True)
def no_remote_commands(monkeypatch):
    original = subprocess.run

    def guarded(argv, *args, **kwargs):
        if isinstance(argv, list | tuple) and Path(argv[0]).name in {
            "ssh",
            "sbatch",
            "sacct",
            "squeue",
            "scontrol",
            "srun",
            "scancel",
        }:
            pytest.fail("transport CPU test attempted remote/Slurm command")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)


@pytest.fixture
def upload_case(request):
    count = getattr(request, "param", 2)
    fixture = legacy_fixtures.ProvenanceTests()
    fixture.setUp()
    try:
        uploader = load("_v2_upload_case", ROOT / "tools/sdsc_provenance_upload_v2.py")
        provenance = uploader.provenance
        fixture.write("tools/sdsc_provenance_v2.py", Path(provenance.__file__).read_bytes())
        # This alphabetically earlier v1 entry must never choose the verifier.
        fixture.write("tools/sdsc_provenance.py", b"raise RuntimeError('v1 verifier must not execute')\n")
        fixture.local_prelude(count - 2)
        successor = fixture.local_pair()
        prepared = provenance.prepare(
            fixture.root,
            fixture.wrapper_path,
            accepted_ancestors=[fixture.ancestor],
            reviewed_local_implementation=successor["implementation"],
            reviewed_local_acceptance=successor["acceptance"],
        )
        wrapper = provenance.load_wrapper(fixture.wrapper_path)
        remote = Path(fixture.temp.name) / "remote"
        release = remote / "releases" / wrapper["run_id"]
        (release / "source/tools").mkdir(parents=True)
        (release / "manifest.json").write_bytes(provenance.canonical(wrapper))
        for name in ("sdsc_provenance_v2.py", "sdsc_provenance.py"):
            (release / "source/tools" / name).write_bytes((fixture.root / "tools" / name).read_bytes())
        artifact = Path(prepared["artifact"])
        archive = uploader.archive_artifact(artifact)
        intent = dict(
            run_id=wrapper["run_id"],
            manifest_sha256=prepared["manifest_sha256"],
            code_sha256=wrapper["code_sha256"],
            archive_sha256=provenance.sha(archive),
            destination=str(remote / "provenance-v2" / prepared["manifest_sha256"]),
        )
        value = SimpleNamespace(
            fixture=fixture,
            uploader=uploader,
            provenance=provenance,
            prepared=prepared,
            artifact=artifact,
            wrapper=wrapper,
            remote=remote,
            release=release,
            archive=archive,
            request=intent,
            successor=successor,
        )
        yield value
    finally:
        fixture.doCleanups()


def receive(case, archive=None, request=None):
    output = io.StringIO()
    source = case.uploader.REMOTE.replace(
        'pathlib.Path("/home/zgao12/quest-runs/OPD")', f"pathlib.Path({str(case.remote)!r})"
    )
    with (
        patch.object(sys, "argv", ["remote-v2", json.dumps(case.request if request is None else request)]),
        patch.object(
            sys, "stdin", SimpleNamespace(buffer=io.BytesIO(case.archive if archive is None else archive))
        ),
        patch("pwd.getpwuid", return_value=SimpleNamespace(pw_name="zgao12")),
        contextlib.redirect_stdout(output),
    ):
        exec(compile(source, "v2-remote-receiver", "exec"), {"__name__": "__main__"})
    return json.loads(output.getvalue())


@pytest.mark.parametrize("upload_case", [129], indirect=True)
def test_actual_129_commit_receiver_dispatches_named_v2_and_verifies_existing_artifact(upload_case):
    case = upload_case
    result = receive(case)
    assert result["verified"] is True and len(result["local_successor"]["commits"]) == 129
    assert result["git_head"] == case.fixture.head
    assert result["local_successor"] == case.successor
    assert result == receive(case)
    destination = Path(result["remote_artifact"])
    assert destination == case.remote / "provenance-v2" / case.prepared["manifest_sha256"]
    assert not (case.remote / "provenance").exists()
    assert {p.name for p in destination.iterdir()} == set(case.uploader.FILES)
    for path in destination.iterdir():
        assert path.read_bytes() == (case.artifact / path.name).read_bytes()
        assert not path.stat().st_mode & 0o222
    assert not destination.stat().st_mode & 0o222


@pytest.mark.parametrize(
    "corruption",
    [
        "v1_namespace",
        "other_digest_namespace",
        "archive_hash",
        "manifest_hash",
        "verifier_bytes",
        "verifier_missing",
        "verifier_manifest",
    ],
)
def test_receiver_rejects_wrong_namespace_anchor_or_verifier(upload_case, corruption):
    case = upload_case
    if corruption == "v1_namespace":
        case.request["destination"] = str(case.remote / "provenance" / case.prepared["manifest_sha256"])
    elif corruption == "other_digest_namespace":
        case.request["destination"] = str(case.remote / "provenance-v2" / ("0" * 64))
    elif corruption == "archive_hash":
        case.request["archive_sha256"] = "0" * 64
    elif corruption == "manifest_hash":
        case.request["manifest_sha256"] = "0" * 64
        case.request["destination"] = str(case.remote / "provenance-v2" / ("0" * 64))
    elif corruption == "verifier_missing":
        (case.release / "source/tools/sdsc_provenance_v2.py").unlink()
    elif corruption == "verifier_bytes":
        (case.release / "source/tools/sdsc_provenance_v2.py").write_text("raise RuntimeError('untrusted')")
    else:
        value = dict(
            case.wrapper,
            files=[row for row in case.wrapper["files"] if row["path"] != "tools/sdsc_provenance_v2.py"],
        )
        value["code_sha256"] = case.provenance.sha(case.provenance.canonical(value["files"]))
        case.request["code_sha256"] = value["code_sha256"]
        (case.release / "manifest.json").write_bytes(case.provenance.canonical(value))
    with pytest.raises((AssertionError, ValueError, StopIteration, FileNotFoundError)):
        receive(case)
    assert not (case.remote / "provenance").exists()
    assert not (case.remote / "provenance-v2" / case.prepared["manifest_sha256"]).exists()


def test_receiver_rejects_rehashed_v1_schema_before_publication(upload_case):
    case = upload_case
    manifest = json.loads((case.artifact / "manifest.json").read_bytes())
    manifest["schema"] = "quest-sdsc-git-provenance-v1"
    raw = case.provenance.canonical(manifest) + b"\n"
    case.fixture.tamper(case.artifact, "manifest.json", raw)
    case.archive = case.uploader.archive_artifact(case.artifact)
    case.request.update(
        manifest_sha256=case.provenance.sha(raw), archive_sha256=case.provenance.sha(case.archive)
    )
    case.request["destination"] = str(case.remote / "provenance-v2" / case.request["manifest_sha256"])
    with pytest.raises(ValueError, match="schema"):
        receive(case)
    assert not Path(case.request["destination"]).exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "extra", "duplicate"])
def test_receiver_requires_exact_three_regular_archive_members(upload_case, kind):
    case = upload_case
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        names = list(case.uploader.FILES)
        if kind == "traversal":
            names[0] = "../manifest.json"
        elif kind == "extra":
            names.append("extra")
        elif kind == "duplicate":
            names[-1] = names[0]
        for name in names:
            entry = tarfile.TarInfo(name)
            data = (case.artifact / name).read_bytes() if name in case.uploader.FILES else b"fixture"
            if kind == "symlink" and name == "history.bundle":
                entry.type, entry.linkname = tarfile.SYMTYPE, "/tmp/outside"
                archive.addfile(entry)
            else:
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    case.archive = buffer.getvalue()
    case.request["archive_sha256"] = case.provenance.sha(case.archive)
    with pytest.raises(AssertionError):
        receive(case)
    assert not Path(case.request["destination"]).exists()


def test_existing_artifact_is_verified_not_overwritten_or_recursively_modified(upload_case):
    case = upload_case
    destination = Path(receive(case)["remote_artifact"])
    destination.chmod(0o700)
    (destination / "history.bundle").chmod(0o600)
    assert receive(case)["verified"] is True
    assert not (destination / "history.bundle").stat().st_mode & 0o222
    destination.chmod(0o700)
    extra = destination / "unexpected"
    extra.write_bytes(b"untouched")
    before = {p.name: (p.read_bytes(), p.stat().st_mode) for p in destination.iterdir()}
    with pytest.raises(AssertionError, match="Unexpected provenance files"):
        receive(case)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mode) for p in destination.iterdir()}


def setup_cli(case, monkeypatch, *, receipt_change=None, failure=False):
    cli, seen = case.uploader.cli, []
    monkeypatch.setattr(cli, "ROOT", case.fixture.root)
    monkeypatch.setattr(cli, "REMOTE_ROOT", str(case.remote))
    monkeypatch.setattr(cli, "run_record", lambda _: dict(state="deployed", manifest=case.wrapper))
    cli.write_json(cli.state_root() / "check.json", dict(connected=True, control_python="/fixture/python"))

    def call(argv, data, timeout):
        seen.append(dict(argv=argv, timeout=timeout))
        if failure:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"fixture connection loss")
        assert argv[:4] == ["/fixture/python", "-I", "-B", "-c"]
        assert argv[4] == case.uploader.REMOTE and timeout == 180
        result = receive(case, archive=data, request=json.loads(argv[5]))
        if receipt_change:
            receipt_change(result)
        return SimpleNamespace(returncode=0, stdout=json.dumps(result).encode(), stderr=b"")

    monkeypatch.setattr(cli, "ssh_call", call)
    return seen


def cli_args(case, mode):
    return [
        "--artifact",
        str(case.artifact),
        "--manifest-sha256",
        case.prepared["manifest_sha256"],
        "--run-id",
        case.wrapper["run_id"],
        mode,
    ]


def test_actual_cli_dry_run_and_upload_bind_v2_destination_and_receipt(upload_case, monkeypatch, capsys):
    case = upload_case
    calls = setup_cli(case, monkeypatch)
    assert case.uploader.main(cli_args(case, "--dry-run")) == 0
    preview = json.loads(capsys.readouterr().out)
    assert not calls and preview["destination"] == case.request["destination"]
    assert case.uploader.main(cli_args(case, "--upload")) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and receipt["remote_artifact"] == preview["destination"]
    path = (
        case.fixture.root / ".sdsc" / ("provenance-v2-upload-" + case.prepared["manifest_sha256"] + ".json")
    )
    assert json.loads(path.read_bytes()) == receipt
    assert not list((case.fixture.root / ".sdsc").glob("provenance-upload-*.json"))


@pytest.mark.parametrize("failure", ["connection", "v1_receipt", "bundle_receipt"])
def test_actual_cli_never_retries_or_saves_unbound_receipt(upload_case, monkeypatch, failure):
    case = upload_case

    def corrupt(value):
        if failure == "v1_receipt":
            value["remote_artifact"] = str(case.remote / "provenance" / case.prepared["manifest_sha256"])
        elif failure == "bundle_receipt":
            value["bundle_sha256"] = "0" * 64

    calls = setup_cli(case, monkeypatch, receipt_change=corrupt, failure=failure == "connection")
    with pytest.raises(ValueError):
        case.uploader.main(cli_args(case, "--upload"))
    assert len(calls) == 1
    assert not list((case.fixture.root / ".sdsc").glob("provenance-v2-upload-*.json"))
