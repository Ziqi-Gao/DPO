"""Real local filesystem/child-process fixtures; no remote or Slurm operation."""

import copy
import importlib.util
import io
import json
import multiprocessing
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "_execution_fence_test", ROOT / "tools/sdsc_student_execution_fence.py"
)
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)


@pytest.fixture(autouse=True)
def private_umask():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    control, project = tmp_path / "control", tmp_path / "project"
    control.mkdir(mode=0o700)
    project.mkdir(mode=0o700)
    monkeypatch.setattr(fence, "CONTROL", control)
    monkeypatch.setattr(fence, "PROJECT", project)
    monkeypatch.setattr(fence, "EXPECTED_UID", os.getuid())
    observed_filesystems = {}
    for role, root in (("control", control), ("project", project)):
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            observed_filesystems[role] = fence._observe_filesystem(fd, root)["portable"]
        finally:
            os.close(fd)
    monkeypatch.setattr(fence, "FILESYSTEMS", observed_filesystems)
    submission = control / "student-name-invariant-v1-submissions" / fence.OLD_INTENT
    result = project / "student-name-invariant-v1" / fence.OLD_INTENT
    release = control / "releases" / "fixture-release"
    node_name = "tools/sdsc_student_name_invariant_job.py"
    node_bytes = b"# explicitly synthetic CPU source fixture\n"
    plan = dict(
        intent_id=fence.OLD_INTENT,
        mode="preflight",
        release=str(release),
        submission_dir=str(submission),
        result_dir=str(result),
        claim=str(control / "old-claims" / "execution.json"),
        scientific_claim=str(control / "old-claims" / "science.json"),
        control_sha256={node_name: fence.sha(node_bytes)},
    )
    monkeypatch.setattr(fence, "OLD_NODE_SHA", fence.sha(node_bytes))
    monkeypatch.setattr(fence, "OLD_PLAN_SHA", fence.sha(fence.canonical(plan)))
    side = {name: (name + ": original fixture evidence\n").encode() for name in fence.SIDEFILES}
    claims = {key: fence.canonical(dict(original_fixture=key)) for key in fence.OLD_CLAIMS}
    monkeypatch.setattr(fence, "SIDEFILES", {k: (len(v), fence.sha(v)) for k, v in side.items()})
    monkeypatch.setattr(fence, "OLD_CLAIMS", {k: (len(v), fence.sha(v)) for k, v in claims.items()})
    files = {submission / "plan.json": fence.canonical(plan), release / "source" / node_name: node_bytes}
    files.update({submission / k: v for k, v in side.items()})
    files.update({Path(plan[k]): v for k, v in claims.items()})
    for path, raw in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o600)
    request = fence.build_request(plan)
    return request, files


def install(request):
    return fence.install(request, authorize=True, dry_run=fence.dry_run(request))


def test_real_install_preserves_original_bytes_and_records_unknown(fixture):
    request, files = fixture
    before = {p: fence._stamp(p.stat()) for p in files}
    assert not Path(request["result_dir"]).parent.exists()
    dry = fence.dry_run(request)
    assert not Path(request["result_dir"]).parent.exists()
    proof = fence.install(request, authorize=True, dry_run=dry)
    assert fence.verify(proof) == proof == fence.verify(request)
    assert fence.validate_proof(proof, request=request) == proof
    assert fence.reconcile(request)["proof"] == proof
    assert proof["old_submission_outcome"] == "unknown"
    assert proof["scheduler_cancellation"] is proof["allocation_absence_proven"] is False
    assert all(proof[k] is False for k in fence.FLAGS)
    assert {p.name for p in Path(request["result_dir"]).iterdir()} == {fence.MARKER_NAME}
    for path, raw in files.items():
        assert path.read_bytes() == raw and fence._stamp(path.stat()) == before[path]
    assert Path(request["proof_path"]).read_bytes() == fence.canonical(proof)
    with pytest.raises((ValueError, OSError)):
        install(request)


def test_sgid_is_accepted_and_retained(fixture):
    request, _ = fixture
    fence.PROJECT.chmod(0o2700)
    fence.CONTROL.chmod(0o2700)
    proof = install(request)
    assert (
        proof["directory_identity"]["mode"]
        == stat.S_IMODE(Path(request["result_dir"]).stat().st_mode)
        == 0o2700
    )
    fence.verify(proof)


@pytest.mark.parametrize("mutation", ["result", "claim", "proof", "receipt", "live-binding"])
def test_preexisting_state_never_adopted(fixture, mutation):
    request, _ = fixture
    if mutation == "result":
        path = Path(request["result_dir"])
        path.mkdir(parents=True)
    else:
        path = (
            Path(request[{"claim": "claim_path", "proof": "proof_path"}[mutation]])
            if mutation in {"claim", "proof"}
            else Path(request["old_plan"]["submission_dir"]) / (mutation + ".json")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve me")
    before = path.stat().st_ino
    with pytest.raises((ValueError, OSError)):
        fence.dry_run(request)
    assert path.stat().st_ino == before
    if mutation != "result":
        assert path.read_bytes() == b"preserve me"


def test_node_wins_after_claim_does_not_become_a_success(fixture, monkeypatch):
    request, _ = fixture
    original = fence._write_once

    def competing_node(path, value):
        row = original(path, value)
        if str(path) == request["claim_path"]:
            Path(request["result_dir"]).mkdir(parents=True)
        return row

    monkeypatch.setattr(fence, "_write_once", competing_node)
    with pytest.raises(FileExistsError):
        install(request)
    assert Path(request["claim_path"]).is_file()
    assert not Path(request["proof_path"]).exists()
    assert not Path(request["marker_path"]).exists()
    assert fence.reconcile(request)["state"] == "unknown"
    with pytest.raises(ValueError):
        install(request)


def test_parent_replaced_after_mkdir_cannot_substitute_another_directory(fixture, monkeypatch):
    request, _ = fixture
    target = Path(request["result_dir"])
    original = fence.os.mkdir
    swapped = []

    def swap_parent(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if str(path) == target.name and kwargs.get("dir_fd") is not None:
            old_parent = target.parent
            old_parent.rename(old_parent.with_name("preserved-winning-parent"))
            original(old_parent, mode=0o700)
            original(target, mode=0o700)
            swapped.append(True)
        return result

    monkeypatch.setattr(fence.os, "mkdir", swap_parent)
    with pytest.raises(ValueError, match="mkdir winner|opened path changed"):
        install(request)
    assert swapped == [True] and Path(request["claim_path"]).exists()
    assert not Path(request["proof_path"]).exists() and not Path(request["marker_path"]).exists()
    assert fence.reconcile(request)["state"] == "unknown"


@pytest.mark.parametrize("failed_key", ["marker_path", "proof_path"])
def test_partial_write_is_preserved_and_never_repaired(fixture, monkeypatch, failed_key):
    request, _ = fixture
    original = fence._write_once

    def partial(path, value):
        if str(path) == request[failed_key]:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(b'{"partial":')
            raise OSError("injected bounded write failure")
        return original(path, value)

    monkeypatch.setattr(fence, "_write_once", partial)
    with pytest.raises(OSError):
        install(request)
    assert Path(request[failed_key]).read_bytes() == b'{"partial":'
    assert fence.reconcile(request)["state"] == "unknown"
    with pytest.raises(ValueError):
        install(request)
    assert Path(request[failed_key]).read_bytes() == b'{"partial":'


def test_lost_ack_only_reconciles_existing_complete_proof(fixture, monkeypatch):
    request, _ = fixture
    original = fence.verify
    monkeypatch.setattr(
        fence, "verify", lambda value: (_ for _ in ()).throw(InterruptedError("lost acknowledgement"))
    )
    with pytest.raises(InterruptedError):
        install(request)
    monkeypatch.setattr(fence, "verify", original)
    rows = {
        Path(request[k]): Path(request[k]).read_bytes() for k in ("claim_path", "marker_path", "proof_path")
    }
    result = fence.reconcile(request)
    assert result["state"] == "fenced" and result["installed"] is True
    assert all(path.read_bytes() == raw for path, raw in rows.items())


@pytest.mark.parametrize(
    "kind", ["bool-stamp", "claim-extra", "marker-extra", "false-scope", "time", "root-type", "root-extra"]
)
def test_pure_proof_validator_rejects_rehashed_metadata(fixture, kind):
    request, _ = fixture
    proof = copy.deepcopy(install(request))
    if kind == "bool-stamp":
        proof["claim"]["stamp"]["nlink"] = True
    elif kind in {"claim-extra", "marker-extra"}:
        key = kind.split("-")[0]
        row = proof[key]
        row["document"]["extra"] = "must reject"
        raw = fence.canonical(row["document"])
        row.update(size=len(raw), sha256=fence.sha(raw))
        row["stamp"]["size"] = len(raw)
    elif kind == "false-scope":
        proof["scheduler_cancellation"] = 0
    elif kind == "root-type":
        proof["roots"][str(fence.CONTROL)]["identity"]["device"] = False
    elif kind == "root-extra":
        proof["roots"]["/unreviewed"] = proof["roots"][str(fence.CONTROL)]
    else:
        proof["completed_at"] = "not-a-time"
    with pytest.raises((ValueError, TypeError)):
        fence.validate_proof(proof)


@pytest.mark.parametrize(
    "mutation", ["marker", "proof", "old-file", "extra", "inode", "symlink", "writable", "root"]
)
def test_verify_rejects_changed_bound_evidence(fixture, mutation):
    request, files = fixture
    proof = install(request)
    target = Path(request["result_dir"])
    if mutation in {"marker", "proof"}:
        path = Path(request[mutation + "_path"])
        value = json.loads(path.read_bytes())
        value["old_submission_outcome"] = "rejected"
        path.write_bytes(fence.canonical(value))
    elif mutation == "old-file":
        next(iter(files)).write_bytes(b"changed")
    elif mutation == "extra":
        (target / "unexpected").write_bytes(b"x")
    elif mutation in {"inode", "symlink"}:
        backup = target.with_name("preserved-original-fence")
        target.rename(backup)
        if mutation == "symlink":
            target.symlink_to(backup, target_is_directory=True)
        else:
            target.mkdir(mode=0o700)
            shutil.copyfile(backup / fence.MARKER_NAME, target / fence.MARKER_NAME)
    elif mutation == "writable":
        target.chmod(0o720)
    elif mutation == "root":
        fence.PROJECT.chmod(0o755)
    with pytest.raises((ValueError, OSError)):
        fence.verify(proof)
    assert fence.reconcile(request)["state"] == "unknown"


@pytest.mark.parametrize(
    "kind", ["parent-symlink", "file-symlink", "hardlink", "group-write", "wrong-uid", "missing-project"]
)
def test_unsafe_preconditions_rejected_readonly(fixture, monkeypatch, kind):
    request, files = fixture
    path = next(iter(files))
    if kind == "parent-symlink":
        parent = path.parent
        moved = parent.with_name("preserved-submission")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    elif kind == "file-symlink":
        moved = path.with_suffix(".preserved")
        path.rename(moved)
        path.symlink_to(moved)
    elif kind == "hardlink":
        os.link(path, path.with_suffix(".link"))
    elif kind == "group-write":
        path.chmod(0o620)
    elif kind == "wrong-uid":
        monkeypatch.setattr(fence, "EXPECTED_UID", os.getuid() + 1)
    else:
        fence.PROJECT.rmdir()
    with pytest.raises((ValueError, OSError)):
        fence.dry_run(request)
    assert not Path(request["claim_path"]).exists()


@pytest.mark.parametrize("kind", ["plan", "request", "dry", "authorize"])
def test_changed_request_dryrun_or_missing_authorization_rejected(fixture, kind):
    request, _ = fixture
    dry = fence.dry_run(request)
    if kind == "plan":
        request = copy.deepcopy(request)
        request["old_plan"]["mode"] = "fit"
    elif kind == "request":
        request = copy.deepcopy(request)
        request[fence.FLAGS[0]] = 0  # False==0 must not bypass typed canonical binding.
    elif kind == "dry":
        dry["old_files"] = {}
    with pytest.raises((ValueError, OSError)):
        fence.install(request, authorize=(kind != "authorize"), dry_run=dry)
    assert not Path(request["claim_path"]).exists()


def _race_child(request, dry, event, queue):
    event.wait(5)
    try:
        fence.install(request, authorize=True, dry_run=dry)
        queue.put("installed")
    except (ValueError, OSError) as error:
        queue.put(type(error).__name__)


def test_two_real_fence_processes_have_only_one_winner(fixture):
    request, _ = fixture
    dry = fence.dry_run(request)
    context = multiprocessing.get_context("fork")
    event, queue = context.Event(), context.Queue()
    children = [context.Process(target=_race_child, args=(request, dry, event, queue)) for _ in range(2)]
    try:
        for child in children:
            child.start()
        event.set()
        results = [queue.get(timeout=10) for _ in children]
        assert results.count("installed") == 1
        assert all(x in {"installed", "FenceError", "FileExistsError"} for x in results)
        for child in children:
            child.join(timeout=5)
            assert child.exitcode == 0
        assert fence.reconcile(request)["state"] == "fenced"
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=2)
            child.close()
        queue.close()
        queue.join_thread()


def test_real_original_plan_build_is_pinned_and_pure():
    # Restore production constants for this one pure/no-filesystem request check.
    path = ROOT / ".sdsc/student-name-invariant-v1/19665dfbd26d7b0024a46d9b7b811ce4/plan.json"
    if not path.is_file():
        pytest.skip("original private submission evidence is not present in this checkout")
    plan = json.loads(path.read_bytes())
    request = fence.build_request(plan)
    assert len(request["old_files"]) == 36
    assert request["old_plan_sha256"] == "b88f012fb8c30a9527d6300680dbabe29a944fdff64b29339a28f0c34764af5f"
    plan["result_dir"] += "-other"
    with pytest.raises(ValueError):
        fence.build_request(plan)


def _other_client(identity, filesystem):
    identity, filesystem = copy.deepcopy(identity), copy.deepcopy(filesystem)
    identity["device"] = os.makedev(240, 33)
    filesystem["mount"]["device_major_minor"] = "240:33"
    filesystem["mount"]["mount_id"] += 500
    return identity, filesystem


def test_all_cross_host_device_numbers_can_differ_only_under_same_bound_filesystem(fixture, monkeypatch):
    request, _ = fixture
    proof = install(request)
    original_bytes = Path(request["proof_path"]).read_bytes()
    original_read, original_roots, original_directory = fence._read, fence._roots, fence._directory

    def read(path, *args, **kwargs):
        raw, row = original_read(path, *args, **kwargs)
        row["stamp"], row["filesystem"] = _other_client(row["stamp"], row["filesystem"])
        return raw, row

    def roots():
        rows = original_roots()
        for row in rows.values():
            row["identity"], row["filesystem"] = _other_client(row["identity"], row["filesystem"])
        return rows

    def directory(path, *, filesystem=False):
        identity, names, fs = original_directory(path, filesystem=True)
        identity, fs = _other_client(identity, fs)
        return (identity, names, fs) if filesystem else (identity, names)

    monkeypatch.setattr(fence, "_read", read)
    monkeypatch.setattr(fence, "_roots", roots)
    monkeypatch.setattr(fence, "_directory", directory)
    observation = fence.verify(proof, return_observation=True)
    assert observation["proof"] == proof and fence.verify(proof) == proof
    current = observation["verification"]
    assert all(row["stamp"]["device"] == os.makedev(240, 33) for row in current["old_files"].values())
    assert current["claim"]["stamp"]["device"] == current["marker"]["stamp"]["device"] == os.makedev(240, 33)
    assert current["directory_identity"]["device"] == os.makedev(240, 33)
    assert all(row["identity"]["device"] == os.makedev(240, 33) for row in current["roots"].values())
    assert Path(request["proof_path"]).read_bytes() == original_bytes


@pytest.mark.parametrize("category", ["old_file", "claim", "marker", "roots", "directory"])
def test_cross_host_same_inode_content_cannot_hide_changed_filesystem_source(fixture, monkeypatch, category):
    request, _ = fixture
    proof = install(request)
    original_read, original_roots, original_directory = fence._read, fence._roots, fence._directory

    def changed(fs):
        fs["portable"]["source"] = fs["mount"]["source"] = "unreviewed-server:/other"

    if category in {"old_file", "claim", "marker"}:
        target = next(iter(request["old_files"])) if category == "old_file" else request[category + "_path"]

        def read(path, *args, **kwargs):
            raw, row = original_read(path, *args, **kwargs)
            if str(path) == target:
                changed(row["filesystem"])
            return raw, row

        monkeypatch.setattr(fence, "_read", read)
    elif category == "roots":

        def roots():
            rows = original_roots()
            changed(rows[str(fence.CONTROL)]["filesystem"])
            return rows

        monkeypatch.setattr(fence, "_roots", roots)
    else:

        def directory(path, *, filesystem=False):
            identity, names, fs = original_directory(path, filesystem=True)
            changed(fs)
            return (identity, names, fs) if filesystem else (identity, names)

        monkeypatch.setattr(fence, "_directory", directory)
    with pytest.raises(ValueError, match="unreviewed backing filesystem"):
        fence.verify(proof)


@pytest.mark.parametrize("kind", ["fstype", "filesystem_path", "device_mapping"])
def test_portable_record_does_not_skip_filesystem_or_local_device_validation(fixture, kind):
    request, _ = fixture
    proof = install(request)
    row = copy.deepcopy(next(iter(proof["old_files"].values())))
    if kind == "device_mapping":
        row["stamp"]["device"] = os.makedev(240, 33)
    else:
        row["filesystem"]["portable"][kind] = "wrong"
    with pytest.raises(ValueError):
        fence._portable_record(row)


@pytest.mark.parametrize("kind", ["valid", "wrong_device", "ambiguous", "malformed_escape"])
def test_mountinfo_parser_binds_real_fd_and_retains_only_selected_mount(fixture, monkeypatch, kind):
    request, files = fixture
    target = next(iter(files))
    fd = os.open(target, os.O_RDONLY)
    original_open = open
    try:
        info = os.fstat(fd)
        device = f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"
        selected_device = "240:33" if kind == "wrong_device" else device
        mountpoint = str(target.parent).replace(" ", r"\040")
        fsroot = r"/export\040root" if kind != "malformed_escape" else r"/export\999root"
        row = f"801 2 {selected_device} {fsroot} {mountpoint} rw - fixturefs fixture-server:/export rw\n"
        raw = ("999 2 241:2 / /unrelated-private-home rw - ignored ignored rw\n" + row).encode()
        if kind == "ambiguous":
            raw += row.replace("801 2", "802 2").encode()

        def fake_open(path, *args, **kwargs):
            return (
                io.BytesIO(raw)
                if str(path) == "/proc/self/mountinfo"
                else original_open(path, *args, **kwargs)
            )

        monkeypatch.setattr("builtins.open", fake_open)
        if kind == "valid":
            result = fence._observe_filesystem(fd, target)
            assert result["portable"] == dict(
                fstype="fixturefs",
                source="fixture-server:/export",
                filesystem_path="/export root/" + target.name,
            )
            assert "unrelated-private-home" not in json.dumps(result)
            assert result["mount"]["device_major_minor"] == device
        else:
            with pytest.raises(ValueError):
                fence._observe_filesystem(fd, target)
    finally:
        os.close(fd)


def test_same_host_device_change_during_read_still_rejected(fixture, monkeypatch):
    request, files = fixture
    path = next(iter(files))
    target_inode = path.stat().st_ino
    original = fence.os.fstat
    calls = []

    def changing(fd):
        info = original(fd)
        if stat.S_ISREG(info.st_mode) and info.st_ino == target_inode:
            calls.append(True)
            if len(calls) >= 4:
                fields = {
                    key: getattr(info, key)
                    for key in (
                        "st_mode",
                        "st_dev",
                        "st_ino",
                        "st_uid",
                        "st_size",
                        "st_nlink",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                }
                fields["st_dev"] = os.makedev(240, 33)
                return SimpleNamespace(**fields)
        return info

    monkeypatch.setattr(fence.os, "fstat", changing)
    with pytest.raises(ValueError, match="changed during read"):
        fence._read(path)
