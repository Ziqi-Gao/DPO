"""Actual CPU class-import capture plus pure boundary corruption; no GPU claim."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/sdsc_torch_generated_origins.py"
spec = importlib.util.spec_from_file_location("_derived_origin", SOURCE)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

CHILD = r"""
import importlib.util, json, os, stat, sys, types
from pathlib import Path

source, work = Path(sys.argv[1]), Path(sys.argv[2])
os.umask(0o077)
spec = importlib.util.spec_from_file_location("helper", source)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
import torch

assert torch.__version__ == "2.8.0+cpu" and torch.version.cuda is None
assert h.MODULE not in sys.modules
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

runtime = Path(sys.prefix).resolve()
manifest = dict(
    runtime_root_sha256="a" * 64,
    entries=[
        dict(
            type="file",
            path="lib/python3.12/site-packages/" + suffix,
            size=size,
            sha256=digest,
            mode=stat.S_IMODE((runtime / "lib/python3.12/site-packages" / suffix).stat().st_mode),
        )
        for suffix, size, digest in h.SOURCES.values()
    ],
)
kwargs = dict(
    work_dir=work,
    runtime_dir=runtime,
    expected_uid=os.getuid(),
    runtime_manifest=manifest,
    context_sha256="b" * 64,
)
e = h.capture_generated_origin(**kwargs)
assert e["file"]["sha256"] == h.SHA256 and e["file"]["size"] == 2355
rows = [
    dict(name=name, file=row["path"], spec_origin=row["path"]) for name, row in e["backing_sources"].items()
] + [dict(name=h.MODULE, file=e["file"]["path"], spec_origin=e["file"]["path"])]
files = [dict(row, uid=os.getuid()) for row in e["backing_sources"].values()] + [
    {k: e["file"][k] for k in ("path", "size", "sha256", "uid", "mode")}
]
validation = dict(
    work_dir=str(work),
    runtime_dir=str(runtime),
    expected_uid=os.getuid(),
    runtime_manifest=manifest,
    context_sha256="b" * 64,
    modules=rows,
    file_rows=files,
    sys_path=list(sys.path),
    native_libraries=[],
    tmpdir=str(work),
    dont_write_bytecode=True,
)
assert h.validate_generated_origin(e, **validation) == e["file"]["path"]
# A second rank may add a sibling while this rank reads its own generated file.
old_read_regular = h.read_regular
sibling = work / "other-rank-owned-sibling"
def sibling_during_read(parent_fd, name, cap):
    result = old_read_regular(parent_fd, name, cap)
    if name == h.FILENAME:
        sibling.mkdir(mode=0o700)
    return result
h.read_regular = sibling_during_read
try:
    concurrent = h.capture_generated_origin(**kwargs)
    assert concurrent["work"]["nlink"] != concurrent["work_after"]["nlink"]
    assert h.validate_generated_origin(concurrent, **validation) == e["file"]["path"]
finally:
    h.read_regular = old_read_regular
    if sibling.exists():
        sibling.rmdir()

# Live mutations are confined to this fresh private CPU fixture; all are restored.
f = Path(e["file"]["path"])
d = f.parent
original = f.read_bytes()
iv = vars(sys.modules[h.INST])
rv = vars(sys.modules[h.REMOTE])
gv = vars(sys.modules[h.MODULE])
failures = []


def reject(name, change, restore):
    change()
    try:
        try:
            h.capture_generated_origin(**kwargs)
        except (ValueError, OSError):
            failures.append(name)
        else:
            raise AssertionError("accepted " + name)
    finally:
        restore()


reject("file_mode", lambda: f.chmod(0o644), lambda: f.chmod(0o600))
reject("directory_mode", lambda: d.chmod(0o755), lambda: d.chmod(0o700))
reject("work_mode", lambda: work.chmod(0o755), lambda: work.chmod(0o700))
reject("bytes", lambda: f.write_bytes(original + b"\n"), lambda: f.write_bytes(original))
extra = d / "unexpected.py"
reject("extra_entry", lambda: extra.write_text(""), lambda: extra.unlink())
saved = f.with_name("held.py")
outside = work / "held.py"


def file_symlink():
    f.rename(outside)
    f.symlink_to(outside)


def undo_file_symlink():
    f.unlink()
    outside.rename(f)


reject("file_symlink", file_symlink, undo_file_symlink)


def hardlink():
    os.link(f, outside)


reject("hardlink", hardlink, lambda: outside.unlink())
old = rv["_NON_SCRIPTABLE_REMOTE_MODULE_MODULE"]
reject(
    "singleton",
    lambda: rv.__setitem__("_NON_SCRIPTABLE_REMOTE_MODULE_MODULE", types.ModuleType(h.MODULE)),
    lambda: rv.__setitem__("_NON_SCRIPTABLE_REMOTE_MODULE_MODULE", old),
)
old_dir = iv["INSTANTIATED_TEMPLATE_DIR_PATH"]
reject(
    "instantiator_directory",
    lambda: iv.__setitem__("INSTANTIATED_TEMPLATE_DIR_PATH", str(work)),
    lambda: iv.__setitem__("INSTANTIATED_TEMPLATE_DIR_PATH", old_dir),
)
reject(
    "second_generated_module",
    lambda: sys.modules.__setitem__("_remote_module_extra", types.ModuleType("_remote_module_extra")),
    lambda: sys.modules.pop("_remote_module_extra"),
)
reject("duplicate_sys_path", lambda: sys.path.append(str(d)), lambda: sys.path.pop())
reject(
    "TMPDIR",
    lambda: os.environ.__setitem__("TMPDIR", str(d)),
    lambda: os.environ.__setitem__("TMPDIR", str(work)),
)
reject(
    "bytecode",
    lambda: setattr(sys, "dont_write_bytecode", False),
    lambda: setattr(sys, "dont_write_bytecode", True),
)
old_code = gv["forward"].__code__
reject(
    "function_code_origin",
    lambda: setattr(gv["forward"], "__code__", old_code.replace(co_filename=str(work / "wrong.py"))),
    lambda: setattr(gv["forward"], "__code__", old_code),
)
old_origin = sys.modules[h.MODULE].__spec__.origin
reject(
    "spec_origin",
    lambda: setattr(sys.modules[h.MODULE].__spec__, "origin", str(work / "wrong.py")),
    lambda: setattr(sys.modules[h.MODULE].__spec__, "origin", old_origin),
)
tv = vars(sys.modules[h.TEMPLATE])
old_template = tv["get_remote_module_template"]
def replace_template():
    wrong = lambda value: "not the template"
    iv["get_remote_module_template"] = tv["get_remote_module_template"] = wrong
def restore_template():
    iv["get_remote_module_template"] = tv["get_remote_module_template"] = old_template
reject("template_callable_substitution", replace_template, restore_template)

# Shared-work identity changes during the read still fail; only sibling churn is allowed.
for change in ("work_inode_during_read", "work_symlink_during_read", "work_mode_during_read"):
    moved = work.with_name(work.name + "-held")
    def identity_during_read(parent_fd, name, cap):
        result = old_read_regular(parent_fd, name, cap)
        if name == h.FILENAME:
            if change == "work_mode_during_read":
                work.chmod(0o755)
            else:
                work.rename(moved)
                if change == "work_symlink_during_read":
                    work.symlink_to(moved, target_is_directory=True)
                else:
                    work.mkdir(mode=0o700)
        return result
    h.read_regular = identity_during_read
    try:
        try:
            h.capture_generated_origin(**kwargs)
        except (ValueError, OSError):
            failures.append(change)
        else:
            raise AssertionError("accepted " + change)
    finally:
        h.read_regular = old_read_regular
        if change == "work_mode_during_read":
            work.chmod(0o700)
        else:
            if work.is_symlink():
                work.unlink()
            else:
                work.rmdir()
            moved.rename(work)

# Directory symlink keeps the visible module path unchanged; nofollow must reject it.
held = work / "moved-generated"


def directory_symlink():
    d.rename(held)
    d.symlink_to(held, target_is_directory=True)


def undo_directory_symlink():
    d.unlink()
    held.rename(d)


reject("directory_symlink", directory_symlink, undo_directory_symlink)
print(
    json.dumps(
        dict(
            evidence=e,
            validation=validation,
            live_rejected=failures,
            scope=(
            "CPU model-class import only; exact three real source hashes and honest CPU modes; "
            "not a runtime/GPU certificate"
        ),
        ),
        sort_keys=True,
    )
)
"""


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    work = tmp_path_factory.mktemp("torch-derived").resolve()
    work.chmod(0o700)
    env = dict(
        os.environ,
        TMPDIR=str(work),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    response = subprocess.run(
        [sys.executable, "-I", "-B", "-c", CHILD, str(SOURCE), str(work)],
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert response.returncode == 0, response.stderr.decode() + response.stdout.decode()
    return json.loads(response.stdout)


def test_actual_qwen_class_import_capture_and_pure_replay(actual):
    e, kw = actual["evidence"], actual["validation"]
    assert h.validate_generated_origin(e, **kw) == e["file"]["path"]
    assert e["file"]["sha256"] == h.SHA256 and e["file"]["size"] == 2355
    assert e["directory_entries"] == [h.FILENAME]


def test_actual_live_capture_rejects_mutations(actual):
    assert set(actual["live_rejected"]) == {
        "file_mode",
        "directory_mode",
        "work_mode",
        "bytes",
        "extra_entry",
        "file_symlink",
        "hardlink",
        "singleton",
        "instantiator_directory",
        "second_generated_module",
        "duplicate_sys_path",
        "TMPDIR",
        "bytecode",
        "function_code_origin",
        "spec_origin",
        "directory_symlink",
        "template_callable_substitution",
        "work_inode_during_read",
        "work_symlink_during_read",
        "work_mode_during_read",
    }


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "size",
        "file_mode",
        "dir_mode",
        "uid",
        "inode_type",
        "device",
        "links",
        "context",
        "root",
        "stable",
        "directory_inventory",
        "backing_hash",
        "manifest_hash",
        "source_file_hash",
        "source_module_origin",
        "module_origin",
        "module_spec",
        "alias",
        "multiple",
        "native_mapping",
        "extra_file",
        "extra_search_path",
        "duplicate_search_path",
        "missing_evidence",
        "missing_backing_module",
        "missing_file",
        "duplicate_file",
        "duplicate_module",
        "singleton_binding",
        "tmpdir",
        "bytecode",
        "wrong_name",
        "nested_dir",
        "traversal",
    ],
)
def test_pure_rejects_corruption(actual, change):
    e, kw = copy.deepcopy(actual["evidence"]), copy.deepcopy(actual["validation"])
    path, directory = e["file"]["path"], e["directory"]["path"]
    if change == "hash":
        e["file"]["sha256"] = "0" * 64
    elif change == "size":
        e["file"]["size"] += 1
    elif change == "file_mode":
        e["file"]["mode"] = 0o644
    elif change == "dir_mode":
        e["directory"]["mode"] = 0o755
    elif change == "uid":
        e["directory"]["uid"] += 1
    elif change == "inode_type":
        e["directory"]["ino"] = True
    elif change == "device":
        e["directory"]["dev"] += 1
    elif change == "links":
        e["file"]["nlink"] = 2
    elif change == "context":
        e["context_sha256"] = "c" * 64
    elif change == "root":
        e["runtime_root_sha256"] = "c" * 64
    elif change == "stable":
        e["stable_before_after"] = 1
    elif change == "directory_inventory":
        e["directory_entries"].append("__pycache__")
    elif change == "backing_hash":
        e["backing_sources"][h.INST]["sha256"] = "0" * 64
    elif change == "manifest_hash":
        kw["runtime_manifest"]["entries"][0]["sha256"] = "0" * 64
    elif change == "source_file_hash":
        kw["file_rows"][0]["sha256"] = "0" * 64
    elif change == "source_module_origin":
        kw["modules"][0]["file"] = path
    elif change == "module_origin":
        kw["modules"][-1]["file"] = path + ".bak"
    elif change == "module_spec":
        kw["modules"][-1]["spec_origin"] = None
    elif change == "alias":
        kw["modules"].append(dict(name="alias", file=path, spec_origin=path))
    elif change == "multiple":
        kw["modules"].append(dict(name="_remote_module_other", file=path, spec_origin=path))
    elif change == "native_mapping":
        kw["native_libraries"].append(path)
    elif change == "extra_file":
        kw["file_rows"].append(dict(kw["file_rows"][-1], path=directory + "/extra.py"))
    elif change == "extra_search_path":
        kw["sys_path"].append(str(Path(directory).with_name("tmpabcdefgh")))
    elif change == "duplicate_search_path":
        kw["sys_path"].append(directory)
    elif change == "missing_evidence":
        e = None
    elif change == "missing_backing_module":
        kw["modules"].pop(0)
    elif change == "missing_file":
        kw["file_rows"].pop()
    elif change == "duplicate_file":
        kw["file_rows"].append(kw["file_rows"][-1])
    elif change == "duplicate_module":
        kw["modules"].append(kw["modules"][-1])
    elif change == "singleton_binding":
        e["live_bindings"]["remote_singleton_module"] = "other"
    elif change == "tmpdir":
        kw["tmpdir"] = directory
    elif change == "bytecode":
        kw["dont_write_bytecode"] = 1
    elif change == "wrong_name":
        e["file"]["path"] = directory + "/_remote_module_other.py"
    elif change == "nested_dir":
        e["file"]["path"] = directory + "/nested/" + h.FILENAME
    elif change == "traversal":
        e["file"]["path"] = directory + "/../" + h.FILENAME
    with pytest.raises(ValueError):
        h.validate_generated_origin(e, **kw)


def test_absent_origin_grants_nothing(actual):
    kw = copy.deepcopy(actual["validation"])
    kw["modules"] = [r for r in kw["modules"] if r["name"] != h.MODULE]
    kw["file_rows"] = [r for r in kw["file_rows"] if Path(r["path"]).name != h.FILENAME]
    kw["sys_path"].remove(actual["evidence"]["directory"]["path"])
    assert h.validate_generated_origin(None, **kw) is None


def test_capture_absent_does_not_import_torch():
    command = """
import importlib.util, sys
s = importlib.util.spec_from_file_location("h", sys.argv[1])
h = importlib.util.module_from_spec(s)
s.loader.exec_module(h)
assert h.capture_generated_origin(
    work_dir="/work", runtime_dir="/runtime", expected_uid=0,
    runtime_manifest={}, context_sha256="0" * 64, modules={}
) is None
assert "torch" not in sys.modules
"""
    response = subprocess.run(
        [sys.executable, "-I", "-B", "-c", command, str(SOURCE)], capture_output=True, timeout=10
    )
    assert response.returncode == 0, response.stderr.decode()


@pytest.mark.parametrize("field", ["dev", "ino", "uid", "mode", "path", "scope"])
def test_shared_work_identity_stays_strict(actual, field):
    evidence = copy.deepcopy(actual["evidence"])
    if field == "scope":
        evidence["work_identity_stable_fields"] = ["dev", "ino", "uid"]
    elif field == "path":
        evidence["work_after"][field] += "-other"
    else:
        evidence["work_after"][field] += 1
    with pytest.raises(ValueError, match="shared work"):
        h.validate_generated_origin(evidence, **actual["validation"])


def test_shared_work_sibling_metadata_changes_are_allowed(actual):
    evidence = copy.deepcopy(actual["evidence"])
    for field in ("size", "nlink", "mtime_ns", "ctime_ns"):
        evidence["work_after"][field] += 1
    assert h.validate_generated_origin(evidence, **actual["validation"]) == evidence["file"]["path"]
