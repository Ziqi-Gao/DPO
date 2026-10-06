"""Exact Torch 2.8 RemoteModule generated-origin evidence, never a tmp-root grant.

This additive helper does not change historical attestations. Old captures lack
its directory evidence and cannot satisfy this contract retrospectively. Callers
must first validate their plan/context and runtime manifest; the returned single
path is the entire exception. CUDA, model acceptance and completion are outside
this contract. Capturing never imports Torch or constructs a model.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import tempfile
import types
from pathlib import Path

SCHEMA = "quest-sdsc-torch-remote-module-derived-origin-v1"
MODULE = "_remote_module_non_scriptable"
FILENAME = MODULE + ".py"
SIZE = 2355
SHA256 = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
INST = "torch.distributed.nn.jit.instantiator"
TEMPLATE = "torch.distributed.nn.jit.templates.remote_module_template"
REMOTE = "torch.distributed.nn.api.remote_module"
SOURCES = {
    INST: (
        "torch/distributed/nn/jit/instantiator.py",
        5510,
        "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615",
    ),
    TEMPLATE: (
        "torch/distributed/nn/jit/templates/remote_module_template.py",
        3463,
        "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc",
    ),
    REMOTE: (
        "torch/distributed/nn/api/remote_module.py",
        31286,
        "d51779fdbafb1ca1d99d002e6f32cf17bc1fef730a6251e3ef0278d8078bd112",
    ),
}
STAMP_KEYS = {"dev", "ino", "uid", "mode", "size", "nlink", "mtime_ns", "ctime_ns"}
WORK_IDENTITY_FIELDS = ("dev", "ino", "uid", "mode")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_path(value):
    require(isinstance(value, str | Path), "path must be text or Path")
    p = Path(value)
    require(p.is_absolute() and ".." not in p.parts and str(p) == str(value), "noncanonical path")
    return p


def expected_sources(runtime_dir, manifest):
    require(
        isinstance(manifest, dict) and re.fullmatch("[a-f0-9]{64}", manifest.get("runtime_root_sha256", "")),
        "runtime identity absent",
    )
    entries = manifest.get("entries")
    require(isinstance(entries, list), "runtime entries absent")
    result = {}
    for name, (suffix, size, digest) in SOURCES.items():
        relative = "lib/python3.12/site-packages/" + suffix
        matches = [r for r in entries if r.get("path") == relative]
        require(len(matches) == 1, "backing runtime source missing or duplicated")
        row = matches[0]
        require(
            row.get("type") == "file"
            and row.get("size") == size
            and row.get("sha256") == digest
            and type(row.get("mode")) is int
            and 0 <= row["mode"] <= 0o777
            and not row["mode"] & 0o002,
            "backing runtime source changed",
        )
        result[name] = dict(path=str(runtime_dir / relative), size=size, sha256=digest, mode=row["mode"])
    return result


def stamp(info):
    return dict(
        dev=info.st_dev,
        ino=info.st_ino,
        uid=info.st_uid,
        mode=stat.S_IMODE(info.st_mode),
        size=info.st_size,
        nlink=info.st_nlink,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def open_directory(path):
    """Open every component without following any symlink, including ancestors."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_regular(parent_fd, name, cap):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "derived/backing source is not regular")
        raw = bytearray()
        while len(raw) <= cap:
            chunk = os.read(fd, min(65536, cap + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(fd)
        require(
            len(raw) <= cap and stamp(before) == stamp(after), "source oversized or changed while reading"
        )
        require(
            stamp(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) == stamp(after),
            "source replaced while reading",
        )
        return bytes(raw), stamp(after)
    finally:
        os.close(fd)


def module_record(name, module, *, canonicalize=False):
    require(type(module) is types.ModuleType, "module is not an ordinary module")
    values = vars(module)
    spec = values.get("__spec__")
    require(
        values.get("__name__") == name and spec is not None and spec.name == name, "module identity differs"
    )
    file, origin = values.get("__file__"), spec.origin
    require(isinstance(file, str) and isinstance(origin, str), "module file/spec absent")
    if canonicalize:
        file, origin = str(Path(file).resolve(strict=True)), str(Path(origin).resolve(strict=True))
    return dict(name=name, file=file, spec_origin=origin)


def live_bindings(modules, path):
    generated, inst, template, remote = (modules.get(k) for k in (MODULE, INST, TEMPLATE, REMOTE))
    require(
        all(type(m) is types.ModuleType for m in (generated, inst, template, remote)),
        "backing modules absent",
    )
    iv, tv, rv, gv = map(vars, (inst, template, remote, generated))
    temporary = iv.get("_TEMP_DIR")
    directory = str(path.parent)
    require(
        type(temporary) is tempfile.TemporaryDirectory and vars(temporary).get("name") == directory,
        "temporary-directory singleton differs",
    )
    require(
        iv.get("INSTANTIATED_TEMPLATE_DIR_PATH") == directory and iv.get("_FILE_PREFIX") == "_remote_module_",
        "instantiator static directory/prefix differs",
    )
    require(
        rv.get("instantiator") is inst and rv.get("_NON_SCRIPTABLE_REMOTE_MODULE_MODULE") is generated,
        "remote-module singleton differs",
    )
    require(
        iv.get("get_remote_module_template") is tv.get("get_remote_module_template")
        and type(tv.get("get_remote_module_template")) is types.FunctionType
        and tv["get_remote_module_template"].__globals__ is tv
        and tv["get_remote_module_template"].__module__ == TEMPLATE,
        "template callable identity differs",
    )
    functions = [gv.get(k) for k in ("forward_async", "forward", "_remote_forward")]
    require(
        all(
            type(f) is types.FunctionType
            and f.__globals__ is gv
            and f.__module__ == MODULE
            and f.__code__.co_filename == str(path)
            for f in functions
        ),
        "generated callable origins differ",
    )
    require(
        gv.get("_generated_methods") == functions[:2] and gv.get("module_interface_cls", object()) is None,
        "non-scriptable generated methods differ",
    )
    return dict(
        instantiator_directory=directory,
        temporary_directory=directory,
        file_prefix="_remote_module_",
        remote_singleton_module=MODULE,
        template_callable_module=TEMPLATE,
        singleton_identity_verified=True,
        generated_function_names=["forward_async", "forward", "_remote_forward"],
    )


def capture_generated_origin(
    *, work_dir, runtime_dir, expected_uid, runtime_manifest, context_sha256, modules=None
):
    """Observe one exact generated file; never manufacture it or import dependencies."""
    work, runtime = canonical_path(work_dir), canonical_path(runtime_dir)
    modules = dict(sys.modules) if modules is None else dict(modules)
    generated_names = {n for n in modules if n.startswith("_remote_module_")}
    if not generated_names:
        require(MODULE not in modules, "generated namespace inconsistency")
        return None
    require(generated_names == {MODULE}, "multiple or scriptable generated modules")
    require(type(expected_uid) is int and expected_uid == os.getuid(), "live owner differs")
    require(
        isinstance(context_sha256, str) and re.fullmatch("[a-f0-9]{64}", context_sha256),
        "context identity absent",
    )
    require(
        os.environ.get("TMPDIR") == str(work) and sys.dont_write_bytecode is True,
        "private TMPDIR/bytecode policy differs",
    )
    source_rows = expected_sources(runtime, runtime_manifest)
    record = module_record(MODULE, modules[MODULE])
    path = canonical_path(record["file"])
    require(
        record["spec_origin"] == str(path)
        and path.name == FILENAME
        and path.parent.parent == work
        and re.fullmatch(r"tmp[a-z0-9_]{8}", path.parent.name),
        "generated origin is not the exact private work child",
    )
    require(sys.path.count(str(path.parent)) == 1, "generated search path missing or duplicated")
    bindings = live_bindings(modules, path)
    for name, expected in source_rows.items():
        row = module_record(name, modules[name], canonicalize=True)
        require(row["file"] == row["spec_origin"] == expected["path"], "backing module origin differs")
        p = Path(expected["path"])
        fd = open_directory(p.parent)
        try:
            raw, observed = read_regular(fd, p.name, expected["size"])
        finally:
            os.close(fd)
        require(
            observed["uid"] == expected_uid
            and observed["mode"] == expected["mode"]
            and len(raw) == expected["size"]
            and hashlib.sha256(raw).hexdigest() == expected["sha256"],
            "backing source bytes/owner/mode differ",
        )
    work_fd = open_directory(work)
    directory_fd = None
    try:
        work_before = stamp(os.fstat(work_fd))
        directory_fd = os.open(
            path.parent.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=work_fd
        )
        before = stamp(os.fstat(directory_fd))
        require(
            work_before["uid"] == before["uid"] == expected_uid
            and work_before["mode"] == before["mode"] == 0o700,
            "work/generated directory owner or mode differs",
        )
        require(sorted(os.listdir(directory_fd)) == [FILENAME], "generated directory contains extra entries")
        raw, file_stamp = read_regular(directory_fd, FILENAME, SIZE)
        require(
            file_stamp["uid"] == expected_uid
            and file_stamp["mode"] == 0o600
            and file_stamp["nlink"] == 1
            and len(raw) == SIZE
            and hashlib.sha256(raw).hexdigest() == SHA256,
            "generated source bytes/owner/mode/links differ",
        )
        require(sorted(os.listdir(directory_fd)) == [FILENAME], "generated directory entries changed")
        require(
            stamp(os.stat(FILENAME, dir_fd=directory_fd, follow_symlinks=False)) == file_stamp,
            "generated file changed",
        )
        require(
            stamp(os.fstat(directory_fd))
            == before
            == stamp(os.stat(path.parent.name, dir_fd=work_fd, follow_symlinks=False)),
            "generated directory changed",
        )
        # Other ranks own sibling children in this shared work directory. Their
        # normal activity may alter size/nlink/timestamps, never its identity.
        work_after = stamp(os.fstat(work_fd))
        path_after = stamp(work.stat(follow_symlinks=False))
        require(
            all(work_after[k] == work_before[k] == path_after[k] for k in WORK_IDENTITY_FIELDS),
            "work directory identity changed",
        )
        # Reopen the full path so replacement of any ancestor by a symlink also fails.
        reopened_work = open_directory(work)
        try:
            reopened = stamp(os.fstat(reopened_work))
            require(
                all(reopened[k] == work_before[k] for k in WORK_IDENTITY_FIELDS),
                "work path identity changed",
            )
        finally:
            os.close(reopened_work)
        require(live_bindings(modules, path) == bindings, "live bindings changed during capture")
        return dict(
            schema=SCHEMA,
            context_sha256=context_sha256,
            runtime_root_sha256=runtime_manifest["runtime_root_sha256"],
            work_dir=str(work),
            runtime_dir=str(runtime),
            expected_uid=expected_uid,
            work=dict(path=str(work), **work_before),
            work_after=dict(path=str(work), **work_after),
            work_identity_stable_fields=list(WORK_IDENTITY_FIELDS),
            directory=dict(path=str(path.parent), **before),
            file=dict(path=str(path), sha256=SHA256, **file_stamp),
            directory_entries=[FILENAME],
            backing_sources=source_rows,
            live_bindings=bindings,
            stable_before_after=True,
        )
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(work_fd)


def validate_generated_origin(
    evidence,
    *,
    work_dir,
    runtime_dir,
    expected_uid,
    runtime_manifest,
    context_sha256,
    modules,
    file_rows,
    sys_path,
    native_libraries,
    tmpdir,
    dont_write_bytecode,
):
    """Pure replay: return one file path, never permit its subtree or native maps."""
    work, runtime = canonical_path(work_dir), canonical_path(runtime_dir)
    require(type(expected_uid) is int and expected_uid >= 0, "invalid expected owner")
    require(
        isinstance(context_sha256, str) and re.fullmatch("[a-f0-9]{64}", context_sha256),
        "context identity absent",
    )
    require(
        isinstance(modules, list)
        and isinstance(file_rows, list)
        and isinstance(sys_path, list)
        and isinstance(native_libraries, list),
        "full inventories absent",
    )
    names = [r["name"] for r in modules]
    paths = [r["path"] for r in file_rows]
    require(
        len(names) == len(set(names)) and len(paths) == len(set(paths)), "duplicate module or file inventory"
    )
    generated = [n for n in names if n.startswith("_remote_module_")]
    temporary_paths = [
        p
        for p in sys_path
        if isinstance(p, str) and Path(p).parent == work and Path(p).name.startswith("tmp")
    ]
    if evidence is None:
        require(
            not generated
            and not temporary_paths
            and not any(Path(p).name.startswith("_remote_module_") for p in paths),
            "derived evidence absent",
        )
        return None
    require(
        set(evidence)
        == {
            "schema",
            "context_sha256",
            "runtime_root_sha256",
            "work_dir",
            "runtime_dir",
            "expected_uid",
            "work",
            "work_after",
            "work_identity_stable_fields",
            "directory",
            "file",
            "directory_entries",
            "backing_sources",
            "live_bindings",
            "stable_before_after",
        },
        "derived evidence fields differ",
    )
    sources = expected_sources(runtime, runtime_manifest)
    require(
        evidence["schema"] == SCHEMA
        and evidence["context_sha256"] == context_sha256
        and evidence["runtime_root_sha256"] == runtime_manifest["runtime_root_sha256"]
        and evidence["work_dir"] == str(work)
        and evidence["runtime_dir"] == str(runtime)
        and evidence["expected_uid"] == expected_uid
        and type(evidence["expected_uid"]) is int,
        "derived context/runtime/owner binding differs",
    )
    require(
        tmpdir == str(work) and dont_write_bytecode is True and evidence["stable_before_after"] is True,
        "derived environment or stability differs",
    )
    f, d, w = evidence["file"], evidence["directory"], evidence["work"]
    work_after = evidence["work_after"]
    for row, extra in ((f, {"sha256"}), (d, set()), (w, set()), (work_after, set())):
        require(
            set(row) == {"path", *STAMP_KEYS, *extra}
            and all(type(row[k]) is int and row[k] >= 0 for k in STAMP_KEYS)
            and row["ino"] > 0
            and row["nlink"] > 0,
            "derived metadata differs",
        )
    require(
        evidence["work_identity_stable_fields"] == list(WORK_IDENTITY_FIELDS)
        and work_after["path"] == w["path"]
        and all(work_after[k] == w[k] for k in WORK_IDENTITY_FIELDS),
        "shared work identity or stability scope differs",
    )
    path = canonical_path(f["path"])
    require(
        path.name == FILENAME
        and path.parent.parent == work
        and re.fullmatch(r"tmp[a-z0-9_]{8}", path.parent.name)
        and d["path"] == str(path.parent)
        and w["path"] == str(work),
        "derived path is not exact private child",
    )
    require(
        f["uid"] == d["uid"] == w["uid"] == expected_uid
        and f["mode"] == 0o600
        and d["mode"] == w["mode"] == 0o700
        and f["nlink"] == 1
        and f["size"] == SIZE
        and f["sha256"] == SHA256
        and f["dev"] == d["dev"] == w["dev"],
        "derived bytes/owner/modes/device differ",
    )
    require(
        evidence["directory_entries"] == [FILENAME] and evidence["backing_sources"] == sources,
        "derived entry inventory/backing sources differ",
    )
    expected_bindings = dict(
        instantiator_directory=str(path.parent),
        temporary_directory=str(path.parent),
        file_prefix="_remote_module_",
        remote_singleton_module=MODULE,
        template_callable_module=TEMPLATE,
        singleton_identity_verified=True,
        generated_function_names=["forward_async", "forward", "_remote_forward"],
    )
    require(
        evidence["live_bindings"] == expected_bindings
        and type(evidence["live_bindings"].get("singleton_identity_verified")) is bool,
        "live singleton evidence differs",
    )
    require(
        generated == [MODULE]
        and temporary_paths == [str(path.parent)]
        and sys_path.count(str(path.parent)) == 1,
        "generated module/search-path multiplicity differs",
    )
    module_map = {r["name"]: r for r in modules}
    file_map = {r["path"]: r for r in file_rows}
    for name, expected in {
        MODULE: dict(path=str(path), size=SIZE, sha256=SHA256, mode=0o600),
        **sources,
    }.items():
        require(
            module_map.get(name) == dict(name=name, file=expected["path"], spec_origin=expected["path"]),
            "derived or backing module origin differs",
        )
        observed = file_map.get(expected["path"], {})
        require(
            observed == dict(expected, uid=expected_uid)
            and all(type(observed.get(k)) is int for k in ("size", "mode", "uid")),
            "derived or backing file inventory differs",
        )
    for row in modules:
        for key in ("file", "spec_origin"):
            origin = row.get(key)
            if isinstance(origin, str) and Path(origin).is_relative_to(path.parent):
                require(row == module_map[MODULE], "generated directory module alias or extra origin")
    require(
        [p for p in paths if Path(p).is_relative_to(path.parent)] == [str(path)],
        "extra generated-directory file",
    )
    require(
        not any(Path(p).is_relative_to(path.parent) for p in native_libraries),
        "generated source cannot be a native library",
    )
    return str(path)
