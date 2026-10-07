#!/usr/bin/env python3
"""Create/deploy an explicitly authorized finite successor plan; never submit jobs."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdsc_cli as cli
import sdsc_pipeline as pipeline
import sdsc_pipeline_remote as remote
import sdsc_pipeline_storage as s

CONTROL_FILES = (
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    "tools/sdsc_pipeline",
    "tools/sdsc_pipeline.py",
    "tools/sdsc_pipeline_remote.py",
    "tools/sdsc_pipeline_storage.py",
    "tools/sdsc_pipeline_inputs.py",
    "tools/sdsc_pipeline_job.py",
    "tools/sdsc_pipeline_job.sh",
    "tools/sdsc_g0.py",
    "tools/sdsc_finalize_g0.py",
    "tools/sdsc_resume.py",
    "tools/sdsc_science_binding.py",
    "tools/sdsc_pilot.py",
    "tools/sdsc_pilot_preflight.py",
    "tools/sdsc_g0_calibration.py",
    "tools/sdsc_provenance.py",
    "tools/sdsc_training_preflight.py",
    "tools/sdsc_teacher_prepare.py",
    "scripts/production/slurm_supervision.sh",
)


def validated_controls(root, manifest, review):
    """Bind reviewed live tools to the already frozen release, before claiming a flow."""
    s.require(manifest["code_sha256"] == s.sha(s.canonical(manifest["files"])), "release digest differs")
    records = {entry["path"]: entry for entry in manifest["files"]}
    s.require(len(records) == len(manifest["files"]), "duplicate release path")
    files = {}
    for name in CONTROL_FILES:
        actual = s.identity(root / name)
        expected = records.get(name)
        s.require(
            expected is not None
            and all(expected[key] == actual[key] for key in ("sha256", "size"))
            and expected["mode"] == (0o755 if (root / name).stat().st_mode & 0o111 else 0o644),
            "reviewed control differs from frozen release: " + name,
        )
        s.require(review["files"].get(name) == actual["sha256"], "reviewed control file changed: " + name)
        files[name] = actual["sha256"]
    return files


DEPLOY_SCRIPT = r"""import sys,json,pathlib,hashlib,os,subprocess,stat
def require(value,message):
    if not value: raise ValueError(message)
def safe(value):
    p=pathlib.Path(value)
    require(p.is_absolute() and '..' not in p.parts and not any(ord(c)<32 for c in str(p)), 'unsafe path')
    require(not any(x.is_symlink() for x in (p,*p.parents)), 'symlink path')
    return p
def read(value,limit=16*1024*1024):
    p=safe(value); info=p.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_size<=limit, 'metadata file bound')
    raw=p.read_bytes(); require(len(raw)==info.st_size,'metadata size changed'); return raw
def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
raw=sys.stdin.buffer.read(1024*1024+1)
require(len(raw)<=1024*1024,'plan bound')
plan=json.loads(raw)
control=pathlib.Path('/home/zgao12/quest-runs/OPD')
require(pathlib.Path.home()==pathlib.Path('/home/zgao12'),'wrong login identity')
require(bool(os.environ.get('SSH_CONNECTION')) and not os.environ.get('SLURM_JOB_ID'),'login SSH required')
import re
require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',plan['flow_id']),'flow ID')
require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',plan['run_id']),'run ID')
root=safe(control/'pipelines'/plan['flow_id'])
require(not root.exists(),'flow already exists; inspect rather than redeploy')
release=safe(control/'releases'/plan['run_id'])
manifest=json.loads(read(release/'manifest.json'))
require(manifest['run_id']==plan['run_id'],'release run differs')
require(manifest['code_sha256']==plan['code_sha256']==sha(canonical(manifest['files'])),'release differs')
records={item['path']:item for item in manifest['files']}
require(len(records)==len(manifest['files']),'duplicate release path')
for name,expected in plan['control_files'].items():
    relative=pathlib.PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name,'control path')
    data=read(release/'source'/name)
    require(sha(data)==expected==records[name]['sha256'] and len(data)==records[name]['size'],name)
provenance=safe(plan['provenance_dir'])
require(provenance==control/'provenance'/plan['provenance_sha256'],'provenance directory')
data=read(provenance/'manifest.json')
require(sha(data)==plan['provenance_sha256'],'provenance SHA')
proof=json.loads(data)
require(proof['git_head']==plan['science_head'],'provenance HEAD')
require(proof['wrapper']['run_id']==plan['run_id'] and
        proof['wrapper']['code_sha256']==plan['code_sha256'],'provenance wrapper')
require(sha(read(provenance/'wrapper-manifest.json'))==proof['wrapper']['manifest_sha256'],'wrapper SHA')
require(proof['bundle']['path']=='history.bundle','bundle path')
bundle=read(provenance/'history.bundle',64*1024*1024)
require(len(bundle)==proof['bundle']['size'] and sha(bundle)==proof['bundle']['sha256'],'bundle SHA')
python=safe(plan['g0_python'])
require(os.access(python,os.X_OK) and
        sha(read(python,128*1024*1024))==plan['g0_python_sha256'],'G0 Python changed')
environment=plan['environment']; container=environment['container']; runtime=environment['runtime']
engine=safe(container['runtime']); image=safe(container['image']); image_info=image.stat()
require(engine.is_file() and os.access(engine,os.X_OK),'container runtime')
require(stat.S_ISREG(image_info.st_mode) and type(container['size']) is int and
        type(container['mtime_ns']) is int and
        (image_info.st_size,image_info.st_mtime_ns)==(container['size'],container['mtime_ns']),
        'image changed')
pilot=safe(runtime['python_path']); base=safe(runtime['base_prefix'])
require(base==python.parent.parent and base.is_dir(),'pilot base is not fixed G0 runtime')
require(os.access(pilot,os.X_OK) and
        sha(read(pilot,128*1024*1024))==runtime['python_sha256'],'pilot Python changed')
check=('import importlib.metadata as m,json,sys,platform; p=json.loads(sys.argv[1]); '
       'print(json.dumps({"packages":{k:m.version(k) for k in p},'
       '"python":platform.python_version(),"base_prefix":sys.base_prefix,"python_path":sys.executable}))')
result=subprocess.run([str(pilot),'-I','-B','-c',check,json.dumps(runtime['packages'])],
                      check=True,capture_output=True,timeout=60)
actual=json.loads(result.stdout)
for key in ('packages','python','base_prefix','python_path'):
    require(canonical(actual[key])==canonical(runtime[key]),'pilot runtime differs: '+key)
vendor=environment['mib']; vendor_manifest=json.loads(read(vendor['manifest']))
require(sha(read(vendor['manifest']))==vendor['manifest_sha256'],'vendor manifest changed')
require(vendor_manifest['head']==vendor['head'] and safe(vendor['path']).is_dir(),'vendor identity')
root.mkdir(parents=True,exist_ok=False)
raw=canonical(plan)+b'\n'
with (root/'plan.json').open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
for folder in (root,root.parent):
    fd=os.open(folder,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
require(read(root/'plan.json')==raw,'plan read-back failed')
print(json.dumps({'plan_sha256':sha(raw),'directory':str(root),'started':False}))
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "flow-id", "g0-python", "g0-python-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("environment", "provenance", "review", "parent-state"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-job-id", required=True)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args(argv)
    s.require(args.authorize, "existing explicit user authorization must be acknowledged")
    environment = s.document(args.environment.absolute())
    provenance = s.document(args.provenance.absolute())
    review = s.document(args.review.absolute())
    record = cli.run_record(args.run_id)
    manifest = record["manifest"]
    root = pipeline.ROOT
    s.require(record["state"] == "deployed", "source snapshot must be deployed after dry-run first")
    s.require(provenance["wrapper_code_sha256"] == manifest["code_sha256"], "provenance wrapper differs")
    s.require(
        environment.get("passed") is True and environment["runtime"]["packages"].get("trl") == "0.22.2",
        "pilot runtime preparation incomplete",
    )
    s.require(
        review.get("independent_review_complete") is True
        and review.get("scope") == "single_invocation_h100_g0_and_seed42_pilot",
        "completed independent review record required",
    )
    files = validated_controls(root, manifest, review)
    allowed = s.helper("sdsc_finalize_g0").REVIEWABLE_ADAPTERS
    adapter_files = {Path(name).name: digest for name, digest in files.items() if Path(name).name in allowed}
    plan = {
        "schema": "quest-sdsc-pipeline-plan-v1",
        "flow_id": args.flow_id,
        "authorized": True,
        "authorization_scope": (
            "user requested formal training and automatic G0 to seed-42 pilot; no factorial/Gemma"
        ),
        "full_factorial_authorized": False,
        "quest_root": str(root),
        "quest_host": socket.gethostname().split(".")[0],
        "run_id": args.run_id,
        "code_sha256": manifest["code_sha256"],
        "science_head": provenance["git_head"],
        "protocol_sha256": "752fa685d795335527c639fb2b7f6cc3e94aa60ffb9a16d4329bf13099b0888e",
        "provenance_sha256": provenance["manifest_sha256"],
        "provenance_dir": str(s.CONTROL / "provenance" / provenance["manifest_sha256"]),
        "parent_state": str(args.parent_state.absolute()),
        "preflight_job_id": args.preflight_job_id,
        "g0_python": args.g0_python,
        "g0_python_sha256": args.g0_python_sha256,
        "hf_home": str(s.PROJECT / "cache/huggingface"),
        "environment": environment,
        "stages": list(remote.STAGES),
        "resources": remote.resource_plan(),
        "poll_seconds": 300,
        "deadline_seconds": 14 * 24 * 3600,
        "control_files": files,
        "review_sha256": s.sha(s.read(args.review.absolute())),
        "adapter_review": {
            "scope": "single_invocation_h100_g0",
            "reviewer": review["reviewers"],
            "files": adapter_files,
        },
    }
    pipeline.validate_plan(plan)
    directory = root / ".sdsc/supervision" / args.flow_id
    s.require(not directory.exists(), "flow name already exists; inspect it rather than re-arming")
    directory.mkdir(parents=True)
    digest = s.atomic(directory / "plan.json", plan)
    s.atomic(directory / "review.json", review)
    result = {
        "plan": str(directory / "plan.json"),
        "plan_sha256": digest,
        "deployed": False,
        "started": False,
    }
    if args.deploy:
        # The only remote mutation is a fresh project control directory and two
        # bounded JSON files. The original user edit workspace is never touched.
        response = cli.ssh_call(
            [args.g0_python, "-I", "-B", "-c", DEPLOY_SCRIPT], data=s.canonical(plan), timeout=120
        )
        s.require(
            response.returncode == 0,
            "remote plan deployment failed; inspect local/remote plan before any retry",
        )
        receipt = json.loads(response.stdout)
        s.require(receipt["plan_sha256"] == digest, "remote plan bytes differ")
        s.atomic(directory / "deployment.json", receipt)
        result["deployed"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
