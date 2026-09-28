"""Finite supervision fixtures: no SSH, scheduler, runtime loading or submission."""

import copy
import fcntl
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "student_supervise_fixture", ROOT / "tools/sdsc_student_supervise.py"
)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


# Actual 2026-09-28 SSH-only CLI responses; copied from local diagnostic records.
REAL_STATUSES = json.loads(
    "{\n"
    '  "final-status": {\n'
    '    "accounting": {\n'
    '      "returncode": 0,\n'
    '      "stderr": "",\n'
    '      "stdout": "54506703|COMPLETED|0:0|00:13:48|\\n54506703.batch|COMPLETED|'
    '0:0|00:13:48|22314444K\\n54506703.extern|COMPLETED|0:0|00:13:48|1524K\\n"\n'
    "    },\n"
    '    "adapted_teacher_sha256": "6928f2537dcca5f2d65c1498659e1ebf011845eb72ef3'
    '64b9544036c2238e9c7",\n'
    '    "bundle_sha256": "877454322d3d67c11b640e3c5aa171f2ba7fed8fb69349b575a374'
    '8019b106e1",\n'
    '    "hf_home": "/expanse/lustre/projects/nwu181/zgao12/OPD/cache/huggingface'
    '",\n'
    '    "job_id": "54506703",\n'
    '    "ok": true,\n'
    '    "preflight_job_id": null,\n'
    '    "prerequisites_path": "/home/zgao12/quest-runs/OPD/submissions/2b6229dd1'
    '19b489d8029724d90c6d3b8/prerequisites.json",\n'
    '    "prerequisites_sha256": "8f3b98aa9cad3a02e4524289f157b810b745879bb030955'
    '8d96e3cb85b638bb6",\n'
    '    "provenance_dir": "/home/zgao12/quest-runs/OPD/provenance/e2a32137229f17'
    '11aa32610d42690afedcd83afad018f0fda74586ed7821bc81",\n'
    '    "provenance_manifest_sha256": "e2a32137229f1711aa32610d42690afedcd83afad'
    '018f0fda74586ed7821bc81",\n'
    '    "queue": {\n'
    '      "returncode": 0,\n'
    '      "stderr": "",\n'
    '      "stdout": ""\n'
    "    },\n"
    '    "result": {\n'
    '      "result_sha256": "307156e33ba88c9a6d98939a4e7a44a661e15278ce4b69ba9cb6'
    '9e155a132bd7",\n'
    '      "verified": true\n'
    "    },\n"
    '    "run_id": "20260928T183316Z-b542b2e7e7fa-1cd4c8d9",\n'
    '    "science_git_head": "28c1026cece772a9e3d167d9cc64a64aaa0fd1b3",\n'
    '    "state": "COMPLETED",\n'
    '    "student_protocol_artifact_sha256": "2c15821442d3ae51da5d86917a5fbf160aa'
    'c9a1bb1b6ef5e8944dab8f8c69274",\n'
    '    "student_protocol_sha256": "9277960e4ff3599c325ac0115888280ad32647891fd3'
    '841d045822bf7db2a320",\n'
    '    "success": true,\n'
    '    "task": "qwen3-v2-adapted-preflight",\n'
    '    "teacher_acceptance_inventory_sha256": "8d53783b9fb1d386de5a0a291c2b28e2'
    '25347168cc7cfed4c0c0aceaf01d9bed",\n'
    '    "teacher_acceptance_sha256": "5d6952823441bde567cdf7f5fad8b4625c58ee7e82'
    '425aad76c10433d0ec5337",\n'
    '    "teacher_job_id": "54496291"\n'
    "  },\n"
    '  "status-monitor-2": {\n'
    '    "accounting": {\n'
    '      "returncode": 0,\n'
    '      "stderr": "",\n'
    '      "stdout": "54506703|RUNNING|0:0|00:08:43|\\n54506703.batch|RUNNING|0:0|'
    '00:08:43|\\n54506703.extern|RUNNING|0:0|00:08:43|\\n"\n'
    "    },\n"
    '    "adapted_teacher_sha256": "6928f2537dcca5f2d65c1498659e1ebf011845eb72ef3'
    '64b9544036c2238e9c7",\n'
    '    "bundle_sha256": "877454322d3d67c11b640e3c5aa171f2ba7fed8fb69349b575a374'
    '8019b106e1",\n'
    '    "hf_home": "/expanse/lustre/projects/nwu181/zgao12/OPD/cache/huggingface'
    '",\n'
    '    "job_id": "54506703",\n'
    '    "ok": true,\n'
    '    "preflight_job_id": null,\n'
    '    "prerequisites_path": "/home/zgao12/quest-runs/OPD/submissions/2b6229dd1'
    '19b489d8029724d90c6d3b8/prerequisites.json",\n'
    '    "prerequisites_sha256": "8f3b98aa9cad3a02e4524289f157b810b745879bb030955'
    '8d96e3cb85b638bb6",\n'
    '    "provenance_dir": "/home/zgao12/quest-runs/OPD/provenance/e2a32137229f17'
    '11aa32610d42690afedcd83afad018f0fda74586ed7821bc81",\n'
    '    "provenance_manifest_sha256": "e2a32137229f1711aa32610d42690afedcd83afad'
    '018f0fda74586ed7821bc81",\n'
    '    "queue": {\n'
    '      "returncode": 0,\n'
    '      "stderr": "",\n'
    '      "stdout": "54506703|RUNNING|exp-19-01"\n'
    "    },\n"
    '    "result": {\n'
    '      "reason": "Expected regular file",\n'
    '      "verified": false\n'
    "    },\n"
    '    "run_id": "20260928T183316Z-b542b2e7e7fa-1cd4c8d9",\n'
    '    "science_git_head": "28c1026cece772a9e3d167d9cc64a64aaa0fd1b3",\n'
    '    "state": "RUNNING",\n'
    '    "student_protocol_artifact_sha256": "2c15821442d3ae51da5d86917a5fbf160aa'
    'c9a1bb1b6ef5e8944dab8f8c69274",\n'
    '    "student_protocol_sha256": "9277960e4ff3599c325ac0115888280ad32647891fd3'
    '841d045822bf7db2a320",\n'
    '    "success": false,\n'
    '    "task": "qwen3-v2-adapted-preflight",\n'
    '    "teacher_acceptance_inventory_sha256": "8d53783b9fb1d386de5a0a291c2b28e2'
    '25347168cc7cfed4c0c0aceaf01d9bed",\n'
    '    "teacher_acceptance_sha256": "5d6952823441bde567cdf7f5fad8b4625c58ee7e82'
    '425aad76c10433d0ec5337",\n'
    '    "teacher_job_id": "54496291"\n'
    "  },\n"
    '  "status-monitor-3": {\n'
    '    "accounting": {\n'
    '      "returncode": 0,\n'
    '      "stderr": "",\n'
    '      "stdout": "54506703|COMPLETED|0:0|00:13:48|\\n54506703.batch|COMPLETED|'
    '0:0|00:13:48|22314444K\\n54506703.extern|COMPLETED|0:0|00:13:48|1524K\\n"\n'
    "    },\n"
    '    "adapted_teacher_sha256": "6928f2537dcca5f2d65c1498659e1ebf011845eb72ef3'
    '64b9544036c2238e9c7",\n'
    '    "bundle_sha256": "877454322d3d67c11b640e3c5aa171f2ba7fed8fb69349b575a374'
    '8019b106e1",\n'
    '    "hf_home": "/expanse/lustre/projects/nwu181/zgao12/OPD/cache/huggingface'
    '",\n'
    '    "job_id": "54506703",\n'
    '    "ok": true,\n'
    '    "preflight_job_id": null,\n'
    '    "prerequisites_path": "/home/zgao12/quest-runs/OPD/submissions/2b6229dd1'
    '19b489d8029724d90c6d3b8/prerequisites.json",\n'
    '    "prerequisites_sha256": "8f3b98aa9cad3a02e4524289f157b810b745879bb030955'
    '8d96e3cb85b638bb6",\n'
    '    "provenance_dir": "/home/zgao12/quest-runs/OPD/provenance/e2a32137229f17'
    '11aa32610d42690afedcd83afad018f0fda74586ed7821bc81",\n'
    '    "provenance_manifest_sha256": "e2a32137229f1711aa32610d42690afedcd83afad'
    '018f0fda74586ed7821bc81",\n'
    '    "queue": {\n'
    '      "returncode": 0,\n'
    '      "stderr": "",\n'
    '      "stdout": "54506703|COMPLETING|exp-19-01"\n'
    "    },\n"
    '    "result": {\n'
    '      "result_sha256": "307156e33ba88c9a6d98939a4e7a44a661e15278ce4b69ba9cb6'
    '9e155a132bd7",\n'
    '      "verified": true\n'
    "    },\n"
    '    "run_id": "20260928T183316Z-b542b2e7e7fa-1cd4c8d9",\n'
    '    "science_git_head": "28c1026cece772a9e3d167d9cc64a64aaa0fd1b3",\n'
    '    "state": "COMPLETED",\n'
    '    "student_protocol_artifact_sha256": "2c15821442d3ae51da5d86917a5fbf160aa'
    'c9a1bb1b6ef5e8944dab8f8c69274",\n'
    '    "student_protocol_sha256": "9277960e4ff3599c325ac0115888280ad32647891fd3'
    '841d045822bf7db2a320",\n'
    '    "success": false,\n'
    '    "task": "qwen3-v2-adapted-preflight",\n'
    '    "teacher_acceptance_inventory_sha256": "8d53783b9fb1d386de5a0a291c2b28e2'
    '25347168cc7cfed4c0c0aceaf01d9bed",\n'
    '    "teacher_acceptance_sha256": "5d6952823441bde567cdf7f5fad8b4625c58ee7e82'
    '425aad76c10433d0ec5337",\n'
    '    "teacher_job_id": "54496291"\n'
    "  }\n"
    "}"
)


class StudentSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in watch.CONTROL_FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("reviewed fixture control " + name)
        (self.root / ".sdsc/submissions").mkdir(parents=True)
        self.preflight_manifest = self.release(watch.PREFLIGHT_RUN)
        self.manifest = self.release("fresh-calibration")
        self.preflight = self.receipt(watch.PREFLIGHT, watch.PREFLIGHT_RUN, "1" * 32, False)
        self.save_receipt(self.preflight)
        self.calibration = self.receipt("54509999", "fresh-calibration", "2" * 32, True)
        self.artifact = self.root / ".sdsc/provenance/fixture"
        self.put(self.artifact / "manifest.json", {"git_head": watch.HEAD})
        self.put(self.artifact / "wrapper-manifest.json", self.manifest)
        self.created = self.root / ".sdsc/proofs/created.json"
        self.uploaded = self.root / ".sdsc/proofs/uploaded.json"
        self.runtime = self.root / ".sdsc/proofs/runtime.json"
        origin = dict(
            git_head=watch.HEAD,
            wrapper_code_sha256=self.manifest["code_sha256"],
            manifest_sha256="a" * 64,
            bundle_sha256="b" * 64,
        )
        self.put(self.created, dict(origin, artifact=str(self.artifact)))
        self.put(
            self.uploaded,
            dict(
                origin,
                verified=True,
                remote_artifact=watch.REMOTE + "/provenance/" + "a" * 64,
                run_id="fresh-calibration",
            ),
        )
        runtime = dict(
            verified=True,
            python=watch.PYTHON,
            version="3.12.13 fixture",
            sha256=watch.PYTHON_SHA,
            packages=watch.PACKAGES,
            gpu_imported=False,
        )
        self.put(self.runtime, {"returncode": 0, "stdout": json.dumps(runtime)})
        self.put(
            self.root / ".sdsc/check.json",
            dict(
                connected=True,
                target="zgao12@login.expanse.sdsc.edu",
                quest_root=str(self.root),
                control_python="/usr/bin/python3.11",
            ),
        )
        self.request = dict(
            schema=watch.REQUEST_SCHEMA,
            flow_id="fixture-flow",
            run_id="fresh-calibration",
            provenance_created=str(self.created),
            provenance_uploaded=str(self.uploaded),
            runtime_evidence=str(self.runtime),
        )
        preflight = watch.binding(watch.PREFLIGHT, self.root)
        paths = [
            self.root / ".sdsc/runs/fresh-calibration.json",
            self.root / preflight["submission"]["path"],
            self.root / preflight["deployment"]["path"],
            self.root / ".sdsc/check.json",
            self.created,
            self.uploaded,
            self.runtime,
            self.artifact / "manifest.json",
            self.artifact / "wrapper-manifest.json",
        ]
        self.plan = dict(
            schema=watch.SCHEMA,
            request=self.request,
            flow_id="fixture-flow",
            quest_root=str(self.root),
            quest_host=socket.gethostname().split(".")[0],
            created_at_unix=1000,
            expires_at_unix=1000 + watch.MAX_SECONDS,
            poll_seconds=watch.POLL,
            control_sha256=watch.controls(self.root),
            evidence=[watch.pin(p, self.root) for p in paths],
            preflight=preflight,
            runtime=runtime,
            scientific_bindings=watch.FIXED_BINDINGS,
            calibration={
                key: self.calibration[key]
                for key in (
                    "run_id",
                    "code_sha256",
                    "provenance_dir",
                    "provenance_manifest_sha256",
                    "bundle_sha256",
                )
            },
            resources=watch.RESOURCES,
            allowed_operations=["status", "fetch", "submit_once"],
            scope="one_canonical_sft_calibration_only_no_G0_pilot_factorial_or_cancellation",
        )
        self.directory = self.root / ".sdsc/supervision/fixture-flow"
        self.directory.mkdir(parents=True)
        self.plan_path = self.directory / "plan.json"
        self.plan_sha = self.put(self.plan_path, self.plan)
        self.time = 1000
        self.calls = []
        self.states = []
        self.statuses = {}
        self.failure = None

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = watch.canonical(value) + b"\n"
        path.write_bytes(raw)
        return watch.sha(raw)

    def release(self, run):
        files = [
            {
                "path": watch.PROTOCOL_PATH,
                "size": 100,
                "mode": 0o644,
                "sha256": watch.FIXED_BINDINGS["student_protocol_artifact_sha256"],
            },
            {"path": "run-identity.txt", "size": len(run), "mode": 0o644, "sha256": watch.sha(run.encode())},
        ]
        manifest = dict(
            run_id=run, git_head=watch.HEAD, files=files, code_sha256=watch.sha(watch.canonical(files))
        )
        self.put(
            self.root / ".sdsc/runs" / (run + ".json"),
            dict(
                state="deployed",
                manifest=manifest,
                deployment=dict(
                    ok=True,
                    run_id=run,
                    code_sha256=manifest["code_sha256"],
                    release=watch.REMOTE + "/releases/" + run,
                ),
            ),
        )
        return manifest

    def receipt(self, job, run, intent, calibration):
        manifest = self.manifest if calibration else self.preflight_manifest
        return dict(
            watch.FIXED_BINDINGS,
            ok=True,
            job_id=job,
            run_id=run,
            intent_id=intent,
            task=watch.TASK if calibration else watch.PREFLIGHT_TASK,
            code_sha256=manifest["code_sha256"],
            resources=watch.RESOURCES if calibration else {**watch.RESOURCES, "time": "01:00:00"},
            python=watch.PYTHON,
            hf_home=watch.HF_HOME,
            container=None,
            preflight_job_id=watch.PREFLIGHT if calibration else None,
            provenance_manifest_sha256="a" * 64,
            provenance_dir=watch.REMOTE + "/provenance/" + "a" * 64,
            bundle_sha256="b" * 64,
            prerequisites_path=watch.REMOTE + "/submissions/" + intent + "/prerequisites.json",
            prerequisites_sha256="c" * 64,
            result_dir=watch.RESULT_ROOT + "/" + run + "/" + intent,
        )

    def save_receipt(self, receipt):
        self.put(
            self.root / ".sdsc/submissions" / (receipt["intent_id"] + ".json"),
            dict(state="submitted", receipt=receipt, request=receipt),
        )

    def report(self, receipt):
        return {"fixture": True, "job_id": receipt["job_id"]}

    def status(self, receipt, state="COMPLETED", verified=True):
        job = receipt["job_id"]
        terminal = state == "COMPLETED"
        raw = watch.canonical(self.report(receipt)) + b"\n"
        return dict(
            {key: receipt.get(key) for key in watch.STATUS_KEYS},
            ok=True,
            state=state,
            success=terminal and verified,
            result=dict(
                verified=verified if terminal else False,
                result_sha256=watch.sha(raw),
                reason="Expected regular file" if not terminal else "",
            ),
            queue=dict(returncode=0, stdout="" if terminal else job + "|RUNNING|None\n"),
            accounting=dict(
                returncode=0,
                stdout="\n".join(
                    name + "|" + state + "|0:0" for name in (job, job + ".batch", job + ".extern")
                ),
            ),
        )

    def call(self, args):
        self.calls.append(args)
        if self.failure:
            self.failure(args)
        if args[0] == "submit":
            self.assertTrue(self.states[-1]["submission_attempted"])
            self.assertEqual(args, watch.submit_arguments(self.plan))
            self.save_receipt(self.calibration)
            return self.calibration
        receipt = self.preflight if args[1] == watch.PREFLIGHT else self.calibration
        if args[0] == "status":
            values = self.statuses.get(args[1], [])
            return values.pop(0) if values else self.status(receipt)
        if args[0] == "fetch":
            destination = self.root / ".sdsc/fetched" / receipt["job_id"] / "fetch-fixture"
            name = "adapted-preflight.json" if receipt == self.preflight else "adapted-calibration.json"
            self.put(destination / name, self.report(receipt))
            return dict(
                job_id=receipt["job_id"],
                intent_id=receipt["intent_id"],
                destination=str(destination),
                files=[name],
                bytes=len(watch.canonical(self.report(receipt))) + 1,
            )
        self.fail("unexpected operation")

    def sleep(self, duration):
        self.assertGreater(duration, 0)
        self.assertLessEqual(duration, 300)
        self.time += duration

    def flow(self):
        return watch.run_flow(
            self.plan,
            self.states.append,
            root=self.root,
            call=self.call,
            sleep=self.sleep,
            clock=lambda: self.time,
        )

    def test_preflight_already_complete_submits_exactly_once_and_finishes(self):
        self.assertEqual(self.flow(), 0)
        self.assertEqual([row[0] for row in self.calls], ["status", "fetch", "submit", "status", "fetch"])
        self.assertEqual(self.states[-1]["phase"], "calibration_complete")
        self.assertFalse(self.states[-1]["g0_passed"])

    def test_running_unpublished_result_waits_five_minutes_then_advances(self):
        self.statuses[watch.PREFLIGHT] = [self.status(self.preflight, "RUNNING")]
        self.assertEqual(self.flow(), 0)
        self.assertEqual(self.time, 1300)
        self.assertEqual(sum(args[0] == "submit" for args in self.calls), 1)

    def test_actual_running_transition_and_completed_statuses(self):
        receipt = {key: REAL_STATUSES["final-status"].get(key) for key in watch.STATUS_KEYS}
        for name, expected in (
            ("status-monitor-2", "active"),
            ("status-monitor-3", "transition"),
            ("final-status", "success"),
        ):
            self.assertEqual(watch.status_outcome(REAL_STATUSES[name], receipt), expected)

    def actual_transition(self):
        value = copy.deepcopy(REAL_STATUSES["status-monitor-3"])
        # Substitute only fixture deployment identities; queue/accounting/result
        # are the actual 54506703 non-atomic terminal observation.
        value.update({key: self.preflight.get(key) for key in watch.STATUS_KEYS})
        return value

    def test_actual_completing_race_waits_then_submits_after_full_success(self):
        self.statuses[watch.PREFLIGHT] = [self.actual_transition()]
        self.assertEqual(self.flow(), 0)
        self.assertEqual(self.time, 1300)
        self.assertEqual([row[0] for row in self.calls[:3]], ["status", "status", "fetch"])
        self.assertEqual(sum(row[0] == "submit" for row in self.calls), 1)

    def test_permanent_actual_completing_race_stops_after_ten_minutes(self):
        self.statuses[watch.PREFLIGHT] = [self.actual_transition() for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "did not clear"):
            self.flow()
        self.assertEqual(self.time, 1600)
        self.assertEqual([row[0] for row in self.calls], ["status"] * 3)
        self.assertFalse(self.states[-1]["submission_attempted"])

    def test_transition_never_hides_failed_or_unverified_completion(self):
        for edit in ("result", "accounting", "queue"):
            value = self.actual_transition()
            if edit == "result":
                value["result"]["verified"] = False
            elif edit == "accounting":
                value["accounting"]["stdout"] = value["accounting"]["stdout"].replace(
                    ".batch|COMPLETED|0:0", ".batch|FAILED|1:0"
                )
            else:
                value["queue"]["stdout"] = "54504895|COMPLETING|exp-19-01"
            self.assertEqual(watch.status_outcome(value, self.preflight), "failure")

    def test_build_plan_accepts_exact_replay_layout_with_local_git_verifier_fixture(self):
        # Git bundle verification itself is covered by test_release_replay and
        # test_provenance; this fixture exercises the actual prepare boundary.
        artifact = self.root / ".sdsc/replays" / ("replay-" + "d" * 32) / "stage/provenance"
        self.put(artifact / "manifest.json", dict(git_head=watch.HEAD, bundle={"sha256": "b" * 64}))
        self.put(artifact / "wrapper-manifest.json", self.manifest)
        origin = watch.document(self.created)
        origin["artifact"] = str(artifact)
        self.put(self.created, origin)
        (self.root / "tools/sdsc_provenance.py").write_text(
            "import json\n"
            "def load_artifact(path, manifest_sha256, code_sha256):\n"
            "    assert manifest_sha256 == '" + "a" * 64 + "'\n"
            "    wrapper = json.loads((path/'wrapper-manifest.json').read_text())\n"
            "    assert wrapper['code_sha256'] == code_sha256\n"
            "    return json.loads((path/'manifest.json').read_text()), wrapper, None\n"
            "def verify(path, manifest_sha256, code_sha256):\n"
            "    return dict(verified=True, git_head='" + watch.HEAD + "')\n"
        )
        plan = watch.build_plan(self.request, self.root, now=1000)
        self.assertEqual(watch.validate_plan(plan, self.root), plan)
        self.assertEqual(plan["calibration"], self.plan["calibration"])
        self.assertTrue(any("/stage/provenance/manifest.json" in row["path"] for row in plan["evidence"]))
        for bad in (
            self.root / ".sdsc/replays/replay-unknown/stage/provenance",
            artifact / "nested",
            artifact.parent / "wrong",
        ):
            with self.assertRaises(ValueError):
                watch.local_artifact(bad, self.root)
        linked = self.root / ".sdsc/provenance/linked"
        linked.symlink_to(artifact, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            watch.local_artifact(linked, self.root)

    def test_failed_or_unverified_terminal_preflight_fetches_and_stops(self):
        for state in ("FAILED", "COMPLETED"):
            with self.subTest(state=state):
                self.calls.clear()
                self.statuses[watch.PREFLIGHT] = [self.status(self.preflight, state, False)]
                with self.assertRaisesRegex(ValueError, "acceptance"):
                    self.flow()
                self.assertEqual([args[0] for args in self.calls], ["status", "fetch"])
                self.assertFalse(self.states[-1]["submission_attempted"])

    def test_wrong_status_id_and_old_preflight_never_submit(self):
        for key, value in (
            ("job_id", "54504895"),
            ("student_protocol_sha256", "e" * 64),
            ("science_git_head", "1" * 40),
            ("provenance_manifest_sha256", "e" * 64),
        ):
            with self.subTest(key=key):
                wrong = self.status(self.preflight)
                wrong[key] = value
                with self.assertRaisesRegex(ValueError, "identity"):
                    watch.status_outcome(wrong, self.preflight)

    def test_success_requires_main_batch_extern_and_every_step_zero(self):
        job = watch.PREFLIGHT
        cases = [
            "",
            f"{job}|COMPLETED|0:0",
            f"{job}|COMPLETED|0:0\n{job}.batch|COMPLETED|0:0",
            f"{job}|COMPLETED|0:0\n{job}.batch|FAILED|1:0\n{job}.extern|COMPLETED|0:0",
            f"{job}|COMPLETED|0:0\n{job}|COMPLETED|0:0",
        ]
        for accounting in cases:
            with self.subTest(accounting=accounting):
                wrong = self.status(self.preflight)
                wrong["accounting"]["stdout"] = accounting
                with self.assertRaises(ValueError):
                    watch.status_outcome(wrong, self.preflight)

    def test_ssh_loss_and_unknown_status_stop_without_retry(self):
        self.failure = lambda args: (_ for _ in ()).throw(RuntimeError("SSH master unavailable"))
        with self.assertRaisesRegex(RuntimeError, "SSH"):
            self.flow()
        self.assertEqual(len(self.calls), 1)
        self.failure = None
        self.statuses[watch.PREFLIGHT] = [self.status(self.preflight, "UNKNOWN")]
        with self.assertRaisesRegex(ValueError, "unknown scheduler"):
            self.flow()

    def test_lost_submission_receipt_retains_attempted_and_never_retries(self):
        def fail(args):
            if args[0] == "submit":
                raise TimeoutError("receipt lost")

        self.failure = fail
        with self.assertRaises(TimeoutError):
            self.flow()
        self.assertTrue(self.states[-1]["submission_attempted"])
        self.assertTrue(self.states[-1]["submission_outcome_unknown"])
        self.assertEqual(sum(args[0] == "submit" for args in self.calls), 1)

    def test_receipt_scope_resource_and_provenance_mismatch_rejected(self):
        for key, value in (
            ("preflight_job_id", "54504895"),
            ("python", "/wrong"),
            ("resources", {**watch.RESOURCES, "gpus": 4}),
            ("run_id", "wrong"),
            ("bundle_sha256", "0" * 64),
        ):
            with self.subTest(key=key):
                receipt = copy.deepcopy(self.calibration)
                receipt[key] = value
                with self.assertRaises(ValueError):
                    watch.validate_receipt(receipt, calibration=True, plan=self.plan)

    def test_control_or_small_evidence_mutation_stops_plan(self):
        watch.validate_plan(self.plan, self.root)
        control = self.root / watch.CONTROL_FILES[0]
        original = control.read_bytes()
        control.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "control"):
            watch.validate_plan(self.plan, self.root)
        control.write_bytes(original)
        self.created.write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence"):
            watch.validate_plan(self.plan, self.root)

    def test_deadline_never_starts_another_operation(self):
        self.time = self.plan["expires_at_unix"]
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.flow()
        self.assertEqual(self.calls, [])

    def test_existing_unknown_or_same_preflight_intent_prevents_new_submit(self):
        self.put(
            self.root / ".sdsc/submissions" / ("f" * 32 + ".json"),
            {"state": "unknown", "request": {"run_id": "other"}},
        )
        with self.assertRaisesRegex(ValueError, "unknown submission"):
            watch.no_competitor(self.root)
        (self.root / ".sdsc/submissions" / ("f" * 32 + ".json")).unlink()
        self.save_receipt(self.calibration)
        with self.assertRaisesRegex(ValueError, "already has"):
            watch.no_competitor(self.root)

    def test_global_claim_blocks_another_flow_even_after_original_stops(self):
        self.statuses[watch.PREFLIGHT] = [self.status(self.preflight, "FAILED", False)]
        with self.assertRaises(ValueError):
            watch.run(
                self.plan_path,
                self.plan_sha,
                root=self.root,
                call=self.call,
                clock=lambda: self.time,
                sleep=self.sleep,
            )
        original = (self.directory / "state.json").read_bytes()
        second = copy.deepcopy(self.plan)
        second["flow_id"] = second["request"]["flow_id"] = "second-flow"
        path = self.root / ".sdsc/supervision/second-flow/plan.json"
        digest = self.put(path, second)
        with self.assertRaises(FileExistsError):
            watch.run(path, digest, root=self.root, call=self.call, clock=lambda: self.time)
        self.assertEqual((self.directory / "state.json").read_bytes(), original)
        self.assertFalse((path.parent / "state.json").exists())

    def test_existing_state_and_global_lock_prevent_rearming(self):
        state = self.directory / "state.json"
        self.put(state, {"phase": "stopped"})
        with self.assertRaisesRegex(ValueError, "already started"):
            watch.run(self.plan_path, self.plan_sha, root=self.root, call=self.call, clock=lambda: self.time)
        state.unlink()
        with (self.root / ".sdsc/student-supervision.lock").open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                watch.run(
                    self.plan_path, self.plan_sha, root=self.root, call=self.call, clock=lambda: self.time
                )
        self.assertEqual(self.calls, [])

    def test_fetched_report_must_match_semantically_verified_status(self):
        status = self.status(self.preflight)
        result = self.call(["fetch", watch.PREFLIGHT])
        report = Path(result["destination"]) / "adapted-preflight.json"
        report.write_bytes(b" " * report.stat().st_size)
        with self.assertRaisesRegex(ValueError, "fetched report"):
            watch.validate_fetch(result, self.preflight, status, self.root)

    def test_fetch_must_report_actual_bounded_bytes(self):
        status = self.status(self.preflight)
        result = self.call(["fetch", watch.PREFLIGHT])
        result["bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "byte count"):
            watch.validate_fetch(result, self.preflight, status, self.root)

    def test_command_cannot_cancel_sync_or_use_arbitrary_argv(self):
        for args in (["cancel", watch.PREFLIGHT], ["sync"], ["submit", "wrong"]):
            with patch.object(watch.subprocess, "run") as run, self.assertRaises(ValueError):
                watch.command(args, self.plan, self.root)
            run.assert_not_called()

    def test_successful_calibration_still_fetches_bounded_results_only(self):
        watch.validate_plan(self.plan, self.root)
        self.assertEqual(self.flow(), 0)
        self.assertFalse(any("cancel" in args or "sync" in args for args in self.calls))
        self.assertEqual(self.states[-1]["last_fetch"]["files"], ["adapted-calibration.json"])


if __name__ == "__main__":
    unittest.main()
