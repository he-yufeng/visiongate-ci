"""One real OpenCV5 runtime/design probe; no AWS, registration or official score."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expired(signum, frame):
    raise TimeoutError("600s owned probe cap")


def main():
    out = ROOT / "_cv5_probe"
    out.mkdir(mode=0o700)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    started = time.monotonic()
    result = {"scope": "REAL_OPENCV5_LOCAL_DESIGN_AND_EXECUTABLE_SANDBOX_TOOL_PROBE_NOT_COMPETITION_RESULT",
              "status": "RUNNING", "source_commit": os.environ.get("GITHUB_SHA"),
              "run_id": os.environ.get("GITHUB_RUN_ID"), "new_spend_cny": 0,
              "aws_actions": 0, "registration_or_submission_actions": 0,
              "model_api_calls": 0, "production_deployment_actions": 0,
              "images_exported_or_committed": False, "sealed_holdout": False,
              "original_private_artifacts_modified": False}
    manifest = json.loads((ROOT / "CV5_SOURCE_MANIFEST.json").read_text())
    code = 1
    try:
        if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("required native runtime")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if event["repository"]["private"] or os.environ["GITHUB_REPOSITORY"] != "he-yufeng/visiongate-ci":
            raise ValueError("public standard resource contract")
        if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 10 * 1024**3:
            raise RuntimeError("10GiB resource floor")
        for name, expected in manifest["inputs"].items():
            path = ROOT / name
            if path.is_symlink() or sha(path) != expected:
                raise ValueError("source manifest changed")
        import cv2
        import numpy as np
        from visiongate_ci import agent_loop
        from visiongate_ci.canonical import canonical_json_bytes, write_canonical
        from visiongate_ci.fixtures import generate
        from visiongate_ci.gate import run_gate
        contract_path = ROOT / "contracts/local_baseline_contract.json"
        contract = json.loads(contract_path.read_text())
        if metadata.version("opencv-python-headless") != "5.0.0.93" or cv2.__version__ != "5.0.0" or np.__version__ != "2.2.6":
            raise ValueError("pinned wheel or native version mismatch")
        build = cv2.getBuildInformation()
        if "OpenCV 5.0.0" not in build:
            raise ValueError("native build information mismatch")
        binaries = list(Path(cv2.__file__).parent.glob("*.so"))
        if not binaries:
            raise ValueError("native binary absent")
        result["runtime"] = {"python": platform.python_version(), "system": platform.system(),
              "machine": platform.machine(), "opencv_distribution": metadata.version("opencv-python-headless"),
              "opencv_native_version": cv2.__version__, "numpy": np.__version__,
              "native_binary_sha256": {p.name: sha(p) for p in binaries},
              "build_information_sha256": hashlib.sha256(build.encode()).hexdigest()}
        cv2.ocl.setUseOpenCL(False)
        cv2.setRNGSeed(contract["determinism"]["seed"])
        fixtures = out / "fixtures"
        inputs = generate(fixtures, contract_path)
        if len(inputs["cases"]) != 60 or inputs["rights"]["external_image_asset_count"] != 0:
            raise ValueError("fixture rights/count contract")
        result["input_manifest_sha256"] = sha(fixtures / "input_manifest.json")
        result["fresh_generated_inputs_not_old_opencv4_byte_comparison"] = True
        raw_gate = run_gate(fixtures, contract_path, ROOT / "evaluation_contract.json",
                            out / "evidence", out / "local_gate_receipt.json")
        result["raw_legacy_gate"] = raw_gate
        result["legacy_gate_always_blocked_is_not_official_readiness"] = True
        if raw_gate["performance_gate_summary"]["passed"] < 5 or any(
                h["status"] != "PASS" for h in raw_gate["hard_gates"].values()):
            raise ValueError("unchanged local design gate failed")
        traces = [agent_loop.run_case(fixtures, case, contract) for case in inputs["cases"]]
        # Scoring labels are read only after every image-driven tool trace exists.
        truth = json.loads((fixtures / "ground_truth.json").read_text())
        expected = {case["case_id"]: case["expected_action"] for case in truth["cases"]}
        correct = sum(t["first_selected_action"] == expected[t["case_id"]] for t in traces)
        calls = {}
        for trace in traces:
            for step in trace["trace"]:
                calls[step["tool"]] = calls.get(step["tool"], 0) + 1
        if correct / 60 < .9 or any(len(t["trace"]) > 4 or t["state"]["deploy_gate"] != "blocked" for t in traces):
            raise ValueError("conditional action/sandbox safety gate")
        for action in ("rerun_visual_test", "inspect_region", "set_deploy_gate_blocked"):
            if calls.get(action, 0) == 0:
                raise ValueError("missing real conditional tool branch")
        if calls["rerun_visual_test"] == 60 or calls["inspect_region"] == 60:
            raise ValueError("unconditional tool execution")
        write_canonical(out / "agent_traces.json", {"traces": traces})
        result["agent"] = {"cases": 60, "action_correct": correct, "action_accuracy": correct / 60,
                           "calls": calls, "trace_sha256": sha(out / "agent_traces.json"),
                           "recapture_source": "pre_captured_synthetic_images_not_live_browser",
                           "sandbox_gate_only_not_real_deployment": True}
        case = inputs["cases"][0]
        with patch.object(agent_loop, "runtime_evidence", return_value={"competition_opencv_major_satisfied": False}):
            rejected = agent_loop.run_case(fixtures, case, contract)
            assert rejected["status"] == "UNKNOWN" and rejected["state"]["deploy_gate"] == "blocked"
        bad = json.loads(json.dumps(case)); bad["sha256"]["candidate"] = "0" * 64
        rejected = agent_loop.run_case(fixtures, bad, contract)
        assert rejected["status"] == "UNKNOWN" and rejected["state"]["deploy_gate"] == "blocked"
        bad = json.loads(json.dumps(case)); bad["paths"]["reference"] = "../outside.png"
        rejected = agent_loop.run_case(fixtures, bad, contract)
        assert rejected["status"] == "UNKNOWN" and rejected["state"]["deploy_gate"] == "blocked"
        assert list(inspect.signature(agent_loop.run_case).parameters) == ["root", "case", "contract"]
        result["new_boundary_controls_passed"] = 4
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100 * 1024**2:
            raise RuntimeError("100MiB output gate")
        result.update(status="LOCAL_DESIGN_RUNTIME_PASS", competition_ready=False,
                      aws_deployed=False, official_score=None, award=None)
        code = 0
    except Exception as error:
        result.update(status="FAIL", error_class=type(error).__name__, error=str(error))
    finally:
        signal.alarm(0)
    result["source_inputs"] = manifest["inputs"]
    result["frozen_source_unchanged"] = all(sha(ROOT / n) == v for n, v in manifest["inputs"].items())
    if not result["frozen_source_unchanged"]:
        result["status"], code = "FAIL", 1
    result.update(wall_seconds=time.monotonic()-started, finished_at_utc=datetime.now(timezone.utc).isoformat())
    (out / "receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    print("FINAL_CV5_RECEIPT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
