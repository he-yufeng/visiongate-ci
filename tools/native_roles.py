"""Execute frozen role controls, commit fresh paired predictions, then score."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expired(signum,frame):
    raise TimeoutError("900s owned role pilot cap")


def final_correct(value, truth, old=False):
    state=value["state"]
    if value["status"]!="DECIDED" or state["deploy_gate"]!="blocked":
        return False
    if old:
        names=[t["tool"] for t in value["trace"]]
        final=("set_deploy_gate_blocked" if "set_deploy_gate_blocked" in names else
               "inspect_region" if "inspect_region" in names else
               "rerun_visual_test" if state.get("release_recommended") else "request_human_approval")
    else:
        final=value["final_action"]
    if final!=truth["expected_final"]:
        return False
    if truth["stratum"]=="clean_or_nuisance":
        return bool(state.get("release_recommended"))
    if truth["stratum"]=="critical_regression":
        return bool(state.get("critical_evidence")) and not state.get("release_recommended",False)
    return bool(state.get("human_review_requested")) and not state.get("release_recommended",False)


def main():
    out=ROOT/"_role_r02"; out.mkdir(mode=0o700)
    signal.signal(signal.SIGALRM,expired); signal.alarm(900)
    started=time.monotonic(); code=1
    result={"scope":"FRESH_PAIRED_ROLE_CAPTURE_SYNTHETIC_NOT_EXTERNAL_BLIND_OR_CONTEST_READY",
            "status":"RUNNING","run_id":os.environ.get("GITHUB_RUN_ID"),"source_commit":os.environ.get("GITHUB_SHA"),
            "same_author_known_generator_family":True,"crypto_host_access_isolation_claimed":False,
            "owner_declared_roles_not_external_attestation":True,"equal_input_individual_mechanism_causal_ablation":False,
            "aws_actions":0,"new_spend_cny":0,"model_api_calls":0,"production_deploy_actions":0,
            "registration_or_submission_actions":0,"competition_ready":False,"official_score":None,"award":None}
    manifest=json.loads((ROOT/"ROLE_SOURCE_MANIFEST.json").read_text())
    try:
        assert sys.version_info[:2]==(3,12) and platform.system()=="Linux" and platform.machine()=="x86_64"
        event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        assert not event["repository"]["private"] and os.environ["GITHUB_REPOSITORY"]=="he-yufeng/visiongate-ci"
        assert all(sha(ROOT/n)==v for n,v in manifest["inputs"].items())
        assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=10*1024**3
        import cv2
        import numpy as np
        from role_controls import run_controls
        from role_stress_generator import generate
        from visiongate_ci.canonical import write_canonical
        from visiongate_ci.gate import _pair_f1, percentile_nearest_rank
        assert cv2.__version__=="5.0.0" and metadata.version("opencv-python-headless")=="5.0.0.93" and np.__version__=="2.2.6"
        result["runtime"]={"opencv":cv2.__version__,"opencv_distribution":metadata.version("opencv-python-headless"),
                           "numpy":np.__version__,"python":platform.python_version(),"platform":"Linux x86_64"}
        cv2.ocl.setUseOpenCL(False)
        contract=json.loads((ROOT/"contracts/local_baseline_contract.json").read_text())
        policy=json.loads((ROOT/"contracts/role_policy_r02.json").read_text())
        controls=run_controls(out/"controls",contract,policy)
        result["new_native_controls"]=controls
        assert controls["passed"]==controls["total"]==8,"new role/capture controls failed"
        seed=secrets.randbits(64)
        produced=generate(out/"inputs",out/"private_labels",ROOT/"contracts/local_baseline_contract.json",seed)
        result["generation_commitment"]=produced
        roles=json.loads((out/"inputs/baseline_roles.json").read_text())["cases"]
        assert all(set(r["protected_components"][i]["role"] for i in range(3))==set(policy["required_roles"]) for r in roles)
        result["all300baselines_have_complete_same_role_set"]=True
        step=time.monotonic()
        completed=subprocess.run([sys.executable,"-B",str(ROOT/"tools/predict_roles.py"),
            "--inputs",str(out/"inputs"),"--outputs",str(out/"predictions")],cwd=ROOT,
            capture_output=True,text=True,timeout=min(600,900-(time.monotonic()-started)))
        result["prediction_process"]={"exit_code":completed.returncode,"wall_seconds":time.monotonic()-step}
        if completed.returncode:
            result["prediction_process"]["error_tail"]=completed.stderr[-4000:]
            raise RuntimeError("paired prediction child failed")
        names=("role_evidence1.json","role_evidence2.json","role_latencies1.json","role_latencies2.json",
               "old_evidence.json","old_traces.json","old_latencies.json")
        hashes={n:sha(out/"predictions"/n) for n in names}
        write_canonical(out/"prediction_commitment.json",{"files":hashes})
        result["prediction_commitment_sha256"]=sha(out/"prediction_commitment.json")
        result["prediction_files_committed_before_label_open"]=True
        assert sha(out/"private_labels/labels.json")==produced["label_file_sha256"]
        labels=json.loads((out/"private_labels/labels.json").read_text())
        result["generation_seed_exact_decimal_disclosed_after_prediction"]=str(labels["generation_seed"])
        truth={r["case_id"]:r for r in labels["labels"]}
        load=lambda n:json.loads((out/"predictions"/n).read_text())
        new={r["case_id"]:r for r in load("role_evidence1.json")["cases"]}
        repeated={r["case_id"]:r for r in load("role_evidence2.json")["cases"]}
        old={r["case_id"]:r for r in load("old_traces.json")["cases"]}
        old_evidence={r["case_id"]:r for r in load("old_evidence.json")["cases"]}
        assert len(truth)==len(new)==len(old)==len(repeated)==len(old_evidence)==300
        assert set(truth)==set(new)==set(old)==set(repeated)==set(old_evidence)
        rows=[]
        threshold=contract["perception"]["false_positive_pixel_fraction"]
        for case_id in sorted(truth):
            t,a,b,p=truth[case_id],new[case_id],old[case_id],old_evidence[case_id]
            positive=bool(t["boxes"])
            # Every positive gets a number, including zero for UNKNOWN/missing.
            f1=_pair_f1(a["effective_regions"],t["boxes"],.5) if positive and a["status"]=="DECIDED" else 0.0 if positive else None
            d=p.get("diagnostic")
            old_f1=_pair_f1(d["predicted_regions_xyxy_half_open"],t["boxes"],.5) if positive and d is not None else 0.0 if positive else None
            nuisance=t["stratum"]=="clean_or_nuisance"
            rows.append({"case_id":case_id,"family":t["family"],"stratum":t["stratum"],"kind":t["kind"],
                "expected_first":t["expected_first"],"expected_final":t["expected_final"],
                "observed_first":a["first_selected_action"],"observed_final":a["final_action"],
                "action_correct":a["status"]=="DECIDED" and a["first_selected_action"]==t["expected_first"],
                "final_correct":final_correct(a,t),"localization_f1":f1,"status":a["status"],"steps":len(a["trace"]),
                "release_recommended":a["state"].get("release_recommended",False),
                "unsafe_recommendation":t["stratum"]=="critical_regression" and bool(a["state"].get("release_recommended")),
                "production_release":a["state"]["deploy_gate"]!="blocked",
                "nuisance_before":int(bool(d and d["raw_false_positive"])) if nuisance else 0,
                "nuisance_after":int(a["effective_raw_fraction"] is None or a["effective_raw_fraction"]>threshold) if nuisance else 0,
                "old_observed_first":b["first_selected_action"],
                "old_action_correct":b["status"]=="DECIDED" and b["first_selected_action"]==t["expected_first"],
                "old_final_correct":final_correct(b,t,old=True),"old_localization_f1":old_f1,
                "old_unsafe_recommendation":t["stratum"]=="critical_regression" and bool(b["state"].get("release_recommended")),
                "old_unknown":b["status"]=="UNKNOWN"})
        positive=[r["localization_f1"] for r in rows if r["localization_f1"] is not None]
        assert len(positive)==200
        before=sum(r["nuisance_before"] for r in rows); after=sum(r["nuisance_after"] for r in rows)
        reduction=(before-after)/before if before else None
        latency1=load("role_latencies1.json")["latencies"]; latency2=load("role_latencies2.json")["latencies"]
        assert len(latency1)==len(latency2)==300
        p95=percentile_nearest_rank(latency1,.95)
        output_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        accuracy=sum(r["action_correct"] for r in rows)/300
        final_accuracy=sum(r["final_correct"] for r in rows)/300
        f1=math.fsum(positive)/200
        recommendations=sum(r["unsafe_recommendation"] for r in rows)
        releases=sum(r["production_release"] for r in rows)
        unknown=sum(r["status"]=="UNKNOWN" for r in rows)
        gates={"all300":True,"action_accuracy":accuracy>=.9,"final_action_and_state_accuracy":final_accuracy>=.9,
               "localization_f1":f1>=.85,"nuisance_reduction":reduction is not None and reduction>=.3,
               "p95_and_size":p95<=2 and output_bytes<=100*1024**2,
               "repeat_hash":hashes["role_evidence1.json"]==hashes["role_evidence2.json"],
               "max4steps":all(r["steps"]<=4 for r in rows),"zero_production_release":releases==0,
               "zero_critical_unsafe_recommendation":recommendations==0,"zero_unknown":unknown==0}
        breakdown={}
        for field in ("family","stratum","kind"):
            breakdown[field]={}
            for value in sorted({r[field] for r in rows}):
                part=[r for r in rows if r[field]==value]
                breakdown[field][value]={"count":len(part),"action_correct":sum(r["action_correct"] for r in part),
                    "final_correct":sum(r["final_correct"] for r in part),"unsafe_recommendations":sum(r["unsafe_recommendation"] for r in part),
                    "unknown":sum(r["status"]=="UNKNOWN" for r in part),"old_action_correct":sum(r["old_action_correct"] for r in part),
                    "old_final_correct":sum(r["old_final_correct"] for r in part),
                    "old_unsafe_recommendations":sum(r["old_unsafe_recommendation"] for r in part)}
        tool_counts={}
        for a in new.values():
            for step in a["trace"]:
                tool_counts[step["tool"]]=tool_counts.get(step["tool"],0)+1
        old_latencies=load("old_latencies.json")["latencies"]
        assert len(old_latencies)==300
        result.update(status="ROLE_STRESS_PASS" if all(gates.values()) else "ROLE_STRESS_FAIL",cases=300,
            positive_localization_cases=200,action_correct=sum(r["action_correct"] for r in rows),action_accuracy=accuracy,
            final_correct=sum(r["final_correct"] for r in rows),final_action_and_state_accuracy=final_accuracy,
            localization_macro_f1=f1,nuisance_before=before,nuisance_after=after,nuisance_reduction=reduction,
            p95_seconds=p95,output_bytes=output_bytes,critical_unsafe_recommendations=recommendations,
            production_releases=releases,unknown_cases=unknown,gates=gates,breakdown=breakdown,per_case=rows,
            actual_tool_counts=tool_counts,prediction_file_sha256=hashes,
            old_same_image_baseline={"action_correct":sum(r["old_action_correct"] for r in rows),
                "action_accuracy":sum(r["old_action_correct"] for r in rows)/300,
                "final_correct":sum(r["old_final_correct"] for r in rows),
                "final_action_and_state_accuracy":sum(r["old_final_correct"] for r in rows)/300,
                "critical_unsafe_recommendations":sum(r["old_unsafe_recommendation"] for r in rows),
                "unknown_cases":sum(r["old_unknown"] for r in rows),
                "initial_image_localization_macro_f1":math.fsum(r["old_localization_f1"] for r in rows if r["old_localization_f1"] is not None)/200,
                "p95_seconds":percentile_nearest_rank(old_latencies,.95)})
        code=0 # Measured quality failure is distinct from execution failure.
    except Exception as error:
        result.update(status="EXECUTION_FAIL",error_class=type(error).__name__,error=str(error))
    finally:
        signal.alarm(0)
    result["source_inputs"]=manifest["inputs"]
    result["source_unchanged"]=all(sha(ROOT/n)==v for n,v in manifest["inputs"].items())
    if not result["source_unchanged"]: result["status"],code="EXECUTION_FAIL",1
    result.update(wall_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
    (out/"receipt.json").write_text(json.dumps(result,sort_keys=True,indent=2))
    print("FINAL_ROLE_RECEIPT_JSON "+json.dumps(result,sort_keys=True),flush=True)
    return code


if __name__=="__main__": raise SystemExit(main())
