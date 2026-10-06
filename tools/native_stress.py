"""Generate sealed-label stress inputs, commit predictions, then score once."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expired(signum, frame):
    raise TimeoutError("900s owned stress cap")


def main():
    out = ROOT / "_stress_r01"
    out.mkdir(mode=0o700)
    signal.signal(signal.SIGALRM, expired); signal.alarm(900)
    started = time.monotonic()
    result = {"scope":"PROCEDURALLY_LABEL_ISOLATED_SYNTHETIC_STRESS_NOT_EXTERNAL_BLIND_DATA",
        "status":"RUNNING","run_id":os.environ.get("GITHUB_RUN_ID"),
        "source_commit":os.environ.get("GITHUB_SHA"),"same_author_generator_family_known":True,
        "crypto_host_access_isolation_claimed":False,"aws_actions":0,"new_spend_cny":0,
        "registration_or_submission_actions":0,"model_api_calls":0,"production_deploy_actions":0,
        "official_score":None,"award":None,"competition_ready":False}
    manifest = json.loads((ROOT / "STRESS_SOURCE_MANIFEST.json").read_text())
    code = 1
    try:
        assert sys.version_info[:2] == (3,12) and platform.system()=="Linux" and platform.machine()=="x86_64"
        event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        assert not event["repository"]["private"] and os.environ["GITHUB_REPOSITORY"]=="he-yufeng/visiongate-ci"
        assert all(sha(ROOT / n)==v for n,v in manifest["inputs"].items())
        assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=10*1024**3
        import cv2
        import numpy as np
        from stress_generator import generate
        from visiongate_ci.canonical import write_canonical
        from visiongate_ci.gate import _pair_f1, percentile_nearest_rank
        assert cv2.__version__=="5.0.0" and metadata.version("opencv-python-headless")=="5.0.0.93" and np.__version__=="2.2.6"
        result["runtime"]={"opencv":cv2.__version__,"opencv_distribution":metadata.version("opencv-python-headless"),
            "numpy":np.__version__,"python":platform.python_version(),"platform":"Linux x86_64"}
        seed=secrets.randbits(64)
        produced=generate(out/"inputs",out/"private_labels",ROOT/"contracts/local_baseline_contract.json",seed)
        result["generation_commitment"]=produced
        step=time.monotonic()
        completed=subprocess.run([sys.executable,"-B",str(ROOT/"tools/predict_stress.py"),
            "--inputs",str(out/"inputs"),"--outputs",str(out/"predictions")],
            cwd=ROOT,capture_output=True,text=True,timeout=min(600,900-(time.monotonic()-started)))
        result["prediction_process"]={"exit_code":completed.returncode,"wall_seconds":time.monotonic()-step}
        if completed.returncode:
            result["prediction_process"]["error_tail"]=completed.stderr[-4000:]
            raise RuntimeError("prediction process failed")
        prediction_names=("evidence1.json","evidence2.json","traces.json","latencies.json")
        prediction_hashes={n:sha(out/"predictions"/n) for n in prediction_names}
        write_canonical(out/"prediction_commitment.json",{"files":prediction_hashes})
        result["prediction_commitment_sha256"]=sha(out/"prediction_commitment.json")
        result["prediction_files_committed_before_label_open"]=True
        # This is the first label read by the evaluator, after the child exits.
        assert sha(out/"private_labels/labels.json")==produced["label_file_sha256"]
        labels=json.loads((out/"private_labels/labels.json").read_text())
        result["generation_seed_disclosed_after_prediction"]=labels["generation_seed"]
        truth={c["case_id"]:c for c in labels["labels"]}
        evidence=json.loads((out/"predictions/evidence1.json").read_text())
        traces=json.loads((out/"predictions/traces.json").read_text())["traces"]
        observed={c["case_id"]:c for c in evidence["cases"]}; actions={c["case_id"]:c for c in traces}
        assert len(truth)==len(observed)==len(actions)==300 and set(truth)==set(observed)==set(actions)
        rows=[]
        for case_id in sorted(truth):
            t,p,a=truth[case_id],observed[case_id],actions[case_id]
            diagnostic=p.get("diagnostic")
            localization=_pair_f1(diagnostic["predicted_regions_xyxy_half_open"],t["boxes"],.5) if diagnostic is not None and t["boxes"] else None
            rows.append({"case_id":case_id,"family":t["family"],"stratum":t["stratum"],"kind":t["kind"],
                "localization_f1":localization,"expected_action":t["expected_action"],
                "observed_action":a["first_selected_action"],"action_correct":a["first_selected_action"]==t["expected_action"],
                "status":a["status"],"steps":len(a["trace"]),
                "production_release":a["state"]["deploy_gate"]!="blocked",
                "unsafe_recommendation":t["stratum"]=="critical_regression" and bool(a["state"].get("release_recommended")),
                "nuisance_before":int(diagnostic["raw_false_positive"]) if diagnostic is not None and t["stratum"]=="clean_or_nuisance" else 0,
                "nuisance_after":int(diagnostic["rerun_false_positive"]) if diagnostic is not None and t["stratum"]=="clean_or_nuisance" else 0})
        positive=[r["localization_f1"] for r in rows if r["localization_f1"] is not None]
        assert len(positive)==200
        accuracy=sum(r["action_correct"] for r in rows)/300
        f1=math.fsum(positive)/len(positive)
        before=sum(r["nuisance_before"] for r in rows); after=sum(r["nuisance_after"] for r in rows)
        reduction=(before-after)/before if before else None
        latency=json.loads((out/"predictions/latencies.json").read_text())
        assert len(latency["first"])==len(latency["second"])==300
        p95=percentile_nearest_rank(latency["first"],.95)
        output_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        recommendations=sum(r["unsafe_recommendation"] for r in rows)
        releases=sum(r["production_release"] for r in rows)
        reproducible=prediction_hashes["evidence1.json"]==prediction_hashes["evidence2.json"]
        gates={"all300":True,"localization_f1":f1>=.85,"action_accuracy":accuracy>=.9,
            "nuisance_reduction":reduction is not None and reduction>=.3,
            "p95_and_size":p95<=2 and output_bytes<=100*1024**2,"repeat_hash":reproducible,
            "max4steps":all(r["steps"]<=4 for r in rows),"zero_production_release":releases==0,
            "zero_critical_unsafe_recommendation":recommendations==0}
        strata={}
        for field in ("family","stratum","kind"):
            strata[field]={}
            for value in sorted({r[field] for r in rows}):
                part=[r for r in rows if r[field]==value]
                strata[field][value]={"count":len(part),"action_correct":sum(r["action_correct"] for r in part),
                    "unsafe_recommendations":sum(r["unsafe_recommendation"] for r in part),
                    "unknown":sum(r["status"]=="UNKNOWN" for r in part)}
        result.update(status="STRESS_PASS" if all(gates.values()) else "STRESS_FAIL",
            cases=300,positive_localization_cases=200,action_correct=sum(r["action_correct"] for r in rows),
            action_accuracy=accuracy,localization_macro_f1=f1,nuisance_before=before,nuisance_after=after,
            nuisance_reduction=reduction,p95_seconds=p95,output_bytes=output_bytes,
            critical_unsafe_recommendations=recommendations,production_releases=releases,
            unknown_cases=sum(r["status"]=="UNKNOWN" for r in rows),gates=gates,breakdown=strata,
            per_case=rows,prediction_file_sha256=prediction_hashes)
        code=0 # A measured quality failure is not an infrastructure failure.
    except Exception as error:
        result.update(status="EXECUTION_FAIL",error_class=type(error).__name__,error=str(error))
    finally:
        signal.alarm(0)
    result["source_inputs"]=manifest["inputs"]
    result["source_unchanged"]=all(sha(ROOT/n)==v for n,v in manifest["inputs"].items())
    if not result["source_unchanged"]: result["status"],code="EXECUTION_FAIL",1
    result.update(wall_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
    (out/"receipt.json").write_text(json.dumps(result,sort_keys=True,indent=2))
    print("FINAL_STRESS_RECEIPT_JSON "+json.dumps(result,sort_keys=True),flush=True)
    return code


if __name__=="__main__":
    raise SystemExit(main())
