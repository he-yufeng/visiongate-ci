"""One bounded owned-web live-capture integration, frozen R02 core unchanged."""
from datetime import datetime,timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expired(signum,frame): raise TimeoutError("900s owned browser integration cap")


def main():
    out=ROOT/"output/playwright/browser-r03"; out.mkdir(parents=True,mode=0o700)
    signal.signal(signal.SIGALRM,expired); signal.alarm(900)
    started=time.monotonic(); code=1; server=None
    result={"scope":"OWNED_REAL_CHROME_CAPTURE_CONTROLLED36CASE_INTEGRATION_NOT_EXTERNAL_REALWORLD_BENCHMARK",
        "status":"RUNNING","run_id":os.environ.get("GITHUB_RUN_ID"),"source_commit":os.environ.get("GITHUB_SHA"),
        "same_author_known_fixture_family":True,"crypto_host_isolation_claimed":False,
        "owner_authored_DOM_roles_not_independent_semantic_attestation":True,
        "user_browser_or_private_site_connection":False,"aws_actions":0,"new_spend_cny":0,"model_api_calls":0,
        "production_deploy_actions":0,"registration_or_submission_actions":0,"competition_ready":False,
        "official_score":None,"award":None}
    manifest=json.loads((ROOT/"BROWSER_SOURCE_MANIFEST.json").read_text())
    try:
        assert sys.version_info[:2]==(3,12) and platform.system()=="Linux" and platform.machine()=="x86_64"
        event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        assert not event["repository"]["private"] and os.environ["GITHUB_REPOSITORY"]=="he-yufeng/visiongate-ci"
        assert all(sha(ROOT/n)==v for n,v in manifest["inputs"].items())
        assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=10*1024**3
        import cv2
        import numpy as np
        from owned_web_fixtures import prepare,serve
        from visiongate_ci.canonical import write_canonical
        from visiongate_ci.gate import _pair_f1,percentile_nearest_rank
        assert cv2.__version__=="5.0.0" and metadata.version("opencv-python-headless")=="5.0.0.93" and np.__version__=="2.2.6"
        chrome=shutil.which("google-chrome")
        assert chrome,"stock hosted-runner Chrome required; do not install fallback browser"
        chrome_version=subprocess.check_output([chrome,"--version"],text=True,timeout=10).strip()
        package_versions={name:json.loads((ROOT/"node_modules"/name/"package.json").read_text())["version"]
                          for name in ("@playwright/cli","playwright","playwright-core")}
        assert package_versions=={"@playwright/cli":"0.1.22","playwright":"1.64.0-alpha-1790635538000","playwright-core":"1.64.0-alpha-1790635538000"}
        result["runtime"]={"opencv":cv2.__version__,"opencv_distribution":metadata.version("opencv-python-headless"),
            "numpy":np.__version__,"python":platform.python_version(),"platform":"Linuxx86_64",
            "stock_chrome":chrome_version,"npm_package_versions":package_versions}
        cv2.ocl.setUseOpenCL(False); cv2.setRNGSeed(20260830)
        seed=secrets.randbits(64)
        views,ids=prepare(out,seed)
        result["label_commitment_sha256"]=sha(out/"private_labels/labels.json")
        server,origin=serve(views,ids,out/"inputs/dom_metadata")
        spec=out/"inputs/navigation.json"
        write_canonical(spec,{"origin":origin,"cases":ids})
        step=time.monotonic()
        process=subprocess.run([sys.executable,"-B",str(ROOT/"tools/predict_browser.py"),
            "--spec",str(spec),"--output",str(out/"predictions")],cwd=ROOT,capture_output=True,text=True,
            timeout=min(600,900-(time.monotonic()-started)))
        result["prediction_process"]={"exit_code":process.returncode,"seconds":time.monotonic()-step}
        if process.returncode:
            result["prediction_process"]["error_tail"]=process.stderr[-6000:]
            raise RuntimeError("owned live-browser prediction child failed")
        files={name:sha(out/"predictions"/name) for name in ("predictions.json","cli_commands.json")}
        write_canonical(out/"prediction_commitment.json",{"files":files})
        result["prediction_commitment_sha256"]=sha(out/"prediction_commitment.json")
        result["predictions_committed_before_score_label_read"]=True
        assert sha(out/"private_labels/labels.json")==result["label_commitment_sha256"]
        truth_file=json.loads((out/"private_labels/labels.json").read_text())
        result["generation_seed_exact_decimal_disclosed_after_prediction"]=truth_file["generation_seed_exact_decimal"]
        truth={t["case_id"]:t for t in truth_file["cases"]}
        predictions=json.loads((out/"predictions/predictions.json").read_text())["cases"]
        observed={p["case_id"]:p for p in predictions}
        assert len(predictions)==len(observed)==len(truth)==36 and set(observed)==set(truth)
        rows=[]; role_binding=True; all_loopback=True; conditional=True; no_prepared=True
        for case_id in sorted(truth):
            t,p=truth[case_id],observed[case_id]; a=p["decision"]; state=a["state"]
            dom=p["baseline_dom_metadata"]; declaration=p["baseline_declaration"]
            parts=declaration["protected_components"]
            role_binding &= declaration["reference_sha256"]==p["input_sha256"]["reference"] and parts==dom["protected_components"]
            role_binding &= len(parts)==3 and {v["role"] for v in parts}=={"primary_deploy","secondary_deploy","critical_warning"}
            all_loopback &= dom["url"]==origin+f"/live/{case_id}/reference" and dom["viewport"]==[1920,1080] and "Chrome/" in dom["user_agent"]
            expected_capture=a["first_selected_action"]=="rerun_visual_test"
            captures=len(p["capture_events"])
            conditional &= captures==int(expected_capture) and p["rerun_exists_after_policy"]==expected_capture
            no_prepared &= not p["rerun_prepared_before_policy"] and all(not e["file_existed_before_request"] for e in p["capture_events"])
            boxes=[]
            if t["stratum"]=="minor_regression": boxes=[dom["status_region"]]
            elif t["stratum"]=="critical_regression": boxes=[next(v["box"] for v in parts if v["role"]==t["removed_role"])]
            f1=_pair_f1(a["effective_regions"],boxes,.5) if boxes and a["status"]=="DECIDED" else 0.0 if boxes else None
            final=a["status"]=="DECIDED" and a["final_action"]==t["expected_final"] and state["deploy_gate"]=="blocked"
            if t["stratum"]=="clean_or_nuisance": final &= bool(state.get("release_recommended"))
            elif t["stratum"]=="critical_regression": final &= bool(state.get("critical_evidence")) and not state.get("release_recommended",False)
            else: final &= bool(state.get("human_review_requested")) and not state.get("release_recommended",False)
            threshold=.0005
            initial=a["trace"][0]["output"].get("raw_difference_fraction") if a["trace"] and a["trace"][0]["tool"]=="inspect_pair" else None
            rows.append({"case_id":case_id,"family":t["family"],"stratum":t["stratum"],"kind":t["kind"],
                "expected_first":t["expected_first"],"expected_final":t["expected_final"],
                "observed_first":a["first_selected_action"],"observed_final":a["final_action"],
                "action_correct":a["status"]=="DECIDED" and a["first_selected_action"]==t["expected_first"],
                "final_correct":bool(final),"status":a["status"],"steps":len(a["trace"]),"localization_f1":f1,
                "unsafe_recommendation":t["stratum"]=="critical_regression" and bool(state.get("release_recommended")),
                "production_release":state["deploy_gate"]!="blocked","actual_conditional_captures":captures,
                "rerun_prepared_before_policy":p["rerun_prepared_before_policy"],
                "nuisance_before":int(initial is None or initial>threshold) if t["stratum"]=="clean_or_nuisance" else 0,
                "nuisance_after":int(a["effective_raw_fraction"] is None or a["effective_raw_fraction"]>threshold) if t["stratum"]=="clean_or_nuisance" else 0,
                "decision_cycle_wall_seconds":p["decision_cycle_wall_seconds"],
                "vision_policy_wall_excluding_sensor_seconds":p["vision_policy_wall_excluding_sensor_seconds"],
                "case_wall_including_initial_acquisition":p["case_wall_including_initial_acquisition"],
                "baseline_reference_sha256":declaration["reference_sha256"],"baseline_components_observed_from_DOM":parts})
        controls={"locked_CLI_stock_isolated_Chrome_loopback_observed":all_loopback,
                  "all36reference_hash_bound_complete_DOM_roles":role_binding,
                  "all_rerun_files_absent_before_visual_policy":no_prepared,
                  "only_selected_rerun_branches_execute_browser_recapture":conditional,
                  "at_least_one_real_recap_and_one_no_recap_branch":0<sum(r["actual_conditional_captures"] for r in rows)<36,
                  "all36baseline_candidate_sources_observed":len(rows)==36}
        first=sum(r["action_correct"] for r in rows)/36; final=sum(r["final_correct"] for r in rows)/36
        positive=[r["localization_f1"] for r in rows if r["localization_f1"] is not None]
        assert len(positive)==24
        f1=math.fsum(positive)/24; unsafe=sum(r["unsafe_recommendation"] for r in rows)
        releases=sum(r["production_release"] for r in rows); unknown=sum(r["status"]=="UNKNOWN" for r in rows)
        before=sum(r["nuisance_before"] for r in rows); after=sum(r["nuisance_after"] for r in rows)
        reduction=(before-after)/before if before else None
        vision_p95=percentile_nearest_rank([r["vision_policy_wall_excluding_sensor_seconds"] for r in rows],.95)
        cycle_p95=percentile_nearest_rank([r["decision_cycle_wall_seconds"] for r in rows],.95)
        acquisition_p95=percentile_nearest_rank([r["case_wall_including_initial_acquisition"] for r in rows],.95)
        output_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        gates={"all36":True,"integration_controls":all(controls.values()),"first_action_accuracy":first>=.9,
               "final_action_state_accuracy":final>=.9,"localization_f1":f1>=.85,
               "nuisance_reduction":reduction is not None and reduction>=.3,"vision_policy_p95":vision_p95<=2,
               "live_decision_cycle_p95":cycle_p95<=10,"output_size":output_bytes<=100*1024**2,
               "max4steps":all(r["steps"]<=4 for r in rows),"zero_critical_unsafe_recommendation":unsafe==0,
               "zero_production_release":releases==0,"zero_unknown":unknown==0}
        breakdown={}
        for field in ("family","stratum","kind"):
            breakdown[field]={}
            for value in sorted({r[field] for r in rows}):
                part=[r for r in rows if r[field]==value]
                breakdown[field][value]={"count":len(part),"action_correct":sum(r["action_correct"] for r in part),
                    "final_correct":sum(r["final_correct"] for r in part),"unsafe_recommendations":sum(r["unsafe_recommendation"] for r in part),
                    "captures":sum(r["actual_conditional_captures"] for r in part)}
        result.update(status="LIVE_BROWSER_PASS" if all(gates.values()) else "LIVE_BROWSER_FAIL",cases=36,
            positive_localization_cases=24,new_integration_controls=controls,action_accuracy=first,final_action_and_state_accuracy=final,
            localization_macro_f1=f1,critical_unsafe_recommendations=unsafe,production_releases=releases,unknown_cases=unknown,
            nuisance_before=before,nuisance_after=after,nuisance_reduction=reduction,vision_policy_p95_seconds=vision_p95,
            live_decision_cycle_p95_seconds=cycle_p95,full_case_acquisition_p95_seconds=acquisition_p95,output_bytes=output_bytes,
            actual_conditional_browser_captures=sum(r["actual_conditional_captures"] for r in rows),
            gates=gates,per_case=rows,breakdown=breakdown,prediction_file_sha256=files)
        code=0
    except Exception as error:
        result.update(status="EXECUTION_FAIL",error_class=type(error).__name__,error=str(error))
    finally:
        if server is not None: server.shutdown(); server.server_close()
        signal.alarm(0)
    result["source_inputs"]=manifest["inputs"]
    result["source_unchanged"]=all(sha(ROOT/n)==v for n,v in manifest["inputs"].items())
    if not result["source_unchanged"]: result["status"],code="EXECUTION_FAIL",1
    result.update(wall_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
    (out/"receipt.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print("FINAL_BROWSER_RECEIPT_JSON "+json.dumps(result,sort_keys=True),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
