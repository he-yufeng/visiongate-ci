"""Build immutable AWS-base image and exercise RIE offline, no AWS account calls."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expired(signum,frame):raise TimeoutError("900s owned Lambda-image probe cap")


def main():
    out=ROOT/"output/lambda-r04";out.mkdir(parents=True,mode=0o700)
    signal.signal(signal.SIGALRM,expired);signal.alarm(900)
    started=time.monotonic();containers=[];code=1
    result={"scope":"OFFLINE_OFFICIAL_AWS_BASE_RIE_API_CONTROLS_NOT_REAL_AWS_DEPLOYMENT_OR_COST_STOP_PROOF",
        "status":"RUNNING","run_id":os.environ.get("GITHUB_RUN_ID"),"source_commit":os.environ.get("GITHUB_SHA"),
        "aws_account_or_resource_calls":0,"registry_pushes":0,"new_spend_cny":0,"competition_ready":False,
        "official_score":None,"award":None,"real_IAM_concurrency_watchdog_teardown_verified":False}
    manifest=json.loads((ROOT/"LAMBDA_SOURCE_MANIFEST.json").read_text())
    def command(args,timeout=240,env=None):
        value=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=timeout,env=env)
        if value.returncode:raise RuntimeError("Owned command failed: "+str(args[:3])+" "+value.stderr[-3000:])
        return value.stdout
    try:
        assert platform.system()=="Linux" and platform.machine()=="x86_64" and sys.version_info[:2]==(3,12)
        event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        assert not event["repository"]["private"] and os.environ["GITHUB_REPOSITORY"]=="he-yufeng/visiongate-ci"
        assert all(sha(ROOT/n)==v for n,v in manifest["inputs"].items())
        assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=10*1024**3
        tag="public.ecr.aws/lambda/python:3.12"
        command(["docker","pull","--platform","linux/amd64",tag])
        base=json.loads(command(["docker","image","inspect",tag]))[0]
        digest=next(x for x in base["RepoDigests"] if re.fullmatch(r"public\.ecr\.aws/lambda/python@sha256:[0-9a-f]{64}",x))
        assert base["Architecture"]=="amd64"
        image="visiongate-lambda-r04:"+os.environ["GITHUB_RUN_ID"]
        command(["docker","build","--platform","linux/amd64","--build-arg","BASE_IMAGE="+digest,
                 "-f","cloud/Dockerfile","-t",image,"."],timeout=360)
        built=json.loads(command(["docker","image","inspect",image]))[0]
        result["base_image_digest_used"]=digest;result["image_id"]=built["Id"];result["image_bytes"]=built["Size"]
        assert built["Size"]<=2*1024**3
        probe_env={**os.environ,"VG_CONTINUATION_KEY":secrets.token_hex(32)}
        def start_container(deadline):
            identifier=command(["docker","run","-d","--read-only","--network","none","--memory","2g","--cpus","1",
                "--tmpfs","/tmp:rw,nosuid,size=64m","-e","VG_CONTINUATION_KEY",
                "-e","VG_SESSION_DEADLINE_UTC="+deadline,"-e","AWS_LAMBDA_FUNCTION_MEMORY_SIZE=2048",image],env=probe_env).strip()
            assert re.fullmatch(r"[0-9a-f]{64}",identifier)
            containers.append(identifier)
            return identifier
        cid=start_container((datetime.now(timezone.utc)+timedelta(seconds=600)).isoformat())
        report=json.loads(command(["docker","exec",cid,"python","-B","-m","cloud.smoke_client"],timeout=180))
        result["new_RIE_API_controls"]=report
        assert report["total"]==report["passed"]==16
        evidence=json.loads(command(["docker","exec",cid,"python","-c",
            "import cv2,numpy as np,platform;from importlib import metadata;import json;print(json.dumps({'opencv':cv2.__version__,'distribution':metadata.version('opencv-python-headless'),'numpy':np.__version__,'python':platform.python_version(),'machine':platform.machine()}))"]))
        assert evidence["opencv"]=="5.0.0" and evidence["distribution"]=="5.0.0.93" and evidence["numpy"]=="2.2.6"
        result["actual_container_runtime"]=evidence
        expired_cid=start_container((datetime.now(timezone.utc)-timedelta(seconds=30)).isoformat())
        expiry_script="import json;from cloud.smoke_client import fixture,invoke,wait_ready;wait_ready();r=invoke(fixture()[1]);assert r['status']=='UNKNOWN' and not r['state']['release_recommended'];print(json.dumps({'expired_lease_refused':True,'production_blocked':r['state']['deploy_gate']=='blocked'}))"
        expiry=json.loads(command(["docker","exec",expired_cid,"python","-c",expiry_script],timeout=40))
        result["separate_expired_environment_control"]=expiry
        assert expiry["expired_lease_refused"] and expiry["production_blocked"]
        result.update(status="OFFLINE_LAMBDA_RIE_PASS",new_controls_passed=17,network_disabled=True,
            filesystem_readonly=True,local_container_memory_bytes=2147483648,local_container_CPUs=1)
        code=0
    except Exception as error:
        result.update(status="EXECUTION_OR_API_CONTROL_FAIL",error_class=type(error).__name__,error=str(error))
    finally:
        cleanup=[]
        for cid in containers:
            try:
                p=subprocess.run(["docker","rm","--force",cid],cwd=ROOT,capture_output=True,text=True,timeout=20)
                cleanup.append({"owned_container_removed":p.returncode==0,"container_id":cid})
            except Exception as error:
                cleanup.append({"owned_container_removed":False,"container_id":cid,"error_class":type(error).__name__})
        result["owned_local_container_cleanup"]=cleanup
        if any(not r["owned_container_removed"] for r in cleanup):result["status"],code="LOCAL_CONTAINER_CLEANUP_FAIL",1
        signal.alarm(0)
    result["source_inputs"]=manifest["inputs"]
    result["source_unchanged"]=all(sha(ROOT/n)==v for n,v in manifest["inputs"].items())
    if not result["source_unchanged"]:result["status"],code="SOURCE_DRIFT_FAIL",1
    result.update(wall_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
    (out/"receipt.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print("FINAL_LAMBDA_RECEIPT_JSON "+json.dumps(result,sort_keys=True),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
