"""Private two-stage CV5 inspection API; no deployment or production release."""
import base64
from datetime import datetime,timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import struct
import time

import cv2
import numpy as np

from visiongate_ci.role_agent import inspect_capture,validate_roles
from visiongate_ci.perception import _diagnostic_action,runtime_evidence

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=json.loads((ROOT/"contracts/local_baseline_contract.json").read_text())
POLICY=json.loads((ROOT/"contracts/role_policy_r02.json").read_text())
cv2.ocl.setUseOpenCL(False)
cv2.setRNGSeed(CONTRACT["determinism"]["seed"])
SCHEMA="visiongate-private-inspection/v1"
MAX_IMAGE_BYTES=1024**2


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def lease():
    deadline=datetime.fromisoformat(os.environ["VG_SESSION_DEADLINE_UTC"].replace("Z","+00:00"))
    if deadline.tzinfo is None:raise ValueError("timezone required")
    remaining=deadline.timestamp()-time.time()
    if not 0<remaining<=900:raise ValueError("expired or excessive lease")
    key=os.environ["VG_CONTINUATION_KEY"].encode()
    if len(key)<32:raise ValueError("continuation key absent")
    return deadline.timestamp(),key


def decode(value):
    if set(value)!={"png_base64","sha256"}:raise ValueError("image fields")
    encoded=value["png_base64"]
    if not isinstance(encoded,str) or len(encoded)>4*((MAX_IMAGE_BYTES+2)//3):raise ValueError("image byte bound")
    raw=base64.b64decode(encoded,validate=True)
    if len(raw)>MAX_IMAGE_BYTES or len(raw)<33 or raw[:8]!=b"\x89PNG\r\n\x1a\n":raise ValueError("PNG required")
    if raw[12:16]!=b"IHDR" or struct.unpack(">II",raw[16:24])!=(1920,1080):raise ValueError("PNG dimensions before allocation")
    if raw[24]!=8 or raw[25] not in (2,6):raise ValueError("8-bit RGB/RGBA only")
    digest=hashlib.sha256(raw).hexdigest()
    if not isinstance(value["sha256"],str) or not hmac.compare_digest(digest,value["sha256"]):raise ValueError("image hash mismatch")
    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype!=np.uint8 or image.shape[:2]!=(1080,1920):raise ValueError("decode failed")
    if image.shape[2]==4:
        if not np.all(image[:,:,3]==255):raise ValueError("nonopaque alpha unsupported")
        image=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
    if image.shape!=(1080,1920,3):raise ValueError("RGB shape")
    return image,digest


def sign(binding,key):
    data=base64.urlsafe_b64encode(canonical(binding)).decode()
    return data+"."+hmac.new(key,data.encode(),hashlib.sha256).hexdigest()


def verify(token,binding,key):
    if not isinstance(token,str) or len(token)>4096:raise ValueError("continuation required")
    data,signature=token.split(".",1)
    if not hmac.compare_digest(signature,hmac.new(key,data.encode(),hashlib.sha256).hexdigest()):raise ValueError("continuation signature")
    if json.loads(base64.urlsafe_b64decode(data))!=binding:raise ValueError("continuation binding")


def lambda_handler(event,context):
    state={"deploy_gate":"blocked","sandbox_only":True,"release_recommended":False}
    response={"schema":SCHEMA,"status":"UNKNOWN","action":"request_human_approval","state":state}
    try:
        deadline,key=lease()
        if not isinstance(event,dict) or len(canonical(event))>3*1024**2:raise ValueError("event bound")
        if event.get("stage") not in ("initial","recapture"):raise ValueError("stage")
        fields={"schema","stage","case_id","images","baseline"}
        if event["stage"]=="recapture":fields.add("continuation_token")
        if set(event)!=fields or event["schema"]!=SCHEMA:raise ValueError("strict event fields")
        if not isinstance(event["case_id"],str) or not re.fullmatch(r"[a-z0-9-]{1,64}",event["case_id"]):raise ValueError("case identifier")
        if not runtime_evidence(CONTRACT)["competition_opencv_major_satisfied"]:raise ValueError("runtime")
        if set(event["images"])!={"reference","observation"}:raise ValueError("image roles")
        reference,reference_sha=decode(event["images"]["reference"])
        observation,observation_sha=decode(event["images"]["observation"])
        case={"case_id":event["case_id"],"width":1920,"height":1080,"sha256":{"reference":reference_sha}}
        parts=validate_roles(case,event["baseline"],POLICY)
        binding={"case_id":event["case_id"],"reference_sha256":reference_sha,
                 "roles_sha256":hashlib.sha256(canonical(event["baseline"])).hexdigest(),"expires_epoch":deadline}
        if event["stage"]=="recapture":verify(event["continuation_token"],binding,key)
        evidence,aligned=inspect_capture(reference,observation,parts,CONTRACT["perception"],POLICY)
        fallback,reason=_diagnostic_action(evidence["changed_fraction"],evidence["alignment_applied"],CONTRACT["perception"])
        if evidence["capture_quality_requires_check"]:
            action="rerun_visual_test" if event["stage"]=="initial" else "request_human_approval"
            reason="distributed_capture_evidence"
        elif evidence["protected_hits"]:
            action="set_deploy_gate_blocked"; state["critical_evidence"]=True; reason="protected_baseline_component_changed"
        else:action=fallback
        if action=="rerun_visual_test":
            if event["stage"]=="initial":response["continuation_token"]=sign(binding,key)
            else:state["release_recommended"]=True
        elif action in ("inspect_region","request_human_approval"):
            state["human_review_requested"]=True
        if action=="inspect_region":
            inspected=[]
            for x1,y1,x2,y2 in evidence["regions"]:
                difference=cv2.absdiff(reference[y1:y2,x1:x2],aligned[y1:y2,x1:x2])
                inspected.append({"box":[x1,y1,x2,y2],"changed_pixels":int(np.count_nonzero(np.max(difference,axis=2)>8))})
            response["actual_inspected_regions"]=inspected
        if time.time()>=deadline:raise ValueError("lease expired while computing")
        response.update(status="DECIDED",action=action,case_id=event["case_id"],stage=event["stage"],reason=reason,
                        evidence=evidence,input_sha256={"reference":reference_sha,"observation":observation_sha},
                        runtime=runtime_evidence(CONTRACT))
    except Exception as error:
        state.update(release_recommended=False,human_review_requested=True)
        response.pop("continuation_token",None)
        response.update(status="UNKNOWN",action="request_human_approval",error_class=type(error).__name__)
    return response
