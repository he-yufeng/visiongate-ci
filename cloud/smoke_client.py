"""New API-boundary controls through real local RIE HTTP, never AWS service."""
import base64
import copy
import hashlib
import json
import socket
import time
import urllib.request

import cv2
import numpy as np

SCHEMA="visiongate-private-inspection/v1"


def invoke(event):
    request=urllib.request.Request("http://127.0.0.1:8080/2015-03-31/functions/function/invocations",
        data=json.dumps(event).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=25) as response:return json.load(response)


def encode(image):
    ok,buffer=cv2.imencode(".png",image,[cv2.IMWRITE_PNG_COMPRESSION,9]);assert ok
    raw=buffer.tobytes()
    return {"png_base64":base64.b64encode(raw).decode(),"sha256":hashlib.sha256(raw).hexdigest()}


def fixture():
    image=np.full((1080,1920,3),236,np.uint8)
    parts=[{"role":role,"box":box} for role,box in
           (("primary_deploy",[1450,950,1700,1030]),("secondary_deploy",[1710,930,1810,968]),("critical_warning",[1410,950,1434,974]))]
    for p in parts:
        x1,y1,x2,y2=p["box"];image[y1:y2,x1:x2]=(30,50,190)
    ref=encode(image)
    event={"schema":SCHEMA,"case_id":"cloud-control","stage":"initial",
           "images":{"reference":ref,"observation":ref},
           "baseline":{"schema":"visiongate-baseline-components/v1","case_id":"cloud-control",
                       "reference_sha256":ref["sha256"],"protected_components":parts}}
    return image,event


def wait_ready():
    for attempt in range(60):
        try:
            with socket.create_connection(("127.0.0.1",8080),timeout=.2):break
        except OSError:
            if attempt==59:raise
            time.sleep(.1) # Connection-readiness only, not request retry.


def main():
    wait_ready()
    image,base=fixture(); records=[]
    def check(name,event,action,status="DECIDED",recommended=False):
        value=invoke(event)
        passed=(value.get("action")==action and value.get("status")==status
                and bool(value.get("state",{}).get("release_recommended"))==recommended
                and value.get("state",{}).get("deploy_gate")=="blocked")
        records.append({"name":name,"passed":passed,"action":value.get("action"),"status":value.get("status"),
                        "release_recommended":value.get("state",{}).get("release_recommended",False)})
        return value
    first=check("initial-clear-requires-live-recapture",base,"rerun_visual_test")
    token=first.get("continuation_token")
    recap=copy.deepcopy(base);recap.update(stage="recapture",continuation_token=token)
    check("signed-clear-recapture-can-recommend",recap,"rerun_visual_test",recommended=True)
    minor=copy.deepcopy(base);changed=image.copy();changed[250:302,600:728]=(20,30,210)
    minor["images"]["observation"]=encode(changed)
    value=check("minor-executes-real-ROI-inspection",minor,"inspect_region")
    records[-1]["passed"] &= bool(value.get("actual_inspected_regions"))
    critical=copy.deepcopy(base);changed=image.copy();changed[955,1415]=(236,236,236)
    critical["images"]["observation"]=encode(changed)
    value=check("single-protected-pixel-cannot-be-waived",critical,"set_deploy_gate_blocked")
    records[-1]["passed"] &= "continuation_token" not in value
    unknown=[]
    wrong=copy.deepcopy(recap);wrong.pop("continuation_token");unknown.append(("missing-continuation",wrong))
    wrong=copy.deepcopy(recap);wrong["continuation_token"]="forged";unknown.append(("forged-continuation",wrong))
    wrong=copy.deepcopy(recap);wrong["case_id"]="other-control";wrong["baseline"]["case_id"]="other-control";unknown.append(("cross-case-token",wrong))
    wrong=copy.deepcopy(recap);wrong["baseline"]["protected_components"][0]["box"][0]+=1;unknown.append(("changed-role-binding-token",wrong))
    wrong=copy.deepcopy(base);wrong["images"]["observation"]["sha256"]="0"*64;unknown.append(("image-byte-hash-mismatch",wrong))
    wrong=copy.deepcopy(base);wrong["images"]["observation"]=encode(np.zeros((8,8,3),np.uint8));unknown.append(("PNG-dimensions-before-allocation",wrong))
    wrong=copy.deepcopy(base);rgba=np.dstack((image,np.full((1080,1920),255,np.uint8)));rgba[100,100,3]=0
    wrong["images"]["observation"]=encode(rgba);unknown.append(("nonopaque-alpha",wrong))
    wrong=copy.deepcopy(base);wrong["stratum"]="critical_regression";unknown.append(("no-scoring-label-event-field",wrong))
    wrong=copy.deepcopy(base);wrong["baseline"]["protected_components"].pop();unknown.append(("incomplete-required-DOM-role-set",wrong))
    wrong=copy.deepcopy(base);wrong["VG_SESSION_DEADLINE_UTC"]="2999-01-01T00:00:00Z";unknown.append(("caller-cannot-extend-environment-lease",wrong))
    wrong=copy.deepcopy(base);wrong["images"]["observation"]["png_base64"]="A"*(4*((1024**2+2)//3)+4);unknown.append(("encoded-image-byte-bound",wrong))
    for name,event in unknown:check(name,event,"request_human_approval",status="UNKNOWN")
    check("same-binding-token-replay-not-claimed-single-use",recap,"rerun_visual_test",recommended=True)
    assert len(records)==16
    print(json.dumps({"controls":records,"passed":sum(r["passed"] for r in records),"total":len(records),
                     "http_interface":"localAWSRIE_not_cloud","tokens_or_images_logged":False}))
    return 0 if all(r["passed"] for r in records) else 1


if __name__=="__main__":raise SystemExit(main())
