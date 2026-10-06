"""Eight newly executed native mechanism controls, not old design-set replay."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from role_stress_generator import baseline, distort
from stress_generator import rect, text
from visiongate_ci.canonical import sha256_file
from visiongate_ci.role_agent import run_case


def run_controls(root, contract, policy):
    root=Path(root); root.mkdir(mode=0o700)
    records=[]
    def check(name, operation):
        try:
            operation()
            records.append({"name":name,"passed":True})
        except Exception as error:
            records.append({"name":name,"passed":False,"error_class":type(error).__name__,"error":str(error)})
    reference,badge,parts=baseline(np.random.default_rng(202610061),"editor")
    def make(name,candidate,rerun):
        paths={}; digests={}
        for role,image in (("reference",reference),("candidate",candidate),("rerun",rerun)):
            path=root/f"{name}-{role}.png"
            assert cv2.imwrite(str(path),image,[cv2.IMWRITE_PNG_COMPRESSION,9])
            paths[role]=path.name; digests[role]=sha256_file(path)
        case={"case_id":name,"height":1080,"width":1920,"paths":paths,"sha256":digests}
        declared={"schema":"visiongate-baseline-components/v1","case_id":name,
                  "reference_sha256":digests["reference"],"protected_components":copy.deepcopy(parts)}
        return case,declared
    def run(name,candidate,rerun):
        case,declared=make(name,candidate,rerun)
        return run_case(root,case,declared,contract,policy)
    def unchanged():
        value=run("clean",reference,reference)
        assert value["first_selected_action"]=="rerun_visual_test",value
        assert value["state"]["release_recommended"] and value["state"]["deploy_gate"]=="blocked",value
    def small_critical():
        image=reference.copy(); x,y=parts[2]["box"][:2]
        image[y+5,x+5]=(246,246,246)
        value=run("one-pixel-critical",image,image)
        assert value["first_selected_action"]==value["final_action"]=="set_deploy_gate_blocked",value
        assert not value["state"]["release_recommended"],value
    def minor():
        image=reference.copy(); rect(image,badge,(210,25,20))
        text(image,"READY",(1540,180),(250,250,250),.55)
        value=run("minor",image,image)
        assert value["first_selected_action"]==value["final_action"]=="inspect_region",value
        assert value["state"]["human_review_requested"] and not value["state"]["release_recommended"],value
        assert any(t["tool"]=="inspect_region" and t["output"]["regions"] for t in value["trace"]),value
    def nuisance():
        value=run("blur",distort(reference,1,np.random.default_rng(12)),reference)
        assert value["first_selected_action"]=="rerun_visual_test" and value["state"]["release_recommended"],value
        assert value["trace"][0]["output"]["capture_quality_requires_check"],value
    def compound():
        image=reference.copy(); rect(image,parts[2]["box"],(246,246,246))
        value=run("blur-critical",distort(image,1,np.random.default_rng(12)),image)
        assert value["first_selected_action"]=="rerun_visual_test",value
        assert value["final_action"]=="set_deploy_gate_blocked" and not value["state"]["release_recommended"],value
        assert [t["tool"] for t in value["trace"]]==["inspect_pair","rerun_visual_test","set_deploy_gate_blocked"],value
    def invariant():
        for _ in range(3):
            other,other_badge,other_parts=baseline(np.random.default_rng(202610061),"editor")
            assert np.array_equal(reference,other) and other_badge==badge and other_parts==parts
        assert {p["role"] for p in parts}==set(policy["required_roles"])
    def invalid_metadata():
        case,declared=make("metadata",reference,reference)
        variants=[]
        wrong=copy.deepcopy(declared); wrong["reference_sha256"]="0"*64; variants.append(wrong)
        wrong=copy.deepcopy(declared); wrong["protected_components"].pop(); variants.append(wrong)
        wrong=copy.deepcopy(declared); wrong["protected_components"][0]["box"][0]=-1; variants.append(wrong)
        wrong=copy.deepcopy(declared); wrong["stratum"]="critical_regression"; variants.append(wrong)
        for wrong in variants:
            value=run_case(root,case,wrong,contract,policy)
            assert value["status"]=="UNKNOWN" and not value["state"]["release_recommended"],value
    def integrity():
        case,declared=make("integrity",reference,reference)
        changed=reference.copy(); changed[200,200]=(0,0,0)
        assert cv2.imwrite(str(root/case["paths"]["candidate"]),changed)
        value=run_case(root,case,declared,contract,policy)
        assert value["status"]=="UNKNOWN" and not value["state"]["release_recommended"],value
    for name,operation in (("all-strata-baseline-role-invariance",invariant),
                           ("clean-recapture-can-recommend-not-always-block",unchanged),
                           ("single-critical-pixel-not-area-threshold",small_critical),
                           ("minor-real-region-and-review",minor),("distributed-blur-triggers-real-recapture",nuisance),
                           ("blur-never-waives-persistent-critical-deletion",compound),
                           ("binding-role-bounds-and-label-field-fail-closed",invalid_metadata),
                           ("candidate-byte-integrity-fail-closed",integrity)):
        check(name,operation)
    return {"passed":sum(r["passed"] for r in records),"total":len(records),"records":records}
