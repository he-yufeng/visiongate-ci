"""New/old paired prediction on identical fresh inputs; no label path accepted."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from visiongate_ci import agent_loop, role_agent
from visiongate_ci.canonical import write_canonical
from visiongate_ci.perception import run_evidence, validate_input_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--outputs",type=Path,required=True)
    args=p.parse_args(); args.outputs.mkdir(mode=0o700)
    contract=json.loads((ROOT/"contracts/local_baseline_contract.json").read_text())
    policy=json.loads((ROOT/"contracts/role_policy_r02.json").read_text())
    inputs=json.loads((args.inputs/"input_manifest.json").read_text())
    assert not validate_input_manifest(inputs)
    roles_list=json.loads((args.inputs/"baseline_roles.json").read_text())["cases"]
    roles={r["case_id"]:r for r in roles_list}
    assert len(roles_list)==len(roles)==len(inputs["cases"])==300
    assert set(roles)=={c["case_id"] for c in inputs["cases"]}
    for case in inputs["cases"]:
        role_agent.validate_roles(case,roles[case["case_id"]],policy)
    for index in (1,2):
        predictions,latencies=role_agent.run_batch(args.inputs,inputs["cases"],roles,contract,policy)
        write_canonical(args.outputs/f"role_evidence{index}.json",predictions)
        write_canonical(args.outputs/f"role_latencies{index}.json",{"latencies":latencies})
    original,_=run_evidence(args.inputs,inputs,contract)
    write_canonical(args.outputs/"old_evidence.json",original)
    old=[]; old_latencies=[]
    for case in inputs["cases"]:
        started=time.perf_counter_ns()
        old.append(agent_loop.run_case(args.inputs,case,contract))
        old_latencies.append((time.perf_counter_ns()-started)/1e9)
    write_canonical(args.outputs/"old_traces.json",{"cases":old})
    write_canonical(args.outputs/"old_latencies.json",{"latencies":old_latencies})
    print("PAIRED_PREDICTIONS_COMMITTED_WITHOUT_LABEL_PATH",flush=True)


if __name__=="__main__": main()
