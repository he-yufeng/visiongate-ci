"""Live prediction subprocess: only public navigation spec, no score labels."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from live_browser_adapter import BrowserClient,decide_live
from visiongate_ci.canonical import sha256_file,write_canonical
from visiongate_ci.role_agent import validate_roles


def main():
    p=argparse.ArgumentParser(); p.add_argument("--spec",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(mode=0o700)
    spec=json.loads(args.spec.read_text()); origin=spec["origin"]
    assert origin.startswith("http://127.0.0.1:") and len(spec["cases"])==36
    contract=json.loads((ROOT/"contracts/local_baseline_contract.json").read_text())
    policy=json.loads((ROOT/"contracts/role_policy_r02.json").read_text())
    images=args.output/"images"; images.mkdir()
    client=BrowserClient(ROOT,args.output,origin)
    evidence=[]
    try:
        client.open(origin+f'/live/{spec["cases"][0]}/reference')
        for case_id in spec["cases"]:
            start=time.monotonic()
            reference=images/f"{case_id}-reference.png"; candidate=images/f"{case_id}-candidate.png"
            client.baseline(case_id,reference)
            metadata=json.loads((args.spec.parent/"dom_metadata"/f"{case_id}.json").read_text())
            assert metadata["url"]==origin+f"/live/{case_id}/reference" and metadata["viewport"]==[1920,1080]
            assert "Chrome/" in metadata["user_agent"]
            client.candidate(case_id,candidate)
            case={"case_id":case_id,"width":1920,"height":1080,
                  "paths":{"reference":str(reference.relative_to(args.output)),"candidate":str(candidate.relative_to(args.output)),
                           "rerun":f"images/{case_id}-live-rerun.png"},
                  "sha256":{"reference":sha256_file(reference),"candidate":sha256_file(candidate),"rerun":"0"*64}}
            declared={"schema":"visiongate-baseline-components/v1","case_id":case_id,
                      "reference_sha256":case["sha256"]["reference"],"protected_components":metadata["protected_components"]}
            validate_roles(case,declared,policy)
            decision=decide_live(client,args.output,case,declared,contract,policy)
            evidence.append({"case_id":case_id,"input_sha256":case["sha256"],"baseline_declaration":declared,
                "baseline_dom_metadata":metadata,"case_wall_including_initial_acquisition":time.monotonic()-start,**decision})
        write_canonical(args.output/"predictions.json",{"cases":evidence})
        write_canonical(args.output/"cli_commands.json",{"commands":client.command_records})
        print("LIVE_PREDICTIONS_COMMITTED_WITHOUT_SCORE_LABEL_ACCESS",flush=True)
    finally:
        client.close()


if __name__=="__main__":main()
