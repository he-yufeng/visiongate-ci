"""Prediction-only subprocess: accepts input images/contracts, no label path."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visiongate_ci import agent_loop
from visiongate_ci.canonical import write_canonical
from visiongate_ci.perception import run_evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    args = p.parse_args()
    args.outputs.mkdir(mode=0o700)
    contract = json.loads((ROOT / "contracts/local_baseline_contract.json").read_text())
    inputs = json.loads((args.inputs / "input_manifest.json").read_text())
    evidence1, latencies1 = run_evidence(args.inputs, inputs, contract)
    write_canonical(args.outputs / "evidence1.json", evidence1)
    evidence2, latencies2 = run_evidence(args.inputs, inputs, contract)
    write_canonical(args.outputs / "evidence2.json", evidence2)
    traces = [agent_loop.run_case(args.inputs, c, contract) for c in inputs["cases"]]
    write_canonical(args.outputs / "traces.json", {"traces":traces})
    write_canonical(args.outputs / "latencies.json", {"first":latencies1,"second":latencies2})
    print("PREDICTION_FILES_COMMITTED_WITHOUT_LABEL_ACCESS", flush=True)


if __name__ == "__main__":
    main()
