"""Run two local evidence passes, score them, and emit a fail-closed receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from . import PIPELINE_REVISION
from .canonical import canonical_json_bytes, sha256_bytes, sha256_file, write_canonical
from .fixtures import TRUTH_SCHEMA
from .perception import run_evidence, validate_input_manifest


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "visiongate-ci-local-gate-receipt/v1"


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def validate_ground_truth(
    truth: Mapping[str, Any], contract: Mapping[str, Any]
) -> List[str]:
    errors: List[str] = []
    if set(truth) != {
        "cases",
        "fixture_revision",
        "pipeline_revision",
        "schema_version",
        "seed",
    }:
        return ["truth_root_fields"]
    if truth.get("schema_version") != TRUTH_SCHEMA:
        errors.append("truth_schema")
    if truth.get("pipeline_revision") != PIPELINE_REVISION:
        errors.append("truth_pipeline_revision")
    cases = truth.get("cases")
    if not isinstance(cases, list):
        return errors + ["truth_cases_not_list"]
    expected_count = int(contract["fixture_contract"]["initial_pairs"])
    if len(cases) != expected_count:
        errors.append("truth_case_count")
    expected_strata = contract["fixture_contract"]["strata"]
    observed_strata = {name: 0 for name in expected_strata}
    seen = set()
    allowed_actions = set(contract["actions"].values())
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "expected_action",
            "ground_truth_regions_xyxy_half_open",
            "stratum",
            "synthetic_generation",
        }:
            errors.append("truth_case_fields")
            continue
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            errors.append("truth_case_id_unique")
        seen.add(case_id)
        stratum = case.get("stratum")
        if stratum not in observed_strata:
            errors.append("truth_stratum")
        else:
            observed_strata[stratum] += 1
        if case.get("expected_action") not in allowed_actions:
            errors.append("truth_expected_action")
        regions = case.get("ground_truth_regions_xyxy_half_open")
        if not isinstance(regions, list):
            errors.append("truth_regions")
            continue
        for region in regions:
            if (
                not isinstance(region, list)
                or len(region) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in region)
                or region[0] < 0
                or region[1] < 0
                or region[0] >= region[2]
                or region[1] >= region[3]
            ):
                errors.append("truth_region_geometry")
    if observed_strata != expected_strata:
        errors.append("truth_strata_counts")
    return sorted(set(errors))


def intersection_over_union(first: Sequence[int], second: Sequence[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _pair_f1(predicted: Sequence[Sequence[int]], expected: Sequence[Sequence[int]], threshold: float) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    unmatched = set(range(len(predicted)))
    matches = 0
    for truth_box in expected:
        best_index = None
        best_iou = -1.0
        for index in sorted(unmatched):
            value = intersection_over_union(predicted[index], truth_box)
            if value > best_iou:
                best_iou = value
                best_index = index
        if best_index is not None and best_iou >= threshold:
            matches += 1
            unmatched.remove(best_index)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def diagnostic_scores(
    evidence: Mapping[str, Any],
    truth: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence_by_id = {case["case_id"]: case for case in evidence["cases"]}
    truth_by_id = {case["case_id"]: case for case in truth["cases"]}
    if set(evidence_by_id) != set(truth_by_id):
        raise ValueError("evidence and truth identifiers do not match")
    iou_threshold = float(
        evaluation_contract["metrics"]["visual_localization"]["match_iou_threshold"]
    )
    positive_f1: List[float] = []
    diagnostic_action_correct = 0
    final_action_correct = 0
    critical_unsafe_release = 0
    unknown_total = 0
    unknown_fail_closed = 0
    nuisance_before = 0
    nuisance_after = 0
    for case_id in sorted(truth_by_id):
        expected = truth_by_id[case_id]
        observed = evidence_by_id[case_id]
        diagnostic = observed.get("diagnostic")
        if expected["ground_truth_regions_xyxy_half_open"] and diagnostic is not None:
            positive_f1.append(
                _pair_f1(
                    diagnostic["predicted_regions_xyxy_half_open"],
                    expected["ground_truth_regions_xyxy_half_open"],
                    iou_threshold,
                )
            )
        if diagnostic is not None and diagnostic["action"] == expected["expected_action"]:
            diagnostic_action_correct += 1
        decision = observed["decision"]
        if decision["action"] == expected["expected_action"]:
            final_action_correct += 1
        if expected["stratum"] == "critical_regression" and decision["deploy_gate"] != "blocked":
            critical_unsafe_release += 1
        if decision["state"] == "UNKNOWN":
            unknown_total += 1
            if decision["deploy_gate"] == "blocked" and decision["action"] == "request_human_approval":
                unknown_fail_closed += 1
        if expected["stratum"] == "clean_or_nuisance" and diagnostic is not None:
            nuisance_before += int(diagnostic["raw_false_positive"])
            nuisance_after += int(diagnostic["rerun_false_positive"])
    pair_count = len(truth_by_id)
    reduction = (
        (nuisance_before - nuisance_after) / nuisance_before
        if nuisance_before
        else None
    )
    return {
        "critical_unsafe_release_count": critical_unsafe_release,
        "diagnostic_action_accuracy": round(diagnostic_action_correct / pair_count, 9),
        "final_action_accuracy": round(final_action_correct / pair_count, 9),
        "localization_positive_pair_count": len(positive_f1),
        "localization_positive_pair_macro_f1": round(sum(positive_f1) / len(positive_f1), 9),
        "nuisance_false_positive_after_rerun": nuisance_after,
        "nuisance_false_positive_before_rerun": nuisance_before,
        "nuisance_false_positive_relative_reduction": (
            round(reduction, 9) if reduction is not None else None
        ),
        "unknown_fail_closed_count": unknown_fail_closed,
        "unknown_total_count": unknown_total,
    }


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_gate(
    fixture_root: Path,
    local_contract_path: Path,
    evaluation_contract_path: Path,
    evidence_dir: Path,
    receipt_path: Path,
) -> Dict[str, Any]:
    local_contract = load_json(local_contract_path)
    evaluation_contract = load_json(evaluation_contract_path)
    input_manifest = load_json(fixture_root / "input_manifest.json")
    ground_truth = load_json(fixture_root / "ground_truth.json")
    input_errors = validate_input_manifest(input_manifest)
    truth_errors = validate_ground_truth(ground_truth, local_contract)
    if input_errors or truth_errors:
        raise ValueError("fixture contract failed closed")
    if len(input_manifest["cases"]) != local_contract["fixture_contract"]["initial_pairs"]:
        raise ValueError("input pair count mismatch")
    if {case["case_id"] for case in input_manifest["cases"]} != {
        case["case_id"] for case in ground_truth["cases"]
    }:
        raise ValueError("input and ground-truth ID sets differ")

    evidence_first, latencies_first = run_evidence(
        fixture_root, input_manifest, local_contract
    )
    evidence_second, latencies_second = run_evidence(
        fixture_root, input_manifest, local_contract
    )
    first_bytes = canonical_json_bytes(evidence_first)
    second_bytes = canonical_json_bytes(evidence_second)
    first_hash = sha256_bytes(first_bytes)
    second_hash = sha256_bytes(second_bytes)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    first_path = evidence_dir / "local_evidence_run1.json"
    second_path = evidence_dir / "local_evidence_run2.json"
    write_canonical(first_path, evidence_first)
    write_canonical(second_path, evidence_second)

    scores = diagnostic_scores(evidence_first, ground_truth, evaluation_contract)
    runtime_compatible = evidence_first["runtime"]["competition_opencv_major_satisfied"]
    artifact_bytes = directory_size_bytes(fixture_root) + first_path.stat().st_size + second_path.stat().st_size
    p95_first = percentile_nearest_rank(latencies_first, 0.95)
    p95_second = percentile_nearest_rank(latencies_second, 0.95)
    metrics_contract = evaluation_contract["metrics"]
    max_artifacts = int(
        evaluation_contract["dataset_contract"]["max_initial_artifact_bytes"]
    )
    reproducible = first_hash == second_hash
    fixture_gate = {
        "id": "fixture_schema",
        "observed": len(input_manifest["cases"]),
        "required": int(local_contract["fixture_contract"]["initial_pairs"]),
        "status": "PASS",
    }
    visual_gate = {
        "diagnostic_observed": scores["localization_positive_pair_macro_f1"],
        "id": "visual_localization_macro_f1",
        "observed": (
            scores["localization_positive_pair_macro_f1"] if runtime_compatible else None
        ),
        "required_min": metrics_contract["visual_localization"]["macro_f1_min"],
        "status": (
            "PASS"
            if runtime_compatible
            and scores["localization_positive_pair_macro_f1"]
            >= metrics_contract["visual_localization"]["macro_f1_min"]
            else "UNKNOWN"
            if not runtime_compatible
            else "FAIL"
        ),
    }
    action_gate = {
        "diagnostic_observed": scores["diagnostic_action_accuracy"],
        "id": "agent_action_accuracy",
        "observed": scores["final_action_accuracy"],
        "required_min": metrics_contract["decision_safety"]["action_accuracy_min"],
        "status": (
            "PASS"
            if scores["final_action_accuracy"]
            >= metrics_contract["decision_safety"]["action_accuracy_min"]
            else "FAIL"
        ),
    }
    reduction_value = scores["nuisance_false_positive_relative_reduction"]
    reduction_gate = {
        "diagnostic_observed": reduction_value,
        "id": "nuisance_false_positive_reduction",
        "observed": reduction_value if runtime_compatible else None,
        "required_min": metrics_contract["adaptive_value"][
            "nuisance_false_positive_relative_reduction_min"
        ],
        "status": (
            "UNKNOWN"
            if not runtime_compatible or reduction_value is None
            else "PASS"
            if reduction_value
            >= metrics_contract["adaptive_value"][
                "nuisance_false_positive_relative_reduction_min"
            ]
            else "FAIL"
        ),
    }
    efficiency_gate = {
        "artifact_bytes_observed": artifact_bytes,
        "artifact_bytes_required_max": max_artifacts,
        "diagnostic_p95_seconds_run1": round(p95_first, 9),
        "diagnostic_p95_seconds_run2": round(p95_second, 9),
        "id": "cpu_efficiency_and_artifact_size",
        "observed_p95_seconds": round(p95_first, 9) if runtime_compatible else None,
        "required_p95_seconds_max": metrics_contract["efficiency_reproducibility"][
            "local_cpu_p95_seconds_max"
        ],
        "status": (
            "UNKNOWN"
            if not runtime_compatible
            else "PASS"
            if p95_first
            <= metrics_contract["efficiency_reproducibility"]["local_cpu_p95_seconds_max"]
            and artifact_bytes <= max_artifacts
            else "FAIL"
        ),
    }
    reproducibility_gate = {
        "evidence_run1_sha256": first_hash,
        "evidence_run2_sha256": second_hash,
        "id": "evidence_reproducibility",
        "required_equal": True,
        "status": "PASS" if reproducible else "FAIL",
    }
    performance_gates = [
        fixture_gate,
        visual_gate,
        action_gate,
        reduction_gate,
        efficiency_gate,
        reproducibility_gate,
    ]
    passed_count = sum(item["status"] == "PASS" for item in performance_gates)
    unknown_policy_passed = (
        scores["unknown_total_count"] == scores["unknown_fail_closed_count"]
        and scores["critical_unsafe_release_count"] == 0
    )
    technology_gate = {
        "id": "required_opencv_major",
        "observed": evidence_first["runtime"]["opencv_version"],
        "required_major": local_contract["runtime_contract"]["required_opencv_major"],
        "status": "PASS" if runtime_compatible else "FAIL",
    }
    receipt = {
        "cost_and_external_actions": {
            "aws_actions": 0,
            "cost_cny": 0,
            "external_model_calls": 0,
            "registration_or_submission_actions": 0,
        },
        "decision": "BLOCKED",
        "diagnostic_metrics_not_competition_evidence": scores,
        "fixture_evidence": {
            "ground_truth_sha256": sha256_file(fixture_root / "ground_truth.json"),
            "input_manifest_sha256": sha256_file(fixture_root / "input_manifest.json"),
            "manifest_sha256": sha256_file(fixture_root / "MANIFEST.sha256"),
            "rights_external_asset_count": input_manifest["rights"][
                "external_image_asset_count"
            ],
        },
        "hard_gates": {
            "required_opencv_major": technology_gate,
            "unknown_and_critical_fail_closed": {
                "critical_unsafe_release_count": scores[
                    "critical_unsafe_release_count"
                ],
                "status": "PASS" if unknown_policy_passed else "FAIL",
                "unknown_fail_closed_count": scores["unknown_fail_closed_count"],
                "unknown_total_count": scores["unknown_total_count"],
            },
        },
        "evaluation_contract_sha256": sha256_file(evaluation_contract_path),
        "local_contract_sha256": sha256_file(local_contract_path),
        "official_metric_claim_allowed": runtime_compatible,
        "performance_gate_summary": {
            "gates": performance_gates,
            "passed": passed_count,
            "required": evaluation_contract["go_no_go"]["performance_gates_required"],
            "total": evaluation_contract["go_no_go"]["performance_gates_total"],
        },
        "pipeline_revision": local_contract["pipeline_revision"],
        "runtime": evidence_first["runtime"],
        "schema_version": RECEIPT_SCHEMA,
        "status": "FAIL_CLOSED",
    }
    write_canonical(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "fixtures")
    parser.add_argument(
        "--local-contract",
        type=Path,
        default=ROOT / "contracts" / "local_baseline_contract.json",
    )
    parser.add_argument(
        "--evaluation-contract", type=Path, default=ROOT / "evaluation_contract.json"
    )
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence")
    parser.add_argument(
        "--receipt", type=Path, default=ROOT / "receipts" / "local_gate_receipt.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_gate(
            args.fixtures,
            args.local_contract,
            args.evaluation_contract,
            args.evidence_dir,
            args.receipt,
        )
    except Exception as exc:
        print(f"LOCAL_GATE_ERROR {type(exc).__name__}")
        return 2
    print(
        f"{receipt['status']} decision={receipt['decision']} "
        f"performance_passed={receipt['performance_gate_summary']['passed']}/"
        f"{receipt['performance_gate_summary']['total']} "
        f"opencv={receipt['runtime']['opencv_version']}"
    )
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
