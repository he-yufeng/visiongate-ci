"""OpenCV perception and deterministic fail-closed action policy."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from . import PIPELINE_REVISION
from .canonical import sha256_file
from .fixtures import INPUT_SCHEMA


EVIDENCE_SCHEMA = "visiongate-ci-evidence/v1"


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def validate_input_manifest(manifest: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    expected_root = {
        "cases",
        "fixture_revision",
        "generator_source_sha256",
        "local_contract_sha256",
        "pipeline_revision",
        "rights",
        "schema_version",
        "seed",
    }
    if set(manifest) != expected_root:
        errors.append("input_manifest_root_fields")
        return errors
    if manifest.get("schema_version") != INPUT_SCHEMA:
        errors.append("input_manifest_schema")
    if manifest.get("pipeline_revision") != PIPELINE_REVISION:
        errors.append("pipeline_revision")
    rights = manifest.get("rights")
    if not isinstance(rights, dict) or rights.get("external_image_asset_count") != 0:
        errors.append("external_asset_rights")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        errors.append("cases_not_list")
        return errors
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "height",
            "paths",
            "sha256",
            "width",
        }:
            errors.append("case_fields")
            continue
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            errors.append("case_id_unique")
        seen.add(case_id)
        paths = case.get("paths")
        hashes = case.get("sha256")
        if not isinstance(paths, dict) or set(paths) != {"reference", "candidate", "rerun"}:
            errors.append("case_paths")
            continue
        if not isinstance(hashes, dict) or set(hashes) != set(paths):
            errors.append("case_hashes")
            continue
        for role, raw_path in paths.items():
            if not isinstance(raw_path, str):
                errors.append("case_path_type")
                continue
            relative = PurePosixPath(raw_path)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append("case_path_unsafe")
            digest = hashes.get(role)
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append("case_hash_format")
    return sorted(set(errors))


def _load_and_verify_images(
    fixture_root: Path, case: Mapping[str, Any]
) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    for role in ("reference", "candidate", "rerun"):
        path = fixture_root / case["paths"][role]
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing_or_symlink_input")
        if sha256_file(path) != case["sha256"][role]:
            raise ValueError("input_hash_mismatch")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("image_decode_failure")
        expected_shape = (int(case["height"]), int(case["width"]), 3)
        if image.shape != expected_shape or image.dtype != np.uint8:
            raise ValueError("image_shape_or_dtype")
        images[role] = image
    return images


def _raw_changed_fraction(reference: np.ndarray, candidate: np.ndarray, threshold: int = 8) -> Tuple[int, float]:
    difference = cv2.absdiff(reference, candidate)
    mask = np.max(difference, axis=2) > threshold
    count = int(np.count_nonzero(mask))
    return count, count / float(mask.size)


def _estimate_translation(
    reference: np.ndarray,
    candidate: np.ndarray,
    downsample: int,
) -> Tuple[float, float, float]:
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    small_size = (reference.shape[1] // downsample, reference.shape[0] // downsample)
    small_reference = cv2.resize(gray_reference, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    small_candidate = cv2.resize(gray_candidate, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    shift, response = cv2.phaseCorrelate(small_reference, small_candidate)
    return shift[0] * downsample, shift[1] * downsample, float(response)


def _align_candidate(
    candidate: np.ndarray,
    shift_x: float,
    shift_y: float,
    response: float,
    config: Mapping[str, Any],
) -> Tuple[np.ndarray, int, int, bool]:
    max_translation = int(config["max_alignment_translation_pixels"])
    response_min = float(config["alignment_response_min"])
    dx = int(round(shift_x))
    dy = int(round(shift_y))
    usable = (
        response >= response_min
        and abs(dx) <= max_translation
        and abs(dy) <= max_translation
    )
    if not usable:
        return candidate, 0, 0, False
    transform = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        candidate,
        transform,
        (candidate.shape[1], candidate.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(244, 246, 248),
    )
    return aligned, dx, dy, dx != 0 or dy != 0


def _localized_regions(
    reference: np.ndarray,
    candidate: np.ndarray,
    config: Mapping[str, Any],
) -> Tuple[List[List[int]], int, float]:
    lab_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    lab_candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB)
    lab_difference = cv2.absdiff(lab_reference, lab_candidate)
    color_mask = np.max(lab_difference, axis=2) > int(config["lab_difference_threshold"])
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    edges_reference = cv2.Canny(gray_reference, 60, 140)
    edges_candidate = cv2.Canny(gray_candidate, 60, 140)
    edge_mask = cv2.absdiff(edges_reference, edges_candidate) > 0
    mask = np.where(color_mask | edge_mask, 255, 0).astype(np.uint8)
    border = int(config["border_ignore_pixels"])
    if border:
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = int(config["minimum_component_area_pixels"])
    regions: List[List[int]] = []
    significant_pixels = 0
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < minimum_area:
            continue
        regions.append([x, y, x + width, y + height])
        significant_pixels += area
    regions.sort()
    fraction = significant_pixels / float(mask.size)
    return regions, significant_pixels, fraction


def _diagnostic_action(
    changed_fraction: float,
    alignment_used: bool,
    config: Mapping[str, Any],
) -> Tuple[str, str]:
    if changed_fraction <= float(config["no_change_fraction_max"]):
        return "rerun_visual_test", "aligned_or_low_evidence_difference"
    if changed_fraction >= float(config["critical_changed_fraction_min"]):
        return "set_deploy_gate_blocked", "large_stable_structural_difference"
    return "inspect_region", "localized_stable_difference"


def _unknown_case(
    case: Mapping[str, Any],
    runtime: Mapping[str, Any],
    error_code: str,
) -> Dict[str, Any]:
    return {
        "case_id": case.get("case_id", "invalid-case"),
        "decision": {
            "action": "request_human_approval",
            "deploy_gate": "blocked",
            "reason": error_code,
            "state": "UNKNOWN",
            "steps": ["validate_inputs", "request_human_approval"],
        },
        "diagnostic": None,
        "input_sha256": case.get("sha256", {}),
        "runtime_compatible": runtime["competition_opencv_major_satisfied"],
    }


def analyze_case(
    fixture_root: Path,
    case: Mapping[str, Any],
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        images = _load_and_verify_images(fixture_root, case)
        config = contract["perception"]
        raw_count, raw_fraction = _raw_changed_fraction(
            images["reference"], images["candidate"]
        )
        shift_x, shift_y, response = _estimate_translation(
            images["reference"],
            images["candidate"],
            int(config["alignment_downsample"]),
        )
        aligned, dx, dy, alignment_used = _align_candidate(
            images["candidate"], shift_x, shift_y, response, config
        )
        regions, changed_pixels, changed_fraction = _localized_regions(
            images["reference"], aligned, config
        )
        diagnostic_action, diagnostic_reason = _diagnostic_action(
            changed_fraction, alignment_used, config
        )
        rerun_count, rerun_fraction = _raw_changed_fraction(
            images["reference"], images["rerun"]
        )
        false_positive_threshold = float(config["false_positive_pixel_fraction"])
        diagnostic = {
            "action": diagnostic_action,
            "action_reason": diagnostic_reason,
            "aligned_changed_fraction": round(changed_fraction, 9),
            "aligned_changed_pixel_count": changed_pixels,
            "alignment": {
                "applied": alignment_used,
                "dx": dx,
                "dy": dy,
                "phase_response": round(response, 8),
            },
            "predicted_regions_xyxy_half_open": regions,
            "raw_changed_fraction": round(raw_fraction, 9),
            "raw_changed_pixel_count": raw_count,
            "raw_false_positive": raw_fraction > false_positive_threshold,
            "rerun_changed_fraction": round(rerun_fraction, 9),
            "rerun_changed_pixel_count": rerun_count,
            "rerun_false_positive": rerun_fraction > false_positive_threshold,
        }
        if not runtime["competition_opencv_major_satisfied"]:
            decision = {
                "action": "request_human_approval",
                "deploy_gate": "blocked",
                "reason": "required_opencv_major_unavailable",
                "state": "UNKNOWN",
                "steps": ["inspect_pair_diagnostic", "request_human_approval"],
            }
        else:
            decision = {
                "action": diagnostic_action,
                "deploy_gate": "blocked",
                "reason": diagnostic_reason,
                "state": "DECIDED",
                "steps": ["inspect_pair", diagnostic_action],
            }
        return {
            "case_id": case["case_id"],
            "decision": decision,
            "diagnostic": diagnostic,
            "input_sha256": case["sha256"],
            "runtime_compatible": runtime["competition_opencv_major_satisfied"],
        }
    except Exception as exc:
        return _unknown_case(case, runtime, f"input_or_perception_{type(exc).__name__}")


def runtime_evidence(contract: Mapping[str, Any]) -> Dict[str, Any]:
    required_major = int(contract["runtime_contract"]["required_opencv_major"])
    observed_major = int(cv2.__version__.split(".", 1)[0])
    return {
        "competition_opencv_major_satisfied": observed_major == required_major,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "required_opencv_major": required_major,
    }


def run_evidence(
    fixture_root: Path,
    input_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[float]]:
    manifest_errors = validate_input_manifest(input_manifest)
    if manifest_errors:
        raise ValueError("input manifest failed closed")
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(int(contract["determinism"]["seed"]))
    runtime = runtime_evidence(contract)
    cases = []
    latencies = []
    for case in input_manifest["cases"]:
        start = time.perf_counter_ns()
        evidence = analyze_case(fixture_root, case, contract, runtime)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000_000.0
        cases.append(evidence)
        latencies.append(elapsed)
    root_status = (
        "DIAGNOSTIC_ONLY_RUNTIME_MISMATCH"
        if not runtime["competition_opencv_major_satisfied"]
        else "COMPETITION_RUNTIME"
    )
    evidence_root = {
        "cases": cases,
        "fixture_input_manifest_sha256": sha256_file(fixture_root / "input_manifest.json"),
        "pipeline_revision": contract["pipeline_revision"],
        "runtime": runtime,
        "schema_version": EVIDENCE_SCHEMA,
        "status": root_status,
    }
    return evidence_root, latencies
