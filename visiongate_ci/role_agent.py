"""Baseline-role protection plus image-triggered capture checking; sandbox only."""
import time

import cv2
import numpy as np

from .agent_loop import _read_image, _tool
from .perception import (_align_candidate, _diagnostic_action, _estimate_translation,
                         _localized_regions, _raw_changed_fraction, runtime_evidence)


def validate_roles(case, declared, policy):
    if set(declared) != {"schema", "case_id", "reference_sha256", "protected_components"}:
        raise ValueError("undeclared role-contract fields")
    if declared["schema"] != "visiongate-baseline-components/v1":
        raise ValueError("role schema")
    if declared["case_id"] != case["case_id"] or declared["reference_sha256"] != case["sha256"]["reference"]:
        raise ValueError("baseline binding mismatch")
    parts = declared["protected_components"]
    if not isinstance(parts, list) or len(parts) != len(policy["required_roles"]):
        raise ValueError("incomplete protected roles")
    roles = []
    for part in parts:
        if set(part) != {"role", "box"}:
            raise ValueError("component fields")
        roles.append(part["role"])
        box = part["box"]
        if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError("component bounds")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= case["width"] and 0 <= y1 < y2 <= case["height"]):
            raise ValueError("out-of-image component")
    if len(set(roles)) != len(roles) or set(roles) != set(policy["required_roles"]):
        raise ValueError("incorrect protected role set")
    return parts


def inspect_capture(reference, image, components, local, policy):
    _, raw_fraction = _raw_changed_fraction(reference, image)
    sx, sy, response = _estimate_translation(reference, image, local["alignment_downsample"])
    aligned, dx, dy, applied = _align_candidate(image, sx, sy, response, local)
    regions, count, fraction = _localized_regions(reference, aligned, local)
    mask = np.max(cv2.absdiff(reference, aligned), axis=2) > policy["pixel_difference_threshold"]
    hits = []
    outside = mask.copy()
    for component in components:
        x1, y1, x2, y2 = component["box"]
        changed = int(np.count_nonzero(mask[y1:y2, x1:x2]))
        if changed >= policy["protected_changed_pixels_min"]:
            hits.append({"role": component["role"], "box": component["box"], "changed_pixels": changed})
        outside[y1:y2, x1:x2] = False
    border = local["border_ignore_pixels"]
    outside[:border, :] = False; outside[-border:, :] = False
    outside[:, :border] = False; outside[:, -border:] = False
    rows, cols = policy["capture_quality_grid"]
    y_edges = np.linspace(0, reference.shape[0], rows + 1, dtype=int)
    x_edges = np.linspace(0, reference.shape[1], cols + 1, dtype=int)
    distributed = sum(int(np.count_nonzero(outside[y_edges[y]:y_edges[y+1], x_edges[x]:x_edges[x+1]]))
                      >= policy["capture_quality_tile_changed_pixels_min"]
                      for y in range(rows) for x in range(cols))
    return {"regions": regions, "changed_pixels": count, "changed_fraction": fraction,
            "raw_difference_fraction": raw_fraction,
            "alignment": [dx, dy], "alignment_applied": applied,
            "protected_hits": hits, "distributed_changed_tiles": int(distributed),
            "capture_quality_requires_check": distributed >= policy["capture_quality_distributed_tiles_min"]}, aligned


def run_case(root, case, declared, contract, policy):
    """No mutation labels accepted; baseline roles are not severity labels."""
    trace = []
    state = {"deploy_gate": "blocked", "sandbox_only": True, "release_recommended": False}
    first_action = "request_human_approval"
    effective = {"regions": [], "protected_hits": [], "raw_difference_fraction": None}
    try:
        if not runtime_evidence(contract)["competition_opencv_major_satisfied"]:
            raise ValueError("runtime mismatch")
        components = validate_roles(case, declared, policy)
        reference = _read_image(root, case, "reference")
        candidate = _read_image(root, case, "candidate")
        local = contract["perception"]
        effective, aligned = inspect_capture(reference, candidate, components, local, policy)
        _tool(trace, "inspect_pair", effective)
        fallback, _ = _diagnostic_action(effective["changed_fraction"], effective["alignment_applied"], local)
        if effective["capture_quality_requires_check"]:
            first_action = "rerun_visual_test"
        elif effective["protected_hits"]:
            first_action = "set_deploy_gate_blocked"
        else:
            first_action = fallback
        final_action = first_action
        if first_action == "rerun_visual_test":
            rerun = _read_image(root, case, "rerun")
            effective, aligned = inspect_capture(reference, rerun, components, local, policy)
            _tool(trace, "rerun_visual_test", {**effective, "capture_source": "pre_captured_synthetic_fixture"})
            if effective["capture_quality_requires_check"]:
                state["human_review_requested"] = True
                final_action = "request_human_approval"
                _tool(trace, final_action, {"reason": "recapture_still_distributed_or_unstable", "deploy_gate": "blocked"})
            elif effective["protected_hits"]:
                final_action = "set_deploy_gate_blocked"
            else:
                final_action, _ = _diagnostic_action(effective["changed_fraction"], effective["alignment_applied"], local)
                if final_action == "rerun_visual_test":
                    state["release_recommended"] = True
        if final_action == "set_deploy_gate_blocked":
            state["critical_evidence"] = True
            _tool(trace, final_action, {"protected_hits": effective["protected_hits"], "deploy_gate": "blocked"})
        elif final_action == "inspect_region":
            if not effective["regions"]:
                raise ValueError("minor decision without regions")
            inspected = []
            for x1, y1, x2, y2 in effective["regions"]:
                delta = cv2.absdiff(reference[y1:y2, x1:x2], aligned[y1:y2, x1:x2])
                inspected.append({"box": [x1,y1,x2,y2], "changed_pixels": int(np.count_nonzero(np.max(delta,axis=2)>8))})
            _tool(trace, "inspect_region", {"regions": inspected})
            state["human_review_requested"] = True
            _tool(trace, "request_human_approval", {"deploy_gate": "blocked"})
        return {"case_id": case["case_id"], "first_selected_action": first_action,
                "final_action": final_action, "status": "DECIDED", "trace": trace, "state": state,
                "effective_regions": effective["regions"], "effective_raw_fraction": effective["raw_difference_fraction"]}
    except Exception as error:
        state["release_recommended"] = False
        state["human_review_requested"] = True
        if len(trace) < 4:
            _tool(trace, "request_human_approval", {"error_class": type(error).__name__, "deploy_gate": "blocked"})
        return {"case_id": case.get("case_id"), "first_selected_action": first_action,
                "final_action": "request_human_approval", "status": "UNKNOWN", "trace": trace,
                "state": state, "effective_regions": [], "effective_raw_fraction": None}


def run_batch(root, cases, roles, contract, policy):
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(int(contract["determinism"]["seed"]))
    results, latencies = [], []
    for case in cases:
        started = time.perf_counter_ns()
        results.append(run_case(root, case, roles[case["case_id"]], contract, policy))
        latencies.append((time.perf_counter_ns()-started)/1e9)
    return {"cases": results}, latencies
