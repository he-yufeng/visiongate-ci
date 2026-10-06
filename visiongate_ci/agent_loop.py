"""Image-driven executable tool branches; sandbox gate only, no live deployment."""
from pathlib import Path

import cv2
import numpy as np

from .canonical import sha256_file
from .perception import (_align_candidate, _diagnostic_action, _estimate_translation,
                         _localized_regions, _raw_changed_fraction, runtime_evidence)


def _read_image(root, case, role):
    root = Path(root).resolve()
    path = root / case["paths"][role]
    path.resolve().relative_to(root)
    if path.is_symlink() or sha256_file(path) != case["sha256"][role]:
        raise ValueError("image integrity failure")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (case["height"], case["width"], 3) or image.dtype != np.uint8:
        raise ValueError("invalid image")
    return image


def _tool(trace, name, output):
    if len(trace) >= 4:
        raise RuntimeError("four-step budget")
    trace.append({"step": len(trace), "tool": name, "output": output})


def run_case(root, case, contract):
    """No scoring labels accepted; future calls depend on actual image results."""
    trace = []
    state = {"deploy_gate": "blocked", "sandbox_only": True}
    try:
        runtime = runtime_evidence(contract)
        if not runtime["competition_opencv_major_satisfied"]:
            raise ValueError("runtime mismatch")
        reference = _read_image(root, case, "reference")
        candidate = _read_image(root, case, "candidate")
        config = contract["perception"]
        sx, sy, response = _estimate_translation(reference, candidate, config["alignment_downsample"])
        aligned, dx, dy, used = _align_candidate(candidate, sx, sy, response, config)
        regions, count, fraction = _localized_regions(reference, aligned, config)
        action, reason = _diagnostic_action(fraction, used, config)
        _tool(trace, "inspect_pair", {"regions": regions, "changed_pixels": count,
              "changed_fraction": fraction, "alignment": [dx, dy], "reason": reason})
        if action == "rerun_visual_test":
            # The third capture is read and analyzed only after this branch is selected.
            # This pilot uses pre-captured synthetic input, not a live browser recapture.
            rerun = _read_image(root, case, "rerun")
            changed, rerun_fraction = _raw_changed_fraction(reference, rerun)
            _tool(trace, "rerun_visual_test", {"changed_pixels": changed,
                  "changed_fraction": rerun_fraction,
                  "capture_source": "pre_captured_synthetic_fixture"})
            state["release_recommended"] = rerun_fraction <= config["no_change_fraction_max"]
            if not state["release_recommended"]:
                state["human_review_requested"] = True
                _tool(trace, "request_human_approval", {"deploy_gate": "blocked"})
        elif action == "inspect_region":
            if not regions:
                raise ValueError("minor branch without usable regions")
            # Execute all region inspections; never select only a favourable crop.
            inspected = []
            for x1, y1, x2, y2 in regions:
                crop_difference = cv2.absdiff(reference[y1:y2, x1:x2], aligned[y1:y2, x1:x2])
                inspected.append({"box": [x1, y1, x2, y2],
                    "changed_pixels": int(np.count_nonzero(np.max(crop_difference, axis=2) > 8))})
            _tool(trace, "inspect_region", {"regions": inspected})
            state["human_review_requested"] = True
            _tool(trace, "request_human_approval", {"deploy_gate": "blocked"})
        elif action == "set_deploy_gate_blocked":
            state["critical_evidence"] = True
            state["deploy_gate"] = "blocked"
            _tool(trace, "set_deploy_gate_blocked", dict(state))
        else:
            raise ValueError("unknown action")
        return {"case_id": case["case_id"], "first_selected_action": action,
                "status": "DECIDED", "trace": trace, "state": state}
    except Exception as error:
        state["human_review_requested"] = True
        if len(trace) < 4:
            _tool(trace, "request_human_approval", {"error_class": type(error).__name__, "deploy_gate": "blocked"})
        return {"case_id": case.get("case_id"), "first_selected_action": "request_human_approval",
                "status": "UNKNOWN", "trace": trace, "state": state}
