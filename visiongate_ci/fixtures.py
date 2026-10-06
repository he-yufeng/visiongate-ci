"""Generate the 60-pair rights-clean synthetic VisionGate fixture set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from . import PIPELINE_REVISION
from .canonical import canonical_json_bytes, sha256_file, write_canonical


ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = "visiongate-ci-fixture-inputs/v1"
TRUTH_SCHEMA = "visiongate-ci-fixture-ground-truth/v1"
FIXTURE_MANIFEST_SCHEMA = "visiongate-ci-fixture-manifest/v1"


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _rectangle(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = -1,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, thickness, cv2.LINE_8)


def _base_ui(seed: int, width: int, height: int) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], Tuple[int, int, int, int]]:
    rng = np.random.default_rng(seed)
    page = (244, 246, 248)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = page
    _rectangle(image, (0, 0, width, 104), (60, 52, 46))
    _rectangle(image, (0, 104, 260, height), (232, 235, 238))
    for row in range(7):
        y = 150 + row * 92
        _rectangle(image, (34, y, 208, y + 18), (170, 176, 181))
        _rectangle(image, (34, y + 29, 150, y + 39), (198, 202, 206))
    for index in range(6):
        x = 36 + index * 54
        cv2.circle(image, (x, 52), 13, (120 + index * 8, 170, 220 - index * 7), -1, cv2.LINE_8)

    badges: List[Tuple[int, int, int, int]] = []
    for row in range(2):
        for column in range(3):
            x1 = 315 + column * 500
            y1 = 160 + row * 330
            x2 = x1 + 430
            y2 = y1 + 270
            shade = int(rng.integers(247, 253))
            _rectangle(image, (x1, y1, x2, y2), (shade, shade, shade))
            cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (205, 209, 213), 3, cv2.LINE_8)
            accent = tuple(int(v) for v in rng.integers(80, 210, size=3))
            cv2.circle(image, (x1 + 66, y1 + 72), 34, accent, -1, cv2.LINE_8)
            for line in range(4):
                line_width = int(rng.integers(150, 285))
                _rectangle(
                    image,
                    (x1 + 120, y1 + 43 + line * 37, x1 + 120 + line_width, y1 + 55 + line * 37),
                    (145 + line * 10, 150 + line * 10, 155 + line * 10),
                )
            badge = (x2 - 106, y1 + 204, x2 - 28, y1 + 246)
            badges.append(badge)
            _rectangle(image, badge, (92, 174, 117))

    critical = (1570, 928, 1870, 1024)
    _rectangle(image, critical, (42, 74, 218))
    _rectangle(image, (1628, 960, 1812, 976), (240, 244, 250))
    return image, badges, critical


def _write_png(path: Path, image: np.ndarray, compression: int) -> None:
    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, compression, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to write a fixture PNG")


def _make_case(
    case_id: str,
    index: int,
    stratum: str,
    seed: int,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    reference, badges, critical = _base_ui(seed + index * 1009, width, height)
    candidate = reference.copy()
    rerun = reference.copy()
    ground_truth_regions: List[List[int]] = []
    if stratum == "clean_or_nuisance":
        local_rng = np.random.default_rng(seed + index * 7919)
        choices = np.array([-3, -2, 2, 3])
        dx = int(local_rng.choice(choices))
        dy = int(local_rng.choice(choices))
        transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        candidate = cv2.warpAffine(
            reference,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(244, 246, 248),
        )
        expected_action = "rerun_visual_test"
        generation = {"kind": "integer_translation", "dx": dx, "dy": dy}
    elif stratum == "minor_regression":
        box = badges[index % len(badges)]
        _rectangle(candidate, box, (189, 102, 74))
        rerun = candidate.copy()
        ground_truth_regions = [list(box)]
        expected_action = "inspect_region"
        generation = {"kind": "localized_badge_color_regression"}
    elif stratum == "critical_regression":
        _rectangle(candidate, critical, (244, 246, 248))
        rerun = candidate.copy()
        ground_truth_regions = [list(critical)]
        expected_action = "set_deploy_gate_blocked"
        generation = {"kind": "critical_control_deletion"}
    else:
        raise ValueError(f"unsupported stratum: {stratum}")
    truth = {
        "case_id": case_id,
        "expected_action": expected_action,
        "ground_truth_regions_xyxy_half_open": ground_truth_regions,
        "stratum": stratum,
        "synthetic_generation": generation,
    }
    return reference, candidate, rerun, truth


def generate(output: Path, contract_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    fixture = contract["fixture_contract"]
    seed = int(contract["determinism"]["seed"])
    pair_count = int(fixture["initial_pairs"])
    strata: List[str] = []
    for name, count in sorted(fixture["strata"].items()):
        strata.extend([name] * int(count))
    if len(strata) != pair_count:
        raise ValueError("fixture strata do not sum to initial_pairs")
    rng = np.random.default_rng(seed)
    rng.shuffle(strata)
    width = int(fixture["width"])
    height = int(fixture["height"])
    compression = int(fixture["png_compression"])
    if output.exists():
        raise FileExistsError("fixture output already exists; generation is non-destructive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        images_dir = temporary / "images"
        images_dir.mkdir()
        input_cases: List[Dict[str, Any]] = []
        truth_cases: List[Dict[str, Any]] = []
        image_manifest_lines: List[str] = []
        for index, stratum in enumerate(strata):
            case_id = f"case-{index:03d}"
            reference, candidate, rerun, truth = _make_case(
                case_id, index, stratum, seed, width, height
            )
            paths = {
                "candidate": f"images/{case_id}-candidate.png",
                "reference": f"images/{case_id}-reference.png",
                "rerun": f"images/{case_id}-rerun.png",
            }
            images = {"candidate": candidate, "reference": reference, "rerun": rerun}
            hashes: Dict[str, str] = {}
            for role in sorted(paths):
                path = temporary / paths[role]
                _write_png(path, images[role], compression)
                hashes[role] = sha256_file(path)
                image_manifest_lines.append(f"{hashes[role]}  {paths[role]}")
            input_cases.append(
                {
                    "case_id": case_id,
                    "height": height,
                    "paths": paths,
                    "sha256": hashes,
                    "width": width,
                }
            )
            truth_cases.append(truth)

        generator_hash = sha256_file(Path(__file__))
        contract_hash = sha256_file(contract_path)
        inputs = {
            "cases": input_cases,
            "fixture_revision": fixture["generator_revision"],
            "generator_source_sha256": generator_hash,
            "local_contract_sha256": contract_hash,
            "pipeline_revision": contract["pipeline_revision"],
            "rights": {
                "external_image_asset_count": 0,
                "statement": fixture["artifact_license_basis"],
            },
            "schema_version": INPUT_SCHEMA,
            "seed": seed,
        }
        truth = {
            "cases": truth_cases,
            "fixture_revision": fixture["generator_revision"],
            "pipeline_revision": contract["pipeline_revision"],
            "schema_version": TRUTH_SCHEMA,
            "seed": seed,
        }
        write_canonical(temporary / "input_manifest.json", inputs)
        write_canonical(temporary / "ground_truth.json", truth)
        manifest_entries = image_manifest_lines + [
            f"{sha256_file(temporary / 'input_manifest.json')}  input_manifest.json",
            f"{sha256_file(temporary / 'ground_truth.json')}  ground_truth.json",
        ]
        (temporary / "MANIFEST.sha256").write_text(
            "\n".join(sorted(manifest_entries, key=lambda line: line.split("  ", 1)[1])) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        temporary = Path()
    finally:
        if temporary != Path() and temporary.exists():
            shutil.rmtree(temporary)
    return inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures")
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "contracts" / "local_baseline_contract.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate(args.output, args.contract)
    except Exception as exc:
        print(f"FIXTURE_GENERATION_FAIL {type(exc).__name__}")
        return 1
    print(
        f"FIXTURE_GENERATION_PASS pairs={len(manifest['cases'])} "
        f"external_assets={manifest['rights']['external_image_asset_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
