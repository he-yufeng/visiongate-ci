"""Independent authored stress family; ephemeral images and separate labels only."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from visiongate_ci import PIPELINE_REVISION
from visiongate_ci.canonical import sha256_file, write_canonical
from visiongate_ci.fixtures import INPUT_SCHEMA


def rect(image, box, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2-1, y2-1), color, -1, cv2.LINE_8)


def text(image, value, xy, color=(65, 65, 65), size=.75):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_8)


def layout(rng, family):
    page = np.full((1080, 1920, 3), 246, np.uint8)
    rect(page, (0, 0, 1920, 90), (42, 53, 61))
    text(page, "Aurora - isolated visual release check", (34, 54), (238, 238, 238), .95)
    accent = tuple(int(x) for x in rng.integers(75, 185, 3))
    badge = (1530, 150, 1640, 196)
    primary = (1490, 915, 1800, 1010)
    icon = (1425, 934, 1447, 956)
    rect(page, badge, accent)
    text(page, "READY", (1540, 180), (250, 250, 250), .55)
    rect(page, primary, (45, 87, 208))
    text(page, "DEPLOY", (1550, 975), (255, 255, 255), 1.1)
    rect(page, icon, (30, 35, 200))
    if family == "kanban":
        for column in range(3):
            x = 90 + column * 550
            text(page, ("BACKLOG", "IN REVIEW", "VERIFIED")[column], (x, 155))
            for row in range(4):
                y = 215 + row * 154
                rect(page, (x, y, x+460, y+123), (225+row, 230, 236))
                text(page, f"Build {int(rng.integers(100,999))} - task {row}", (x+18, y+34), size=.65)
                rect(page, (x+18, y+58, x+330, y+68), accent)
                rect(page, (x+18, y+83, x+230, y+92), (164, 172, 180))
    elif family == "table":
        for row in range(12):
            y = 220 + row * 52
            rect(page, (74, y, 1830, y+43), (238, 239, 241) if row%2 else (222, 229, 235))
            for col in range(5):
                text(page, f"R{row}-C{col}-{int(rng.integers(10,99))}", (100+col*335, y+28), size=.6)
    elif family == "editor":
        rect(page, (60, 212, 1210, 837), (51, 57, 64))
        for row in range(18):
            text(page, f"{row+1:02}  verify(stage_{int(rng.integers(10,99))}, safe=True)",
                 (90, 248+row*31), (197, 220, 204), .65)
        for row in range(6):
            y = 225+row*98
            rect(page, (1280, y, 1810, y+70), (219, 229, 237))
            text(page, f"Control {row} = pending", (1300, y+40), size=.7)
    else:
        raise ValueError("unsupported layout")
    return page, badge, primary, icon


def generate(input_root, label_root, contract_path, generation_seed):
    input_root, label_root = Path(input_root), Path(label_root)
    input_root.mkdir(mode=0o700); label_root.mkdir(mode=0o700)
    (input_root / "images").mkdir()
    contract = json.loads(Path(contract_path).read_text())
    rng = np.random.default_rng(generation_seed)
    # Every family has100 cases; cross-family stratum totals are exactly100 each.
    work = []
    for family, counts in zip(("kanban", "table", "editor"), ((34,33,33),(33,34,33),(33,33,34))):
        for stratum, count in zip(("clean_or_nuisance","minor_regression","critical_regression"), counts):
            work.extend((family,stratum,i) for i in range(count))
    rng.shuffle(work)
    cases, labels = [], []
    for position, (family, stratum, index) in enumerate(work):
        case_id = f"stress-{position:03}"
        reference, badge, primary, icon = layout(rng, family)
        candidate = reference.copy(); rerun = reference.copy(); boxes = []
        if stratum == "clean_or_nuisance":
            mode = index % 3
            if mode == 0:
                dx, dy = int(rng.choice([-6,-4,4,6])), int(rng.choice([-6,-4,4,6]))
                candidate = cv2.warpAffine(reference, np.array([[1,0,dx],[0,1,dy]], np.float32),
                                          (1920,1080), flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(246,246,246))
                kind = "capture_translation"
            elif mode == 1:
                candidate = cv2.GaussianBlur(reference, (3,3), .7)
                kind = "capture_blur"
            else:
                ok, encoded = cv2.imencode(".jpg", reference, [cv2.IMWRITE_JPEG_QUALITY, 72])
                assert ok
                candidate = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                kind = "capture_jpeg"
            expected_action = "rerun_visual_test"
        elif stratum == "minor_regression":
            rect(candidate, badge, (180,85,70))
            text(candidate, "READY", (1540,180), (250,250,250), .55)
            boxes = [list(badge)]; kind = "local_status_color"
            rerun = candidate.copy(); expected_action = "inspect_region"
        else:
            mode = index % 3
            if mode == 0:
                rect(candidate, primary, (246,246,246)); boxes = [list(primary)]
                kind = "large_deploy_control_deleted"
            elif mode == 1:
                # Small critical control: severity is not a function of image area.
                small = (1730,930,1820,962)
                rect(reference, small, (22,48,190)); text(reference,"GO",(1745,953),(255,255,255),.5)
                candidate = reference.copy(); rect(candidate, small, (246,246,246))
                boxes = [list(small)]; kind = "small_deploy_control_deleted"
            else:
                rect(candidate, icon, (246,246,246)); boxes = [list(icon)]
                kind = "critical_warning_icon_deleted"
            rerun = candidate.copy(); expected_action = "set_deploy_gate_blocked"
        paths, digests = {}, {}
        for role, image in (("reference",reference),("candidate",candidate),("rerun",rerun)):
            name = f"images/{case_id}-{role}.png"; path = input_root / name
            assert cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION,9])
            paths[role] = name; digests[role] = sha256_file(path)
        cases.append({"case_id":case_id,"height":1080,"width":1920,"paths":paths,"sha256":digests})
        labels.append({"case_id":case_id,"family":family,"stratum":stratum,"kind":kind,
                       "expected_action":expected_action,"boxes":boxes})
    manifest = {"schema_version":INPUT_SCHEMA,"pipeline_revision":PIPELINE_REVISION,
                "fixture_revision":"visiongate-independent-authored-stress/v1",
                "generator_source_sha256":sha256_file(Path(__file__)),
                "local_contract_sha256":sha256_file(Path(contract_path)),
                "seed":contract["determinism"]["seed"],
                "rights":{"external_image_asset_count":0,"statement":"Owned geometric/Hershey primitives; runtime internal only"},
                "cases":cases}
    write_canonical(input_root / "input_manifest.json", manifest)
    write_canonical(label_root / "labels.json", {"generation_seed":generation_seed,"labels":labels,
                    "source_sha256":sha256_file(Path(__file__))})
    return {"cases":300,"input_manifest_sha256":sha256_file(input_root / "input_manifest.json"),
            "label_file_sha256":sha256_file(label_root / "labels.json")}
