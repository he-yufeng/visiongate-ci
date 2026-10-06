"""Fresh role-invariant baselines; roles exist in every stratum before mutation."""
import json
from pathlib import Path

import cv2
import numpy as np

from stress_generator import layout, rect, text
from visiongate_ci import PIPELINE_REVISION
from visiongate_ci.canonical import sha256_file, write_canonical
from visiongate_ci.fixtures import INPUT_SCHEMA


def baseline(rng, family):
    reference, badge, old_primary, old_icon = layout(rng, family)
    rect(reference, old_primary, (246,246,246)); rect(reference, old_icon, (246,246,246))
    dx, dy = int(rng.integers(-25,26)), int(rng.integers(-8,9))
    primary = [1490+dx,915+dy,1800+dx,1010+dy]
    secondary = [1740+dx,853+dy,1830+dx,885+dy]
    warning = [1425+dx,934+dy,1447+dx,956+dy]
    rect(reference, primary, (45,87,208))
    text(reference,"DEPLOY",(primary[0]+60,primary[1]+60),(255,255,255),1.1)
    rect(reference, secondary, (22,48,190))
    text(reference,"GO",(secondary[0]+15,secondary[1]+23),(255,255,255),.5)
    rect(reference, warning, (30,35,200))
    components = [{"role":role,"box":box} for role,box in
                  (("primary_deploy",primary),("secondary_deploy",secondary),("critical_warning",warning))]
    return reference, list(badge), components


def distort(image, mode, rng):
    if mode == 0:
        dx,dy = int(rng.choice([-6,-4,4,6])),int(rng.choice([-6,-4,4,6]))
        return cv2.warpAffine(image,np.array([[1,0,dx],[0,1,dy]],np.float32),(1920,1080),
                             flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=(246,246,246))
    if mode == 1:
        return cv2.GaussianBlur(image,(3,3),.7)
    ok, encoded = cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,72])
    assert ok
    return cv2.imdecode(encoded,cv2.IMREAD_COLOR)


def generate(input_root, label_root, contract_path, generation_seed):
    input_root,label_root = Path(input_root),Path(label_root)
    input_root.mkdir(mode=0o700); label_root.mkdir(mode=0o700)
    (input_root/"images").mkdir()
    contract=json.loads(Path(contract_path).read_text())
    streams=np.random.SeedSequence(generation_seed).spawn(601)
    schedule=np.random.default_rng(streams[0])
    work=[]
    for family,counts in zip(("kanban","table","editor"),((34,33,33),(33,34,33),(33,33,34))):
        for stratum,count in zip(("clean_or_nuisance","minor_regression","critical_regression"),counts):
            work.extend((family,stratum,i) for i in range(count))
    schedule.shuffle(work)
    cases,labels,roles=[],[],[]
    for position,(family,stratum,index) in enumerate(work):
        base_rng=np.random.default_rng(streams[1+2*position])
        mutation_rng=np.random.default_rng(streams[2+2*position])
        # baseline() cannot see the stratum, index, mutation or labels.
        reference,badge,components=baseline(base_rng,family)
        candidate=reference.copy(); boxes=[]
        case_id=f"role-{position:03}"
        if stratum=="clean_or_nuisance":
            candidate=distort(reference,index%3,mutation_rng)
            rerun=reference.copy()
            kind=("capture_translation","capture_blur","capture_jpeg")[index%3]
            expected_first=expected_final="rerun_visual_test"
        else:
            if stratum=="minor_regression":
                rect(candidate,badge,(210,25,20))
                text(candidate,"READY",(1540,180),(250,250,250),.55)
                boxes=[badge]; kind="local_status_color"
                expected_first=expected_final="inspect_region"
            else:
                removed=components[index%3]
                rect(candidate,removed["box"],(246,246,246))
                boxes=[removed["box"]]; kind=removed["role"]+"_deleted"
                expected_first=expected_final="set_deploy_gate_blocked"
            rerun=candidate.copy()
            # Compound regressions require capture checking but must stay blocked
            # or reviewed after recapture; nuisance is never permission to release.
            if index%4 in (2,3):
                mode=1 if index%4==2 else 2
                candidate=distort(candidate,mode,mutation_rng)
                kind += ("_with_blur" if mode==1 else "_with_jpeg")
                expected_first="rerun_visual_test"
        paths,digests={},{}
        for role,image in (("reference",reference),("candidate",candidate),("rerun",rerun)):
            name=f"images/{case_id}-{role}.png"; path=input_root/name
            assert cv2.imwrite(str(path),image,[cv2.IMWRITE_PNG_COMPRESSION,9])
            paths[role]=name; digests[role]=sha256_file(path)
        cases.append({"case_id":case_id,"height":1080,"width":1920,"paths":paths,"sha256":digests})
        roles.append({"schema":"visiongate-baseline-components/v1","case_id":case_id,
                      "reference_sha256":digests["reference"],"protected_components":components})
        labels.append({"case_id":case_id,"family":family,"stratum":stratum,"kind":kind,
                       "expected_first":expected_first,"expected_final":expected_final,"boxes":boxes})
    write_canonical(input_root/"input_manifest.json",{
        "schema_version":INPUT_SCHEMA,"pipeline_revision":PIPELINE_REVISION,
        "fixture_revision":"visiongate-role-invariant-stress/v2",
        "generator_source_sha256":sha256_file(Path(__file__)),"local_contract_sha256":sha256_file(Path(contract_path)),
        "seed":contract["determinism"]["seed"],
        "rights":{"external_image_asset_count":0,"statement":"Owned primitives; internal runtime only"},"cases":cases})
    write_canonical(input_root/"baseline_roles.json",{"cases":roles})
    write_canonical(label_root/"labels.json",{"generation_seed":generation_seed,"labels":labels})
    return {"cases":300,"input_manifest_sha256":sha256_file(input_root/"input_manifest.json"),
            "baseline_roles_sha256":sha256_file(input_root/"baseline_roles.json"),
            "label_file_sha256":sha256_file(label_root/"labels.json")}
