"""Owned HTML UI fixtures and loopback-only routes; no external/private sites."""
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import threading

ROLES=("primary_deploy","secondary_deploy","critical_warning")


def baseline(rng,family):
    return {"family":family,"dx":rng.randint(-20,20),"dy":rng.randint(-6,6),
            "builds":[rng.randint(100,999) for _ in range(48)]}


def render(base,removed=None,minor=False,effect="none"):
    dx,dy=base["dx"],base["dy"]
    if base["family"]=="board":
        body=''.join(f'<article style="left:{70+c*580}px;top:{210+r*148}px"><h3>Build {base["builds"][c*4+r]}</h3><p>Review dependency and release evidence</p><div class="bar"></div></article>'
                     for c in range(3) for r in range(4))
    else:
        body='<table>'+''.join('<tr>'+''.join(f'<td>R{r} C{c} build {base["builds"][r*4+c]}</td>' for c in range(4))+'</tr>' for r in range(12))+'</table>'
    protected=''.join(f'<button id="{role}" data-protected-role="{role}" style="left:{x+dx}px;top:{y+dy}px;width:{w}px;height:{h}px;{("visibility:hidden;" if removed==role else "")}">{label}</button>'
        for role,x,y,w,h,label in ((ROLES[0],1490,925,310,90,"DEPLOY"),
                                   (ROLES[1],1730,855,96,36,"GO"),(ROLES[2],1425,950,24,24,"!")))
    background="#d21920" if minor else "#448a72"
    # Only the fixture server can choose a perturbation. The controller receives
    # screenshots and reference DOM declarations, not this render function's args.
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Aurora build control</title>
<style>
*{{box-sizing:border-box}}html,body{{margin:0;width:1920px;height:1080px;overflow:hidden;background:#f6f6f6;font:22px Arial,sans-serif}}
#shell{{position:relative;width:1920px;height:1080px;background:#f6f6f6}}header{{height:90px;background:#2b353f;color:white;padding:26px 40px;font-size:30px}}
h2{{margin:28px 70px}}article{{position:absolute;width:480px;height:120px;background:#dce4ed;padding:10px 18px}}
h3{{margin:3px 0 12px}}p{{font-size:18px;margin:0 0 12px}}.bar{{height:8px;width:330px;background:#549184}}
table{{position:absolute;left:70px;top:210px;width:1780px;border-collapse:collapse}}td{{height:48px;padding:8px 20px;border-bottom:5px solid #f6f6f6;background:#dfe8ef}}
button{{position:absolute;border:0;border-radius:0;color:white;background:#255bd0;font:bold 24px Arial,sans-serif}}
#critical_warning{{background:#ce2030;font-size:18px}}#secondary_deploy{{font-size:18px}}#status{{position:absolute;left:1520px;top:132px;width:128px;height:52px;background:{background};color:white;padding:15px;font-size:18px}}
.capture-blur{{filter:blur(.7px)}}.capture-shift{{transform:translate(4px,-4px)}}
</style></head><body><main id="shell"><header>Aurora / isolated visual release control</header><h2>Release evidence workspace</h2>{body}<div id="status">READY</div>{protected}</main>
<script>
window.captureSettled=true;
window.startCapture=()=>{{
  const shell=document.getElementById('shell');
  const effect={json.dumps(effect)};
  window.captureSettled=false;
  if(effect!=='none')shell.classList.add('capture-'+effect);
  setTimeout(()=>{{shell.classList.remove('capture-blur','capture-shift');window.captureSettled=true;}},350);
}};
</script></body></html>'''


def prepare(out,generation_seed):
    out=Path(out); (out/"inputs").mkdir(mode=0o700); (out/"private_labels").mkdir(mode=0o700)
    schedule=random.Random(generation_seed)
    work=[(family,stratum,i) for family in ("board","table")
          for stratum in ("clean_or_nuisance","minor_regression","critical_regression") for i in range(6)]
    schedule.shuffle(work)
    views={}; labels=[]; ids=[]
    for position,(family,stratum,index) in enumerate(work):
        case_id=f"live-{position:03}"; ids.append(case_id)
        base=baseline(random.Random(generation_seed ^ ((position+1)*0x9e3779b97f4a7c15)),family)
        views[f"/live/{case_id}/reference"]=render(base)
        removed=None; minor=False
        if stratum=="clean_or_nuisance":
            effect="shift" if index%2==0 else "blur"
            expected_first=expected_final="rerun_visual_test"; kind="transient_"+effect
        else:
            effect="blur" if index%2 else "none"
            if stratum=="minor_regression":
                minor=True; expected_first=expected_final="inspect_region"; kind="status_color"
            else:
                removed=ROLES[index%3]; expected_first=expected_final="set_deploy_gate_blocked"; kind=removed+"_hidden"
            if effect!="none": expected_first="rerun_visual_test"; kind+="_with_transient_blur"
        views[f"/live/{case_id}/candidate"]=render(base,removed,minor,effect)
        labels.append({"case_id":case_id,"family":family,"stratum":stratum,"kind":kind,
                       "removed_role":removed,"expected_first":expected_first,"expected_final":expected_final})
    (out/"private_labels/labels.json").write_text(json.dumps({"generation_seed_exact_decimal":str(generation_seed),"cases":labels}))
    return views,ids


def serve(views,ids,metadata_dir):
    metadata_dir=Path(metadata_dir); metadata_dir.mkdir(mode=0o700)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            if self.path not in views:
                self.send_error(404); return
            body=views[self.path].encode()
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Cache-Control","no-store"); self.send_header("Content-Length",str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            case_id=self.path.removeprefix("/metadata/")
            if self.path!="/metadata/"+case_id or case_id not in ids:
                self.send_error(404); return
            length=int(self.headers.get("Content-Length",0))
            if not 0<length<=65536:
                self.send_error(413); return
            value=json.loads(self.rfile.read(length))
            if set(value)!={"protected_components","status_region","url","user_agent","viewport"}:
                self.send_error(400); return
            path=metadata_dir/f"{case_id}.json"
            if path.exists(): self.send_error(409); return
            with path.open("x") as stream: json.dump(value,stream)
            self.send_response(200); self.send_header("Content-Length","2"); self.end_headers(); self.wfile.write(b"OK")
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start()
    return server,f"http://127.0.0.1:{server.server_port}"
