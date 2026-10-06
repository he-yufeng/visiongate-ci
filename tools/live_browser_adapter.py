"""Playwright CLI sensor, lazy rerun reader for unmodified frozen R02 policy."""
import json
from pathlib import Path
import subprocess
import time

from visiongate_ci import role_agent
from visiongate_ci.canonical import sha256_file


class BrowserClient:
    def __init__(self,root,out,origin):
        self.root,self.out,self.origin=Path(root),Path(out),origin
        self.session="visiongate_owned_r03"
        config=json.loads((self.root/"browser/playwright-cli.json").read_text())
        config["network"]={"allowedOrigins":[origin]}
        self.config=self.out/"effective_cli_config.json"
        self.config.write_text(json.dumps(config))
        self.command_records=[]
    def command(self,*args):
        start=time.monotonic()
        command=[str(self.root/"tools/browser_cli.sh"),"-s="+self.session]
        # CLI0.1.22 accepts configuration only on open. The named session
        # retains it; goto/snapshot/run-code/close reject that option.
        if args[0]=="open":
            command.append("--config="+str(self.config))
        value=subprocess.run([*command,*args],cwd=self.out,capture_output=True,text=True,timeout=30)
        self.command_records.append({"command":args[0],"exit_code":value.returncode,"seconds":time.monotonic()-start})
        if value.returncode:
            raise RuntimeError("Owned CLI command failed: "+value.stderr[-1800:]+value.stdout[-1800:])
        return value.stdout
    def open(self,url): self.command("open",url)
    def close(self): self.command("close")
    def baseline(self,case_id,path):
        url=self.origin+f"/live/{case_id}/reference"
        self.command("goto",url)
        self.command("snapshot","--filename="+str(self.out/f"{case_id}-baseline.yaml"))
        # run-code is required for font-ready screenshot + DOM rectangles,
        # not a replacement for a stale element ref; fresh snapshot above.
        script="""async page => {
          await page.evaluate(() => document.fonts.ready);
          await page.screenshot({path:PATH,animations:'disabled'});
          const data=await page.evaluate(() => {
            const box=el=>{const r=el.getBoundingClientRect();return [Math.floor(r.left),Math.floor(r.top),Math.ceil(r.right),Math.ceil(r.bottom)];};
            return {protected_components:[...document.querySelectorAll('[data-protected-role]')].map(el=>({role:el.dataset.protectedRole,box:box(el)})),
              status_region:box(document.getElementById('status')),url:location.href,user_agent:navigator.userAgent,viewport:[innerWidth,innerHeight]};
          });
          const response=await page.request.post(METADATA,{data});
          if(response.status()!==200)throw new Error('baseline DOM metadata rejected');
          return {baselineRecorded:true};
        }""".replace("PATH",json.dumps(str(path))).replace("METADATA",json.dumps(self.origin+"/metadata/"+case_id))
        self.command("run-code",script)
    def candidate(self,case_id,path):
        self.command("goto",self.origin+f"/live/{case_id}/candidate")
        self.command("snapshot","--filename="+str(self.out/f"{case_id}-candidate.yaml"))
        script="""async page => {
          await page.evaluate(() => document.fonts.ready);
          await page.evaluate(() => window.startCapture());
          await page.screenshot({path:PATH,animations:'disabled'});
          return {initialCaptureRecorded:true};
        }""".replace("PATH",json.dumps(str(path)))
        self.command("run-code",script)
    def recapture(self,path):
        script="""async page => {
          await page.waitForFunction(() => window.captureSettled===true,{},{timeout:2000});
          await page.screenshot({path:PATH,animations:'disabled'});
          return {actualConditionalRecapture:true};
        }""".replace("PATH",json.dumps(str(path)))
        self.command("run-code",script)


def decide_live(client,root,case,declared,contract,policy):
    root=Path(root)
    original=role_agent._read_image
    reads=[]; capture_events=[]; capture_wall=0.0
    rerun_path=root/case["paths"]["rerun"]
    prepared=rerun_path.exists()
    def reader(read_root,read_case,role):
        nonlocal capture_wall
        if role=="rerun":
            existed=rerun_path.exists()
            if existed: raise ValueError("pre-captured rerun forbidden in live pilot")
            if reads!=["reference","candidate"]: raise ValueError("recapture before image inspection")
            step=time.monotonic()
            client.recapture(rerun_path)
            elapsed=time.monotonic()-step; capture_wall+=elapsed
            case["sha256"]["rerun"]=sha256_file(rerun_path)
            capture_events.append({"file_existed_before_request":existed,"capture_seconds":elapsed,
                                   "sha256":case["sha256"]["rerun"]})
        reads.append(role)
        return original(read_root,read_case,role)
    start=time.monotonic()
    role_agent._read_image=reader
    try:
        decision=role_agent.run_case(root,case,declared,contract,policy)
    finally:
        role_agent._read_image=original
    wall=time.monotonic()-start
    # The inherited trace literal is corrected for this intentionally adapted
    # sensor. The callback executed a new browser screenshot, not a cached file.
    for step in decision["trace"]:
        if step["tool"]=="rerun_visual_test":
            step["output"]["capture_source"]="live_owned_browser_after_visual_selection"
    return {"decision":decision,"read_order":reads,"capture_events":capture_events,
            "rerun_prepared_before_policy":prepared,"rerun_exists_after_policy":rerun_path.exists(),
            "decision_cycle_wall_seconds":wall,"live_capture_wall_seconds":capture_wall,
            "vision_policy_wall_excluding_sensor_seconds":max(0.0,wall-capture_wall)}
