# VisionGate CI: bounded OpenCV 5 runtime pilot

Participant-owned prototype for reproducible image-driven release-gating tools.
Not a registered competition entry, official score, AWS deployment or award.

This pilot preserves the August 30 perception, generator and local contracts.
It runs fresh design fixtures on a hash-pinned OpenCV 5.0.0.93 headless wheel
and NumPy 2.2.6, Linux x86_64 Python3.12. The previous OpenCV4 artifacts stay
in the private workspace. No screenshot, fixture image, customer data,
credential or personal document is committed here. Geometric design fixtures
are generated only inside the ephemeral runner; these are not a sealed holdout.

`agent_loop.py` adds real conditional operations. Pair inspection selects a
rerun-image analysis, all-region crop analysis, or sandbox gate mutation.
Rerun inputs are pre-captured synthetic fixtures, not live browser captures.
The controller receives no ground-truth labels and never releases production.
Actual OpenCV outputs determine subsequent calls, not just text explanations.

The original gate intentionally returns `FAIL_CLOSED/BLOCKED` even when its
local compatibility/performance gates pass. A verified OpenCV5 version is not
an official metric, blind-generalization proof, or competition readiness.
AWS execution, independent holdout, deployment safeguards, registration,
legal/platform steps, report and judge-accessible video remain separate work.

## One bounded run

The workflow installs only two exact official PyPI wheels with required hashes,
then executes `python -B tools/native_cv5_probe.py`. It has a20minute job cap,
600second owned-step cap,10GiB free-disk floor and100MiB output cap. Public
standard CPU only: no paid runner, cloud resource, model API, cache or artifact
storage. It emits one aggregate receipt; images remain internal to the runner.
Do not restart on observation timeout or change thresholds after outcomes.

Source code is owned by Yufeng He; no additional public license is granted by
this pilot. OpenCV and NumPy retain their own Apache-2.0/BSD licenses.
