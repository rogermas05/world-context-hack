"""
Local web dashboard to start/stop the full-body demo (play_full_body.py)
without touching the terminal each time.

    python hands/dashboard.py
    open http://localhost:8420

Stop always sends SIGINT (not SIGKILL) to the running demo first -- Python
turns that into a KeyboardInterrupt inside play_full_body.py, which is exactly
the path that eases all three arms down to rest slowly (see move_to_rest_slowly
in play_full_body.py) instead of just freezing mid-motion. Only if the process
doesn't exit within a generous timeout does this fall back to a hard kill,
and even then it explicitly releases torque on all three arms afterward so
nothing is left stiff/holding a position with no supervising process.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HANDS = REPO / "hands"
PORT = 8420

# Prefer the project's own persistent venv; fall back to whatever's running this.
_venv_python = REPO / ".venv" / "bin" / "python"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

# Only presets with a real, full-rendered overlay video and a confirmed-good
# run this session. clip_start matches what each trajectory was extracted at.
PRESETS = {
    "binding-pre-fold-stitching (50s, best)": {
        "traj": "trajectories/clip_uihmkro2gdyoa_0_50.npz",
        "overlay": "demo_binding_pre_fold_stitching_50s.mp4",
        "clip_start": 0,
    },
    "binding-pre-fold-stitching (30s)": {
        "traj": "trajectories/clip_uihmkro2gdyoa_0_30.npz",
        "overlay": "demo_binding_pre_fold_stitching.mp4",
        "clip_start": 0,
    },
    "metal-grinding": {
        "traj": "trajectories/clip_hcvl74foq7j2g_0_60.npz",
        "overlay": "demo_metal_grinding.mp4",
        "clip_start": 0,
    },
    "oil-seal-pressing": {
        "traj": "trajectories/clip_54mdrpbksrge2_0_30.npz",
        "overlay": "demo_oil_seal_pressing.mp4",
        "clip_start": 0,
    },
    "notebook-cover-assembly (colleague's original)": {
        "traj": "trajectories/z45uhnxrkwr5s_60_120.npz",
        "overlay": "demo_notebook_cover_assembly_z45.mp4",
        "clip_start": 60,
    },
}

STOP_GRACE_S = 20.0  # time to let the gentle rest-easing finish before hard-killing

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_log_path: Path | None = None


def _release_all_torque() -> None:
    """Last-resort safety net if a hard kill was needed -- never leave an arm
    stiff with no process supervising it."""
    subprocess.run(
        [PYTHON, "-c", (
            "from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig\n"
            "from arm_config import ARMS\n"
            "for role, (port, arm_id) in ARMS.items():\n"
            "    r = SO101Follower(SO101FollowerConfig(port=port, id=arm_id))\n"
            "    r.connect(calibrate=False)\n"
            "    r.bus.disable_torque()\n"
            "    r.disconnect()\n"
        )],
        cwd=HANDS, capture_output=True,
    )


def _start(preset_name: str, scale: float | None, speed: float | None) -> dict:
    global _proc, _log_path
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return {"ok": False, "error": "already running"}
        preset = PRESETS.get(preset_name)
        if not preset:
            return {"ok": False, "error": f"unknown preset {preset_name!r}"}

        cmd = [PYTHON, "play_full_body.py", "--traj", preset["traj"],
               "--overlay-video", preset["overlay"],
               "--clip-start", str(preset["clip_start"])]
        if scale is not None:
            cmd += ["--scale", str(scale)]
        if speed is not None:
            cmd += ["--speed", str(speed)]

        _log_path = HANDS / "dashboard_last_run.log"
        log_file = open(_log_path, "w")
        _proc = subprocess.Popen(cmd, cwd=HANDS, stdout=log_file, stderr=subprocess.STDOUT)
        return {"ok": True, "pid": _proc.pid, "cmd": " ".join(cmd)}


def _stop() -> dict:
    global _proc
    with _lock:
        if _proc is None or _proc.poll() is not None:
            return {"ok": True, "note": "not running"}
        proc = _proc

    # SIGINT -> KeyboardInterrupt in the child -> its own graceful, slow
    # ease-down-to-rest path. Give it real time to finish that, don't rush it.
    proc.send_signal(signal.SIGINT)
    deadline = time.monotonic() + STOP_GRACE_S
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)

    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
        _release_all_torque()
        return {"ok": True, "note": "graceful stop timed out -- hard-killed and released torque"}
    return {"ok": True, "note": "stopped gracefully"}


def _status() -> dict:
    with _lock:
        running = _proc is not None and _proc.poll() is None
        pid = _proc.pid if running else None
    log_tail = ""
    if _log_path and _log_path.exists():
        lines = _log_path.read_text(errors="replace").splitlines()
        log_tail = "\n".join(lines[-40:])
    return {"running": running, "pid": pid, "log_tail": log_tail}


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Full-Body Demo Control</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 20px; }
  select, input, button { font-size: 15px; padding: 6px 10px; margin: 4px 0; }
  button { cursor: pointer; border-radius: 6px; border: 1px solid #ccc; }
  #start { background: #1a7f37; color: white; border: none; }
  #stop { background: #c0392b; color: white; border: none; }
  #status { margin-top: 16px; font-weight: bold; }
  #log { background: #111; color: #ddd; font-family: ui-monospace, monospace; font-size: 12px;
         padding: 10px; height: 320px; overflow-y: auto; white-space: pre-wrap; border-radius: 6px; }
  label { display: block; margin-top: 10px; font-size: 13px; color: #555; }
</style></head>
<body>
<h1>SO-101 Full-Body Demo</h1>

<label>Clip preset</label>
<select id="preset"></select>

<label>Scale override (blank = script default)</label>
<input id="scale" type="number" step="0.05" placeholder="e.g. 0.95">

<label>Speed override (blank = 1.0)</label>
<input id="speed" type="number" step="0.1" placeholder="e.g. 1.0">

<div>
  <button id="start">Start</button>
  <button id="stop">Stop (gentle)</button>
</div>

<div id="status">idle</div>
<div id="log"></div>

<script>
const presets = REPLACED_BY_SERVER;
const sel = document.getElementById('preset');
for (const name of Object.keys(presets)) {
  const o = document.createElement('option'); o.value = name; o.textContent = name; sel.appendChild(o);
}

document.getElementById('start').onclick = async () => {
  const scale = document.getElementById('scale').value;
  const speed = document.getElementById('speed').value;
  const r = await fetch('/start', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({preset: sel.value, scale: scale ? parseFloat(scale) : null,
                           speed: speed ? parseFloat(speed) : null})});
  const j = await r.json();
  if (!j.ok) alert(j.error);
};
document.getElementById('stop').onclick = async () => {
  document.getElementById('status').textContent = 'stopping (easing arms down to rest)...';
  await fetch('/stop', {method: 'POST'});
};

async function poll() {
  const r = await fetch('/status'); const j = await r.json();
  document.getElementById('status').textContent = j.running ? `running (pid ${j.pid})` : 'idle';
  document.getElementById('log').textContent = j.log_tail || '';
  const log = document.getElementById('log'); log.scrollTop = log.scrollHeight;
}
setInterval(poll, 1000); poll();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            html = PAGE.replace("REPLACED_BY_SERVER", json.dumps(PRESETS))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            self._json(_status())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {}
        if self.path == "/start":
            self._json(_start(data.get("preset"), data.get("scale"), data.get("speed")))
        elif self.path == "/stop":
            self._json(_stop())
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"Dashboard at http://localhost:{PORT}  (python: {PYTHON})")
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
