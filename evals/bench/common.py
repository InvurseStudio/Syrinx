"""Shared pieces of Vireo Bench: paths, the config, result rows, memory watchers, HTTP and
the machine's state. Standard library only, Python 3.9+."""
import json
import os
import platform
import re
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE.parents[1]          # VireoPrototype/
FIX = HERE / "fixtures"
WIN = os.name == "nt"
PLATFORM = "windows" if WIN else "mac"
TOKEN = "bench"


def config():
    return json.loads((HERE / "suite.json").read_text())


def path(p):
    """A config path: '~' is home, relative is from VireoPrototype/."""
    p = os.path.expanduser(p)
    return Path(p) if os.path.isabs(p) else PROTO / p


def fixture_json(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def fixture_text(name):
    return (FIX / name).read_text(encoding="utf-8")


def llama_env(folder):
    env = dict(os.environ)
    if WIN:
        env["PATH"] = str(folder) + os.pathsep + env.get("PATH", "")
    else:
        env["DYLD_LIBRARY_PATH"] = str(folder)
    return env


def nonce():
    """A unique first line, so no engine reuses a cache it shouldn't (rules, item 5)."""
    return f"[Request {uuid.uuid4().hex[:8]}]\n"


def exe(p):
    p = Path(p)
    return p if p.exists() or not WIN else p.with_suffix(".exe")


# ---- Results ----------------------------------------------------------------------------

class Recorder:
    def __init__(self, out, base):
        self.out, self.base = Path(out), dict(base)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def __call__(self, **kw):
        row = dict(self.base, **kw, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        line = json.dumps(row, ensure_ascii=False)
        with self.lock:
            print(_short(row), flush=True)
            with open(self.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return row


def _short(row):
    keep = {k: v for k, v in row.items() if k not in ("run", "platform", "tier", "commit", "suite_version", "time", "text", "texts", "turns")}
    s = json.dumps(keep, ensure_ascii=False)
    return s if len(s) < 300 else s[:297] + "..."


# ---- Memory -----------------------------------------------------------------------------

UNITS = {"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}


class MemWatch:
    """Mac: a server process's memory every 2 s. total = phys_footprint + the resident part
    of files it maps (RSS misses MLX's KV cache; the footprint misses an mmapped GGUF)."""

    def __init__(self, pid):
        self.pid, self.peak, self.stop_ = pid, 0, False
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def sample(self):
        try:
            out = subprocess.run(["footprint", "-p", str(self.pid)], capture_output=True, text=True).stdout
            m = re.search(r"Footprint: ([\d.]+) (B|KB|MB|GB)", out)
            fp = float(m[1]) * UNITS[m[2]] if m else 0
            mf = re.search(r"^\s*\S+ \S+\s+([\d.]+) (B|KB|MB|GB)\s+\S+ \S+\s+\d+\s+mapped file$", out, re.M)
            total = fp + (float(mf[1]) * UNITS[mf[2]] if mf else 0)
        except Exception:
            total = 0
        self.peak = max(self.peak, total)
        return total

    def loop(self):
        while not self.stop_:
            self.sample()
            time.sleep(2)

    def reset(self):
        self.peak = 0
        return self.sample()

    def stop(self):
        self.stop_ = True


class GpuWatch(MemWatch):
    """Windows: the whole GPU's memory in use (nvidia-smi), read only when asked (between
    tests, never during one: polling moves the clocks)."""

    def __init__(self, pid):
        self.pid, self.peak, self.stop_ = pid, 0, True

    def sample(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            used = int(out.split()[0]) << 20
        except Exception:
            used = 0
        self.peak = max(self.peak, used)
        return used


def watch(pid):
    return (GpuWatch if WIN else MemWatch)(pid)


def gb(b):
    return round(b / 2**30, 2) if b else None


# ---- HTTP -------------------------------------------------------------------------------

def post(url, body, token=None, timeout=3600):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), h), timeout=timeout)


def sse(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                pass


def wait_health(url, proc, secs=900):
    t = time.time()
    while time.time() - t < secs:
        if proc.poll() is not None:
            raise RuntimeError("server exited during load")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not come up")


def stop_proc(p):
    p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(3)


# ---- Machine ----------------------------------------------------------------------------

def run_out(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def can_sudo():
    return not WIN and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def machine_state():
    s = {"os": platform.platform(), "python": platform.python_version(),
         "commit": run_out(["git", "-C", str(PROTO), "rev-parse", "--short", "HEAD"]),
         "dirty": bool(run_out(["git", "-C", str(PROTO), "status", "--porcelain", "--untracked-files=no"]))}
    if WIN:
        s["gpu"] = run_out(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm", "--format=csv,noheader"])
        s["power_plan"] = run_out(["powercfg", "/getactivescheme"])
        s["on_mains"] = run_out(["powershell", "-NoProfile", "-Command",
                                 "(Get-CimInstance Win32_Battery).BatteryStatus"]) in ("2", "")
    else:
        s["chip"] = run_out(["sysctl", "-n", "machdep.cpu.brand_string"])
        s["memory_gb"] = int(run_out(["sysctl", "-n", "hw.memsize"]) or 0) // 2**30
        s["macos"] = run_out(["sw_vers", "-productVersion"])
        batt = run_out(["pmset", "-g", "batt"])
        s["on_mains"] = "AC Power" in batt
        pm = run_out(["pmset", "-g"])
        m = re.search(r"powermode\s+(\d)", pm) or re.search(r"lowpowermode\s+(\d)", pm)
        s["power_mode"] = m.group(0) if m else ""
        mp = run_out(["memory_pressure"])
        m = re.search(r"System-wide memory free percentage: (\d+)%", mp)
        s["memory_free_pct"] = int(m[1]) if m else None
        s["uptime"] = run_out(["uptime"])
        s["sudo"] = can_sudo()
    return s
