"""The Gemmas at a 131,072-token context with the Mac's `large` prompt (~90K tokens), on
Vireo and llama.cpp, through bench-2026-11-13/bench.py's own server classes and workload.
Before and after the prompt, Windows' per-process GPU counters are read while the server
still holds its memory (dedicated GB, shared GB): shared growing means the context spilled
into the PC's memory. Nothing runs on the GPU while a counter is read.

    CONTEXT=131072 python -X utf8 -I probe_131k.py <out.jsonl> e2b,e4b vireo,llama
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench", HERE.parent / "bench-2026-11-13" / "bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

out = Path(sys.argv[1])
keys = sys.argv[2].split(",")
engines = sys.argv[3].split(",")


def counters(pid):
    """Dedicated and shared GPU memory of one process, GB, from Windows' counters."""
    ps = (f"$s = Get-Counter '\\GPU Process Memory(pid_{pid}*)\\Dedicated Usage','\\GPU Process Memory(pid_{pid}*)\\Shared Usage';"
          "$d = 0; $h = 0; foreach ($c in $s.CounterSamples) { if ($c.Path -like '*dedicated usage') { $d += $c.CookedValue } else { $h += $c.CookedValue } };"
          "Write-Output \"$d $h\"")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
        d, h = r.stdout.split()
        return [round(float(d) / 2**30, 2), round(float(h) / 2**30, 2)]
    except Exception as e:
        return [None, repr(e)[:100]]


def smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return round(int(r.stdout.split()[0]) / 1024, 2)


for key in keys:
    model = next(m for m in bench.MODELS if m[1] == key)
    for kind in engines:
        row = {"engine": kind, "model": model[0], "context": model[4], "work": "large"}
        try:
            eng = bench.start(kind, model)
            eng.key = key
        except Exception as e:
            row["error"] = f"start: {e!r}"
            print(json.dumps(row), flush=True)
            out.open("a").write(json.dumps(row) + "\n")
            continue
        try:
            time.sleep(2)
            row["load"], row["load_smi_gb"] = counters(eng.pid), smi()
            eng.chat([{"role": "user", "content": bench.nonce() + "Say hello in five words."}], 16, False, bench.Desk())
            t = time.time()
            rows = bench.run_workload(eng, "large", None)
            row["wall"] = round(time.time() - t, 2)
            time.sleep(2.2)
            row["mem"], row["smi_gb"] = counters(eng.pid), smi()
            r = rows[0]
            r["text"] = (r.get("text") or "")[:600]
            row["rows"] = rows
            row["log"] = str(eng.log_path)
        except Exception as e:
            row["error"] = repr(e)[:500]
        finally:
            eng.stop()
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(json.dumps(row)[:600], flush=True)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
print("DONE", flush=True)
