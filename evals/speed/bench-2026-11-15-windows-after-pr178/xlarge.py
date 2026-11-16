"""The xlarge row: the largest prompt each model holds on this 8 GB GPU on both engines,
through bench-2026-11-13/bench.py's own server classes (the same system turn, request and
log parsing as the short / medium / large rows), after probe_131k.py of the afternoon.

The prompt is the Mac's xlarge.txt (a ~120K-token repository dump with the recall
question at its end), its repository part cut at a line's end to CHARS characters when
CHARS is given, so it fits the context; the question is kept. The same text goes to
every engine. Windows' per-process GPU counters (dedicated GB, shared GB) are read after
loading and again after the prompt while the server still holds its memory: shared
growing past its value at load means the context spilled into the PC's memory, and the
row is not a valid measurement.

    python -X utf8 -I xlarge.py <out.jsonl> <model key> <context> <chars or 0> vireo,vireo-plain,llama
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
bench.LOGS = HERE / "xlarge-logs"
bench.LOGS.mkdir(exist_ok=True)

out = Path(sys.argv[1])
key = sys.argv[2]
context = int(sys.argv[3])
chars = int(sys.argv[4])
engines = sys.argv[5].split(",")

text = (bench.PROMPTS / "xlarge.txt").read_text()
end = text.rindex("</repository>")
body, tail = text[:end], text[end:]
if chars and chars < len(body):
    body = body[:body.rindex("\n", 0, chars) + 1]
PROMPT = body + tail


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


m = next(m for m in bench.MODELS if m[1] == key)
model = (m[0], m[1], m[2], m[3], context, m[5])
for kind in engines:
    row = {"engine": kind, "model": model[0], "context": context, "chars": len(body), "work": "xlarge"}
    try:
        eng = bench.start(kind, model)
        eng.key = key
    except Exception as e:
        row["error"] = f"start: {e!r}"
        print(json.dumps(row), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        continue
    try:
        time.sleep(2)
        row["load"], row["load_smi_gb"] = counters(eng.pid), smi()
        eng.chat([{"role": "user", "content": bench.nonce() + "Say hello in five words."}], 16, False, bench.Desk())
        t = time.time()
        r = eng.chat([{"role": "user", "content": bench.nonce() + PROMPT}], 128, False, bench.Desk())
        row["wall"] = round(time.time() - t, 2)
        r["recall"] = sum(x.lower() in r["text"].lower() for x in ["OSPREY-41", "Dagny Vell"])
        r["text"] = (r.get("text") or "")[:600]
        time.sleep(2.2)
        row["mem"], row["smi_gb"] = counters(eng.pid), smi()
        row["rows"] = [r]
        row["log"] = str(eng.log_path)
    except Exception as e:
        row["error"] = repr(e)[:500]
    finally:
        eng.stop()
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps(row)[:700], flush=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
print("DONE", flush=True)
