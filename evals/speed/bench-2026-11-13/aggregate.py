"""Summarises results.jsonl (bench.py) into summary.json and summary.md: per model,
engine and workload, the mean over rounds of time to first token, prompt reading speed,
generation speed, whole-task time, tokens read, right answers and peak memory.

    python3 -I aggregate.py [results.jsonl]
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.jsonl"
ENGINES = ["vireo", "vireo-plain", "llama", "mlx"]
LABEL = {"vireo": "Vireo", "vireo-plain": "Vireo (no drafting)", "llama": "llama.cpp", "mlx": "mlx-vlm"}
MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B", "Gemma 4 26B-A4B",
          "Gemma 4 31B", "Qwen 3.8 27B", "Muse Glimmer 30B"]


def right(work, text, row):
    t = (text or "").replace(",", "").lower()
    if work in ("long", "large", "xlarge"):
        return row.get("recall", 0) / 2
    if work == "tools":
        return float("26742.25" in t or "26742.2" in t)
    if work == "skill":
        return float("78" in t and bool(row.get("wrote")))
    if work == "longtools":
        return float("inventory-sync" in t)
    if work == "multiturn":
        return float(bool(row.get("right")))
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


acc = defaultdict(lambda: defaultdict(list))
loads = defaultdict(lambda: defaultdict(list))
errors = []
for line in open(src):
    d = json.loads(line)
    key = (d["model"], d["engine"])
    if d.get("error"):
        errors.append(f"{d['model']} / {d['engine']} / {d.get('work', '?')}: {d['error'][:160]}")
        continue
    if d.get("work") == "load":
        loads[key]["load_s"].append(d["load_s"])
        loads[key]["idle_gb"].append(d.get("idle_total", 0) / 2**30)
        continue
    if d.get("work") == "warmup":
        continue
    w, rows = d["work"], d["rows"]
    a = acc[(d["model"], d["engine"], w)]
    a["wall"].append(d["wall"])
    a["peak_gb"].append(d.get("peak_total", 0) / 2**30)
    for r in rows:
        if r.get("error"):
            errors.append(f"{d['model']} / {d['engine']} / {w}: {r['error'][:160]}")
    if w == "multiturn":
        later = rows[1:]
        a["ttft"].append(mean([r["ttft"] for r in later]))
        a["ttft_first"].append(rows[0]["ttft"])
        a["read_per_turn"].append(mean([sum(s.get("read") or 0 for s in r["steps"]) for r in later]))
        a["total"].append(sum(r["total"] for r in rows))
        a["right"].append(mean([right(w, r["text"], r) for r in rows]))
        gen = sum(s.get("gen") or 0 for r in rows for s in r["steps"])
        gs = sum(s.get("gen_s") or 0 for r in rows for s in r["steps"])
        a["gen_tps"].append(gen / gs if gs else None)
        continue
    r = rows[0]
    steps = r["steps"]
    a["ttft"].append(r["ttft"])
    a["total"].append(r["total"])
    a["right"].append(right(w, r["text"], r))
    read = sum(s.get("read") or 0 for s in steps)
    read_s = sum(s.get("read_s") or 0 for s in steps)
    gen = sum(s.get("gen") or 0 for s in steps)
    gen_s = sum(s.get("gen_s") or 0 for s in steps)
    a["read"].append(read)
    a["gen"].append(gen)
    a["calls"].append(len(r["calls"]))
    a["requests"].append(len(steps))
    if w in ("short", "medium", "long", "large", "xlarge"):
        s0 = steps[0] if steps else {}
        a["prompt_tokens"].append(s0.get("prompt"))
        a["read_tps"].append(s0["read"] / s0["read_s"] if s0.get("read_s") else None)
        a["gen_tps"].append(s0["gen"] / s0["gen_s"] if s0.get("gen_s") else None)

out = {"engines": ENGINES, "labels": LABEL, "models": [], "errors": errors}
md = ["# Nov 13 benchmark summary", ""]
for m in MODELS:
    if not any(k[0] == m for k in acc):
        continue
    entry = {"model": m, "engines": {}}
    for e in ENGINES:
        ee = {"load_s": mean(loads[(m, e)]["load_s"]), "idle_gb": mean(loads[(m, e)]["idle_gb"]), "work": {}}
        for (mm, en, w), a in acc.items():
            if mm == m and en == e:
                ee["work"][w] = {k: mean(v) for k, v in a.items()}
        if ee["work"]:
            ee["peak_gb"] = max(v.get("peak_gb") or 0 for v in ee["work"].values())
            entry["engines"][e] = ee
    out["models"].append(entry)
    md += [f"## {m}", "", "| Workload | Metric | " + " | ".join(LABEL[e] for e in entry["engines"]) + " |",
           "|---|---|" + "---|" * len(entry["engines"])]
    rows = [("short", "ttft"), ("short", "gen_tps"), ("medium", "ttft"), ("medium", "read_tps"), ("medium", "gen_tps"),
            ("long", "ttft"), ("long", "read_tps"), ("long", "gen_tps"), ("long", "right"),
            ("large", "prompt_tokens"), ("large", "ttft"), ("large", "read_tps"), ("large", "gen_tps"), ("large", "right"),
            ("xlarge", "prompt_tokens"), ("xlarge", "ttft"), ("xlarge", "read_tps"), ("xlarge", "gen_tps"), ("xlarge", "right"),
            ("multiturn", "ttft"), ("multiturn", "read_per_turn"), ("multiturn", "right"),
            ("tools", "total"), ("tools", "requests"), ("tools", "calls"), ("tools", "right"),
            ("skill", "total"), ("skill", "calls"), ("skill", "right"),
            ("longtools", "ttft"), ("longtools", "total"), ("longtools", "right"),
            ("large", "peak_gb"), ("xlarge", "peak_gb")]
    for w, k in rows:
        vals = [entry["engines"][e]["work"].get(w, {}).get(k) for e in entry["engines"]]
        md.append(f"| {w} | {k} | " + " | ".join("" if v is None else f"{v:.3g}" if isinstance(v, float) else str(v) for v in vals) + " |")
    md.append("| all | load_s | " + " | ".join(str(entry["engines"][e]["load_s"]) for e in entry["engines"]) + " |")
    md.append("| all | peak_gb | " + " | ".join(f"{entry['engines'][e]['peak_gb']:.1f}" for e in entry["engines"]) + " |")
    md.append("")
if errors:
    md += ["## Errors", ""] + [f"- {x}" for x in errors]
(HERE / "summary.json").write_text(json.dumps(out, indent=1))
(HERE / "summary.md").write_text("\n".join(md))
print("\n".join(md))
