#!/usr/bin/env python3
"""vireo-eval on main (6bc5d1d) against the branch, per model: passes by part, speeds, and
every case whose result or answer changed.   python3 -I compare.py [--answers]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = [("e2b", "Gemma 4 E2B"), ("e4b", "Gemma 4 E4B"), ("q4b", "Qwen 3.5 4B"), ("q9b", "Qwen 3.5 9B"),
         ("g26", "Gemma 4 26B-A4B"), ("q27", "Qwen 3.8 27B"), ("g31", "Gemma 4 31B"), ("muse", "Muse Glimmer 30B")]


def load(key, build):
    p = HERE / f"{key}-{build}.json"
    if not p.exists():
        return None
    return next(iter(json.loads(p.read_text())["models"].values()))


def part(c):
    return "skills" if c["skillCase"] else "changes" if c["changes"] else "more" if c["moreCase"] else "hard" if c["hard"] else "easy"


print("| Model | Passed | Easy | Hard | Changes | Skills | More | s a case | Read tok/s | Written tok/s | From cache |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
changed = []
for key, name in NAMES:
    a, b = load(key, "main"), load(key, "branch")
    if not a or not b:
        continue
    cells = [f"{a['passed']} → {b['passed']} of {b['total']}"]
    for p in ["easy", "hard", "changes", "skills", "more"]:
        n = lambda m: sum(c["passed"] for c in m["cases"] if part(c) == p)
        cells.append(f"{n(a)} → {n(b)}")
    sa, sb = a["speeds"], b["speeds"]
    cells.append(f"{a['seconds'] / a['total']:.1f} → {b['seconds'] / b['total']:.1f}")
    cells.append(f"{sa['readTokensPerSecond']:.0f} → {sb['readTokensPerSecond']:.0f}")
    cells.append(f"{sa['writtenTokensPerSecond']:.1f} → {sb['writtenTokensPerSecond']:.1f}")
    cells.append(f"{100 * sa['keptTokens'] / sa['promptTokens']:.0f}% → {100 * sb['keptTokens'] / sb['promptTokens']:.0f}%")
    print(f"| {name} | " + " | ".join(cells) + " |")
    cb = {c["id"]: c for c in b["cases"]}
    for c in a["cases"]:
        d = cb.get(c["id"])
        if d and c["passed"] != d["passed"]:
            changed.append((name, c["id"], part(c), c["passed"], d["passed"], d["problem"] if c["passed"] else c["problem"]))
        elif d and "--answers" in sys.argv and c["answer"] != d["answer"]:
            changed.append((name, c["id"], part(c), c["passed"], d["passed"], "answer worded differently"))
print()
for name, cid, p, was, now, why in changed:
    print(f"- {name}, `{cid}` ({p}): {'pass' if was else 'fail'} → {'pass' if now else 'fail'}. {str(why)[:300]}")
