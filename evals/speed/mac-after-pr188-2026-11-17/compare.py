"""Old (deck, 14-15 Nov) vs new (main after PR #188, 17 Nov) for the Mac cards 2-9.
Reads only; prints markdown tables to stdout.

    python3 -I compare.py > tables.md
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEED = HERE.parent
ROOT = SPEED.parents[1]


def pct(new, old):
    if new is None or old in (None, 0):
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


def lines(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ---- cards 2, 3, 8: long context ------------------------------------------------------
def long_ctx(rows):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("work") != "long" or r.get("error"):
            continue
        x = r["rows"][0]
        st = x["steps"][0]
        if "read" not in st:      # mlx-vlm's 256K first try, timed out
            continue
        k =(r["engine"], r["size"])
        out[k]["read"].append(st["read"] / st["read_s"])
        out[k]["gen"].append(st["gen"] / st["gen_s"])
        out[k]["peak"].append(r["peak_total"] / 1e9)
        out[k]["ttft"].append(x["ttft"])
        out[k]["recall"].append(x.get("recall"))
    return {k: {m: (v if m == "recall" else max(v) if m == "peak" else statistics.mean(v)) for m, v in d.items()} for k, d in out.items()}


OLD_LC = long_ctx(lines(SPEED / "long-context-2026-11-15/results.jsonl"))
NEW_LC = long_ctx(lines(HERE / "long-context.jsonl"))
COOL_LC = long_ctx(lines(HERE / "rerun-long-context-128k.jsonl"))
SIZES = ["32k", "64k", "128k"]


def card_lc(title, metric, fmt, lower_better=False):
    print(f"\n### {title}\n")
    print("| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SIZES:
        o, n = OLD_LC.get(("vireo", s)), NEW_LC.get(("vireo", s))
        l, m = OLD_LC.get(("llama", s)), OLD_LC.get(("mlx", s))
        c = COOL_LC.get(("vireo", s))
        ov, mv, cv = o and o[metric], n and n[metric], c and c[metric]
        nv = cv or mv
        facts = ", ".join(f"{x}/2" for x in (n["recall"] if n else []) + (c["recall"] if c else [])) or "not run"
        print(f"| {s.upper()} | {fmt.format(ov) if ov else '-'} | {fmt.format(mv) if mv else '-'} | {fmt.format(cv) if cv else ''} | {pct(nv, ov)} | "
              f"{fmt.format(l[metric]) if l else '-'} | {fmt.format(m[metric]) if m else '-'} | {facts} |")


# ---- cards 4, 5, 9: eight models -------------------------------------------------------
def bench(rows):
    R = defaultdict(dict)
    for d in rows:
        m, e, w = d.get("model"), d.get("engine"), d.get("work")
        if w in ("warmup", "load") or not d.get("rows"):
            if d.get("error"):
                R[m][(w, e)] = {"err": str(d["error"])[:120]}
            continue
        rs = d["rows"]
        st = [s for x in rs for s in x["steps"]]
        g = sum(s.get("gen") or 0 for s in st)
        gs = sum(s.get("gen_s") or 0 for s in st)
        t = (rs[0].get("text") or "").replace(",", "").lower()
        right = {"xlarge": rs[0].get("recall"), "tools": int("26742.25" in t or "26742.2" in t),
                 "skill": int("78" in t and bool(rs[0].get("wrote"))), "longtools": int("inventory-sync" in t)}.get(w)
        prev = R[m].get((w, e))
        cur = dict(gen=g / gs if gs else None, total=sum(x.get("total") or 0 for x in rs), right=right,
                   err="; ".join((x.get("error") or "")[:80] for x in rs if x.get("error")), n=1)
        if prev and prev.get("n"):          # several reps (the cool-down reruns): average the speeds
            k = prev["n"]
            for f in ("gen", "total"):
                if prev.get(f) is not None and cur[f] is not None:
                    cur[f] = (prev[f] * k + cur[f]) / (k + 1)
            cur["n"] = k + 1
        R[m][(w, e)] = cur
    return R


OLD_B = bench(lines(SPEED / "bench-2026-11-13/results-q6-128k.jsonl"))
NEW_B = bench(lines(HERE / "bench-8models.jsonl"))
COOL_B = bench(lines(HERE / "rerun-short.jsonl") + lines(HERE / "rerun-xlarge.jsonl"))
MODELS = ["Gemma 4 31B", "Qwen 3.8 27B", "Muse Glimmer 30B", "Gemma 4 26B-A4B", "Gemma 4 E4B", "Gemma 4 E2B",
          "Qwen 3.5 4B", "Qwen 3.5 9B"]


def card_b(title, work, metric, fmt, models=MODELS, lower_better=False, right_label=None):
    print(f"\n### {title}\n")
    rl = f" | {right_label} deck → now" if right_label else ""
    print(f"| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other{rl} |")
    print("|---|---|---|---|---|---|---|---|" + ("---|" if right_label else ""))
    for m in models:
        o, n = OLD_B[m].get((work, "vireo"), {}), NEW_B[m].get((work, "vireo"), {})
        l, x = OLD_B[m].get((work, "llama"), {}), OLD_B[m].get((work, "mlx"), {})
        c = COOL_B[m].get((work, "vireo"), {})
        ov, mv, cv, lv, xv = o.get(metric), n.get(metric), c.get(metric), l.get(metric), x.get(metric)
        nv = cv if cv is not None else mv
        others = [v for v in (lv, xv) if v]
        best = (min(others) if lower_better else max(others)) if others else None
        lead = (best / nv if lower_better else nv / best) if (nv and best) else None
        f = lambda v: fmt.format(v) if v is not None else "-"
        cells = f"| {m} | {f(ov)} | {f(mv)}{' ⚠ ' + n['err'] if n.get('err') else ''} | {f(cv) if cv is not None else ''} | {pct(nv, ov)} | {f(lv)} | {f(xv)} | {f'{lead:.2f}×' if lead else '-'} |"
        if right_label:
            cells += f" {o.get('right', '-')} → {n.get('right', 'not run')} (llama {l.get('right', '-')}, mlx {x.get('right', '-')}) |"
        print(cells)


# ---- card 6: long chat -----------------------------------------------------------------
def long_run(p):
    p = Path(p)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    R = d["rows"]
    rereads = sum(1 for r in R if r.get("turn", 0) > 1 and r.get("kind") != "side-chat" and r.get("kept") == 0)
    scored = [r for r in R if r.get("right") is not None]
    return dict(s=d["seconds"], rereads=rereads, right=sum(bool(r["right"]) for r in scored), of=len(scored),
                turns=d["turns"], errors=sum(1 for r in R if r.get("error")))


def card6():
    o = long_run(ROOT / "evals/long-run/gemma-4-31B-vireo.json")
    n = long_run(HERE / "long-run-gemma-4-31B-vireo.json")
    l = long_run(ROOT / "evals/long-run/gemma-4-31B-llama.json")
    print("\n### Card 6: 64-turn chat, Gemma 4 31B, 32K context\n")
    print("| | Deck (new engine) | Now | Change | llama.cpp (stored) |")
    print("|---|---|---|---|---|")
    if not n:
        print("| not run | | | | |")
        return
    print(f"| Whole chat (s) | {o['s']} | {n['s']} | {pct(n['s'], o['s'])} | {l['s']} |")
    print(f"| Full re-reads | {o['rereads']} | {n['rereads']} | | 12 (its client drops old turns; per the deck) |")
    print(f"| Answers right | {o['right']}/{o['of']} | {n['right']}/{n['of']} | | {l['right']}/{l['of']} |")
    print(f"| Turns / errors | {o['turns']} / {o['errors']} | {n['turns']} / {n['errors']} | | {l['turns']} / {l['errors']} |")
    print(f"| vs llama.cpp | {l['s'] / o['s']:.2f}× | {l['s'] / n['s']:.2f}× | | |")


# ---- card 7: same answers --------------------------------------------------------------
def card7():
    s6 = lines(HERE / "bench-s6-e2b.jsonl")
    s3 = lines(HERE / "bench-s3-e2b.jsonl")
    print("\n### Card 7: same answers (Gemma 4 E2B)\n")
    print("| Check | Deck | Now | Notes |")
    print("|---|---|---|---|")
    ppl = {r["engine"]: r for r in s6 if r.get("test") == "ppl"}
    v, l = ppl.get("vireo", {}), ppl.get("llama", {})
    pv, pl = v.get("ppl"), l.get("ppl")
    if pv and pl:
        print(f"| Perplexity, Vireo vs llama.cpp | 120.78 vs 120.76 (0.01%) | {pv:.2f} vs {pl:.2f} ({abs(pv - pl) / pl * 100:.2f}%) | llama-perplexity rerun (b12228) |")
    else:
        print(f"| Perplexity | 120.78 vs 120.76 | {v.get('error') or l.get('error') or 'not run'} | |")
    g = next((r for r in s6 if r.get("test") == "greedy-match" and "identical" in r), None)
    if g:
        print(f"| Greedy 256-token continuations identical | 19/20 | {g['identical']}/{g['prompts']} | median words agreeing {g['median_words']}, min {g['min_words']} |")
    else:
        print("| Greedy match | 19/20 | not run | |")
    for r in s3:
        if r.get("suite") == "S3" and r.get("engine") == "vireo":
            if r.get("recall") is not None:
                print(f"| Planted facts, {r['test'].upper()} context | 5/5 | {r['recall']}/5 | read {r.get('read_tps', 0):.0f} tok/s, write {r.get('gen_tps', 0):.1f} tok/s |")
            else:
                print(f"| Planted facts, {r['test'].upper()} context | 5/5 | error | {r.get('error', '')[:120]} |")
    lc = [f"{s.upper()} {', '.join(f'{x}/2' for x in NEW_LC[('vireo', s)]['recall'])}" for s in SIZES if ('vireo', s) in NEW_LC]
    print(f"| Gemma 4 31B facts at every size | 2/2 to 256K | {', '.join(lc) or 'not run'} (256K not run) | card 2's run |")


if __name__ == "__main__":
    print("## Cards 2, 3, 8: Gemma 4 31B at long context")
    card_lc("Card 2: reading speed (tok/s)", "read", "{:.0f}")
    card_lc("Card 3: writing speed after the prompt (tok/s)", "gen", "{:.1f}")
    card_lc("Card 8: peak memory (GB, lower is better)", "peak", "{:.1f}", lower_better=True)
    card_lc("Time to first token (s), not on a card", "ttft", "{:.0f}")
    print("\n## Cards 4, 5, 9: eight models, 131,072-token context")
    card_b("Card 4: writing after the ~125K-token prompt (tok/s)", "xlarge", "gen", "{:.1f}", right_label="Facts (of 2)")
    card_b("Card 5: tool task, total time (s, lower is better)", "tools", "total", "{:.1f}", lower_better=True, right_label="Right")
    card_b("Skill task, total time (s), the other agent task", "skill", "total", "{:.1f}", lower_better=True, right_label="Right")
    card_b("Long tool task, total time (s)", "longtools", "total", "{:.1f}", lower_better=True, right_label="Right")
    card_b("Card 9: writing a short reply (tok/s)", "short", "gen", "{:.1f}")
    card6()
    card7()
