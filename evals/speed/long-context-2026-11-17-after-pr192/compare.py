"""Compares this folder's long-context runs (Gemma 4 31B, new engine at d3bedf5, after PR #192)
with the 15 Nov deck runs and the 17 Nov afternoon runs at PR #188, with the stored llama.cpp
and mlx-vlm rows beside them. Prints markdown tables plus each run's own values.

    python3 -I compare.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEED = HERE.parent
NOV15 = SPEED / "long-context-2026-11-15"
PR188 = SPEED / "mac-after-pr188-2026-11-17"
SIZES = ("32k", "64k", "128k", "256k")
GB = 1e9
REPLY = re.compile(r"Reply: (\d+) prompt tokens .*?(\d+) tokens in .*?, 0 tool calls(.*)\.$")
GUESS = re.compile(r"(\d+) of (\d+) guesses from the (chat|draft model) taken")


def mem(path, engine="vireo"):
    """Per size: idle memory after load, and each long run's peak and after-reply memory (GB)."""
    out = defaultdict(lambda: dict(idle=[], peak=[], after=[]))
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if d["engine"] != engine or d.get("error"):
            continue
        if d["work"] == "load" and d.get("idle_total"):
            out[d["size"]]["idle"].append(d["idle_total"] / GB)
        elif d["work"] == "long" and d.get("peak_total") and not (d.get("rows") or [{}])[0].get("error"):
            out[d["size"]]["peak"].append(d["peak_total"] / GB)
            out[d["size"]]["after"].append(d["after_total"] / GB)
    return out


def load(path, engine="vireo"):
    """Long rows per size, in file order, for one engine."""
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if d["engine"] != engine or d["work"] != "long":
            continue
        if d.get("error"):
            out[d["size"]].append(dict(error=d["error"]))
            continue
        r = d["rows"][0]
        if r.get("error") or not r.get("steps") or "read" not in r["steps"][0]:
            out[d["size"]].append(dict(error=str(r.get("error"))))
            continue
        s = r["steps"][0]
        out[d["size"]].append(dict(time=d["time"], read=s["read"], read_tps=s["read"] / s["read_s"],
                                   ttft=r["ttft"], gen=s["gen"], gen_tps=s["gen"] / s["gen_s"],
                                   peak=d["peak_total"] / GB, recall=r["recall"]))
    return out


def ok(rows):
    return [r for r in rows if "error" not in r]


def guesses(log_dir, pattern="vireo-g31-*"):
    """Draft/chat guesses from each long request's "Reply:" line, per size, in time order."""
    out = defaultdict(list)
    for log in sorted(Path(log_dir).glob(pattern), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        size = log.stem.split("-")[2]
        for line in log.read_text(errors="replace").splitlines():
            m = REPLY.search(line)
            if m and int(m.group(1)) > 20000:
                g = {k: (int(a), int(b)) for a, b, k in GUESS.findall(m.group(3))}
                out[size].append(g)
    return out


def agg(key, xs):
    # Peak memory: the highest of the runs (as the deck's report.py); everything else the mean.
    xs = [x for x in xs if x is not None]
    return (max(xs) if xs else None) if key == "peak" else avg(xs)


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def gstr(gs):
    if not gs:
        return "-"
    parts = []
    for kind, label in (("draft model", "draft"), ("chat", "chat")):
        taken = sum(g[kind][0] for g in gs if kind in g)
        made = sum(g[kind][1] for g in gs if kind in g)
        if made or any(kind in g for g in gs):
            parts.append(f"{label} {taken}/{made} ({100 * taken / made:.0f}%)" if made else f"{label} 0/0")
    return ", ".join(parts)


new = load(HERE / "results.jsonl")
deck = load(NOV15 / "results.jsonl")
llama = load(NOV15 / "results.jsonl", "llama")
mlx = load(NOV15 / "results.jsonl", "mlx")
pr188 = load(PR188 / "long-context.jsonl")
pr188_rerun = load(PR188 / "rerun-long-context-128k.jsonl")
g_new = guesses(HERE / "logs" / "bench-logs")
g_deck = guesses(NOV15 / "logs" / "bench-logs") or guesses(NOV15 / "logs")
g_188 = guesses(PR188 / "logs" / "bench-logs")
# Keep only the last N long requests per size, N = recorded runs (drops the 15 Nov smoke run's).
for g, rows in ((g_deck, deck), (g_188, {k: pr188.get(k, []) + pr188_rerun.get(k, []) for k in SIZES})):
    for k in SIZES:
        g[k] = g[k][len(g[k]) - len(rows.get(k, [])):] if rows.get(k) else []


def pr188_val(size, key):
    # As in PR #188's results.md: the cool-down rerun where there is one (128K), else the main run.
    rows = pr188_rerun.get(size) or pr188.get(size)
    return agg(key, [r.get(key) for r in ok(rows or [])])


METRICS = [  # key, title, format, better
    ("read_tps", "Reading speed (tok/s)", "{:.0f}", "high"),
    ("ttft", "Time to first token (s)", "{:.0f}", "low"),
    ("gen_tps", "Writing speed (tok/s)", "{:.1f}", "high"),
    ("peak", "Peak memory (GB)", "{:.1f}", "low"),
]


def fmt(f, v):
    return "-" if v is None else f.format(v)


for key, title, f, better in METRICS:
    print(f"\n### {title}\n")
    ratio_head = ("new ÷ faster other" if better == "high" and key != "peak"
                  else "faster other ÷ new" if key == "ttft" else "new ÷ lowest other")
    print(f"| Context | Deck (15 Nov) | PR #188 (17 Nov pm) | Now (PR #192) | Now vs deck | llama.cpp | mlx-vlm | {ratio_head} |")
    print("|---|---|---|---|---|---|---|---|")
    for size in SIZES:
        d, p, n = agg(key, [r.get(key) for r in ok(deck[size])]), pr188_val(size, key), agg(key, [r.get(key) for r in ok(new[size])])
        l, m = agg(key, [r.get(key) for r in ok(llama[size])]), agg(key, [r.get(key) for r in ok(mlx[size])])
        if n is None:
            ratio = None
        elif key in ("read_tps", "gen_tps"):
            ratio = n / max(l, m)
        elif key == "ttft":
            ratio = min(l, m) / n
        else:
            ratio = n / min(l, m)
        chg = None if n is None or d is None else 100 * (n - d) / d
        print(f"| {size.upper()} | {fmt(f, d)} | {fmt(f, p)} | **{fmt(f, n)}** | "
              f"{'-' if chg is None else f'{chg:+.1f}%'} | {fmt(f, l)} | {fmt(f, m)} | "
              f"{'-' if ratio is None else f'{ratio:.2f}×'} |")

print("\n### Facts recalled (of 2) and draft guesses taken\n")
print("| Context | Facts deck → PR #188 → now (each run) | Guesses deck | Guesses PR #188 | Guesses now |")
print("|---|---|---|---|---|")
for size in SIZES:
    fs = lambda rows: ", ".join(str(r.get("recall", "err")) for r in rows) or "-"
    p_rows = pr188.get(size, []) + pr188_rerun.get(size, [])
    print(f"| {size.upper()} | {fs(deck[size])} → {fs(p_rows)} → **{fs(new[size])}** | "
          f"{gstr(g_deck[size])} | {gstr(g_188[size])} | {gstr(g_new[size])} |")

print("\n### Each run now\n")
print("| Context | Run | Finished | Read | Read tok/s | First token (s) | Written | Write tok/s | Peak GB | Facts | Guesses |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for size in SIZES:
    for i, r in enumerate(new[size]):
        g = gstr([g_new[size][i]]) if i < len(g_new[size]) else "-"
        if "error" in r:
            print(f"| {size.upper()} | {i + 1} | error: {r['error'][:120]} |||||||||")
            continue
        print(f"| {size.upper()} | {i + 1} | {r['time'][11:]} | {r['read']:,} | {r['read_tps']:.0f} | {r['ttft']:.1f} | "
              f"{r['gen']} | {r['gen_tps']:.1f} | {r['peak']:.1f} | {r['recall']}/2 | {g} |")


def memcell(sets, key, size, each=False):
    xs = [x for m in sets for x in m[size][key]]
    if not xs:
        return "-"
    return " / ".join(f"{x:.1f}" for x in xs) if each else f"{(max(xs) if key == 'peak' else avg(xs)):.1f}"


m_deck, m_new = mem(NOV15 / "results.jsonl"), mem(HERE / "results.jsonl")
m_188 = mem(PR188 / "long-context.jsonl")
m_188r = mem(PR188 / "rerun-long-context-128k.jsonl")
m_l, m_m = mem(NOV15 / "results.jsonl", "llama"), mem(NOV15 / "results.jsonl", "mlx")
print("\n### Memory (GB): footprint + resident model file, sampled every 2 s\n")
print("Each run's value, in run order (PR #188 at 128K: main run / cool-down rerun).\n")
print("| Context | What | Deck (15 Nov) | PR #188 (17 Nov pm) | Now (PR #192) | Now − deck (max peak) | llama.cpp | mlx-vlm |")
print("|---|---|---|---|---|---|---|---|")
for size in SIZES:
    for key, label in (("idle", "Idle after load"), ("peak", "Peak, reading + writing"), ("after", "After the reply")):
        diff = ""
        if key == "peak" and m_new[size]["peak"] and m_deck[size]["peak"]:
            diff = f"{max(m_new[size]['peak']) - max(m_deck[size]['peak']):+.1f}"
        print(f"| {size.upper() if key == 'idle' else ''} | {label} | {memcell([m_deck], key, size, True)} | "
              f"{memcell([m_188, m_188r], key, size, True)} | **{memcell([m_new], key, size, True)}** | {diff} | "
              f"{memcell([m_l], key, size, True)} | {memcell([m_m], key, size, True)} |")
