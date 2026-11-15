"""Builds the report page for the Mac's own harness (bench-2026-11-13/bench.py) run on
Windows, in the layout of windows-rtx5060-2026-11-15-tasks.html (bench_tasks_report.py):
Vireo against llama.cpp on the four small models, the Mac's percentages for the same
models beside them, and Vireo before and after the merge of PR #167 from two runs of
bench_tasks.py. Every number and percentage on the page is worked out here from the
result files; the prose is in <run>/report.json, with {key}s filled from them.

    python evals/speed/bench_mac_harness_report.py evals/speed/bench-2026-11-15-windows-mac-harness \
        evals/speed/bench-2026-11-15-windows-tasks evals/speed/bench-2026-11-15-windows-tasks-after-merge \
        evals/speed/windows-rtx5060-2026-11-15-mac-harness.html [--fill]

Then print it: msedge --headless=new --no-pdf-header-footer --print-to-pdf=<out.pdf> "file:///<page>?theme=dark"
"""
import html
import json
import math
import re
import statistics
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
run, before_dir, after_dir, out = (Path(a) for a in sys.argv[1:5])
MAC = here / "bench-2026-11-13" / "results-q6-128k.jsonl"
MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B"]
QUANT = {m: ("Q4_0" if m.startswith("Gemma") else "Q4_K_M") for m in MODELS}
ENGINES = [("vireo", "Vireo"), ("llama", "llama.cpp")]
esc = html.escape


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


info = json.loads((run / "report.json").read_text(encoding="utf-8"))
WIN_ROWS = load(run / "results.jsonl")
MAC_ROWS = load(MAC)


# ---- the Mac harness's rows, read the way its aggregate.py reads them ---------------------

def right(work, text, row):
    """aggregate.py's scoring, as on the Mac."""
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


def record(rows, model, engine, work):
    of = [d for d in rows if d.get("model") == model and d.get("engine") == engine and d.get("work") == work]
    return of[-1] if of else None


def metrics(rows, model, engine, work):
    """One workload's numbers: aggregate.py's, with a task that ended in an error not timed."""
    d = record(rows, model, engine, work)
    if d is None:
        return None
    if work == "load":
        return {"load_s": d["load_s"], "gb": d.get("idle_total", 0) / 2**30}
    if d.get("error"):
        return {"error": d["error"], "right": 0.0}
    rs = d["rows"]
    o = {"wall": d["wall"], "gb": d.get("peak_total", 0) / 2**30}
    errs = [x["error"] for x in rs if x.get("error")]
    if work == "multiturn":
        later = [x["ttft"] for x in rs[1:] if x.get("ttft") is not None]
        o.update(ttft=statistics.mean(later) if later else None, total=sum(x["total"] for x in rs),
                 right=statistics.mean(right(work, x["text"], x) for x in rs),
                 turns_right=sum(bool(x.get("right")) for x in rs), turns=len(rs),
                 read_turn=statistics.mean(sum(s.get("read") or 0 for s in x["steps"]) for x in rs[1:]))
    else:
        r, st = rs[0], rs[0]["steps"]
        o.update(ttft=r["ttft"], total=r["total"], right=right(work, r["text"], r), calls=r["calls"],
                 requests=len(st), read=sum(s.get("read") or 0 for s in st), gen=sum(s.get("gen") or 0 for s in st),
                 text=r["text"], wrote=r.get("wrote"))
        if work in ("short", "medium", "long", "large", "xlarge") and st:
            s0 = st[0]
            o.update(prompt=s0.get("prompt"), read_tps=s0["read"] / s0["read_s"] if s0.get("read_s") else None,
                     gen_tps=s0["gen"] / s0["gen_s"] if s0.get("gen_s") else None)
    if errs:
        o["error"] = errs[0]
        o["right"] = 0.0
    return o


def val(model, engine, work, field, rows=None):
    rows = WIN_ROWS if rows is None else rows
    if work == "memory":  # the most GPU memory in use after any workload
        gbs = [m["gb"] for w in ("short", "medium", "long", "large", "xlarge", "multiturn", "tools", "skill", "longtools")
               if (m := metrics(rows, model, engine, w)) and m.get("gb")]
        return max(gbs) if gbs else None
    m = metrics(rows, model, engine, work)
    if m is None or (m.get("error") and field in ("total", "ttft", "wall", "read_tps", "gen_tps")):
        return None
    return m.get(field)


# kind "speed": Vireo ÷ llama.cpp − 1; "time" and "memory": llama.cpp ÷ Vireo − 1, as on the Mac's page.
STATS = [
    ("Short reply writing", "Short reply\nwriting", "short", "gen_tps", "speed"),
    ("Reading ~3K", "Reading\n~3K", "medium", "read_tps", "speed"),
    ("Reading ~32K", "Reading\n~32K", "long", "read_tps", "speed"),
    ("Writing after ~32K", "Writing\nafter ~32K", "long", "gen_tps", "speed"),
    ("First token after ~32K", "First token\nafter ~32K", "long", "ttft", "time"),
    ("Tool task time", "Tool task\ntime", "tools", "total", "time"),
    ("Skill task time", "Skill task\ntime", "skill", "total", "time"),
    ("Long tool task time", "Long tool\ntask time", "longtools", "total", "time"),
    ("GPU memory, highest", "GPU memory,\nhighest", "memory", "gb", "memory"),
]


def diff(model, work, field, kind, rows=None):
    v, o = val(model, "vireo", work, field, rows), val(model, "llama", work, field, rows)
    if not v or not o:
        return None
    return v / o - 1 if kind == "speed" else o / v - 1


def signed(p, digits=0):
    p = round(100 * p, digits)
    return ("+" if p > 0 else "−" if p < 0 else "") + f"{abs(p):.{digits}f}%"


def span(ps):
    if not ps:
        return "–"
    lo, hi = min(ps), max(ps)
    return signed(lo) if round(100 * lo) == round(100 * hi) else f"{signed(lo)} to {signed(hi)}"


def counts(ps):
    return sum(p > 0.005 for p in ps), sum(p < -0.005 for p in ps), len(ps)


def keystat(work, field, kind, rows=None):
    return [p for m in MODELS if (p := diff(m, work, field, kind, rows)) is not None]


def n(v):
    if v is None:
        return "–"
    return f"{v:,.0f}" if v >= 1000 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v < 1 else f"{v:.1f}"


SCORED = ["long", "multiturn", "tools", "skill", "longtools"]
SCORED_NAMES = {"long": "long recall", "multiturn": "multiturn", "tools": "tool", "skill": "skill", "longtools": "long tool"}


def scored(model, engine, rows=None):
    return {w: (val(model, engine, w, "right", rows) or 0) >= 1 for w in SCORED}


# ---- Vireo before and after the merge: bench_tasks.py's two runs ---------------------------
BA_ROWS = {"before": load(before_dir / "results.jsonl"), "after": load(after_dir / "results.jsonl")}


def ba(when, model, engine, work, field):
    of = [r for r in BA_ROWS[when] if r["model"] == model and r["engine"] == engine and r.get("work") == work]
    r = of[-1] if of else None
    if r is None or "error" in r or "stream_error" in r or r.get(field) is None:
        return None
    return r[field]


def ba_error(when, model, engine, work):
    of = [r for r in BA_ROWS[when] if r["model"] == model and r["engine"] == engine and r.get("work") == work]
    r = of[-1] if of else None
    if r is None:
        return "not measured"
    if "stream_error" in r:
        return "task errored: " + r["stream_error"][:50]
    if "error" in r:
        return "task errored: " + str(r["error"]).replace("<", "").replace(">", "")[:50]
    return None


def ba_score(when, model, engine="vireo"):
    of = lambda w: [r for r in BA_ROWS[when] if r["model"] == model and r["engine"] == engine and r.get("work") == w]  # noqa: E731
    return sum(bool(of(w) and of(w)[-1].get("right")) for w in ("long", "multiturn", "tool", "skill", "longtool"))


BA = [("Short reply writing (tok/s)", "short", "gen_tps", "speed"), ("Reading ~3K (tok/s)", "medium", "prompt_tps", "speed"),
      ("Reading the long prompt (tok/s)", "long", "prompt_tps", "speed"), ("Tool task (s)", "tool", "wall_s", "time"),
      ("Skill task (s)", "skill", "wall_s", "time"), ("Long tool task (s)", "longtool", "wall_s", "time")]


def ba_cell(model, work, field, engine="vireo"):
    b, a = ba("before", model, engine, work, field), ba("after", model, engine, work, field)
    fmt = lambda v: "error" if v is None else n(v)  # noqa: E731
    return f"{fmt(b)} → <b>{fmt(a)}</b>"


# ---- bar panels -----------------------------------------------------------------------

def nice_max(v):
    p = 10 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * p >= v:
            return m * p
    return 10 * p


def bars(values, unit, fmt=n, max_value=None, keys=("vireo", "llama")):
    """values: per model, a pair; a value may be a string (an error's text)."""
    W, LEFT, RIGHT, TOP, BAR, GAP, ROWGAP = 520, 104, 58, 6, 13, 2, 22
    GROUP = 2 * BAR + GAP
    H = TOP + len(MODELS) * GROUP + (len(MODELS) - 1) * ROWGAP + 26
    nums = [x for pair in values for x in pair if isinstance(x, (int, float))]
    top = max_value or nice_max(max(nums))
    x = lambda v: LEFT + (W - LEFT - RIGHT) * v / top  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(unit)}">']
    bottom = H - 22
    for i in range(5):
        v = top * i / 4
        s.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{TOP - 2}" y2="{bottom}" stroke="var(--grid)" stroke-width="1"/>')
        label = f"{v:,.0f}" if float(v).is_integer() else f"{v:,.4g}"
        s.append(f'<text x="{x(v):.1f}" y="{H - 6}" text-anchor="middle" font-size="11" fill="var(--fg-3)" class="num">{label}</text>')
    for i, m in enumerate(MODELS):
        y0 = TOP + i * (GROUP + ROWGAP)
        s.append(f'<text x="0" y="{y0 + GROUP / 2 - 3}" font-size="13" font-weight="500" fill="var(--fg)">{esc(m)}</text>')
        s.append(f'<text x="0" y="{y0 + GROUP / 2 + 12}" font-size="11" fill="var(--fg-3)" class="num">{QUANT[m]}</text>')
        for j, (k, v) in enumerate(zip(keys, values[i])):
            y = y0 + j * (BAR + GAP)
            if not isinstance(v, (int, float)):
                s.append(f'<text x="{LEFT + 2}" y="{y + BAR - 2.5}" font-size="11" font-style="italic" fill="var(--behind-fg)">{esc(v or "not measured")}</text>')
                continue
            w = max(2, x(v) - LEFT)
            r = min(4, w / 2)
            s.append(f'<path d="M{LEFT},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{BAR - 2 * r} a{r},{r} 0 0 1 -{r},{r} h{-(w - r):.1f} z" fill="var(--{k})"/>')
            s.append(f'<text x="{x(v) + 6:.1f}" y="{y + BAR - 2.5}" font-size="11.5" fill="var(--fg-2)" class="num">{fmt(v)}</text>')
    s.append("</svg>")
    return "".join(s)


def verdict(work, field, kind):
    ps = keystat(work, field, kind)
    if not ps:
        return "", "even"
    lo, hi = min(ps), max(ps)
    if lo >= 0.05:
        return f"Vireo ahead on all {len(ps)} · {span(ps)}", "ahead"
    if hi <= -0.05:
        return f"Vireo behind on all {len(ps)} · {span(ps)}", "behind"
    if lo > -0.05 and hi < 0.05:
        return f"Level · {span(ps)}", "even"
    a, b, t = counts(ps)
    return f"{a} ahead · {b} behind · {span(ps)}", "even"


def error_text(model, engine, work):
    m = metrics(WIN_ROWS, model, engine, work)
    if m is None:
        return "not measured"
    if m.get("error"):
        return "task errored: " + re.sub(r"[<>]", "", str(m["error"]))[:55]
    return None


def panel(title, sub, work, field, kind, note, fmt=n):
    values = [tuple(val(m, e, work, field) if val(m, e, work, field) is not None else error_text(m, e, work)
                    for e, _ in ENGINES) for m in MODELS]
    text, cls = verdict(work, field, kind)
    return (f'<article class="panel"><div class="panel-head"><div><h3>{esc(title)}</h3><div class="sub">{esc(sub)}</div></div>'
            f'<span class="verdict {cls}">{esc(text)}</span></div>{bars(values, title, fmt)}'
            + (f'<p class="note">{note}</p>' if note else "") + "</article>")


def score_panel():
    values = [tuple(sum(scored(m, e).values()) for e, _ in ENGINES) for m in MODELS]
    return ('<article class="panel"><div class="panel-head"><div><h3>Scored tasks right</h3>'
            '<div class="sub">fully right, of 5 · higher is better</div></div></div>'
            + bars(values, "Scored tasks right", fmt=lambda v: f"{v:.0f}", max_value=5)
            + f'<p class="note">{t(info["panels"]["score"])}</p></article>')


def ba_panel(title, sub, work, field, note):
    values = [tuple(ba(w, m, "vireo", work, field) if ba(w, m, "vireo", work, field) is not None else ba_error(w, m, "vireo", work)
                    for w in ("before", "after")) for m in MODELS]
    return (f'<article class="panel"><div class="panel-head"><div><h3>{esc(title)}</h3><div class="sub">{esc(sub)}</div></div></div>'
            f'{bars(values, title, keys=("was", "vireo"))}' + (f'<p class="note">{note}</p>' if note else "") + "</article>")


# ---- tables ---------------------------------------------------------------------------

def stat_rows():
    out_rows = []
    for title, _, work, field, kind in STATS:
        ps, mac = keystat(work, field, kind), keystat(work, field, kind, MAC_ROWS)
        a, b, tt = counts(ps)
        ma, mb, mt = counts(mac)
        out_rows.append(f'<tr><td>{esc(title)}</td><td><b>{span(ps)}</b> <span class="med">· median {signed(statistics.median(ps)) if ps else "–"}</span></td>'
                        f'<td>{a} ahead · {b} behind <span class="med">of {tt}</span></td>'
                        f'<td class="mac">{span(mac)} <span class="med">· median {signed(statistics.median(mac)) if mac else "–"} · {ma} ahead · {mb} behind of {mt}</span></td></tr>')
    return "".join(out_rows)


def heat_cell(p):
    if p is None:
        return '<td class="heat">–</td>'
    strength = min(abs(p), 0.6) / 0.6
    colour = "var(--vireo)" if p > 0 else "var(--llama)"
    mix = round(12 + 78 * strength)
    ink = "#fff" if mix > 55 else "var(--fg)"
    return f'<td class="heat" style="background:color-mix(in srgb, {colour} {mix}%, var(--surface));color:{ink}">{signed(p)}</td>'


def heatmap(rows=None):
    head = "".join(f"<th>{esc(short).replace(chr(10), '<br>')}</th>" for _, short, *_ in STATS)
    body = "".join(f"<tr><td>{esc(m)}</td>" + "".join(heat_cell(diff(m, w, f, k, rows)) for _, _, w, f, k in STATS) + "</tr>"
                   for m in MODELS)
    return f'<table class="heatmap"><tr><th></th>{head}</tr>{body}</table>'


def score_table():
    body = ""
    for m in MODELS:
        cells = ""
        for e, _ in ENGINES:
            s = scored(m, e)
            wrong = [SCORED_NAMES[w] for w, ok in s.items() if not ok]
            cells += f'<td><b>{sum(s.values())}</b> <span class="med">/ 5</span></td><td class="wrong">{esc(", ".join(wrong) or "all right")}</td>'
        mac = " · ".join(f"{sum(scored(m, e, MAC_ROWS).values())}" for e, _ in ENGINES)
        body += f'<tr><td>{esc(m)}</td>{cells}<td class="mac">{mac}</td></tr>'
    return ('<table class="scores"><tr><th>Model</th><th>Vireo</th><th>Vireo got wrong</th><th>llama.cpp</th>'
            f'<th>llama.cpp got wrong</th><th>Mac, Vireo · llama.cpp</th></tr>{body}</table>')


def detail_table():
    body = ""
    for m in MODELS:
        for e, label in ENGINES:
            ld = metrics(WIN_ROWS, m, e, "load") or {}
            mt = metrics(WIN_ROWS, m, e, "multiturn") or {}
            body += (f"<tr><td>{esc(m)}</td><td>{label}</td><td>{n(ld.get('load_s'))}</td><td>{CONTEXT[m]:,}</td>"
                     f"<td>{val(m, e, 'long', 'prompt') or 0:,}</td><td>{n(val(m, e, 'short', 'ttft'))}</td>"
                     f"<td>{n(val(m, e, 'long', 'ttft'))}</td><td>{n(mt.get('ttft'))}</td><td>{n(mt.get('total'))}</td>"
                     f"<td>{ld.get('gb', 0):.1f}</td><td>{n(val(m, e, 'memory', 'gb'))}</td></tr>")
    return ('<table class="detail"><tr><th>Model</th><th>Engine</th><th>Load (s)</th><th>Context</th>'
            '<th>~32K prompt, tokens</th><th>First token, short (s)</th><th>First token, ~32K (s)</th>'
            '<th>Multiturn, first token a turn (s)</th><th>Multiturn, 8 turns (s)</th>'
            f'<th>GPU memory loaded (GB)</th><th>GPU memory, highest (GB)</th></tr>{body}</table>')


def task_tokens():
    body = ""
    for m in MODELS:
        cells = ""
        for w in ("tools", "skill", "longtools"):
            for e, _ in ENGINES:
                x = metrics(WIN_ROWS, m, e, w)
                cells += (f"<td>{x['read']:,} · {x['gen']:,} · {len(x['calls'])}</td>" if x and "read" in x else "<td>–</td>")
        body += f"<tr><td>{esc(m)}</td>{cells}</tr>"
    head = "".join(f"<th>{w}, {e}</th>" for w in ("Tool", "Skill", "Long tool") for e in ("Vireo", "llama.cpp"))
    return f'<table class="detail tokens"><tr><th>Model</th>{head}</tr>{body}</table>'


def ba_table():
    head = "".join(f"<th>{esc(title)}</th>" for title, *_ in BA) + "<th>Scored tasks right, of 5</th>"
    body = ""
    for m in MODELS:
        body += (f"<tr><td>{esc(m)}</td>" + "".join(f"<td>{ba_cell(m, w, f)}</td>" for _, w, f, _ in BA)
                 + f"<td>{ba_score('before', m)} → <b>{ba_score('after', m)}</b></td></tr>")
    return f'<table class="detail ba"><tr><th>Vireo</th>{head}</tr>{body}</table>'


# ---- the fill for report.json's {key}s -----------------------------------------------------
CONTEXT = {}
for m in MODELS:
    # The context both engines ran with: Vireo's log says it; the harness's table sets it.
    CONTEXT[m] = info["contexts"][m]

fill = {}
for _, _, w, f, k in STATS:
    key = f"{w}.{f}"
    for prefix, rows in (("", WIN_ROWS), ("mac:", MAC_ROWS)):
        ps = keystat(w, f, k, rows)
        a, b, tt = counts(ps)
        fill[prefix + key] = span(ps)
        fill[prefix + key + ".ahead"], fill[prefix + key + ".behind"], fill[prefix + key + ".of"] = a, b, tt
        fill[prefix + key + ".median"] = signed(statistics.median(ps)) if ps else "–"
        for m in MODELS:
            p = diff(m, w, f, k, rows)
            fill[f"{prefix}{m}|{key}"] = signed(p) if p is not None else "–"
for m in MODELS:
    for e, _ in ENGINES:
        for prefix, rows in (("", WIN_ROWS), ("mac:", MAC_ROWS)):
            fill[f"{prefix}{m}|{e}|score"] = sum(scored(m, e, rows).values())
            for w in ("short", "medium", "long", "multiturn", "tools", "skill", "longtools", "memory"):
                for f in (("gb",) if w == "memory" else ("gen_tps", "read_tps", "ttft", "total", "prompt", "gb", "wall")):
                    if (v := val(m, e, w, f, rows)) is not None:
                        fill[f"{prefix}{m}|{e}|{w}.{f}"] = f"{v:,}" if f == "prompt" else f"{v:.1f}" if f == "gb" else n(v)
        ld = metrics(WIN_ROWS, m, e, "load")
        if ld:
            fill[f"{m}|{e}|load.gb"] = f"{ld['gb']:.1f}"
    fill[f"{m}|context"] = f"{CONTEXT[m]:,}"
    for _, w, f, kind in BA:
        for e, _ in ENGINES:
            b, a = ba("before", m, e, w, f), ba("after", m, e, w, f)
            fill[f"ba:{m}|{e}|{w}.{f}"] = f"{n(b) if b is not None else 'error'} → {n(a) if a is not None else 'error'}"
            if a and b:
                fill[f"ba:{m}|{e}|{w}.{f}.change"] = signed(a / b - 1 if kind == "speed" else b / a - 1)
    fill[f"ba:{m}|score"] = f"{ba_score('before', m)} → {ba_score('after', m)}"
    fill[f"ba:{m}|llama|score"] = f"{ba_score('before', m, 'llama')} → {ba_score('after', m, 'llama')}"
for prefix, rows in (("", WIN_ROWS), ("mac:", MAC_ROWS)):
    for e, _ in ENGINES:
        fill[f"{prefix}{e}_total"] = sum(fill[f"{prefix}{m}|{e}|score"] for m in MODELS)
fill["ba:vireo_total"] = f"{sum(ba_score('before', m) for m in MODELS)} → {sum(ba_score('after', m) for m in MODELS)}"
fill["ba:llama_total"] = f"{sum(ba_score('before', m, 'llama') for m in MODELS)} → {sum(ba_score('after', m, 'llama') for m in MODELS)}"
for _, w, f, kind in BA:
    for e, _ in ENGINES:
        ch = [(a / b - 1 if kind == "speed" else b / a - 1) for m in MODELS
              if (a := ba("after", m, e, w, f)) and (b := ba("before", m, e, w, f))]
        fill[f"ba:{e}|{w}.{f}"] = span(ch)


def t(text):
    def one(mm):
        if mm[1] not in fill:
            raise KeyError(f"report.json names {mm[1]!r}, which the rows don't have")
        return str(fill[mm[1]])
    return re.sub(r"\{([^{}]+)\}", one, text)


# ---- the page -------------------------------------------------------------------------
style = (here / "windows-rtx5060-2026-11-15.html").read_text(encoding="utf-8")
style = style[style.index("<style>") + 7: style.index("</style>")]
EXTRA = """
:root { --was: #8b94a1; }
.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 760px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; display: grid; gap: 4px; align-content: start; }
.card b { font-family: var(--display); font-weight: 600; font-size: 26px; line-height: 1.15; }
.card span { font-size: 13px; color: var(--fg-2); }
section.block { display: grid; gap: 14px; }
section.block > h2 { font-family: var(--display); font-weight: 600; font-size: 24px; margin: 0; }
section.block > p { margin: 0; color: var(--fg-2); font-size: 14px; max-width: 100ch; }
.panel h3 { font-family: var(--display); font-weight: 600; font-size: 18px; margin: 0; }
.box { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; }
.box ul, .box ol { margin: 0; padding-left: 20px; display: grid; gap: 6px; font-size: 14px; }
.box li b { color: var(--fg); }
.box li { color: var(--fg-2); }
table.keystats td:first-child { font-weight: 500; }
table.keystats td { font-family: var(--body); text-align: left; white-space: normal; }
table.keystats td:nth-child(n+2) { font-family: var(--mono); }
table.keystats td.mac { color: var(--fg-2); }
table.keystats th.mac, table.scores th:last-child { color: var(--fg-3); }
.med { color: var(--fg-3); font-size: 12px; }
table.heatmap { min-width: 0; table-layout: fixed; border-collapse: separate; border-spacing: 3px; }
table.heatmap th { text-align: center; white-space: normal; font-size: 10.5px; line-height: 1.25; vertical-align: bottom; border: 0; padding: 4px 2px; }
table.heatmap td { border: 0; }
table.heatmap td:first-child { width: 130px; font-weight: 500; font-family: var(--body); }
table.heatmap td.heat { text-align: center; font-weight: 600; padding: 12px 2px; border-radius: 3px; font-family: var(--mono); }
.scale { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--fg-3); }
.scale .ramp { width: 220px; height: 10px; border-radius: 3px; background: linear-gradient(90deg, var(--llama), var(--surface), var(--vireo)); border: 1px solid var(--line); }
table.scores, table.detail { min-width: 0; font-size: 13px; }
table.scores td, table.detail td { font-family: var(--body); }
table.scores td.wrong { color: var(--fg-2); white-space: normal; text-align: left; }
table.scores td.mac { color: var(--fg-2); font-family: var(--mono); }
table.scores th, table.scores td, table.detail th { text-align: left; }
table.detail th { white-space: normal; vertical-align: bottom; text-align: right; }
table.detail td:nth-child(n+3) { font-family: var(--mono); }
table.detail th:nth-child(-n+2) { text-align: left; }
table.tokens td:nth-child(n+2) { font-family: var(--mono); text-align: right; }
table.ba td:nth-child(n+2) { font-family: var(--mono); text-align: right; }
table.ba th:first-child, table.ba td:first-child { text-align: left; }
h3.sub2 { font-family: var(--display); font-weight: 600; font-size: 16px; margin: 6px 0 0; color: var(--fg-2); }
.panel-head .verdict { white-space: normal; text-align: right; }
@media print {
  .pagebreak { break-before: page; }
  .box, table, .cards, .panel { break-inside: avoid; }
  section.block > h2 { break-after: avoid; }
}
"""


def legend(pairs=(("vireo", "Vireo"), ("llama", "llama.cpp"))):
    return '<div class="legend">' + "".join(
        f'<span class="key"><span class="sw" style="background:var(--{k})"></span>{esc(lab)}</span>' for k, lab in pairs) + "</div>"


def bullets(items, tag="ul"):
    return f"<{tag}>" + "".join(f"<li>{t(i)}</li>" for i in items) + f"</{tag}>"


P = info["panels"]
cards = "".join(f'<div class="card"><b>{esc(t(c["big"]))}</b><span>{esc(t(c["text"]))}</span></div>' for c in info["cards"])
rig = "".join(f'<div><dt>{esc(x["label"])}</dt><dd>{esc(x["value"])}<small>{esc(x["detail"])}</small></dd></div>' for x in info["rig"])
BA_LEGEND = legend((("was", "Vireo, before the merge (28ddaeb)"), ("vireo", "Vireo, after (7e00a19)")))

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(info["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap">
<style>{style}{EXTRA}</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="top"><div class="eyebrow">{esc(info["eyebrow"])}</div><button class="theme" id="theme" type="button"></button></div>
    <h1>{esc(t(info["headline"]))}</h1>
    <p class="lede">{esc(t(info["lede"]))}</p>
  </header>
  <div class="cards">{cards}</div>
  <dl class="rig" aria-label="Test machine">{rig}</dl>

  <section class="block">
    <h2>Key stats: Vireo's difference, across the four models</h2>
    <p>Positive = Vireo better. Speeds: Vireo ÷ llama.cpp − 1. Times and memory: llama.cpp ÷ Vireo − 1 (so +100% on a task time means llama.cpp took twice as long). Lowest to highest of the models, with the median; "ahead/behind" counts models more than 0.5% either way. The last column is the Mac's run of the same harness on the same four models (M5 Max, Q6, a 131,072-token context; its memory is the server process's peak, its 128K prompt included): a different machine, for the direction of each difference, not its size.</p>
    <div class="panel"><div class="tablewrap" style="margin-top:0"><table class="keystats"><tr><th>Workload</th><th>Windows: Vireo vs llama.cpp</th><th>Models</th><th class="mac">Mac, 14 Nov (Q6, 131,072 context)</th></tr>{stat_rows()}</table></div></div>
  </section>

  <section class="block">
    <h2>Where Vireo is ahead and where it is behind</h2>
    <div class="box">{bullets(info["where"])}</div>
  </section>

  <section class="block">
    <h2>How this run differs from the Mac's</h2>
    <div class="box">{bullets(info["differs"])}</div>
    <p class="method">{t(info["method"])}</p>
  </section>

  <section class="block pagebreak">
    <h2>Vireo relative to llama.cpp, per model</h2>
    <p>Blue: Vireo ahead. Orange: Vireo behind. Same formulas as the key stats; – = not measured (a task that ended in an error).</p>
    <div class="panel">{heatmap()}
    <h3 class="sub2">The Mac, same four models (M5 Max, Q6, 131,072 context)</h3>{heatmap(MAC_ROWS)}
    <div class="scale"><span>−60% or worse</span><span class="ramp"></span><span>+60% or better</span></div></div>
  </section>

  <section class="block pagebreak">
    <h2>Writing speed, first token and memory</h2>
    {legend()}
    <div class="panels">
      {panel("Writing, short reply", "generation after a ~40-token question · tokens/s · higher is better", "short", "gen_tps", "speed", t(P["short"]))}
      {panel("Writing after the ~32K prompt", "generation after long.txt · tokens/s · higher is better", "long", "gen_tps", "speed", t(P["longwrite"]))}
      {panel("Time to first token, ~32K prompt", "seconds until the first streamed token · lower is better", "long", "ttft", "time", t(P["ttft"]))}
      {panel("GPU memory, highest", "the whole GPU's in use after any workload · GB · lower is better", "memory", "gb", "memory", t(P["memory"]), fmt=lambda v: f"{v:.1f}")}
    </div>
  </section>

  <section class="block pagebreak">
    <h2>Prompt reading speed</h2>
    {legend()}
    <div class="panels">
      {panel("~3K-token prompt", "medium.txt · tokens/s · higher is better", "medium", "read_tps", "speed", t(P["medium"]))}
      {panel("~32K-token prompt", "long.txt whole, two facts hidden in it · tokens/s · higher is better", "long", "read_tps", "speed", t(P["long"]))}
    </div>
  </section>

  <section class="block">
    <h2>Whole agent tasks and answers</h2>
    {legend()}
    <p>Time for the whole task, every tool call included (the Mac harness's "total").</p>
    <div class="panels">
      {panel("Tool task", "list, read the quarterly files, calculate, answer · s · lower is better", "tools", "total", "time", t(P["tool"]))}
      {panel("Skill task", "weekly-report skill: list, read, calculate, write a file · s · lower is better", "skill", "total", "time", t(P["skill"]))}
      {panel("Long tool task", "a ~55K-token log pasted in, then tool calls · s · lower is better", "longtools", "total", "time", t(P["longtool"]))}
      {score_panel()}
    </div>
  </section>

  <section class="block pagebreak">
    <h2>Vireo before and after the merge</h2>
    {BA_LEGEND}
    <p>{t(info["ba_intro"])}</p>
    <div class="panels">
      {ba_panel("Tool task", "whole-task wall time · s · lower is better", "tool", "wall_s", t(P["ba_tool"]))}
      {ba_panel("Skill task", "whole-task wall time · s · lower is better", "skill", "wall_s", t(P["ba_skill"]))}
      {ba_panel("Long tool task", "whole-task wall time · s · lower is better", "longtool", "wall_s", t(P["ba_longtool"]))}
      {ba_panel("Writing, short reply", "generation speed · tokens/s · higher is better", "short", "gen_tps", t(P["ba_short"]))}
    </div>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{ba_table()}</div>
    <p class="note">{t(info["ba_note"])}</p></div>
  </section>

  <section class="block pagebreak">
    <h2>What failed or looked odd</h2>
    <div class="box">{bullets(info["odd"], "ol")}</div>
  </section>

  <section class="block">
    <h2>Scored tasks right, per model and engine</h2>
    <p>Fully right out of 5, scored as the Mac's aggregate.py scores them: long recall (both hidden facts), multiturn (all 8 turns), tool (the year's total, $26,742.25), skill (78 hours said and a file written), long tool (inventory-sync named). The Mac's 6 included its ~96K recall prompt; its scores here are the same five.</p>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{score_table()}</div></div>
  </section>

  <section class="block">
    <h2>Load, context, first tokens and memory</h2>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{detail_table()}</div>
    <p class="note">{t(info["detail_note"])}</p></div>
  </section>

  <section class="block">
    <h2>Agent tasks: tokens read, tokens written, tool calls</h2>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{task_tokens()}</div>
    <p class="note">{t(info["tokens_note"])}</p></div>
  </section>

</div>
<script>
const themeBtn = document.getElementById("theme"), root = document.documentElement;
const systemDark = () => matchMedia("(prefers-color-scheme: dark)").matches;
const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : systemDark();
function setTheme(t, keep) {{
  if (t) root.dataset.theme = t; else delete root.dataset.theme;
  if (keep) {{ try {{ localStorage.setItem("vireo-speed-theme", t); }} catch (e) {{}} }}
  themeBtn.textContent = isDark() ? "Light mode" : "Dark mode";
}}
let saved = new URLSearchParams(location.search).get("theme");
if (!saved) {{ try {{ saved = localStorage.getItem("vireo-speed-theme"); }} catch (e) {{}} }}
setTheme(saved === "dark" || saved === "light" ? saved : null, false);
themeBtn.addEventListener("click", () => setTheme(isDark() ? "light" : "dark", true));
</script>
</body>
</html>
"""
out.write_text(page, encoding="utf-8", newline="\n")
if "--fill" in sys.argv:
    print(json.dumps(fill, indent=1, ensure_ascii=False))
print("ok", out)
