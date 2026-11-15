"""Builds the report page for the Mac's own harness (bench-2026-11-13/bench.py) run on
Windows after the three changes of 15 November (Qwen's saved state at the system turn's
end, sliding-window rings on CUDA, the prompt attention kernel's f16 queries), in the
layout of windows-rtx5060-2026-11-15-mac-harness.html (bench_mac_harness_report.py) but
Windows only: Vireo against llama.cpp on the four small models now; Vireo this morning
(7e00a19) against now; the project's eval (vireo-eval) on both builds; and the Gemmas at a
131,072-token context. Every number and percentage on the page is worked out here from the
result files; the prose is in <run>/report.json, with {key}s filled from them.

    python evals/speed/bench_after_changes_report.py evals/speed/bench-2026-11-15-windows-after-changes \
        evals/speed/bench-2026-11-15-windows-mac-harness evals/windows-cuda-changes.json \
        evals/windows-cuda-changes-before.json evals/speed/windows-rtx5060-2026-11-15-after-changes.html [--fill]

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
run, morning, ev_after_path, ev_before_path, out = (Path(a) for a in sys.argv[1:6])
MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B"]
GGUF = {"Gemma 4 E2B": "gemma-4-E2B_q4_0-it", "Gemma 4 E4B": "gemma-4-E4B_q4_0-it",
        "Qwen 3.5 4B": "Qwen3.5-4B-Q4_K_M", "Qwen 3.5 9B": "Qwen3.5-9B-Q4_K_M"}
QUANT = {m: ("Q4_0" if m.startswith("Gemma") else "Q4_K_M") for m in MODELS}
ENGINES = [("vireo", "Vireo"), ("llama", "llama.cpp")]
esc = html.escape


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


info = json.loads((run / "report.json").read_text(encoding="utf-8"))
NEW = load(run / "results.jsonl")
OLD = load(morning / "results.jsonl")
PROBE_PATH = run / "probe-131k.jsonl"
PROBE = load(PROBE_PATH) if PROBE_PATH.exists() else []
# The eval's before run may still be going: the page is built without its section then.
EV_DONE = ev_after_path.exists() and ev_before_path.exists()
EV = {"after": json.loads(ev_after_path.read_text(encoding="utf-8")),
      "before": json.loads(ev_before_path.read_text(encoding="utf-8"))} if EV_DONE else {}


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
            o.update(prompt=s0.get("prompt"), kept=s0.get("cached"), read_tps=s0["read"] / s0["read_s"] if s0.get("read_s") else None,
                     gen_tps=s0["gen"] / s0["gen_s"] if s0.get("gen_s") else None)
    if errs:
        o["error"] = errs[0]
        o["right"] = 0.0
    return o


def val(model, engine, work, field, rows=None):
    rows = NEW if rows is None else rows
    if work == "memory":  # the most GPU memory in use after any workload
        gbs = [m["gb"] for w in ("short", "medium", "long", "multiturn", "tools", "skill", "longtools")
               if (m := metrics(rows, model, engine, w)) and m.get("gb")]
        return max(gbs) if gbs else None
    m = metrics(rows, model, engine, work)
    if m is None or (m.get("error") and field in ("total", "ttft", "wall", "read_tps", "gen_tps")):
        return None
    return m.get(field)


# kind "speed": Vireo ÷ llama.cpp − 1; "time" and "memory": llama.cpp ÷ Vireo − 1.
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


# ---- Vireo this morning against now (the same harness, the same contexts) -----------------
# (label, work, field, kind, which change it comes from: a key into report.json's "ba_from")
BA = [("Reading ~3K (tokens/s)", "medium", "read_tps", "speed"),
      ("Reading ~32K (tokens/s)", "long", "read_tps", "speed"),
      ("First token after ~32K (s)", "long", "ttft", "time"),
      ("Writing, short reply (tokens/s)", "short", "gen_tps", "speed"),
      ("Writing after ~32K (tokens/s)", "long", "gen_tps", "speed"),
      ("Tool task (s)", "tools", "total", "time"),
      ("Tool task, tokens read", "tools", "read", "count"),
      ("Skill task (s)", "skill", "total", "time"),
      ("Skill task, tokens read", "skill", "read", "count"),
      ("Long tool task (s)", "longtools", "total", "time"),
      ("Long tool task, tokens read", "longtools", "read", "count"),
      ("GPU memory loaded (GB)", "load", "gb", "memory"),
      ("GPU memory, highest (GB)", "memory", "gb", "memory")]


def change(b, a, kind):
    """Now ÷ this morning − 1, for every kind: a speed up, or a time, a count or memory down,
    is the change; `better` says which way is better."""
    if not a or not b:
        return None
    return a / b - 1


def better(c, kind):
    return c > 0.005 if kind == "speed" else c < -0.005 if kind in ("time", "memory") else False


def gb(v):
    return "–" if v is None else f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def ba_vals(model, engine, work, field):
    return val(model, engine, work, field, OLD), val(model, engine, work, field, NEW)


def ba_cell(model, engine, work, field, kind):
    b, a = ba_vals(model, engine, work, field)
    fmt = gb if kind == "memory" else (lambda v: "–" if v is None else f"{v:,}") if kind == "count" else n
    c = change(b, a, kind)
    tag = "" if c is None else f' <span class="med">{signed(c)}</span>'
    return f"{fmt(b)} → <b>{fmt(a)}</b>{tag}"


def ba_score(model, rows, engine="vireo"):
    return sum(scored(model, engine, rows).values())


# ---- the eval on both builds ----------------------------------------------------------------

def ev_model(when, model):
    return EV[when]["models"][GGUF[model]]


def ev_cases(when, model):
    return {c["id"]: c for c in ev_model(when, model)["cases"]}


def same_answer(a, b):
    return a.get("answer") == b.get("answer") and a.get("calls") == b.get("calls")


EVAL = {}
for m in (MODELS if EV_DONE else []):
    b, a = ev_cases("before", m), ev_cases("after", m)
    flips = [(i, b[i], a[i]) for i in a if i in b and a[i]["passed"] != b[i]["passed"]]
    differ = [i for i in a if i in b and not same_answer(a[i], b[i])]
    EVAL[m] = {"before": ev_model("before", m), "after": ev_model("after", m), "flips": flips, "differ": differ,
               "same": sum(1 for i in a if i in b and same_answer(a[i], b[i])), "cases": len(a)}


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


def error_text(model, engine, work, rows=None):
    m = metrics(NEW if rows is None else rows, model, engine, work)
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


def ba_panel(title, sub, work, field, kind, note, fmt=n):
    values = []
    for m in MODELS:
        b, a = ba_vals(m, "vireo", work, field)
        values.append((b if b is not None else error_text(m, "vireo", work, OLD),
                       a if a is not None else error_text(m, "vireo", work)))
    cs = [c for m in MODELS if (c := change(*ba_vals(m, "vireo", work, field), kind)) is not None]
    return (f'<article class="panel"><div class="panel-head"><div><h3>{esc(title)}</h3><div class="sub">{esc(sub)}</div></div>'
            f'<span class="verdict {"ahead" if cs and all(better(c - (0.025 if kind == "speed" else -0.025), kind) for c in cs) else "even"}">now {esc(span(cs))}</span></div>'
            f'{bars(values, title, fmt, keys=("was", "vireo"))}' + (f'<p class="note">{note}</p>' if note else "") + "</article>")


# ---- tables ---------------------------------------------------------------------------

def stat_rows():
    out_rows = []
    for title, _, work, field, kind in STATS:
        ps, old = keystat(work, field, kind), keystat(work, field, kind, OLD)
        a, b, tt = counts(ps)
        oa, ob, ot = counts(old)
        out_rows.append(f'<tr><td>{esc(title)}</td><td><b>{span(ps)}</b> <span class="med">· median {signed(statistics.median(ps)) if ps else "–"}</span></td>'
                        f'<td>{a} ahead · {b} behind <span class="med">of {tt}</span></td>'
                        f'<td class="mac">{span(old)} <span class="med">· median {signed(statistics.median(old)) if old else "–"} · {oa} ahead · {ob} behind of {ot}</span></td></tr>')
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
        old = " · ".join(f"{sum(scored(m, e, OLD).values())}" for e, _ in ENGINES)
        body += f'<tr><td>{esc(m)}</td>{cells}<td class="mac">{old}</td></tr>'
    return ('<table class="scores"><tr><th>Model</th><th>Vireo</th><th>Vireo got wrong</th><th>llama.cpp</th>'
            f'<th>llama.cpp got wrong</th><th>This morning, Vireo · llama.cpp</th></tr>{body}</table>')


def detail_table():
    body = ""
    for m in MODELS:
        for e, label in ENGINES:
            ld = metrics(NEW, m, e, "load") or {}
            mt = metrics(NEW, m, e, "multiturn") or {}
            body += (f"<tr><td>{esc(m)}</td><td>{label}</td><td>{n(ld.get('load_s'))}</td><td>{CONTEXT[m]:,}</td>"
                     f"<td>{val(m, e, 'long', 'prompt') or 0:,}</td><td>{n(val(m, e, 'short', 'ttft'))}</td>"
                     f"<td>{n(val(m, e, 'long', 'ttft'))}</td><td>{n(mt.get('ttft'))}</td><td>{n(mt.get('total'))}</td>"
                     f"<td>{ld.get('gb', 0):.1f}</td><td>{gb(val(m, e, 'memory', 'gb'))}</td></tr>")
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
                x = metrics(NEW, m, e, w)
                cells += (f"<td>{x['read']:,} · {x['gen']:,} · {len(x['calls'])}</td>" if x and "read" in x else "<td>–</td>")
        body += f"<tr><td>{esc(m)}</td>{cells}</tr>"
    head = "".join(f"<th>{w}, {e}</th>" for w in ("Tool", "Skill", "Long tool") for e in ("Vireo", "llama.cpp"))
    return f'<table class="detail tokens"><tr><th>Model</th>{head}</tr>{body}</table>'


def ba_table(engine="vireo", from_col=True):
    head = "".join(f"<th>{esc(m)}</th>" for m in MODELS) + ("<th>From</th>" if from_col else "")
    body = ""
    for label, w, f, kind in BA:
        frm = info["ba_from"].get(f"{w}.{f}", "") if from_col else None
        body += (f"<tr><td>{esc(label)}</td>" + "".join(f"<td>{ba_cell(m, engine, w, f, kind)}</td>" for m in MODELS)
                 + (f'<td class="from">{t(frm)}</td>' if from_col else "") + "</tr>")
    body += ("<tr><td>Scored tasks right, of 5</td>" + "".join(
        f"<td>{ba_score(m, OLD, engine)} → <b>{ba_score(m, NEW, engine)}</b></td>" for m in MODELS)
        + (f'<td class="from">{t(info["ba_from"].get("score", ""))}</td>' if from_col else "") + "</tr>")
    return f'<table class="detail ba"><tr><th>{"Vireo" if engine == "vireo" else "llama.cpp"}: this morning → now</th>{head}</tr>{body}</table>'


def eval_table():
    body = ""
    for m in MODELS:
        e = EVAL[m]
        b, a = e["before"], e["after"]
        bs, as_ = b["speeds"], a["speeds"]
        body += (f"<tr><td>{esc(m)}</td><td>{b['passed']} → <b>{a['passed']}</b> <span class='med'>of {a['total']}</span></td>"
                 f"<td>{len(e['flips'])}</td><td>{e['same']} <span class='med'>of {e['cases']}</span></td>"
                 f"<td>{bs['readTokens']:,} → <b>{as_['readTokens']:,}</b></td>"
                 f"<td>{bs['readTokensPerSecond']:,} → <b>{as_['readTokensPerSecond']:,}</b></td>"
                 f"<td>{bs['writtenTokensPerSecond']:.1f} → <b>{as_['writtenTokensPerSecond']:.1f}</b></td>"
                 f"<td>{b['seconds']:.0f} → <b>{a['seconds']:.0f}</b> <span class='med'>{signed(b['seconds'] / a['seconds'] - 1)}</span></td></tr>")
    return ('<table class="detail ev"><tr><th>Model</th><th>Passed, before → after</th><th>Pass/fail changed</th>'
            '<th>Identical answers and calls</th><th>Tokens read (not kept)</th><th>Reading (tokens/s)</th>'
            f'<th>Writing (tokens/s)</th><th>All 43 cases (s)</th></tr>{body}</table>')


def flips_table():
    rows = ""
    for m in MODELS:
        for cid, b, a in EVAL[m]["flips"]:
            note = info.get("eval_flips", {}).get(f"{m}|{cid}", "")
            rows += (f"<tr><td>{esc(m)}</td><td>{esc(cid)}</td><td>{'pass' if b['passed'] else 'fail'} → <b>{'pass' if a['passed'] else 'fail'}</b></td>"
                     f"<td class='wrong'>{esc(b.get('problem') or '–')}</td><td class='wrong'>{esc(a.get('problem') or '–')}</td>"
                     f"<td class='wrong'>{t(note)}</td></tr>")
    if not rows:
        return "<p>No case's pass or fail changed.</p>"
    return ('<table class="scores flips"><tr><th>Model</th><th>Case</th><th>Before → after</th><th>Problem before</th>'
            f'<th>Problem after</th><th>Wrong, or only different</th></tr>{rows}</table>')


def probe_table():
    body = ""
    for r in PROBE:
        if r.get("error"):
            body += f"<tr><td>{esc(r['model'])}</td><td>{'Vireo' if r['engine'] == 'vireo' else 'llama.cpp'}</td><td colspan='8' class='wrong'>{esc(r['error'][:120])}</td></tr>"
            continue
        x = r["rows"][0]
        s0 = x["steps"][0] if x.get("steps") else {}
        rt = s0["read"] / s0["read_s"] if s0.get("read_s") else None
        gt = s0["gen"] / s0["gen_s"] if s0.get("gen_s") else None
        body += (f"<tr><td>{esc(r['model'])}</td><td>{'Vireo' if r['engine'] == 'vireo' else 'llama.cpp'}</td>"
                 f"<td>{s0.get('prompt', 0):,}</td><td>{n(rt)}</td><td>{n(gt)}</td><td>{n(x.get('ttft'))}</td>"
                 f"<td>{r['load'][0]:.2f} · {r['load'][1]:.2f}</td><td>{r['mem'][0]:.2f} · {r['mem'][1]:.2f}</td>"
                 f"<td>{r['smi_gb']:.1f}</td><td>{x.get('recall', 0)} of 2</td></tr>")
    return ('<table class="detail probe"><tr><th>Model</th><th>Engine</th><th>Prompt, tokens</th><th>Reading (tokens/s)</th>'
            '<th>Writing after it (tokens/s)</th><th>First token (s)</th><th>Loaded: dedicated · shared (GB)</th>'
            f'<th>After the prompt: dedicated · shared (GB)</th><th>Whole GPU in use (GB)</th><th>Facts recalled</th></tr>{body}</table>')


# ---- the fill for report.json's {key}s -----------------------------------------------------
CONTEXT = {m: info["contexts"][m] for m in MODELS}

fill = {}
for _, _, w, f, k in STATS:
    key = f"{w}.{f}"
    for prefix, rows in (("", NEW), ("old:", OLD)):
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
        for prefix, rows in (("", NEW), ("old:", OLD)):
            fill[f"{prefix}{m}|{e}|score"] = sum(scored(m, e, rows).values())
            for w in ("short", "medium", "long", "multiturn", "tools", "skill", "longtools", "memory", "load"):
                for f in (("gb",) if w in ("memory", "load") else ("gen_tps", "read_tps", "ttft", "total", "prompt", "gb", "wall", "read", "kept", "calls")):
                    if (v := val(m, e, w, f, rows)) is not None:
                        if f == "calls":
                            v = len(v)
                        fill[f"{prefix}{m}|{e}|{w}.{f}"] = f"{v:,}" if f in ("prompt", "read", "kept", "calls") else f"{v:.1f}" if f == "gb" else n(v)
        for label, w, f, kind in BA:
            b, a = ba_vals(m, e, w, f)
            if (c := change(b, a, kind)) is not None:
                fill[f"ba:{m}|{e}|{w}.{f}"] = signed(c)
    fill[f"{m}|context"] = f"{CONTEXT[m]:,}"
for prefix, rows in (("", NEW), ("old:", OLD)):
    for e, _ in ENGINES:
        fill[f"{prefix}{e}_total"] = sum(fill[f"{prefix}{m}|{e}|score"] for m in MODELS)
for label, w, f, kind in BA:
    for e, _ in ENGINES:
        cs = [c for m in MODELS if (c := change(*ba_vals(m, e, w, f), kind)) is not None]
        fill[f"ba:{e}|{w}.{f}"] = span(cs)
        for group, ms in (("gemma", MODELS[:2]), ("qwen", MODELS[2:])):
            cs = [c for m in ms if (c := change(*ba_vals(m, e, w, f), kind)) is not None]
            fill[f"ba:{group}|{e}|{w}.{f}"] = span(cs)
# llama.cpp's drift across every speed and time it has, as the check the machine was the same
for name, kinds, works in (("speeds", ("speed", "time"), ("medium", "long", "short")),
                           ("tasks", ("time",), ("tools", "skill", "longtools")),
                           ("memory", ("memory",), ("load", "memory"))):
    drift = [abs(c) for m in MODELS for _, w, f, kind in BA if kind in kinds and w in works
             if (c := change(*ba_vals(m, "llama", w, f), kind)) is not None]
    fill[f"llama_drift.{name}.median"] = f"{100 * statistics.median(drift):.1f}%" if drift else "–"
    fill[f"llama_drift.{name}.max"] = f"{100 * max(drift):.1f}%" if drift else "–"
drift_read = [abs(c) for m in MODELS for w in ("medium", "long")
              if (c := change(*ba_vals(m, "llama", w, "read_tps"), "speed")) is not None]
fill["llama_drift.read.max"] = f"{100 * max(drift_read):.1f}%" if drift_read else "–"
for m in (MODELS if EV_DONE else []):
    e = EVAL[m]
    b, a = e["before"], e["after"]
    fill[f"ev:{m}|passed"] = f"{b['passed']} → {a['passed']}"
    fill[f"ev:{m}|same"] = e["same"]
    fill[f"ev:{m}|differ"] = len(e["differ"])
    fill[f"ev:{m}|flips"] = len(e["flips"])
    fill[f"ev:{m}|read"] = f"{b['speeds']['readTokens']:,} → {a['speeds']['readTokens']:,}"
    fill[f"ev:{m}|read.change"] = signed(a['speeds']['readTokens'] / b['speeds']['readTokens'] - 1)
    fill[f"ev:{m}|seconds"] = f"{b['seconds']:.0f} → {a['seconds']:.0f}"
    fill[f"ev:{m}|seconds.change"] = signed(b['seconds'] / a['seconds'] - 1)
    fill[f"ev:{m}|read_tps"] = f"{b['speeds']['readTokensPerSecond']:,} → {a['speeds']['readTokensPerSecond']:,}"
    fill[f"ev:{m}|write_tps"] = f"{b['speeds']['writtenTokensPerSecond']:.1f} → {a['speeds']['writtenTokensPerSecond']:.1f}"
if EV_DONE:
  fill["ev:passed"] = f"{sum(EVAL[m]['before']['passed'] for m in MODELS)} → {sum(EVAL[m]['after']['passed'] for m in MODELS)}"
  fill["ev:same"] = sum(EVAL[m]["same"] for m in MODELS)
  fill["ev:cases"] = sum(EVAL[m]["cases"] for m in MODELS)
  fill["ev:flips"] = sum(len(EVAL[m]["flips"]) for m in MODELS)
for r in PROBE:
    if r.get("error"):
        continue
    x, s0 = r["rows"][0], r["rows"][0]["steps"][0]
    k = f"probe:{r['model']}|{r['engine']}"
    fill[k + "|read_tps"] = n(s0["read"] / s0["read_s"])
    fill[k + "|gen_tps"] = n(s0["gen"] / s0["gen_s"])
    fill[k + "|ttft"] = n(x["ttft"])
    fill[k + "|prompt"] = f"{s0['prompt']:,}"
    fill[k + "|smi"] = f"{r['smi_gb']:.1f}"
    fill[k + "|dedicated"] = f"{r['mem'][0]:.1f}"
    fill[k + "|shared"] = f"{r['mem'][1]:.2f}"
    fill[k + "|shared.load"] = f"{r['load'][1]:.2f}"


# the re-check of the Gemmas' short and medium rows, and the full-speed check before the run
RECHECK_PATH = run / "recheck-gemma-medium.jsonl"
for m in MODELS[:2]:
    for w, f in (("medium", "read_tps"), ("short", "gen_tps")):
        vs = []
        for d in (load(RECHECK_PATH) if RECHECK_PATH.exists() else []):
            if d.get("model") == m and d.get("work") == w and d.get("rows"):
                s0 = d["rows"][0]["steps"][0]
                vs.append(s0["read"] / s0["read_s"] if f == "read_tps" else s0["gen"] / s0["gen_s"])
        if vs:
            fill[f"re:{m}|{w}.{f}"] = " and ".join(dict.fromkeys(n(v) for v in vs))
SPEED = re.findall(r"\(([\d.]+) tokens/s\)", (run / "speed-check.txt").read_text(encoding="utf-8"))
fill["check:read"] = " and ".join(f"{float(v):,.0f}" for v in SPEED[0::2])
fill["check:write"] = " and ".join(f"{float(v):.1f}" for v in SPEED[1::2])


def t(text):
    def one(mm):
        if mm[1].startswith("ev:") and not EV_DONE:
            return "(eval still running)"
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
table.ba td:nth-child(n+2), table.ev td:nth-child(n+2) { font-family: var(--mono); text-align: right; white-space: nowrap; }
table.ba th, table.ev th { text-align: right; }
table.ba th:first-child, table.ba td:first-child, table.ev th:first-child, table.ev td:first-child { text-align: left; }
table.ba td.from { font-family: var(--body); text-align: left; white-space: normal; color: var(--fg-2); font-size: 12px; min-width: 190px; }
table.ba th:last-child { text-align: left; }
table.ba td { font-size: 12.5px; }
table.flips td { white-space: normal; font-size: 12.5px; }
table.probe td:nth-child(n+3) { font-family: var(--mono); text-align: right; }
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
rig = "".join(f'<div><dt>{esc(x["label"])}</dt><dd>{esc(x["value"])}<small>{esc(t(x["detail"]))}</small></dd></div>' for x in info["rig"])
BA_LEGEND = legend((("was", "Vireo this morning (7e00a19)"), ("vireo", "Vireo now (7e00a19 + today's changes)")))

probe_block = (f"""
  <section class="block">
    <h2>{esc(info["probe_title"])}</h2>
    <p>{t(info["probe_intro"])}</p>
    {f'<div class="panel"><div class="tablewrap" style="margin-top:0">{probe_table()}</div><p class="note">{t(info["probe_note"])}</p></div>' if PROBE else ""}
  </section>""")

EVAL_SECTION = (f"""<section class="block pagebreak">
    <h2>The project's eval on both builds</h2>
    <p>{t(info["eval_intro"])}</p>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{eval_table()}</div>
    <p class="note">{t(info["eval_note"])}</p></div>
    <h3 class="sub2">Cases whose pass or fail changed</h3>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{flips_table()}</div></div>
  </section>""" if EV_DONE else
    """<section class="block pagebreak"><h2>The project's eval on both builds</h2><p>The eval on the build before today's changes is still running; this section is added when it ends.</p></section>""")

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
    <p>Positive = Vireo better. Speeds: Vireo ÷ llama.cpp − 1. Times and memory: llama.cpp ÷ Vireo − 1 (so +100% on a task time means llama.cpp took twice as long). Lowest to highest of the models, with the median; "ahead/behind" counts models more than 0.5% either way. The last column is this morning's run of the same harness on this laptop, before today's changes.</p>
    <div class="panel"><div class="tablewrap" style="margin-top:0"><table class="keystats"><tr><th>Workload</th><th>Now: Vireo vs llama.cpp</th><th>Models</th><th class="mac">This morning (7e00a19)</th></tr>{stat_rows()}</table></div></div>
  </section>

  <section class="block">
    <h2>Where Vireo is ahead and where it is behind</h2>
    <div class="box">{bullets(info["where"])}</div>
    <p class="method">{t(info["method"])}</p>
  </section>

  <section class="block">
    <h2>Today's three changes</h2>
    <div class="box">{bullets(info["changes"])}</div>
  </section>

  <section class="block pagebreak">
    <h2>Vireo relative to llama.cpp, per model</h2>
    <p>Blue: Vireo ahead. Orange: Vireo behind. Same formulas as the key stats; – = not measured (a task that ended in an error).</p>
    <div class="panel">{heatmap()}
    <h3 class="sub2">This morning, before today's changes</h3>{heatmap(OLD)}
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
    <p>Time for the whole task, every tool call included (the harness's "total").</p>
    <div class="panels">
      {panel("Tool task", "list, read the quarterly files, calculate, answer · s · lower is better", "tools", "total", "time", t(P["tool"]))}
      {panel("Skill task", "weekly-report skill: list, read, calculate, write a file · s · lower is better", "skill", "total", "time", t(P["skill"]))}
      {panel("Long tool task", "a ~55K-token log pasted in, then tool calls · s · lower is better", "longtools", "total", "time", t(P["longtool"]))}
      {score_panel()}
    </div>
  </section>

  <section class="block pagebreak">
    <h2>Vireo before and after today's changes</h2>
    {BA_LEGEND}
    <p>{t(info["ba_intro"])}</p>
    <div class="panels">
      {ba_panel("Reading ~32K", "long.txt · tokens/s · higher is better", "long", "read_tps", "speed", t(P["ba_long"]))}
      {ba_panel("Long tool task", "whole task · s · lower is better", "longtools", "total", "time", t(P["ba_longtool"]))}
      {ba_panel("Tool task", "whole task · s · lower is better", "tools", "total", "time", t(P["ba_tool"]))}
      {ba_panel("GPU memory, highest", "the whole GPU's in use after any workload · GB · lower is better", "memory", "gb", "memory", t(P["ba_memory"]), fmt=lambda v: f"{v:.1f}")}
    </div>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{ba_table()}</div>
    <p class="note">{t(info["ba_note"])}</p></div>
    <h3 class="sub2">llama.cpp, unchanged, this morning → now: the check that the laptop was in the same state</h3>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{ba_table("llama", False)}</div>
    <p class="note">{t(info["llama_note"])}</p></div>
  </section>

  {EVAL_SECTION}
{probe_block}

  <section class="block pagebreak">
    <h2>What failed or looked odd</h2>
    <div class="box">{bullets(info["odd"], "ol")}</div>
  </section>

  <section class="block">
    <h2>Scored tasks right, per model and engine</h2>
    <p>Fully right out of 5, scored as the Mac's aggregate.py scores them: long recall (both hidden facts), multiturn (all 8 turns), tool (the year's total, $26,742.25), skill (78 hours said and a file written), long tool (inventory-sync named).</p>
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
