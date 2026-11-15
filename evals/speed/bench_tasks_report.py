"""Builds the report page for bench_tasks.py's run (the Mac's 14 November benchmark, on
Windows) from its results.jsonl, in the house style of windows-rtx5060-2026-11-15.html
(its colours, fonts and print page). Every number and percentage on the page comes from
the rows; the prose around them is here.

    python evals/speed/bench_tasks_report.py evals/speed/bench-2026-11-15-windows-tasks evals/speed/windows-rtx5060-2026-11-15-tasks.html

Then print it: msedge --headless=new --no-pdf-header-footer --print-to-pdf=<out.pdf> "file:///<page>?theme=dark"
"""
import html
import json
import re
import statistics
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
bench, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in (bench / "results.jsonl").read_text().splitlines() if l.strip()]
info = json.loads((bench / "report.json").read_text(encoding="utf-8"))
MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B"]
QUANT = {m: ("Q4_0" if m.startswith("Gemma") else "Q4_K_M") for m in MODELS}
ENGINES = [("vireo", "Vireo"), ("llama", "llama.cpp")]
esc = html.escape


def row(model, engine, work):
    """The last row for a model, engine and workload (a re-run's row replaces the first)."""
    of = [r for r in rows if r["model"] == model and r["engine"] == engine and r.get("work") == work]
    return of[-1] if of else None


def val(model, engine, work, field):
    r = row(model, engine, work)
    # A task that ended in the server's error (a stream error, or an HTTP error) is not
    # measured: its time is how long the failure took, as on the Mac's page.
    if r is None or "error" in r or "stream_error" in r or r.get(field) is None:
        return None
    return r[field]


# ---- the comparisons: (title, short title, workload, field, kind) ------------------------
# kind "speed": Vireo ÷ llama.cpp − 1; "time" and "memory": llama.cpp ÷ Vireo − 1 (so
# positive is Vireo better either way, as on the Mac's page).
long_n = {m: val(m, "llama", "long", "prompt_n") for m in MODELS}
long_k = sorted(round(n / 1000) for n in long_n.values() if n)
LONG = f"~{long_k[-1]}K" if long_k[0] == long_k[-1] else f"~{long_k[0]}–{long_k[-1]}K"
STATS = [
    ("Short reply writing", "Short reply\nwriting", "short", "gen_tps", "speed"),
    ("Reading ~3K", "Reading\n~3K", "medium", "prompt_tps", "speed"),
    (f"Reading the long prompt ({LONG})", f"Reading\n{LONG}", "long", "prompt_tps", "speed"),
    ("Writing after the long prompt", "Writing after\nlong prompt", "long", "gen_tps", "speed"),
    ("First token, long prompt", "First token,\nlong prompt", "long", "ttft_s", "time"),
    ("Tool task time", "Tool task\ntime", "tool", "wall_s", "time"),
    ("Skill task time", "Skill task\ntime", "skill", "wall_s", "time"),
    ("Long tool task time", "Long tool\ntask time", "longtool", "wall_s", "time"),
    ("GPU memory after the long prompt", "GPU memory,\nlong prompt", "memory", "gpu_mb", "memory"),
]


def diff(model, work, field, kind):
    v, o = val(model, "vireo", work, field), val(model, "llama", work, field)
    if not v or not o:
        return None
    return v / o - 1 if kind == "speed" else o / v - 1


def signed(p, digits=0):
    p = round(100 * p, digits)
    return ("+" if p > 0 else "−" if p < 0 else "") + f"{abs(p):.{digits}f}%"


def span(ps):
    lo, hi = min(ps), max(ps)
    return signed(lo) if round(100 * lo) == round(100 * hi) else f"{signed(lo)} to {signed(hi)}"


def counts(ps):
    ahead = sum(p > 0.005 for p in ps)
    behind = sum(p < -0.005 for p in ps)
    return ahead, behind, len(ps)


def keystat(work, field, kind):
    ps = [p for m in MODELS if (p := diff(m, work, field, kind)) is not None]
    return ps


# ---- numbers in text -----------------------------------------------------------------

def n(v):
    if v is None:
        return "–"
    return f"{v:,.0f}" if v >= 1000 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v < 1 else f"{v:.1f}"


def gb(mb):
    return f"{mb / 1024:.1f} GB"


def scored(model, engine):
    """The five scored tasks: long recall (2/2), multiturn (every turn), tool, skill, long tool."""
    works = ["long", "multiturn", "tool", "skill", "longtool"]
    return {w: bool(val(model, engine, w, "right")) for w in works}


SCORED_NAMES = {"long": "long recall", "multiturn": "multiturn", "tool": "tool", "skill": "skill", "longtool": "long tool"}

# ---- bar panels (SVG, drawn here; the house page's bar shape and colours) -------------------

def niceMax(v):
    import math
    p = 10 ** math.floor(math.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * p >= v:
            return m * p
    return 10 * p


def bars(values, unit, fmt=n, max_value=None):
    """values: per model, a (vireo, llama) pair; a value may be a string (an error's text)."""
    W, LEFT, RIGHT, TOP, BAR, GAP, ROWGAP = 520, 104, 58, 6, 13, 2, 22
    GROUP = 2 * BAR + GAP
    H = TOP + len(MODELS) * GROUP + (len(MODELS) - 1) * ROWGAP + 26
    nums = [x for pair in values for x in pair if isinstance(x, (int, float))]
    top = max_value or niceMax(max(nums))
    x = lambda v: LEFT + (W - LEFT - RIGHT) * v / top  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(unit)}">']
    bottom = H - 22
    for i in range(5):
        v = top * i / 4
        s.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{TOP - 2}" y2="{bottom}" stroke="var(--grid)" stroke-width="1"/>')
        label = f"{v:,.0f}" if v >= 10 or v == 0 else f"{v:g}"
        s.append(f'<text x="{x(v):.1f}" y="{H - 6}" text-anchor="middle" font-size="11" fill="var(--fg-3)" class="num">{label}</text>')
    for i, m in enumerate(MODELS):
        y0 = TOP + i * (GROUP + ROWGAP)
        s.append(f'<text x="0" y="{y0 + GROUP / 2 - 3}" font-size="13" font-weight="500" fill="var(--fg)">{esc(m)}</text>')
        s.append(f'<text x="0" y="{y0 + GROUP / 2 + 12}" font-size="11" fill="var(--fg-3)" class="num">{QUANT[m]}</text>')
        for j, (k, v) in enumerate(zip(("vireo", "llama"), values[i])):
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
    r = row(model, engine, work)
    if r is None:
        return "not measured"
    if "stream_error" in r:
        return "task errored: " + r["stream_error"][:60]
    if "error" in r:
        return "task errored: " + r["error"].replace("<", "").replace(">", "")[:60]
    return None


def panel(title, sub, work, field, kind, note, fmt=n):
    values = [tuple(val(m, e, work, field) if val(m, e, work, field) is not None else error_text(m, e, work)
                    for e, _ in ENGINES) for m in MODELS]
    if kind == "memory":
        values = [tuple(v / 1024 if isinstance(v, (int, float)) else v for v in pair) for pair in values]
    text, cls = verdict(work, field, kind)
    return (f'<article class="panel"><div class="panel-head"><div><h3>{esc(title)}</h3><div class="sub">{esc(sub)}</div></div>'
            f'<span class="verdict {cls}">{esc(text)}</span></div>{bars(values, title, fmt)}'
            + (f'<p class="note">{note}</p>' if note else "") + "</article>")


def score_panel():
    values = [tuple(sum(scored(m, e).values()) for e, _ in ENGINES) for m in MODELS]
    return ('<article class="panel"><div class="panel-head"><div><h3>Scored tasks right</h3>'
            '<div class="sub">fully right, of 5 · higher is better</div></div></div>'
            + bars(values, "Scored tasks right", fmt=lambda v: f"{v:.0f}", max_value=5)
            + '<p class="note">Long recall (both planted facts), multiturn (every turn), tool, skill (the total said and written to the file), long tool (the service and its retry limit). The Mac scored 6, with its 96K recall prompt.</p></article>')


# ---- the page -----------------------------------------------------------------------

def stat_rows():
    out_rows = []
    for title, _, work, field, kind in STATS:
        ps = keystat(work, field, kind)
        a, b, t = counts(ps)
        med = statistics.median(ps)
        out_rows.append(f'<tr><td>{esc(title)}</td><td><b>{span(ps)}</b> <span class="med">· median {signed(med)}</span></td>'
                        f'<td>{a} ahead · {b} behind <span class="med">of {t}</span></td></tr>')
    return "".join(out_rows)


def heat_cell(p):
    if p is None:
        return '<td class="heat">–</td>'
    strength = min(abs(p), 0.6) / 0.6
    colour = "var(--vireo)" if p > 0 else "var(--llama)"
    mix = round(12 + 78 * strength)
    ink = "#fff" if mix > 55 else "var(--fg)"
    return f'<td class="heat" style="background:color-mix(in srgb, {colour} {mix}%, var(--surface));color:{ink}">{signed(p)}</td>'


def heatmap():
    head = "".join(f"<th>{esc(short).replace(chr(10), '<br>')}</th>" for _, short, *_ in STATS)
    body = "".join(f"<tr><td>{esc(m)}</td>" + "".join(heat_cell(diff(m, w, f, k)) for _, _, w, f, k in STATS) + "</tr>"
                   for m in MODELS)
    return (f'<table class="heatmap"><tr><th></th>{head}</tr>{body}</table>'
            '<div class="scale"><span>−60% or worse</span><span class="ramp"></span><span>+60% or better</span></div>')


def score_table():
    body = ""
    for m in MODELS:
        cells = ""
        for e, _ in ENGINES:
            s = scored(m, e)
            wrong = [SCORED_NAMES[w] for w, ok in s.items() if not ok]
            cells += f'<td><b>{sum(s.values())}</b> <span class="med">/ 5</span></td><td class="wrong">{esc(", ".join(wrong) or "all right")}</td>'
        body += f"<tr><td>{esc(m)}</td>{cells}</tr>"
    return ('<table class="scores"><tr><th>Model</th><th>Vireo</th><th>Vireo got wrong</th><th>llama.cpp</th>'
            f'<th>llama.cpp got wrong</th></tr>{body}</table>')


def detail_table():
    body = ""
    for m in MODELS:
        for e, label in ENGINES:
            ctx = val(m, e, "load", "context")
            body += (f"<tr><td>{esc(m)}</td><td>{label}</td><td>{n(val(m, e, 'load', 'wall_s'))}</td><td>{ctx:,}</td>"
                     f"<td>{val(m, e, 'long', 'prompt_n'):,}</td><td>{n(val(m, e, 'short', 'ttft_s'))}</td>"
                     f"<td>{n(val(m, e, 'long', 'ttft_s'))}</td><td>{n(val(m, e, 'multiturn', 'wall_s'))}</td>"
                     f"<td>{gb(val(m, e, 'memory', 'gpu_mb'))}</td></tr>")
    return ('<table class="detail"><tr><th>Model</th><th>Engine</th><th>Load (s)</th><th>Context</th>'
            '<th>Long prompt tokens read</th><th>First token, short (s)</th><th>First token, long (s)</th>'
            f'<th>Multiturn, 3 turns (s)</th><th>GPU memory</th></tr>{body}</table>')


def task_tokens():
    body = ""
    for m in MODELS:
        cells = ""
        for w in ("tool", "skill", "longtool"):
            for e, _ in ENGINES:
                r = row(m, e, w)
                ok = r and val(m, e, w, "wall_s") is not None
                cells += (f"<td>{r.get('read_n', 0):,} · {r.get('gen_n', 0):,} · {len(r.get('calls') or [])}</td>" if ok else "<td>–</td>")
        body += f"<tr><td>{esc(m)}</td>{cells}</tr>"
    head = "".join(f"<th>{w}, {e}</th>" for w in ("Tool", "Skill", "Long tool") for e in ("Vireo", "llama.cpp"))
    return f'<table class="detail tokens"><tr><th>Model</th>{head}</tr>{body}</table>'


style = (here / "windows-rtx5060-2026-11-15.html").read_text(encoding="utf-8")
style = style[style.index("<style>") + 7: style.index("</style>")]
EXTRA = """
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
table.keystats td:nth-child(2), table.keystats td:nth-child(3) { font-family: var(--mono); }
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
table.scores th, table.scores td, table.detail th { text-align: left; }
table.detail th { white-space: normal; vertical-align: bottom; text-align: right; }
table.detail td:nth-child(n+3) { font-family: var(--mono); }
table.detail th:nth-child(-n+2) { text-align: left; }
table.tokens td:nth-child(n+2) { font-family: var(--mono); text-align: right; }
.panel-head .verdict { white-space: normal; text-align: right; }
@media print {
  .pagebreak { break-before: page; }
  .box, table, .cards { break-inside: avoid; }
  section.block > h2 { break-after: avoid; }
}
"""


def legend():
    return ('<div class="legend"><span class="key"><span class="sw" style="background:var(--vireo)"></span>Vireo</span>'
            '<span class="key"><span class="sw" style="background:var(--llama)"></span>llama.cpp</span></div>')


def bullets(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# ---- the summary's numbers, worked out -------------------------------------------------
K = {w + "." + f: keystat(w, f, k) for _, _, w, f, k in STATS}
per = {(m, w + "." + f): diff(m, w, f, k) for m in MODELS for _, _, w, f, k in STATS}
fill = {"LONG": LONG}
for key, ps in K.items():
    a, b, t = counts(ps)
    fill[key] = span(ps)
    fill[key + ".ahead"] = a
    fill[key + ".behind"] = b
    fill[key + ".of"] = t
    fill[key + ".median"] = signed(statistics.median(ps))
for (m, key), p in per.items():
    fill[f"{m}|{key}"] = signed(p) if p is not None else "–"
for m in MODELS:
    for e, _ in ENGINES:
        fill[f"{m}|{e}|score"] = sum(scored(m, e).values())
        for w in ("short", "medium", "long", "tool", "skill", "longtool", "multiturn", "load"):
            for f in ("gen_tps", "prompt_tps", "ttft_s", "wall_s", "prompt_n", "context"):
                if (v := val(m, e, w, f)) is not None:
                    fill[f"{m}|{e}|{w}.{f}"] = f"{v:,}" if f in ("prompt_n", "context") else n(v)
        if (v := val(m, e, "memory", "gpu_mb")) is not None:
            fill[f"{m}|{e}|gpu"] = gb(v)
fill["vireo_total"] = sum(fill[f"{m}|vireo|score"] for m in MODELS)
fill["llama_total"] = sum(fill[f"{m}|llama|score"] for m in MODELS)


def t(text):
    """report.json's text, with {key} filled from the rows (see `fill`)."""
    def one(m):
        if m[1] not in fill:
            raise KeyError(f"report.json names {m[1]!r}, which the rows don't have")
        return str(fill[m[1]])
    return re.sub(r"\{([^{}]+)\}", one, text)


cards = "".join(f'<div class="card"><b>{esc(t(c["big"]))}</b><span>{esc(t(c["text"]))}</span></div>' for c in info["cards"])
rig = "".join(f'<div><dt>{esc(x["label"])}</dt><dd>{esc(x["value"])}<small>{esc(x["detail"])}</small></dd></div>' for x in info["rig"])
P = info["panels"]

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
    <p>Positive = Vireo better. Speeds: Vireo ÷ llama.cpp − 1. Times and memory: llama.cpp ÷ Vireo − 1 (so +100% on a task time means llama.cpp took twice as long). Lowest to highest of the models, with the median; "ahead/behind" counts models more than 0.5% either way.</p>
    <div class="panel"><div class="tablewrap" style="margin-top:0"><table class="keystats"><tr><th>Workload</th><th>Vireo vs llama.cpp</th><th>Models</th></tr>{stat_rows()}</table></div></div>
  </section>

  <section class="block">
    <h2>Where Vireo is ahead and where it is behind</h2>
    <div class="box">{bullets([t(b) for b in info["where"]])}</div>
  </section>

  <section class="block">
    <h2>How this run differs from the Mac's</h2>
    <div class="box">{bullets([t(b) for b in info["differs"]])}</div>
  </section>

  <section class="block pagebreak">
    <h2>Vireo relative to llama.cpp, per model</h2>
    <p>Blue: Vireo ahead. Orange: Vireo behind. Same formulas as the key stats; – = not measured.</p>
    <div class="panel">{heatmap()}</div>
  </section>

  <section class="block">
    <h2>Writing speed, first token and memory</h2>
    {legend()}
    <div class="panels">
      {panel("Writing, short reply", "generation after a ~40-token question · tokens/s · higher is better", "short", "gen_tps", "speed", t(P["short"]))}
      {panel("Writing after the long prompt", f"generation after the {LONG}-token prompt · tokens/s · higher is better", "long", "gen_tps", "speed", t(P["longwrite"]))}
      {panel("Time to first token, long prompt", "seconds until the first streamed token · lower is better", "long", "ttft_s", "time", t(P["ttft"]))}
      {panel("GPU memory, after the long prompt", "the whole GPU's memory in use · GB · lower is better", "memory", "gpu_mb", "memory", t(P["memory"]), fmt=lambda v: f"{v:.1f}")}
    </div>
  </section>

  <section class="block pagebreak">
    <h2>Prompt reading speed</h2>
    {legend()}
    <div class="panels">
      {panel("~3K-token prompt", "medium: a ~2.5K-token document · tokens/s · higher is better", "medium", "prompt_tps", "speed", t(P["medium"]))}
      {panel(f"Long prompt ({LONG} tokens)", "long.txt cut to the context, two facts planted · tokens/s · higher is better", "long", "prompt_tps", "speed", t(P["long"]))}
    </div>
  </section>

  <section class="block">
    <h2>Whole agent tasks and answers</h2>
    {legend()}
    <p>Wall time for the whole task, every tool call included.</p>
    <div class="panels">
      {panel("Tool task", "list, read four files, calculate, answer · s · lower is better", "tool", "wall_s", "time", t(P["tool"]))}
      {panel("Skill task", "weekly-report skill: list, read, calculate, write a file · s · lower is better", "skill", "wall_s", "time", t(P["skill"]))}
      {panel("Long tool task", "a long log pasted in, then tool calls · s · lower is better", "longtool", "wall_s", "time", t(P["longtool"]))}
      {score_panel()}
    </div>
  </section>

  <section class="block pagebreak">
    <h2>What failed or looked odd</h2>
    <div class="box"><ol>{"".join(f"<li>{t(b)}</li>" for b in info["odd"])}</ol></div>
  </section>

  <section class="block">
    <h2>Scored tasks right, per model and engine</h2>
    <p>Fully right out of the 5 scored tasks (the Mac's 6 less its 96K recall prompt).</p>
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

  <p class="method">{t(info["method"])}</p>
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
