"""Builds the report page for the speed run after PR 178 on Windows (main at b5ff3ae),
the run after PRs 171 and 172 repeated: Vireo against
llama.cpp on the four small models, prompt reading, writing and time to first token only,
at four prompt sizes (small, medium ~3K, large ~32K, xlarge: the largest that fits the
8 GB GPU for each model on both engines). In the style of
windows-rtx5060-2026-11-15-after-pr172.html (bench_after_pr172_report.py, which this adapts), with
the change since that run in every cell, xlarge included (the same contexts and prompts). Every
number and percentage on the page is worked out here from the result files; the prose is
in <run>/report.json, with {key}s filled from them.

    python -X utf8 evals/speed/bench_after_pr178_report.py evals/speed/bench-2026-11-15-windows-after-pr178 \
        evals/speed/bench-2026-11-15-windows-after-pr172 evals/speed/windows-rtx5060-2026-11-15-after-pr178.html [--fill]

Then print it: msedge --headless=new --no-pdf-header-footer --print-to-pdf=<out.pdf> "file:///<page>?theme=dark"
"""
import html
import json
import re
import statistics
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
run, earlier, out = (Path(a) for a in sys.argv[1:4])
MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B"]
QUANT = {m: ("Q4_0" if m.startswith("Gemma") else "Q4_K_M") for m in MODELS}
SIZES = [("short", "Small", "a ~40-token question"), ("medium", "Medium", "~3K-token document"),
         ("long", "Large", "~32K-token repository dump"), ("xlarge", "Xlarge", "the largest that fits")]
# (field, title, unit, kind): kind "speed" is Vireo ÷ llama.cpp − 1, "time" llama.cpp ÷ Vireo − 1.
MEASURES = [("read_tps", "Prompt reading speed", "tokens/s · higher is better", "speed"),
            ("gen_tps", "Writing speed", "tokens/s · higher is better", "speed"),
            ("ttft", "Time to first token", "seconds, client-side, streamed · lower is better", "time")]
ENGINE_NAME = {"vireo": "Vireo", "vireo-plain": "Vireo, drafting off", "llama": "llama.cpp"}
esc = html.escape


def load(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


info = json.loads((run / "report.json").read_text(encoding="utf-8"))
XL = [d for d in load(run / "xlarge.jsonl") if not d.get("error")]
NEW = load(run / "results.jsonl") + XL
OLD = load(earlier / "results.jsonl") + [d for d in load(earlier / "xlarge.jsonl") if not d.get("error")]
PROBES = load(run / "xlarge-probes.jsonl")


def record(rows, model, engine, work):
    of = [d for d in rows if d.get("model") == model and d.get("engine") == engine and d.get("work") == work]
    return of[-1] if of else None


def metrics(rows, model, engine, work):
    d = record(rows, model, engine, work)
    if d is None or d.get("error") or not d.get("rows"):
        return None
    r = d["rows"][0]
    if r.get("error") or not r.get("steps"):
        return None
    s = r["steps"][0]
    return {"ttft": r["ttft"], "prompt": s.get("prompt"), "kept": s.get("cached"), "read": s["read"], "gen": s["gen"],
            "read_tps": s["read"] / s["read_s"] if s.get("read_s") else None,
            "gen_tps": s["gen"] / s["gen_s"] if s.get("gen_s") else None,
            "context": d.get("context") or info["contexts"][model], "recall": r.get("recall"),
            "load": d.get("load"), "mem": d.get("mem"), "smi": d.get("smi_gb"), "chars": d.get("chars")}


def val(model, engine, work, field, rows=None):
    m = metrics(NEW if rows is None else rows, model, engine, work)
    return None if m is None else m.get(field)


def diff(model, work, field, kind, rows=None, engine="vireo"):
    v, o = val(model, engine, work, field, rows), val(model, "llama", work, field, rows)
    if not v or not o:
        return None
    return v / o - 1 if kind == "speed" else o / v - 1


def since(model, engine, work, field, kind):
    """Now against the run after PRs 171 and 172, with better positive (a speed up, a time down)."""
    b, a = val(model, engine, work, field, OLD), val(model, engine, work, field)
    if not a or not b:
        return None
    return a / b - 1 if kind == "speed" else b / a - 1


def signed(p, digits=0):
    p = round(100 * p, digits)
    return ("+" if p > 0 else "−" if p < 0 else "") + f"{abs(p):.{digits}f}%"


def span(ps):
    if not ps:
        return "–"
    lo, hi = min(ps), max(ps)
    return signed(lo) if round(100 * lo) == round(100 * hi) else f"{signed(lo)} to {signed(hi)}"


def n(v):
    if v is None:
        return "–"
    return f"{v:,.0f}" if v >= 1000 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v < 1 else f"{v:.1f}"


def tint(p):
    if p is None:
        return ""
    strength = min(abs(p), 0.4) / 0.4
    colour = "var(--vireo)" if p > 0 else "var(--llama)"
    return f' style="background:color-mix(in srgb, {colour} {round(6 + 34 * strength)}%, var(--surface))"'


def word(p):
    return "level" if p is None or abs(p) < 0.03 else "ahead" if p > 0 else "behind"


# ---- tables ---------------------------------------------------------------------------

def measure_table(field, kind, engine="vireo"):
    head = "".join(f"<th>{lab}<small>{esc(sub)}</small></th>" for _, lab, sub in SIZES)
    body = ""
    for m in MODELS:
        cells = ""
        for w, _, _ in SIZES:
            v, l = val(m, engine, w, field), val(m, "llama", w, field)
            if v is None and l is None:
                cells += '<td class="cell">–</td>'
                continue
            p = diff(m, w, field, kind, engine=engine)
            sv, sl = since(m, engine, w, field, kind), since(m, "llama", w, field, kind)
            since_txt = ((f'<div class="since">{val(m, engine, w, "context"):,} context · {val(m, engine, w, "prompt"):,} tokens on Vireo</div>'
                          if w == "xlarge" else "") +
                         f'<div class="since">since after PR 172: Vireo {signed(sv) if sv is not None else "–"} · '
                         f'llama.cpp {signed(sl) if sl is not None else "–"}</div>')
            cells += (f'<td class="cell"{tint(p)}><div class="pair"><span><i class="sw" style="background:var(--vireo)"></i>{n(v)}</span>'
                      f'<span><i class="sw" style="background:var(--llama)"></i>{n(l)}</span></div>'
                      f'<div class="d">{signed(p) if p is not None else "–"} <span class="med">{word(p)}</span></div>{since_txt}</td>')
        body += f'<tr><td class="model">{esc(m)}<small>{QUANT[m]}</small></td>{cells}</tr>'
    return f'<table class="measure"><tr><th></th>{head}</tr>{body}</table>'


def plain_table():
    """Writing with Vireo's drafting off (no draft model, no guesses), against shipped and llama.cpp."""
    head = "".join(f"<th>{lab}</th>" for _, lab, _ in SIZES)
    body = ""
    for m in MODELS:
        cells = ""
        for w, _, _ in SIZES:
            v, pl, l = (val(m, e, w, "gen_tps") for e in ("vireo", "vireo-plain", "llama"))
            if pl is None:
                cells += "<td>–</td>"
                continue
            p = pl / l - 1 if l else None
            dv = v / pl - 1 if v else None
            cells += (f'<td{tint(p)}><b>{n(pl)}</b> <span class="med">vs llama.cpp</span> {signed(p) if p is not None else "–"}'
                      f'<div class="since">drafting on: {n(v)} ({signed(dv) if dv is not None else "–"})</div>'
                      f'<div class="since">since after PR 172: off {signed(sp) if (sp := since(m, "vireo-plain", w, "gen_tps", "speed")) is not None else "–"}'
                      f' · on {signed(so) if (so := since(m, "vireo", w, "gen_tps", "speed")) is not None else "–"}</div></td>')
        body += f'<tr><td class="model">{esc(m)}</td>{cells}</tr>'
    return f'<table class="measure plain"><tr><th></th>{head}</tr>{body}</table>'


def settings_table():
    body = ""
    for m in MODELS:
        for w, lab, _ in SIZES:
            v, l, pl = (metrics(NEW, m, e, w) for e in ("vireo", "llama", "vireo-plain"))
            if not v and not l:
                continue
            ctx = (v or l)["context"]
            mem = ""
            if w == "xlarge":
                mem = " · ".join(f"{x['load'][1]:.2f} → {x['mem'][1]:.2f}" if x and x.get("mem") else "–" for x in (v, l))
            body += (f"<tr><td>{esc(m)}</td><td>{lab}</td><td>{ctx:,}</td>"
                     f"<td>{(v or {}).get('prompt') or 0:,} <span class='med'>({(v or {}).get('kept') or 0:,} kept)</span></td>"
                     f"<td>{(l or {}).get('prompt') or 0:,} <span class='med'>({(l or {}).get('kept') or 0:,} kept)</span></td>"
                     f"<td>{(v or {}).get('gen') or 0:,} · {(pl or {}).get('gen') or 0:,} · {(l or {}).get('gen') or 0:,}</td>"
                     f"<td>{mem or '<span class=med>not read</span>'}</td></tr>")
    return ('<table class="detail settings"><tr><th>Model</th><th>Size</th><th>Context</th><th>Vireo prompt, tokens</th>'
            '<th>llama.cpp prompt, tokens</th><th>Tokens written: Vireo · drafting off · llama.cpp</th>'
            f'<th>xlarge: shared GPU memory, loaded → after the prompt (GB), Vireo · llama.cpp</th></tr>{body}</table>')


def probe_table():
    body = ""
    for r in PROBES:
        x = (r.get("rows") or [{}])[0]
        s = (x.get("steps") or [{}])[0]
        rt = s["read"] / s["read_s"] if s.get("read_s") else None
        mem = f"{r['load'][1]:.2f} → {r['mem'][1]:.2f}" if r.get("mem") and r.get("load") else "–"
        body += (f"<tr><td>{esc(r['model'])}</td><td>{ENGINE_NAME.get(r['engine'], r['engine'])}</td><td>{r['context']:,}</td>"
                 f"<td>{s.get('prompt') or 0:,}</td><td>{n(rt)}</td><td>{mem}</td>"
                 f"<td class='wrong'>{esc(r.get('verdict') or r.get('error', '')[:120])}</td></tr>")
    return ('<table class="detail probes"><tr><th>Model</th><th>Engine</th><th>Context</th><th>Prompt, tokens</th>'
            f'<th>Reading (tokens/s)</th><th>Shared GPU memory, loaded → after (GB)</th><th>Verdict</th></tr>{body}</table>')


# ---- the fill for report.json's {key}s -----------------------------------------------------
fill = {}
for field, _, _, kind in MEASURES:
    for w, _, _ in SIZES:
        ps = [p for m in MODELS if (p := diff(m, w, field, kind)) is not None]
        fill[f"{w}.{field}"] = span(ps)
        fill[f"old:{w}.{field}"] = span([p for m in MODELS if (p := diff(m, w, field, kind, OLD)) is not None])
        for m in MODELS:
            p = diff(m, w, field, kind)
            fill[f"{m}|{w}.{field}"] = signed(p) if p is not None else "–"
            pp = diff(m, w, field, kind, engine="vireo-plain")
            fill[f"plain:{m}|{w}.{field}"] = signed(pp) if pp is not None else "–"
            for e in ("vireo", "vireo-plain", "llama"):
                v = val(m, e, w, field)
                fill[f"{m}|{e}|{w}.{field}"] = n(v)
                s = since(m, e, w, field, kind)
                fill[f"since:{m}|{e}|{w}.{field}"] = signed(s) if s is not None else "–"
                o = val(m, e, w, field, OLD)
                fill[f"old:{m}|{e}|{w}.{field}"] = n(o)
        for e in ("vireo", "vireo-plain", "llama"):
            cs = [c for m in MODELS if (c := since(m, e, w, field, kind)) is not None]
            fill[f"since:{e}|{w}.{field}"] = span(cs)
for m in MODELS:
    for w, _, _ in SIZES:
        for e in ("vireo", "llama"):
            x = metrics(NEW, m, e, w)
            if x:
                fill[f"{m}|{e}|{w}.prompt"] = f"{x['prompt']:,}"
                fill[f"{m}|{w}.context"] = f"{x['context']:,}"
DRIFT = [abs(c) for field, _, _, kind in MEASURES for w, _, _ in SIZES for m in MODELS
         if (c := since(m, "llama", w, field, kind)) is not None and w != "short"]
fill["llama_drift.max"] = f"{100 * max(DRIFT):.1f}%" if DRIFT else "–"
fill["llama_drift.median"] = f"{100 * statistics.median(DRIFT):.1f}%" if DRIFT else "–"
SPEED = {}
for when in ("before", "after"):
    p = run / f"speed-check-{when}.txt"
    vs = re.findall(r"\(([\d.]+) tokens/s\)", p.read_text(encoding="utf-8")) if p.exists() else []
    fill[f"check:{when}:read"] = " and ".join(f"{float(v):,.0f}" for v in vs[0::2]) or "–"
    fill[f"check:{when}:write"] = " and ".join(f"{float(v):.1f}" for v in vs[1::2]) or "–"


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
.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 760px) { .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; display: grid; gap: 4px; align-content: start; }
.card b { font-family: var(--display); font-weight: 600; font-size: 24px; line-height: 1.15; }
.card span { font-size: 13px; color: var(--fg-2); }
section.block { display: grid; gap: 14px; }
section.block > h2 { font-family: var(--display); font-weight: 600; font-size: 24px; margin: 0; }
section.block > p { margin: 0; color: var(--fg-2); font-size: 14px; max-width: 100ch; }
.box { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; }
.box ul, .box ol { margin: 0; padding-left: 20px; display: grid; gap: 6px; font-size: 14px; }
.box li { color: var(--fg-2); }
.box li b { color: var(--fg); }
.med { color: var(--fg-3); font-size: 12px; }
table.measure { min-width: 760px; table-layout: fixed; border-collapse: separate; border-spacing: 3px; }
table.measure th { text-align: left; font-size: 13px; vertical-align: bottom; border: 0; }
table.measure th small, table.measure td.model small { display: block; color: var(--fg-3); font-weight: 400; font-size: 11px; }
table.measure td { border: 0; font-family: var(--body); vertical-align: top; border-radius: 4px; padding: 8px 10px; text-align: left; white-space: normal; }
table.measure td.model { width: 120px; font-weight: 500; }
table.measure .pair { display: flex; gap: 12px; font-family: var(--mono); font-size: 13px; flex-wrap: wrap; }
table.measure .pair .sw { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; }
table.measure .d { font-family: var(--mono); font-weight: 600; font-size: 15px; margin-top: 2px; }
table.measure .since { font-size: 11px; color: var(--fg-3); margin-top: 2px; }
table.detail { min-width: 0; font-size: 13px; }
table.detail td { font-family: var(--body); }
table.detail th { white-space: normal; vertical-align: bottom; text-align: right; }
table.detail th:nth-child(-n+2), table.detail td:nth-child(-n+2) { text-align: left; }
table.detail td:nth-child(n+3) { font-family: var(--mono); text-align: right; }
table.detail td.wrong { font-family: var(--body); text-align: left; white-space: normal; color: var(--fg-2); }
@media print {
  .pagebreak { break-before: page; }
  .box, table, .cards, .panel { break-inside: avoid; }
  section.block > h2 { break-after: avoid; }
}
"""


def legend():
    return ('<div class="legend">' + "".join(
        f'<span class="key"><span class="sw" style="background:var(--{k})"></span>{esc(lab)}</span>'
        for k, lab in (("vireo", "Vireo"), ("llama", "llama.cpp"))) +
        '<span class="key med">cell colour: blue Vireo ahead, orange behind; the bold figure is Vireo\'s difference</span></div>')


def bullets(items, tag="ul"):
    return f"<{tag}>" + "".join(f"<li>{t(i)}</li>" for i in items) + f"</{tag}>"


cards = "".join(f'<div class="card"><b>{esc(t(c["big"]))}</b><span>{esc(t(c["text"]))}</span></div>' for c in info["cards"])
rig = "".join(f'<div><dt>{esc(x["label"])}</dt><dd>{esc(x["value"])}<small>{esc(t(x["detail"]))}</small></dd></div>' for x in info["rig"])
P = info["notes"]
sections = ""
for i, (field, title, unit, kind) in enumerate(MEASURES):
    sections += f"""
  <section class="block{' pagebreak' if i else ''}">
    <h2>{esc(title)}</h2>
    <p>{esc(unit)}. {t(P[field])}</p>
    {legend()}
    <div class="panel"><div class="tablewrap" style="margin-top:0">{measure_table(field, kind)}</div></div>
  </section>"""

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
    <h2>Where Vireo is ahead, level and behind</h2>
    <div class="box">{bullets(info["where"])}</div>
    <p class="method">{t(info["method"])}</p>
  </section>
{sections}

  <section class="block">
    <h2>Writing with Vireo's drafting off</h2>
    <p>{t(P["plain"])}</p>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{plain_table()}</div></div>
  </section>

  <section class="block pagebreak">
    <h2>Settings of each row</h2>
    <p>{t(P["settings"])}</p>
    <div class="panel"><div class="tablewrap" style="margin-top:0">{settings_table()}</div></div>
  </section>

  <section class="block">
    <h2>The xlarge contexts</h2>
    <p>{t(P["probes"])}</p>
    {f'<div class="panel"><div class="tablewrap" style="margin-top:0">{probe_table()}</div></div>' if PROBES else ""}
  </section>

  <section class="block">
    <h2>What failed or looked odd</h2>
    <div class="box">{bullets(info["odd"], "ol")}</div>
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
