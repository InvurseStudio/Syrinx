"""Builds a run's report page from its results file, with the previous run of the same tier
and platform beside it, and keeps baselines.json.

    python3 -I report.py results/<run>.jsonl                  -> results/<run>.html
    python3 -I report.py results/<run>.jsonl --prev <other.jsonl>
    python3 -I report.py results/<run>.jsonl --save-baseline  (a standard run: what quick compares with)

Every number is the median over reps (and over a test's repeats). "Vireo vs best" compares
Vireo with the better of llama.cpp and mlx-vlm on that row. In a quick run llama.cpp and
mlx-vlm come from baselines.json, marked "(base)". Regressions: a Vireo speed more than 5%
worse than the previous run (or the baseline), or a lower score; listed at the top and
printed.
"""
import argparse
import html
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE / "baselines.json"
ENGINES = ["vireo", "vireo-plain", "llama", "mlx"]
LABEL = {"vireo": "Vireo", "vireo-plain": "Vireo, no drafting", "llama": "llama.cpp", "mlx": "mlx-vlm"}

# (suite, tests (None = every test in the suite), field, title, higher is better?, unit/format)
VIEWS = [
    ("S1", None, "tps", "S1 Raw speed (tokens/s)", True, "{:.1f}"),
    ("S2", ["load-cold", "load-warm"], "ready_s", "S2 Start to first answer (s)", False, "{:.2f}"),
    ("S2", ["ttft-short", "ttft-3k"], "ttft", "S2 Time to first token (s)", False, "{:.3f}"),
    ("S2", ["reply-400"], "gen_tps", "S2 Whole replies, writing speed (tokens/s)", True, "{:.1f}"),
    ("S3", None, "read_tps", "S3 Context ladder: reading (tokens/s)", True, "{:.0f}"),
    ("S3", None, "gen_tps", "S3 Context ladder: writing at that depth (tokens/s)", True, "{:.1f}"),
    ("S3", None, "ttft", "S3 Context ladder: time to first token (s)", False, "{:.1f}"),
    ("S3", None, "recall", "S3 Context ladder: facts found (of 5)", True, "{:.0f}"),
    ("S4", ["chat-16"], "ttft_mean", "S4 16-turn chat: mean time to first token (s)", False, "{:.3f}"),
    ("S4", ["chat-16"], "right", "S4 16-turn chat: right answers (of 16)", True, "{:.0f}"),
    ("S4", ["regenerate", "edit-turn"], "read_tokens", "S4 Tokens read again (fewer is better)", False, "{:.0f}"),
    ("S4", ["chat-64"], "seconds", "S4 64-turn overflowing chat: total time (s)", False, "{:.0f}"),
    ("S4", ["chat-64"], "probes_right", "S4 64-turn chat: recall probes right (of 8)", True, "{:.0f}"),
    ("S5", None, "right", "S5 Agent tasks: right (1) or wrong (0)", True, "{:.0f}"),
    ("S5", None, "wall", "S5 Agent tasks: wall time (s)", False, "{:.1f}"),
    ("S6", ["ppl"], "ppl", "S6 Perplexity (lower is better; should match llama.cpp)", False, "{:.4f}"),
    ("S6", ["greedy-match"], "median_words", "S6 Greedy continuations: words agreeing with llama.cpp (median of 20)", True, "{:.0f}"),
    ("S6", ["greedy-match"], "identical", "S6 Greedy continuations identical to llama.cpp (of 20)", True, "{:.0f}"),
    ("S6", ["qa-50"], "right", "S6 Exact-answer questions right (of 50)", True, "{:.0f}"),
    ("S6", ["grammar"], "right", "S6 JSON requests right (of 20)", True, "{:.0f}"),
    ("S6", ["grammar"], "strict", "S6 JSON replies that parse as sent (of 20)", True, "{:.0f}"),
    ("S7", None, "peak_gb", "S7 Peak memory (GB)", False, "{:.1f}"),
    ("S7", None, "idle_gb", "S7 Memory after load, idle (GB)", False, "{:.1f}"),
    ("S7", None, "settle_gb", "S7 Memory 5 s after the request (GB)", False, "{:.1f}"),
    ("S8", None, "gen_tps_first5", "S8 Soak: writing speed, first 5 minutes", True, "{:.1f}"),
    ("S8", None, "gen_tps_last5", "S8 Soak: writing speed, last 5 minutes", True, "{:.1f}"),
    ("S8", None, "mem_end_gb", "S8 Soak: memory at the end (GB)", False, "{:.1f}"),
    ("S8", None, "held_pct", "S8 Heat: last minute's writing speed as % of the first", True, "{:.0f}"),
    ("S9", None, "combined_tps", "S9 Several at once: combined writing (tokens/s)", True, "{:.1f}"),
    ("S9", None, "slowest_ttft", "S9 Several at once: slowest first token (s)", False, "{:.2f}"),
    ("S10", None, "j_per_1k", "S10 Energy: joules per 1,000 tokens", False, "{:.1f}"),
    ("S10", None, "watts", "S10 Energy: average watts", False, "{:.1f}"),
]
SCORES = {"recall", "right", "probes_right", "identical", "median_words", "strict"}
LADDER = ["4k", "16k", "32k", "64k", "128k", "256k"]


def load(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    return rows


def table(rows):
    """{(suite, test, model, engine, field): median}"""
    acc = defaultdict(list)
    for r in rows:
        if r.get("suite") in (None, "run") or r.get("test") == "error":
            continue
        for k, v in r.items():
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, (int, float)) and k not in ("rep", "i"):
                acc[(r["suite"], r["test"], r["model"], r["engine"], k)].append(v)
    return {k: statistics.median(v) for k, v in acc.items()}


def previous(path):
    rows = load(path)
    head = next((r for r in rows if r.get("test") == "start"), {})
    cands = sorted(p for p in path.parent.glob("*.jsonl")
                   if p.name < path.name and f"-{head.get('platform')}-{head.get('tier')}-" in p.name)
    return cands[-1] if cands else None


def save_baseline(path):
    rows = load(path)
    head = next(r for r in rows if r.get("test") == "start")
    base = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}
    t = table(rows)
    base[head["platform"]] = {"from": head["run"], "commit": head["commit"],
                              "values": {"|".join(map(str, k)): v for k, v in t.items()}}
    BASELINES.write_text(json.dumps(base, indent=1, sort_keys=True))
    return BASELINES


def baseline(platform):
    if not BASELINES.exists():
        return None, {}
    b = json.loads(BASELINES.read_text()).get(platform)
    if not b:
        return None, {}
    return b["from"], {tuple(k.split("|")): v for k, v in b["values"].items()}


def fmt(v, f):
    return "" if v is None else f.format(v)


def pct(a, b, higher):
    if a is None or b is None or not b:
        return None
    return (a / b - 1) * 100 if higher else (b / a - 1) * 100 if a else None


def sort_tests(tests):
    return sorted(tests, key=lambda t: (LADDER.index(t) if t in LADDER else 99, t))


def build(path, prev_path=None, out=None):
    path = Path(path)
    rows = load(path)
    head = next((r for r in rows if r.get("test") == "start"), {})
    end = next((r for r in rows if r.get("test") == "end"), {})
    cur = table(rows)
    prev_path = Path(prev_path) if prev_path else previous(path)
    prev = table(load(prev_path)) if prev_path else {}
    base_from, base = baseline(head.get("platform"))
    if head.get("tier") != "quick":
        base = {}
    models = []
    for r in rows:
        if r.get("model") and r["model"] not in [m for m, _ in models]:
            models.append((r["model"], r.get("model_name", r["model"])))
    engines_run = [e for e in ENGINES if any(r.get("engine") == e for r in rows)]
    engines_base = [e for e in ("llama", "mlx") if e not in engines_run and any(k[3] == e for k in base)]
    cols = engines_run + engines_base

    def val(key):
        if key in cur:
            return cur[key], False
        if key[3] in engines_base:
            b = base.get(key)
            return b, b is not None
        return None, False

    regress, sections = [], []
    for suite, tests, field, title, higher, f in VIEWS:
        if not any(k[0] == suite and k[4] == field for k in cur):
            continue
        all_tests = sort_tests({k[1] for k in cur if k[0] == suite and k[4] == field and (not tests or k[1] in tests)})
        shown = [e for e in cols if any(val((suite, t, m, e, field))[0] is not None for m, _ in models for t in all_tests)]
        body = []
        for model, mname in models:
            for test in all_tests:
                cells, vals = [], {}
                for e in shown:
                    v, from_base = val((suite, test, model, e, field))
                    vals[e] = v
                    cells.append(f"<td class='num{' base' if from_base else ''}'>{fmt(v, f)}{' <small>(base)</small>' if from_base and v is not None else ''}</td>")
                if all(v is None for v in vals.values()):
                    continue
                others = [vals.get(e) for e in ("llama", "mlx") if vals.get(e) is not None]
                best = (max(others) if higher else min(others)) if others else None
                vs = pct(vals.get("vireo"), best, higher) if field not in SCORES else None
                ref = prev if prev else {k: v for k, v in base.items() if k[3] == "vireo"}
                pv = ref.get((suite, test, model, "vireo", field))
                d = pct(vals.get("vireo"), pv, higher) if pv is not None else None
                if vals.get("vireo") is not None and pv is not None:
                    worse = (vals["vireo"] < pv if higher else vals["vireo"] > pv) if field in SCORES else (d is not None and d < -5)
                    if worse:
                        regress.append(f"{suite} {test} {field} on {mname}: {fmt(pv, f)} -> {fmt(vals['vireo'], f)}")
                cls = "" if vs is None else (" good" if vs >= 5 else " bad" if vs <= -5 else "")
                dcls = "" if d is None else (" good" if d >= 5 else " bad" if d <= -5 else "")
                body.append(f"<tr><td>{html.escape(mname)}</td><td>{html.escape(test)}</td>{''.join(cells)}"
                            f"<td class='num{cls}'>{'' if vs is None else f'{vs:+.0f}%'}</td>"
                            f"<td class='num'>{fmt(pv, f)}</td><td class='num{dcls}'>{'' if d is None else f'{d:+.0f}%'}</td></tr>")
        if body:
            heads = "".join(f"<th class='num'>{LABEL[e]}</th>" for e in shown)
            sections.append(f"<section><h2>{html.escape(title)}</h2><div class='scroll'><table><thead><tr><th>Model</th><th>Test</th>"
                            f"{heads}<th class='num'>Vireo vs best</th><th class='num'>Vireo before</th><th class='num'>Change</th>"
                            f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>")

    s5 = [r for r in rows if r.get("suite") == "S5" and r.get("test") != "error" and not r.get("right")]
    wrong = "".join(f"<li><b>{html.escape(r.get('model_name', ''))}</b> / {LABEL.get(r['engine'], r['engine'])} / "
                    f"{html.escape(r['test'])}{' (thinking)' if r.get('thinking') else ''}: expected {html.escape(r.get('checks', ''))}"
                    f"{'; error ' + html.escape(r['error']) if r.get('error') else ''}</li>" for r in s5)
    errs = [r for r in rows if r.get("error") and r.get("suite") != "S5"]
    err_html = "".join(f"<li>{html.escape(r.get('suite', ''))} {html.escape(str(r.get('test', '')))} / "
                       f"{html.escape(r.get('model_name', '') or '')} / {LABEL.get(r.get('engine'), r.get('engine'))}: "
                       f"{html.escape(str(r['error'])[:300])}</li>" for r in errs)
    skips = [r for r in rows if r.get("skipped")]
    skip_html = "".join(f"<li>{html.escape(r.get('suite', ''))} {html.escape(str(r.get('test', '')))} / "
                        f"{html.escape(r.get('model_name', '') or '')} / {LABEL.get(r.get('engine'), r.get('engine'))}: "
                        f"{html.escape(r['skipped'])}</li>" for r in skips)
    m = head.get("machine", {})
    rig = " · ".join(html.escape(str(x)) for x in [m.get("chip") or m.get("gpu"), f"{m.get('memory_gb')} GB" if m.get("memory_gb") else None,
                                                   m.get("macos") or m.get("os"), "mains" if m.get("on_mains") else "BATTERY",
                                                   m.get("power_mode") or m.get("power_plan")] if x)
    pinned = ", ".join(f"{k} {v}" for k, v in (head.get("pinned") or {}).items() if v)
    reg_html = ("<section class='alert'><h2>Regressions</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in regress) +
                "</ul></section>") if regress else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vireo Bench {html.escape(head.get('tier', ''))} run</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#fff;--good:#1f7a4d;--bad:#b3261e;--base:#8a8780;--accent:#2b5d8a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151514;--fg:#ecebe6;--muted:#a3a198;--line:#2e2d2a;--card:#1d1c1a;--good:#5cc28d;--bad:#ff8a80;--base:#8d8a82;--accent:#8ab8e6}}}}
:root[data-theme="dark"]{{--bg:#151514;--fg:#ecebe6;--muted:#a3a198;--line:#2e2d2a;--card:#1d1c1a;--good:#5cc28d;--bad:#ff8a80;--base:#8d8a82;--accent:#8ab8e6}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px;color:var(--accent)}}
.meta{{color:var(--muted);font-size:13px}} .scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}} th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{color:var(--muted);font-weight:600}} .num{{text-align:right;font-variant-numeric:tabular-nums}} .good{{color:var(--good)}} .bad{{color:var(--bad);font-weight:600}}
.base{{color:var(--base)}} .alert{{border-left:3px solid var(--bad);padding-left:12px}} small{{color:var(--muted)}} ul{{padding-left:20px}}
</style></head><body><main>
<h1>Vireo Bench {html.escape(head.get('suite_version', ''))}: {html.escape(head.get('tier', ''))} run</h1>
<p class="meta">{html.escape(head.get('run', ''))} · commit {html.escape(str(head.get('commit')))}{' (uncommitted changes)' if m.get('dirty') else ''} · {end.get('minutes', '?')} min<br>
{rig}<br>Pinned: {html.escape(pinned)}{'<br>Note: ' + html.escape(head['note']) if head.get('note') else ''}<br>
Compared with: {html.escape(prev_path.name) if prev_path else 'no earlier run of this tier'}{' · llama.cpp / mlx-vlm from baseline ' + html.escape(base_from) if engines_base else ''}</p>
{reg_html}
{''.join(sections)}
{'<section><h2>S5 tasks answered wrong</h2><ul>' + wrong + '</ul></section>' if wrong else ''}
{'<section><h2>Errors</h2><ul>' + err_html + '</ul></section>' if err_html else ''}
{'<section><h2>Skipped</h2><ul>' + skip_html + '</ul></section>' if skip_html else ''}
<p class="meta">Medians over reps. "Vireo vs best": against the better of llama.cpp and mlx-vlm on that row (speed and time rows only). "Vireo before": the previous run of this tier on this platform{', or the baseline' if not prev_path else ''}. See BENCH-SUITE.md for what each test does.</p>
</main></body></html>"""
    out = Path(out) if out else path.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    if regress:
        print("Regressions:\n  " + "\n  ".join(regress))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--prev")
    ap.add_argument("--out")
    ap.add_argument("--save-baseline", action="store_true")
    a = ap.parse_args()
    print(build(a.results, a.prev, a.out))
    if a.save_baseline:
        print(f"Baseline saved: {save_baseline(Path(a.results))}")


if __name__ == "__main__":
    sys.exit(main())
