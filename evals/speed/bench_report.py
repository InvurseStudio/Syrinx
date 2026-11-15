"""Builds a speed report page from bench_windows.py's results.jsonl, on the page of
windows-rtx5060-2026-11-15.html (its charts, tables and print styles).

    python evals/speed/bench_report.py evals/speed/bench-2026-11-15-windows evals/speed/windows-rtx5060-2026-11-15-llama.html

Then print it: msedge --headless=new --no-pdf-header-footer --print-to-pdf=<out.pdf> "file:///<page>?theme=dark"
"""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
bench, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in (bench / "results.jsonl").read_text().splitlines() if l.strip()]
page = (here / "windows-rtx5060-2026-11-15.html").read_text(encoding="utf-8")

RAW = ["Qwen 3.5 0.8B", "Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B"]
CHAT = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B"]
QUANT = {m: ("Q4_0" if m.startswith("Gemma") else "Q4_K_M") for m in RAW}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def raw(model, engine, work, field):
    return mean([r[field] for r in rows if r["part"] == "raw" and r["model"] == model
                 and r["engine"] == engine and r.get("work") == work and "error" not in r])


def reply(model, engine, work, field="gen_tps"):
    """A reply's speed in the last round (the first reads and records what later ones keep)."""
    of = [r for r in rows if r["part"] == "replies" and r["model"] == model and r["engine"] == engine
          and r.get("work") == work and "error" not in r]
    return max(of, key=lambda r: r["rep"])[field] if of else 0.0


def tokens(model, work):
    return int(raw(model, "vireo", work, "prompt_n"))


def r1(v):
    return round(v, 1)


def panel(title, sub, models, vireo, other, note):
    return {"title": title, "sub": sub, "models": models, "vireo": [r1(v) for v in vireo],
            "other": [r1(v) for v in other], "note": note}


long_range = f"{min(tokens(m, 'long') for m in RAW):,}–{max(tokens(m, 'long') for m in RAW):,}"
panels = [
    panel("Writing a reply", "short chat · 128 tokens · no drafting", RAW,
          [raw(m, "vireo", "short", "gen_tps") for m in RAW], [raw(m, "llama", "short", "gen_tps") for m in RAW],
          "One token at a time on both engines: each of Vireo's is one replay of a recorded graph of its own kernels."),
    panel("Writing, deep in a long chat", f"after {long_range} tokens of context · 64 tokens · no drafting", RAW,
          [raw(m, "vireo", "long", "gen_tps") for m in RAW], [raw(m, "llama", "long", "gen_tps") for m in RAW],
          "A token's attention over the whole chat is most of the difference from a short chat on both engines."),
    panel("Reading a prompt", "~2,300 tokens", RAW,
          [raw(m, "vireo", "medium", "prompt_tps") for m in RAW], [raw(m, "llama", "medium", "prompt_tps") for m in RAW],
          "The size of a system prompt with its tools, or a document attached. Both Gemmas now read ahead of llama.cpp here; the Qwens' matrix products are still behind."),
    panel("Reading a long prompt", f"{long_range} tokens, by tokenizer", RAW,
          [raw(m, "vireo", "long", "prompt_tps") for m in RAW], [raw(m, "llama", "long", "prompt_tps") for m in RAW],
          "As long a chat as this 8 GB GPU holds for every model. Prompt attention is a third of Vireo's time here, and where it trails most."),
]
REPLIES = [
    ("list", "Rewriting a document as a list", "~450-token request · up to 400 tokens",
     "Part copied, part new. Vireo takes the chat's long runs as guesses and checks several in one pass; the Gemmas' draft models guess between them."),
    ("code", "Returning a code file with an edit", "~500-token request · up to 400 tokens",
     "Nearly all copying: almost every token is guessed from the chat and checked a few at a time."),
    ("story", "Writing a new story", "short request · 400 tokens",
     "Nothing to copy. The Gemmas' draft models still guess about a third of it; Qwen 3.5 has no draft model, so it writes at one token's speed."),
    ("explain", "Explaining, with a code example", "short request · 400 tokens",
     "New text again: the draft models guess about half of the Gemmas' replies."),
]
for key, title, sub, note in REPLIES:
    share = [round(100 * reply(m, "vireo", key, "guessed_share")) for m in CHAT]
    panels.append(panel(title, f"through the servers, as shipped · greedy · {sub}", CHAT,
                        [reply(m, "vireo", key) for m in CHAT], [reply(m, "llama", key) for m in CHAT],
                        note + " Guessed tokens: " + ", ".join(f"{s}% ({m.split()[-1]})" for s, m in zip(share, CHAT)) + "."))


def n(v):
    return f"{v:,.0f}" if v >= 1000 else f"{v:.1f}"


def pair(a, b):
    return f"{n(a)} / {n(b)} ({round(100 * a / b) if b else 0}%)"


reading = [[m, pair(raw(m, "vireo", "short", "prompt_tps"), raw(m, "llama", "short", "prompt_tps")),
            pair(raw(m, "vireo", "medium", "prompt_tps"), raw(m, "llama", "medium", "prompt_tps")),
            pair(raw(m, "vireo", "long", "prompt_tps"), raw(m, "llama", "long", "prompt_tps")),
            f"{tokens(m, 'short')} · {tokens(m, 'medium'):,} · {tokens(m, 'long'):,}"] for m in RAW]
writing = [[m, pair(raw(m, "vireo", "short", "gen_tps"), raw(m, "llama", "short", "gen_tps")),
            pair(raw(m, "vireo", "medium", "gen_tps"), raw(m, "llama", "medium", "gen_tps")),
            pair(raw(m, "vireo", "long", "gen_tps"), raw(m, "llama", "long", "gen_tps"))] for m in RAW]


def first_and_last(model, work):
    of = sorted((r for r in rows if r["part"] == "replies" and r["model"] == model and r["engine"] == "vireo"
                 and r.get("work") == work and "error" not in r), key=lambda r: r["rep"])
    if len(of) > 1 and abs(of[0]["gen_tps"] - of[-1]["gen_tps"]) > 0.15 * of[-1]["gen_tps"]:
        return f"{n(of[-1]['gen_tps'])} (first round {n(of[0]['gen_tps'])})"
    return n(of[-1]["gen_tps"]) if of else "–"


replies = [[m] + [f"{first_and_last(m, k)} / {n(reply(m, 'llama', k))}" for k, *_ in REPLIES] for m in CHAT]


def memory(model, engine, part):
    of = [r for r in rows if r["part"] == "memory" and r["model"] == model and r["engine"] == engine
          and part in r["after"]]
    return of[-1] if of else None


mem_rows = []
for m in RAW:
    a, b, c = memory(m, "llama", "raw"), memory(m, "vireo", "replies"), memory(m, "llama", "replies")
    mem_rows.append([m, f"{a['gpu_mb']:,} MB" if a else "–",
                     f"{b['gpu_mb']:,} MB ({b['context']:,} tokens)" if b else "–",
                     f"{c['gpu_mb']:,} MB" if c else "–"])

report = json.loads((bench / "report.json").read_text(encoding="utf-8"))
report["models"] = [{"name": m, "quant": QUANT[m]} for m in RAW]
report["panels"] = panels
report["tables"] = [
    {"title": "Reading, tokens/s: Vireo / llama.cpp (Vireo as a share of llama.cpp)",
     "head": ["Model", "Short prompt", "~2,300 tokens", "Long prompt", "Tokens: short · medium · long"],
     "rows": reading,
     "note": "The same three prompt files as the Mac's benchmark. A short prompt's reading is mostly each engine's start-up for a request, not its kernels."},
    {"title": "Writing with no drafting, tokens/s: Vireo / llama.cpp",
     "head": ["Model", "After the short prompt (128 tokens)", "After ~2,300 (64)", "After the long prompt (64)"],
     "rows": writing, "note": ""},
    {"title": "Whole replies through the servers, tokens/s: Vireo as shipped / llama.cpp",
     "head": ["Model", "A document as a list", "A code file with an edit", "A new story", "An explanation"],
     "rows": replies,
     "note": "vireo-server at its defaults (guesses from the chat, and Google's draft models for the Gemmas) against llama-server. The last round of each; a first round is shown where it differed by more than 15%. Qwen 3.5 9B fills this GPU: it guesses only when the memory to undo a wrong guess is free as a reply starts."},
    {"title": "GPU memory in use",
     "head": ["Model", "llama.cpp, 36,000-token context", "vireo-server (its own context)", "llama-server, 8,192-token context"],
     "rows": mem_rows,
     "note": "Read once each server's rounds were done, the whole GPU's (about 0.6 GB of it is Windows'). vireo-server sizes its context to the memory free: the server also holds its picker sessions and, for the Gemmas, a draft model."},
]
i, j = page.index("const REPORT = {"), page.index("// ---- Rendering:")
page = page[:i] + "const REPORT = " + json.dumps(report, indent=2, ensure_ascii=False) + ";\n" + page[page.rindex("</script>", 0, j):]


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:60])
    return s.replace(a, b)


# A panel may chart some of the models only (`models`: their names).
page = rep(page, "const PANELS = REPORT.panels.map(p => ({ ...p, llama: p.other }));",
           "const PANELS = REPORT.panels.map(p => ({ ...p, llama: p.other, names: p.models || MODELS }));\n"
           "const quantOf = name => QUANT[MODELS.indexOf(name)] || \"\";")
page = rep(page, "const H = TOP + MODELS.length * GROUP + (MODELS.length - 1) * ROWGAP + 26;",
           "const H = TOP + p.names.length * GROUP + (p.names.length - 1) * ROWGAP + 26;")
page = rep(page, "  MODELS.forEach((m, i) => {\n    const y0", "  p.names.forEach((m, i) => {\n    const y0")
page = rep(page, "q.textContent = QUANT[i];", "q.textContent = quantOf(m);")
page = rep(page, "${esc(MODELS[i])} · ${esc(p.title.toLowerCase())}", "${esc(p.names[i])} · ${esc(p.title.toLowerCase())}")
page = rep(page, "for (const p of PANELS) MODELS.forEach((m, i) => {", "for (const p of PANELS) p.names.forEach((m, i) => {")
out.write_text(page, encoding="utf-8", newline="\n")
print("ok", len(panels), "panels")
