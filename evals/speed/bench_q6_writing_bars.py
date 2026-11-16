"""Builds the bar-panel page for the Windows run on the Q6_K files after the writing work of
16 November (Q6_K rows without padding, fewer loads a weight in the one-token kernels):
Vireo against llama.cpp on the three small models, a panel a workload, in the layout of
windows-rtx5060-2026-11-12.html (bench_q6_bars.py with this run's words). A mark shows
where Vireo was in the run before, when its folder is given. Every number comes from the
runs' result files; a row of recheck-e4b-medium.jsonl stands in for the one it re-ran.

    python -X utf8 evals/speed/bench_q6_writing_bars.py evals/speed/bench-2026-11-16-windows-q6-writing         evals/speed/windows-rtx5060-2026-11-16-q6-writing-bars.html evals/speed/bench-2026-11-16-windows-q6

Then a picture of it: msedge --headless=new --hide-scrollbars --window-size=1240,2300
    --screenshot=<out.png> "file:///<page>?theme=dark"
"""
import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
run, out = Path(sys.argv[1]), Path(sys.argv[2])
earlier = Path(sys.argv[3]) if len(sys.argv) > 3 else None
MODELS = [("Gemma 4 E2B", "Q6_K"), ("Gemma 4 E4B", "Q6_K"), ("Qwen 3.5 4B", "Q6_K")]


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def speeds(folder, model, engine, work):
    """(reading, writing) tokens per second and the prompt's tokens, of the last such row."""
    rows = load(folder / "results.jsonl") + load(folder / "xlarge.jsonl") + load(folder / "recheck-e4b-medium.jsonl")
    of = [d for d in rows if d.get("model") == model and d.get("engine") == engine and d.get("work") == work
          and not d.get("error") and d.get("rows") and d["rows"][0].get("steps")]
    s = of[-1]["rows"][0]["steps"][0]
    return s["read"] / s["read_s"], s["gen"] / s["gen_s"], s["prompt"]


def value(folder, model, engine, work, which):
    v = speeds(folder, model, engine, work)[which]
    return round(v, 1) if v < 1000 else round(v)


def panel(title, sub, work, which, note):
    cell = lambda folder, engine: [value(folder, m, engine, work, which) for m, _ in MODELS]
    p = {"title": title, "sub": sub, "vireo": cell(run, "vireo"), "other": cell(run, "llama"), "note": note}
    if earlier:
        p["before"] = cell(earlier, "vireo")
    return p


READ, WRITE = 0, 1
full = [speeds(run, m, "vireo", "xlarge")[2] for m, _ in MODELS]
plain = [value(run, m, "vireo-plain", "xlarge", WRITE) for m, _ in MODELS]
panels = [
    panel("Reading a document", "~3K tokens", "medium", READ,
          "Nothing changed in reading since the run before. One run each moves these by a tenth: the prompt is read in a fifth of a second."),
    panel("Reading a repository dump", "~32K tokens", "long", READ,
          "4% to 9% behind, as before. Prompt attention is half of Gemma 4 E2B's time here, and llama.cpp's is quicker."),
    panel("Reading a prompt that fills the context", f"{min(full) // 1000}K to {max(full) // 1000}K tokens in a 65,536 context", "xlarge", READ,
          "The cap for an 8 GB GPU. Qwen 3.5 4B is level; Gemma 4 E2B, the most attention-bound, is a tenth behind."),
    panel("Writing a short reply", "~40-token question", "short", WRITE,
          "Qwen 3.5 4B was a tenth behind and is level: it writes a token at a time, which is what got faster. Gemma 4 E2B is level and E4B ahead."),
    panel("Writing after a document", "~3K tokens of context", "medium", WRITE,
          "The Gemmas' lead is their draft models' guesses on text copied from the document. Qwen 3.5 4B, with no guesses, went from 10% behind to level."),
    panel("Writing after a repository dump", "~32K tokens of context", "long", WRITE,
          "Qwen 3.5 4B is still a tenth behind this deep: each token reads a longer cache, which this work didn't touch. Gemma 4 E2B's short reply drew few guesses."),
    panel("Writing after a context-filling prompt", "53K to 57K tokens of context", "xlarge", WRITE,
          f"With drafting off the three write {plain[0]}, {plain[1]} and {plain[2]}. All three are faster than the run before; Gemma 4 E2B and Qwen 3.5 4B still trail."),
]

report = {
    "title": "Vireo Engine vs llama.cpp, Q6_K",
    "eyebrow": "Vireo core prototype · Windows · 16 Nov 2026 · Q6_K, after the writing work",
    "headline": "Vireo's own engine against llama.cpp on the Q6_K files",
    "lede": ("The second run on the Q6_K files, after the one-token Q6_K kernels were made to read each weight with "
             "fewer loads and its rows lost their padding. Three models that fit an 8 GB GPU, contexts capped at "
             "65,536 tokens, same GGUF files, same GPU, one server at a time, one run each. With drafting off, writing "
             "is 4% to 15% faster than the run before in every row but Gemma 4 E2B's short reply (level), which "
             "closes Qwen 3.5 4B's gap at short and medium prompts. "
             "Reading is unchanged: level at ~3K tokens, 4% to 10% behind on long prompts."),
    "rig": [
        {"label": "Laptop", "value": "ASUS TUF Gaming A14", "detail": "FA401GM · on mains, Turbo power plan"},
        {"label": "GPU", "value": "GeForce RTX 5060 Laptop", "detail": "8 GB · Blackwell, 26 SMs · driver 621.14"},
        {"label": "CPU · memory", "value": "AMD Ryzen AI 9 465", "detail": "31 GB system memory"},
        {"label": "Software", "value": "Windows 11 · CUDA 13.5",
         "detail": "llama.cpp b12262 (CUDA build) · Vireo branch windows/product-kernel with the writing work (uncommitted), 8-bit mode"},
    ],
    "labels": {"now": "Vireo engine", "other": "llama.cpp", "before": "Vireo in the first Q6_K run"},
    "models": [{"name": m, "quant": q} for m, q in MODELS],
    "panels": panels,
    "method": ("How it was measured: both engines' servers (<code>vireo-server</code> as shipped, <code>llama-server "
               "-ngl 99 -fa on</code>), the same request to each, one run, speeds from the tokens and seconds each "
               "reports (<code>bench-2026-11-13/bench.py</code>; the context-filling prompt by <code>xlarge.py</code>, "
               "the repository dump cut to 190,000 characters). Every model at a 65,536-token context, nothing "
               "spilled into the PC's memory. Reading ~40 tokens is left out (about 30 ms: noise). Qwen 3.5 9B is not "
               "here: at Q6_K it is too large for this GPU. Gemma 4 E4B's ~3K row is its re-run (the first read it "
               "at 4,196 tokens/s once; two more gave 5,107 and 5,192). These are not comparable with the Q4 runs up "
               "to “after PR 178”."),
}

page = (here / "windows-rtx5060-2026-11-12.html").read_text(encoding="utf-8")
block = re.compile(r"const REPORT = \{.*?\n\};\n", re.S)
assert len(block.findall(page)) == 1, "the page's REPORT block wasn't found"
page = block.sub(lambda _: "const REPORT = " + json.dumps(report, ensure_ascii=False, indent=2) + ";\n", page)
out.write_text(page, encoding="utf-8")
for p in panels:
    print(p["title"], "| vireo", p["vireo"], "| llama", p["other"])
