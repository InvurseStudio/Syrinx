"""Builds the bar-panel page for the Windows run on the Q6_K files after PRs 182 and 183
(Q6_K and Q8_0 rows and one-token kernels, Gemma 4's per-layer matrices, prompt attention):
Vireo against llama.cpp on the three small models, a panel a workload, in the layout of
windows-rtx5060-2026-11-12.html. Only the two engines are drawn: no mark for an earlier
Vireo run. Every number comes from the run's result files.

    python -X utf8 evals/speed/bench_q6_after_pr183_bars.py evals/speed/bench-2026-11-16-windows-q6-after-pr183         evals/speed/windows-rtx5060-2026-11-16-q6-after-pr183-bars.html

Then a picture of it: msedge --headless=new --hide-scrollbars --window-size=1240,2130
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
    rows = load(folder / "results.jsonl") + load(folder / "xlarge.jsonl")
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
plain_m = [value(run, m, "vireo-plain", "medium", WRITE) for m, _ in MODELS]
panels = [
    panel("Reading a document", "~3K tokens", "medium", READ,
          "Level on Gemma 4 E4B and Qwen 3.5 4B, a tenth ahead on Gemma 4 E2B. One run each moves these by a few percent: the prompt is read in a fifth of a second."),
    panel("Reading a repository dump", "~32K tokens", "long", READ,
          "Level or just ahead on all three. This was 4% to 9% behind before prompt attention was rebuilt the way llama.cpp shares a head between warps."),
    panel("Reading a prompt that fills the context", f"{min(full) // 1000}K to {max(full) // 1000}K tokens in a 65,536 context", "xlarge", READ,
          "The cap for an 8 GB GPU. Ahead on all three, Qwen 3.5 4B by the most."),
    panel("Writing a short reply", "~40-token question", "short", WRITE,
          "Qwen 3.5 4B writes a token at a time and is 8% ahead. The Gemmas' wider lead is their draft models' guesses."),
    panel("Writing after a document", "~3K tokens of context", "medium", WRITE,
          f"Where drafting pays most: text copied from the document is nearly all guessed right. With drafting off the three write {plain_m[0]}, {plain_m[1]} and {plain_m[2]}."),
    panel("Writing after a repository dump", "~32K tokens of context", "long", WRITE,
          "Qwen 3.5 4B is 5% behind here, on a reply of 43 tokens. The Gemmas are ahead, E4B by its guesses."),
    panel("Writing after a context-filling prompt", "53K to 57K tokens of context", "xlarge", WRITE,
          f"With drafting off the three write {plain[0]}, {plain[1]} and {plain[2]}. Gemma 4 E2B is still 6% behind this deep; the other two are level or ahead."),
]

report = {
    "title": "Vireo Engine vs llama.cpp, Q6_K",
    "eyebrow": "Vireo core prototype · Windows · 16 Nov 2026 · Q6_K, after PRs 182 and 183",
    "headline": "Vireo's own engine against llama.cpp on the Q6_K files",
    "lede": ("The third run on the Q6_K files, on main after the day's kernel work: Q6_K and Q8_0 weights read with "
             "fewer loads, Gemma 4's per-layer matrices made smaller, and prompt attention rebuilt. Three models that "
             "fit an 8 GB GPU, contexts capped at 65,536 tokens, same GGUF files, same GPU, one server at a time, one "
             "run each. Vireo reads level with or ahead of llama.cpp at every size and writes level or ahead in ten of "
             "twelve rows; it trails on Qwen 3.5 4B's writing at ~32K and Gemma 4 E2B's at 57K."),
    "rig": [
        {"label": "Laptop", "value": "ASUS TUF Gaming A14", "detail": "FA401GM · on mains, Turbo power plan"},
        {"label": "GPU", "value": "GeForce RTX 5060 Laptop", "detail": "8 GB · Blackwell, 26 SMs · driver 621.14"},
        {"label": "CPU · memory", "value": "AMD Ryzen AI 9 465", "detail": "31 GB system memory"},
        {"label": "Software", "value": "Windows 11 · CUDA 13.5",
         "detail": "llama.cpp b12262 (CUDA build) · Vireo main at 02a72c0 (after PR 183), 8-bit mode"},
    ],
    "labels": {"now": "Vireo engine", "other": "llama.cpp"},
    "models": [{"name": m, "quant": q} for m, q in MODELS],
    "panels": panels,
    "method": ("How it was measured: both engines' servers (<code>vireo-server</code> as shipped, <code>llama-server "
               "-ngl 99 -fa on</code>), the same request to each, one run, speeds from the tokens and seconds each "
               "reports (<code>bench-2026-11-13/bench.py</code>; the context-filling prompt by <code>xlarge.py</code>, "
               "the repository dump cut to 190,000 characters). Every model at a 65,536-token context, nothing "
               "spilled into the PC's memory. Reading ~40 tokens is left out (about 30 ms: noise). Qwen 3.5 9B is not "
               "here: at Q6_K it is too large for this GPU. These are not comparable with the Q4 runs up "
               "to “after PR 178”."),
}

page = (here / "windows-rtx5060-2026-11-12.html").read_text(encoding="utf-8")
block = re.compile(r"const REPORT = \{.*?\n\};\n", re.S)
assert len(block.findall(page)) == 1, "the page's REPORT block wasn't found"
page = block.sub(lambda _: "const REPORT = " + json.dumps(report, ensure_ascii=False, indent=2) + ";\n", page)
out.write_text(page, encoding="utf-8")
for p in panels:
    print(p["title"], "| vireo", p["vireo"], "| llama", p["other"])
