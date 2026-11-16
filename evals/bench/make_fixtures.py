"""Builds Vireo Bench's generated fixtures. The output is committed, so every run (and the
Windows laptop) uses the same text; run this again only on a version bump.

    python3 -I make_fixtures.py              corpus, ppl text, notes, copies, then cuts
    python3 -I make_fixtures.py --cuts-only  only the token cuts (after adding a model)

  corpus.txt       ~1.4 MB of this repository's own source, frozen, ASCII only: the body of
                   every long prompt (S1, S3, the deep tool task)
  ppl.txt          ~120 KB of the project's own prose, for perplexity (S6)
  chat_notes.json  64 made-up shift reports, each with facts and a question (S4)
  replies.json     the four whole-reply prompts from tools/reply_speed.py (S2)
  sandbox/, skills/, incident-log.txt, medium.txt, short.txt
                   copied from evals/speed/bench-2026-11-13 and prompts-mac (S1, S2, S5)
  cuts.json        per model key: how many characters of corpus.txt make N tokens on that
                   model's tokenizer (llama-tokenize), for every S1 size and S3 rung. The
                   Windows Q4 files share their tokenizers with the Mac's Q6, so the Mac's
                   cuts serve both.
"""
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

FIX = HERE / "fixtures"
PROTO = common.PROTO
OLD = PROTO / "evals/speed/bench-2026-11-13"

S1_SIZES = {"pp512": 512, "pp4096": 4096, "d8192": 8192, "d32768": 32768}


def ladder_body_tokens(ctx):
    """Tokens of corpus in a ladder prompt for a context of `ctx`: room left for Vireo's
    ~1,300-token system turn, the question and 200 tokens of reply (rules, item 9)."""
    return int(ctx * 0.92) - 1700


def corpus():
    rng = random.Random(1115)
    files = []
    for root in ["crates", "Mac", "Windows", "tools"]:
        for p in sorted((PROTO / root).rglob("*")):
            if (p.is_file() and p.suffix in (".rs", ".swift", ".py", ".md", ".cs", ".metal", ".cu", ".h", ".c")
                    and 2_000 < p.stat().st_size < 400_000
                    and not any(x in p.parts for x in ("target", ".build", "build", "Vendor", "vendor", "node_modules", "obj", "bin"))):
                files.append(p)
    rng.shuffle(files)
    out, total = [], 0
    for p in files:
        text = p.read_text(errors="replace").encode("ascii", "ignore").decode()
        piece = f"\n\n===== {p.relative_to(PROTO).as_posix()} =====\n{text}"
        out.append(piece)
        total += len(piece)
        if total > 1_400_000:
            break
    (FIX / "corpus.txt").write_text("".join(out)[:1_400_000])


def ppl_text():
    parts = []
    for p in [PROTO / "PLAN.md", PROTO / "HANDOFF.md", PROTO / "README.md"] + sorted((PROTO.parent / "docs").glob("*.md")):
        if p.exists():
            parts.append(p.read_text(errors="replace").encode("ascii", "ignore").decode())
    (FIX / "ppl.txt").write_text("\n\n".join(parts)[:120_000])


NAMES = ["Yuki", "Bram", "Ines", "Leif", "Caspian", "Tobias", "Marisol", "Dagny"]
JOBS = ["fork services", "wheel truings", "brake bleeds", "tubeless conversions", "e-bike diagnostics", "fittings"]
FILLER = [
    "Three rental bikes came back late and the late fees were waived as a courtesy.",
    "One customer asked for a same-day service and was booked in for the morning instead.",
    "The front counter was busy from opening until noon, mostly with walk-in repairs.",
    "The parts wall was reorganised so the common sizes sit at eye height.",
    "Rain kept the test-ride area closed for most of the afternoon.",
    "A courier collected the online orders just before four.",
    "The compressor in the back room was serviced and runs quietly again.",
    "Two customers asked about winter tyres and were shown the studded range.",
    "The till was counted twice at close because the first count was interrupted.",
    "A school group booked a safety-check morning for next month.",
    "The workshop radio was turned down after a complaint from next door.",
    "Stock of inner tubes in the common sizes is getting low.",
]


def chat_notes():
    rng = random.Random(64)
    notes = []
    for i in range(1, 65):
        lead, job = rng.choice(NAMES), rng.choice(JOBS)
        n, till, bay = rng.randint(3, 29), rng.randint(900, 4800), rng.randint(1, 9)
        body = [f"Shift report {i}: {lead} led the shift and the team did {n} {job}.",
                f"The till closed at ${till}.", f"Bay {bay} was kept free for e-bikes."]
        while sum(len(s) for s in body) < 2300:
            body.append(rng.choice(FILLER))
        rng.shuffle(body[1:])
        notes.append({"n": i, "text": " ".join(body), "lead": lead, "job": job, "count": n, "till": till, "bay": bay,
                      "question": f"How many {job} did the team do in shift report {i}? Answer with the number only.",
                      "expect": str(n)})
    (FIX / "chat_notes.json").write_text(json.dumps({
        "_about": "S4: made-up shift reports, ~600 tokens each. A turn pastes one and asks about it; recall turns ask for an earlier report's till total.",
        "notes": notes}, indent=1))


def replies():
    sys.path.insert(0, str(PROTO / "tools"))
    import reply_speed  # noqa: E402
    (FIX / "replies.json").write_text(json.dumps({
        "_about": "S2 reply-400: tools/reply_speed.py's four prompts, a whole reply each (400 tokens at most).",
        "prompts": reply_speed.PROMPTS}, indent=1))


def copies():
    for name in ["sandbox", "skills"]:
        if (FIX / name).exists():
            shutil.rmtree(FIX / name)
        shutil.copytree(OLD / name, FIX / name)
    shutil.copy(OLD / "prompts/incident-log.txt", FIX / "incident-log.txt")
    shutil.copy(OLD / "prompts/medium.txt", FIX / "medium.txt")
    shutil.copy(PROTO / "evals/speed/prompts-mac/short.txt", FIX / "short.txt")


def count(gguf, text, tokenize):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
    try:
        out = subprocess.run([str(tokenize), "-m", str(gguf), "-f", f.name, "--ids", "--show-count", "--log-disable"],
                             capture_output=True, text=True, env=common.llama_env(tokenize.parent)).stdout
        return int(out.rsplit("Total number of tokens:", 1)[1].split()[0])
    finally:
        Path(f.name).unlink()


def cut_for(gguf, text, target, tokenize, ratio):
    """Characters of `text` (cut at a line's end) that make at most `target` tokens."""
    chars = min(len(text), int(target * ratio))
    for _ in range(8):
        end = text.rfind("\n", 0, chars) if chars < len(text) else len(text)
        n = count(gguf, text[:end], tokenize)
        if target * 0.985 <= n <= target:
            return end, n
        chars = int(end * target / n * (0.995 if n > target else 1.0))
        if chars >= len(text) and n < target:
            return len(text), n
    return end, n


def cuts():
    cfg = common.config()
    tokenize = PROTO / cfg["platforms"]["mac"]["engines"]["llama"]["dir"] / "llama-tokenize"
    text = (FIX / "corpus.txt").read_text()
    path = FIX / "cuts.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out["_about"] = ("Characters of corpus.txt (cut at a line's end) and the tokens they make, per model key. "
                     "S1 sizes are exact prompt lengths; S3 rungs are the body of a ladder prompt "
                     "(0.92 of the context, less 1,700 tokens of room).")
    for m in cfg["platforms"]["mac"]["models"]:
        gguf = common.path(m["gguf"])
        if not gguf.exists():
            print(f"skip {m['key']}: no {gguf}")
            continue
        ratio = len(text[:200_000]) / count(gguf, text[:200_000], tokenize)
        row = {}
        for name, n in S1_SIZES.items():
            row[name] = cut_for(gguf, text, n - 1, tokenize, ratio)
        for rung, ctx in cfg["ladder"].items():
            if ctx <= m["max_context"]:
                row[rung] = cut_for(gguf, text, ladder_body_tokens(ctx), tokenize, ratio)
        row["deep"] = cut_for(gguf, text, int(min(131072, m["max_context"]) * 0.7), tokenize, ratio)
        # The Windows laptop's smaller contexts: the deep tool task at 70% of those too.
        for wctx in (65536, 32768):
            row[f"deep{wctx}"] = cut_for(gguf, text, int(wctx * 0.7), tokenize, ratio)
        out[m["key"]] = row
        print(m["key"], json.dumps(row))
        path.write_text(json.dumps(out, indent=1))


def main():
    FIX.mkdir(exist_ok=True)
    if "--cuts-only" not in sys.argv:
        corpus()
        ppl_text()
        chat_notes()
        replies()
        copies()
    cuts()


if __name__ == "__main__":
    main()
