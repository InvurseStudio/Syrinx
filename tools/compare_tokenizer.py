"""Checks Vireo's tokenizer against llama.cpp's, token for token.

Runs llama.cpp's `llama-tokenize` and `vireo-cli tokenize` on the same texts and reports
every text where the ids differ. The texts:

- every text file in the Vireo repository (code, docs, data) up to 100 KB, cut into
  pieces of a few thousand characters
- random text built from a mix of scripts, numbers, punctuation, whitespace, emoji,
  combining marks, contractions and special tokens (the same every run: seeded)

Each text is tokenized twice: as a prompt (BOS added, special tokens read) and raw
(neither).

    python -I tools/compare_tokenizer.py models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

Needs a release build of vireo-cli (`cargo build --release -p vireo-cli`) and llama.cpp
in reference/ (see PLAN.md).
"""

import argparse
import concurrent.futures
import os
import pathlib
import random
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
# Windows (.exe, reference/llama.cpp-*) or the Mac (reference/llama-b*).
VIREO = next((p for p in (ROOT / "target/release/vireo-cli.exe", ROOT / "target/release/vireo-cli")
              if p.exists()), ROOT / "target/release/vireo-cli")
LLAMA = next((p for pattern in ("reference/llama.cpp-*/llama-tokenize.exe",
                                "reference/llama-*/llama-tokenize")
              for p in ROOT.glob(pattern)), None)
TEXT_SUFFIXES = {".md", ".txt", ".rs", ".swift", ".cs", ".py", ".js", ".jsx", ".ts", ".json",
                 ".yml", ".yaml", ".toml", ".xaml", ".sh", ".cmd", ".ps1", ".html", ".css", ".csv"}


def repo_texts(piece_chars=3000):
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True,
                            text=True, check=True).stdout.splitlines()
    for name in listed:
        path = REPO / name
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        if path.stat().st_size > 100_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i in range(0, len(text), piece_chars):
            yield f"{name} [{i}]", text[i:i + piece_chars]


POOL = [
    "the", " quick", " brown", "Fox", "JUMPS", "über", "naïve", "façade", "Ελληνικά", "русский",
    "日本語", "中文", "한국어", "हिन्दी", "العربية", "עברית", "ไทย", "Tiếng Việt", "😀", "👍🏽", "👨‍👩‍👧",
    "🇳🇿", "1", "22", "333", "4444", "55555", "3.14", "-7", "½", "²", "٣", "Ⅻ", "'s", "'S", "'re", "'VE",
    "'ll", "'d", "'m", "'t", "'x", "''", "!", "?", "...", "—", "(", ")", "{", "}", "[", "]", "<", ">",
    "/", "\\", "\"", "#", "$", "%", "&", "*", "+", "=", "_", "|", "~", "`", "@", "^", ":", ";", ",", ".",
    " ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", "\r", " \n", " ", "　", " ", "​",
    "́", "ा", "\u0000", "\u001f", "\u007f", "﻿", "<|eot_id|>", "<|start_header_id|>",
    "<|end_header_id|>", "<|begin_of_text|>", "<|im_start|>", "<|im_end|>", "<|eot_id", "|>",
    "<|turn>", "<turn|>", "<bos>", "<eos>", "<|channel>", "<channel|>", "▁", "▁▁", "</s>",
]


def fuzz_texts(count, seed=20261110):
    rng = random.Random(seed)
    for i in range(count):
        parts = rng.choices(POOL, k=rng.randint(1, 40))
        yield f"random #{i}", "".join(parts)


def tokenize(program_args, text, workdir):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".txt",
                                     dir=workdir, delete=False) as f:
        f.write(text)
        path = f.name
    try:
        args = [a.replace("{file}", path) for a in program_args]
        # llama-tokenize now and then prints nothing when many copies start at once
        # (seen once in 3,484 runs; the same text ran fine on its own), so try again.
        for _ in range(3):
            out = subprocess.run(args, capture_output=True, timeout=120)
            if out.returncode != 0:
                return f"error {out.returncode}: {out.stderr.decode('utf-8', 'replace')[-300:]}"
            lines = out.stdout.decode("utf-8", "replace").strip().splitlines()
            if lines:
                return [int(x) for x in lines[-1].strip("[]").split(",") if x.strip()]
        return f"no output: {out.stderr.decode('utf-8', 'replace')[-300:]!r}"
    finally:
        os.unlink(path)


def compare(model, name, text, raw, workdir):
    llama = [str(LLAMA), "-m", model, "-f", "{file}", "--ids", "--no-escape", "--log-disable"]
    vireo = [str(VIREO), "tokenize", model, "{file}", "--file", "--ids"]
    if raw:
        llama += ["--no-bos", "--no-parse-special"]
        vireo += ["--raw"]
    return name, raw, text, tokenize(llama, text, workdir), tokenize(vireo, text, workdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--fuzz", type=int, default=400, help="random texts (default 400)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if LLAMA is None or not VIREO.exists():
        sys.exit("Needs llama.cpp's llama-tokenize in reference/ and a release vireo-cli.")

    texts = list(repo_texts()) + list(fuzz_texts(args.fuzz))
    work = [(name, text, raw) for name, text in texts for raw in (False, True)]
    print(f"Comparing {len(texts)} texts ({len(work)} runs) with {LLAMA.parent.name}…")

    failures = []
    with tempfile.TemporaryDirectory() as workdir, \
            concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(compare, args.model, n, t, r, workdir) for n, t, r in work]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, raw, text, llama_ids, vireo_ids = future.result()
            if llama_ids != vireo_ids:
                failures.append((name, raw, text, llama_ids, vireo_ids))
            if done % 200 == 0:
                print(f"  {done}/{len(work)}, {len(failures)} different")

    tokens = 0
    for name, raw, text, llama_ids, vireo_ids in failures[:15]:
        mode = "raw" if raw else "prompt"
        print(f"\nDIFFERENT: {name} ({mode}) {text[:120]!r}")
        if isinstance(llama_ids, str) or isinstance(vireo_ids, str):
            print(f"  llama.cpp: {llama_ids}\n  vireo:     {vireo_ids}")
            continue
        first = next((i for i, (a, b) in enumerate(zip(llama_ids, vireo_ids)) if a != b),
                     min(len(llama_ids), len(vireo_ids)))
        print(f"  first difference at token {first}")
        print(f"  llama.cpp: {llama_ids[max(0, first - 3):first + 6]}")
        print(f"  vireo:     {vireo_ids[max(0, first - 3):first + 6]}")
    print(f"\n{len(work) - len(failures)} of {len(work)} runs identical.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
