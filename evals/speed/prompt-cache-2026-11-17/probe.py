#!/usr/bin/env python3
"""How much of a new chat's prompt vireo-server takes from its cache, chat after chat.

New chats in a row, each one message: plain questions, a long one, ones that ask for a
file to be changed (the tools that change files get declared, so the system turn differs),
and one that saves a memory. For each: the prompt's tokens, how many were kept from the
session, and the time to read the rest (the server's "Reply:" line).

    python3 -I evals/speed/prompt-cache-2026-11-17/probe.py <model.gguf> [--context 32768]
        [--server <another build of vireo-server>]

Standard library only. The server is started as Vireo Bench starts it (evals/bench/engines.py),
with the bench's file tools and its sandbox folder, which is where the files are written.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE.parents[2]
sys.path.insert(0, str(PROTO / "evals" / "bench"))
import engines  # noqa: E402
from desk import Desk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--long-chars", type=int, default=24000, help="the long chats' pasted text")
    ap.add_argument("--server", help="another build of vireo-server, to compare two")
    a = ap.parse_args()
    if a.server:
        config = engines.engine_cfg
        engines.engine_cfg = lambda name: dict(config(name), server=a.server) if name == "vireo" else config(name)
    desk = Desk()
    work = desk.posix
    doc = (PROTO / "evals/bench/fixtures/corpus.txt").read_text(encoding="utf-8")[:a.long_chars]
    chats = [
        ("plain", "What is the capital of France?"),
        ("plain", "Name three primary colours."),
        ("long, change", f"{doc}\n\nWrite a two-line summary of the text above to the file {work}/summary.txt."),
        ("plain", "What is 12 times 12?"),
        ("change", f"Create the file {work}/hello.txt containing the word hello."),
        ("plain", "Which is larger, a litre or a pint?"),
        ("long", f"{doc}\n\nIn one sentence, what is the text above?"),
        ("plain", "How many days are in a leap year?"),
        ("change", f"Create the file {work}/bye.txt containing the word bye."),
        ("plain", "Name a mammal that lays eggs."),
        ("remember", "Remember that my favourite colour is teal."),
        ("plain", "What is the boiling point of water in Celsius?"),
        ("plain", "Name a bird that cannot fly."),
        ("remember", "Remember that I live in Tampere."),
        ("plain", "What is the freezing point of water in Fahrenheit?"),
    ]
    server = engines.Vireo({"key": "probe", "gguf": str(Path(a.model).resolve())}, a.context)
    try:
        total_read = total = 0
        print(f"{'chat':14} {'prompt':>7} {'kept':>7} {'read':>7} {'read s':>7}  {'calls':22} reply")
        for kind, text in chats:
            r = server.chat([{"role": "user", "content": text}], 120, tools=True, desk=desk)
            if r.get("error") or not r["steps"]:
                print(f"{kind:14} error: {r.get('error')}")
                continue
            # The reply's first read: the new chat's prompt.
            step = r["steps"][0]
            total += step["prompt"]
            total_read += step["read"]
            said = " ".join(r["text"].split())[:70]
            print(f"{kind:14} {step['prompt']:7} {step['cached']:7} {step['read']:7} {step['read_s']:7.2f}  {','.join(r['calls']) or '-':22} {said}")
        print(f"\nFrom the cache: {100 * (total - total_read) / max(total, 1):.0f}% of {total} prompt tokens.")
    finally:
        server.stop()
        desk.close()


if __name__ == "__main__":
    main()
