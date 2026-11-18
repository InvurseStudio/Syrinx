#!/usr/bin/env python3
"""Several people asking vireo-server at the same moment: how fast each is answered.

`--people` chats, each starting with a different stretch of text. Round 1: everyone's
first message at once (each prompt is read in turn). Rounds 2 on: everyone asks for a
few paragraphs at once. For each round: when each person's first token came and when
their reply ended, the tokens a second each reply was written at (the server's "Reply:"
lines), and the tokens a second the server wrote in all. Run it with `--slots 1` (one at
a time, the others queue) and `--slots <people>` to compare.

    python3 -I evals/speed/multi-chat-2026-11-17/together.py <model.gguf> [--people 4]
        [--slots 4] [--chars 12000] [--rounds 2] [--max-tokens 200] [--context 32768]

Standard library only; the server is started as Vireo Bench starts it.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE.parents[2]
sys.path.insert(0, str(PROTO / "evals" / "bench"))
import engines  # noqa: E402

FIRST = "In two sentences, what is the text above about?"
ASKS = [
    "Write three paragraphs on what a reader might learn from it.",
    "Write three paragraphs on how it could be made clearer.",
    "Write three paragraphs comparing its start with its end.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--people", type=int, default=4)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--chars", type=int, default=12000, help="each chat's opening text")
    ap.add_argument("--rounds", type=int, default=2, help="rounds after the first")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--context", type=int, default=32768)
    a = ap.parse_args()
    corpus = (PROTO / "evals/bench/fixtures/corpus.txt").read_text(encoding="utf-8")
    docs = [corpus[i * a.chars:(i + 1) * a.chars] for i in range(a.people)]
    chats = [[] for _ in range(a.people)]
    extra = ["--slots", str(a.slots)] if a.slots > 1 else []
    server = engines.Vireo({"key": "together", "gguf": str(Path(a.model).resolve())}, a.context, extra=extra)
    try:
        print(f"{a.people} people at once, {a.slots} slots, {a.chars} characters each to start, {a.max_tokens}-token replies")
        for r in range(a.rounds + 1):
            ask = FIRST if r == 0 else ASKS[(r - 1) % len(ASKS)]
            for c in range(a.people):
                chats[c].append({"role": "user", "content": f"{docs[c]}\n\n{ask}" if r == 0 else ask})
            results = [None] * a.people
            server.new_log()

            def one(c):
                # Each person's chat has its id, as the app's chats do.
                results[c] = server.chat(chats[c], a.max_tokens if r else 60, log=False, chat_id=f"person-{c}")

            t0 = time.time()
            threads = [threading.Thread(target=one, args=(c,)) for c in range(a.people)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.time() - t0
            time.sleep(0.3)
            log = server.new_log()
            errors = [x["error"] for x in results if x.get("error")]
            if errors:
                print(f"round {r + 1}: error: {errors[0]}")
                return
            for c in range(a.people):
                chats[c].append({"role": "assistant", "content": results[c]["text"]})
            replies = [(int(m[5]), float(m[6])) for m in server.REPLY.finditer(log)]
            written = sum(n for n, _ in replies)
            firsts = sorted(x["ttft"] for x in results)
            ends = sorted(x["total"] for x in results)
            rates = sorted(n / max(s, 1e-9) for n, s in replies)
            print(f"round {r + 1}: first tokens at {', '.join(f'{t:.1f}' for t in firsts)} s; "
                  f"replies done at {', '.join(f'{t:.1f}' for t in ends)} s")
            print(f"         each written at {', '.join(f'{x:.1f}' for x in rates)} tok/s; "
                  f"{written} tokens in {wall:.1f} s: {written / wall:.1f} tok/s in all")
            words = [len(x["text"].split()) for x in results]
            print(f"         words in each reply: {words}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
