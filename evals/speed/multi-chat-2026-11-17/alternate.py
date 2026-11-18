#!/usr/bin/env python3
"""What it costs vireo-server to go from one person's chat to another's, turn by turn.

`--chats` people, each with a chat of their own that starts with a different stretch of
text, take turns: person 1's first message, person 2's, ..., then everyone's second, and so
on. For each turn: the prompt's tokens, how many came from the cache, the time to the first
token, and what the server logged about saving and loading chats. With `--chats 1` the same
chat's turns follow each other, which is the cost with nobody else on the server.

    python3 -I evals/speed/multi-chat-2026-11-17/alternate.py <model.gguf> [--chats 2]
        [--chars 32000] [--turns 4] [--context 65536] [--server <another build>]

Standard library only; the server is started as Vireo Bench starts it.
"""
import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE.parents[2]
sys.path.insert(0, str(PROTO / "evals" / "bench"))
import engines  # noqa: E402

ASKS = [
    "In one sentence, what is the text above about?",
    "Name one person or place it mentions.",
    "What happens at the very start of it? One sentence.",
    "What happens at the very end of it? One sentence.",
    "Give it a title of five words or fewer.",
    "Is its tone formal or informal? One word.",
]
NOTES = re.compile(r"^(Loaded .*|Saved .*|Couldn't .*)$", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--chats", type=int, default=2)
    ap.add_argument("--chars", type=int, default=32000, help="each chat's opening text")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--server", help="another build of vireo-server, to compare two")
    a = ap.parse_args()
    if a.server:
        config = engines.engine_cfg
        engines.engine_cfg = lambda name: dict(config(name), server=a.server) if name == "vireo" else config(name)
    corpus = (PROTO / "evals/bench/fixtures/corpus.txt").read_text(encoding="utf-8")
    chats = [[] for _ in range(a.chats)]
    docs = [corpus[i * a.chars:(i + 1) * a.chars] for i in range(a.chats)]
    server = engines.Vireo({"key": "multi", "gguf": str(Path(a.model).resolve())}, a.context)
    try:
        print(f"{a.chats} chats, {a.chars} characters each to start, {a.turns} turns each")
        print(f"{'turn':>4} {'chat':>4} {'prompt':>7} {'cached':>7} {'read':>6} {'read s':>7} {'first s':>8} {'tok/s':>6}  server")
        later = []
        for turn in range(a.turns):
            for c in range(a.chats):
                ask = ASKS[turn % len(ASKS)]
                chats[c].append({"role": "user", "content": f"{docs[c]}\n\n{ask}" if turn == 0 else ask})
                server.new_log()
                r = server.chat(chats[c], a.max_tokens, log=False)
                time.sleep(0.2)
                log = server.new_log()
                if r.get("error"):
                    print(f"{turn + 1:4} {c + 1:4} error: {r['error']}")
                    return
                chats[c].append({"role": "assistant", "content": r["text"]})
                m = server.REPLY.search(log)
                prompt, cached, read, read_s, gen, gen_s = (float(x) for x in m.groups()[:6])
                notes = "; ".join(n.strip() for n in NOTES.findall(log))
                print(f"{turn + 1:4} {c + 1:4} {prompt:7.0f} {cached:7.0f} {read:6.0f} {read_s:7.2f} {r['ttft']:8.2f} {gen / max(gen_s, 1e-9):6.1f}  {notes}")
                if turn > 0:
                    later.append((r["ttft"], read))
        if later:
            firsts = sorted(t for t, _ in later)
            print(f"\nTurns after the first: first token in {sum(firsts) / len(firsts):.2f} s on average "
                  f"(median {firsts[len(firsts) // 2]:.2f}, worst {firsts[-1]:.2f}); "
                  f"{sum(n for _, n in later) / len(later):.0f} tokens read a turn.")
        files = [f for f in (server.dir / "chat-cache").rglob("*") if f.is_file()]
        print(f"On disk at the end, before the server stops: {len(files)} files, "
              f"{sum(f.stat().st_size for f in files) / 1e9:.1f} GB.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
