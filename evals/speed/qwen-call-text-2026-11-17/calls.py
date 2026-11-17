#!/usr/bin/env python3
"""Does the model call write_file when asked to create a small file? Twelve new chats.

Qwen 3.8 27B with thinking off began such a reply with "[" as often as with <tool_call>,
wrote "[Tool call: write_file]" (or "I'll create that file for you.") and stopped: 8 of 12
on main at 6bc5d1d. Each line here: the file's word, the tools called, the reply's start.
With VIREO_DEBUG_CALLS=1 the server's log (its path is printed) has the reply's three
likeliest first tokens ("[calls] turn starts with ...").

    python3 -I evals/speed/qwen-call-text-2026-11-17/calls.py <model.gguf> [--thinking]

Standard library only; the server is started as Vireo Bench starts it, with its file tools
and its sandbox folder.
"""
import sys
from pathlib import Path

PROTO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROTO / "evals" / "bench"))
import engines  # noqa: E402
from desk import Desk  # noqa: E402

WORDS = ["bye", "zebra", "ok", "cat", "dog", "sun", "moon", "tree", "blue", "green", "seven", "milk"]


def main():
    model = str(Path(sys.argv[1]).resolve())
    desk = Desk()
    server = engines.Vireo({"key": "calls", "gguf": model}, 32768, thinking="--thinking" in sys.argv[2:])
    missed = 0
    try:
        for word in WORDS:
            ask = f"Create the file {desk.posix}/{word}.txt containing the word {word}."
            r = server.chat([{"role": "user", "content": ask}], 120, tools=True, desk=desk)
            missed += "write_file" not in r["calls"]
            print(f"{word:6} {','.join(r['calls']) or '-':12} {' '.join(r['text'].split())[:80]}", flush=True)
        print(f"\n{missed} of {len(WORDS)} without a write_file call. Server log: {server.log_path}")
    finally:
        server.stop()
        desk.close()


if __name__ == "__main__":
    main()
