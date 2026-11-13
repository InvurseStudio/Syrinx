#!/usr/bin/env python3
"""What vireo-server does with more than one chat at a time.

Three things are timed, on one model:

  together    two chats sent at the same moment: how long the second waits for the first
  mid-reply   a short question sent while a long reply is being written
  alternating two long chats taking turns: what each switch between them costs (the
              tokens kept and read, the saved chat loaded, the chat cache's size on disk)

  python3 tools/concurrent_run.py --model <file.gguf> --out evals/long-run/concurrent.json

Standard library only; the server's data goes in a temporary folder (see long_run.py).
"""

import argparse
import json
import random
import shutil
import threading
import time
from pathlib import Path

import long_run


def ask_timed(server, messages, max_tokens=None):
    """`ask` on a connection of its own (the log lines are read by the caller)."""
    started = time.time()
    body = {"systemPrompt": long_run.SYSTEM, "messages": messages,
            "generation": {"max_tokens": max_tokens or server.max_tokens, "temperature": 0}}
    first, reply = None, ""
    response = long_run.post(f"http://127.0.0.1:{server.port}/api/chat/completions", body, server.token)
    for event in long_run.events(response):
        if "content" in event:
            first = first or time.time() - started
            reply += event["content"]
    return {"sent": started, "first_word_s": first, "total_s": time.time() - started,
            "words": len(reply.split())}


def side_by_side(server, first_messages, second_messages, delay, first_tokens=None):
    """Sends the first chat, then the second `delay` seconds later; both rows."""
    rows = [None, None]

    def run(i, messages, tokens):
        rows[i] = ask_timed(server, messages, tokens)

    a = threading.Thread(target=run, args=(0, first_messages, first_tokens))
    b = threading.Thread(target=run, args=(1, second_messages, None))
    a.start()
    time.sleep(delay)
    b.start()
    a.join()
    b.join()
    for row in rows:
        row.pop("sent")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--notes", type=int, default=14, help="notes in each long chat (default 14, ~12K tokens)")
    parser.add_argument("--switches", type=int, default=6)
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()

    server = long_run.Vireo(args.model, 0, 200, args.extra)
    result = {"model": Path(args.model).name, "context": server.context}
    try:
        user = lambda text: [{"role": "user", "content": text}]
        # Warm up: the system prompt and the questions the server asks itself.
        server.ask(user("Say hello in five words."))

        alone = ask_timed(server, user("Name three rivers in Europe, one line."))
        alone.pop("sent")
        together = side_by_side(server, user("Name three lakes in Africa, one line."),
                                user("Name three mountains in Asia, one line."), 0.0)
        result["together"] = {"alone": alone, "first": together[0], "second": together[1]}
        print(f"alone: first word {alone['first_word_s']:.2f}s, done {alone['total_s']:.2f}s")
        print(f"together: first chat first word {together[0]['first_word_s']:.2f}s, done {together[0]['total_s']:.2f}s; "
              f"second chat first word {together[1]['first_word_s']:.2f}s, done {together[1]['total_s']:.2f}s")

        long_reply = user("Write a 300-word story about a lighthouse keeper.")
        mid = side_by_side(server, long_reply, user("What is 12 times 12? One line."), 3.0, first_tokens=450)
        result["mid_reply"] = {"long": mid[0], "short": mid[1]}
        print(f"mid-reply: the long reply took {mid[0]['total_s']:.1f}s; the short question, sent 3 s in, "
              f"got its first word after {mid[1]['first_word_s']:.1f}s")

        # Two long chats built up, then taking turns.
        chats = []
        for seed in (11, 23):
            rng = random.Random(seed)
            messages = []
            for i in range(1, args.notes + 1):
                text, facts = long_run.note(i, rng)
                messages.append({"role": "user", "content": f"{text}\n\nFrom note {i}: how many {facts['item']} were shipped?"})
                row = server.ask(messages)
                messages.append({"role": "assistant", "content": row["reply"]})
            chats.append(messages)
            print(f"chat {len(chats)} built: {row.get('prompt_tokens')} tokens")
        switches = []
        for n in range(args.switches):
            messages = chats[n % 2]
            messages.append({"role": "user", "content": "In one line: which note mentioned the most units shipped?"})
            row = server.ask(messages)
            messages.append({"role": "assistant", "content": row["reply"]})
            time.sleep(1.5)  # the save is written on a thread of its own
            row = {k: row.get(k) for k in ("prompt_tokens", "kept", "read", "read_s", "loaded_tokens",
                                           "load_ms", "first_word_s", "total_s")}
            row.update(chat=n % 2 + 1, cache_bytes=server.cache_bytes())
            switches.append(row)
            print(f"switch {n + 1} to chat {row['chat']}: prompt {row['prompt_tokens']}, kept {row['kept']}, "
                  f"read {row['read']} in {row['read_s']}s, loaded {row['loaded_tokens']} tokens in {row['load_ms']} ms, "
                  f"first word {row['first_word_s']:.2f}s, cache on disk {row['cache_bytes'] / 1e9:.2f} GB")
        result["alternating"] = switches
    finally:
        server.stop()
        shutil.rmtree(server.dir, ignore_errors=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"Saved to {out}.")


if __name__ == "__main__":
    main()
