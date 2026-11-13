#!/usr/bin/env python3
"""A long chat, turn by turn, through vireo-server or llama.cpp's llama-server.

Each turn the user pastes a made-up status note (a few hundred words with facts of its
own) and asks about it, so the chat grows by several hundred tokens a turn and crosses
the context limit partway. Recorded for every turn: the prompt's tokens and how many the
server kept from its cache, the time to the first word, the writing speed, and the reply.
Some turns ask for a number from an earlier note (an early one and a recent one), to see
what the chat still knows.

With vireo-server (--engine vireo) two more things are tried: a turn that asks the model
to remember a fact (the remembered facts sit in the system prompt), and short side chats
partway, after which the long chat is picked up again (the saved chat's size and how long
it takes to come back).

llama-server takes no trimming from us, so for it the script drops the oldest turns
itself once the chat no longer fits, a turn at a time, as a plain client would.

  python3 tools/long_run.py --engine vireo --model <file.gguf> --out evals/long-run/x.json
  python3 tools/long_run.py --engine llama --model <file.gguf> --out evals/long-run/y.json

Standard library only. The server's own data (chat cache, memory file) goes in a
temporary folder, never the user's.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SYSTEM = (
    "You are helping the user keep track of a logistics project. Answer briefly, in one "
    "to three sentences, from the notes in this chat."
)
SITES = ["Lyon", "Porto", "Gdansk", "Tampere", "Graz", "Leeds", "Bilbao", "Aarhus", "Turin", "Cork"]
ITEMS = ["valve kits", "sensor boards", "cable reels", "pump housings", "filter packs",
         "relay modules", "gasket sets", "mounting rails"]
PEOPLE = ["Imani", "Soren", "Marta", "Kenji", "Odile", "Tomas", "Priya", "Lukas"]
FILLER = [
    "The carrier confirmed the pickup window after two calls, and the paperwork went out the same afternoon.",
    "Customs asked for a corrected packing list, which {p} sent before the end of the day.",
    "The warehouse ran a second shift to clear the backlog left over from the previous week.",
    "A pallet was found damaged on arrival and has been set aside for the insurer to inspect.",
    "{p} walked the floor with the site lead and agreed a new layout for the loading bays.",
    "The forecast for the next fortnight was revised after the customer moved a delivery date.",
    "Two temporary staff finished their training and will start on the packing line on Monday.",
    "The label printer failed again, so the team wrote the labels by hand for most of the morning.",
    "An audit of the cold store found nothing missing, though one shelf was wrongly tagged.",
    "{p} asked for the weekly call to move an hour earlier so the other sites can join.",
    "The supplier's invoice did not match the delivery note, and the difference is being checked.",
    "Fuel costs rose again this week, and the route through the northern depot is under review.",
]


def note(i, rng):
    """Note `i`: its text, and the facts it holds (site, item, units, person)."""
    site, item, person = rng.choice(SITES), rng.choice(ITEMS), rng.choice(PEOPLE)
    units = rng.randrange(120, 980)
    late = rng.randrange(1, 9)
    lines = [
        f"Note {i}. This week {site} shipped {units} {item}, with {person} in charge of the run.",
        f"The shipment left {late} days later than planned.",
    ]
    for _ in range(40):
        lines.append(rng.choice(FILLER).format(p=rng.choice(PEOPLE)))
    return " ".join(lines), {"site": site, "item": item, "units": units, "person": person}


def plan(turns, seed):
    """The user's messages, each with what a right answer must contain (or None)."""
    rng = random.Random(seed)
    facts, out = {}, []
    for i in range(1, turns + 1):
        if i % 8 == 0:
            # A number from an early note (gone once the chat is trimmed) and from
            # a recent one, with nothing new pasted.
            # (A different early note each round: an earlier round's answer stays in the
            # chat, and would give the number away.)
            for k, kind in ((i // 8, "early"), (i - 3, "recent")):
                f = facts[k]
                out.append({
                    "turn": len(out) + 1, "kind": f"recall-{kind}",
                    "text": f"Going by note {k} earlier in this chat: how many {f['item']} did {f['site']} ship? Give the number.",
                    "expect": str(f["units"]),
                })
            continue
        text, f = note(i, rng)
        facts[i] = f
        out.append({
            "turn": len(out) + 1, "kind": "note",
            "text": f"{text}\n\nFrom note {i}: who was in charge of the run, and how many {f['item']} were shipped?",
            "expect": str(f["units"]),
        })
    return out


def post(url, body, token=None, timeout=1800):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def events(response):
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                pass


class Vireo:
    name = "vireo"
    REPLY = re.compile(
        r"Reply: (\d+) prompt tokens \((\d+) kept, (\d+) read in ([\d.]+)s, \d+/s\), "
        r"(\d+) tokens in ([\d.]+)s"
    )
    LOADED = re.compile(r"Loaded (\d+) tokens of a saved chat in (\d+) ms")

    def __init__(self, model, context, max_tokens, extra):
        self.max_tokens = max_tokens
        self.dir = Path(tempfile.mkdtemp(prefix="vireo-long-run-"))
        self.cache = self.dir / "chat-cache"
        self.log_path = self.dir / "server.log"
        self.log = open(self.log_path, "w")
        self.read_to = 0
        self.token = "long-run"
        args = [str(HERE / "target/release/vireo-server"), model, "--port", "0",
                "--token", self.token, "--chat-cache", str(self.cache)]
        if context:
            args += ["--context", str(context)]
        env = dict(os.environ, VIREO_MEMORY_FILE=str(self.dir / "memories.txt"))
        self.process = subprocess.Popen(args + extra, stdout=subprocess.PIPE, stderr=self.log,
                                        env=env, text=True)
        self.port = None
        for line in self.process.stdout:
            if line.startswith("Listening on http://127.0.0.1:"):
                self.port = int(line.strip().rsplit(":", 1)[1])
                break
        if not self.port:
            raise SystemExit(f"vireo-server didn't start; see {self.log_path}")
        metrics = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/metrics",
            headers={"Authorization": f"Bearer {self.token}"})))
        self.context = metrics.get("contextLimit")

    def new_log(self):
        self.log.flush()
        text = self.log_path.read_text(errors="replace")
        new, self.read_to = text[self.read_to:], len(text)
        return new

    def ask(self, messages):
        self.new_log()
        started = time.time()
        response = post(f"http://127.0.0.1:{self.port}/api/chat/completions", {
            "systemPrompt": SYSTEM, "messages": messages,
            "generation": {"max_tokens": self.max_tokens, "temperature": 0},
        }, self.token)
        first, reply, calls, error = None, "", [], None
        for event in events(response):
            if "content" in event:
                first = first or time.time() - started
                reply += event["content"]
            elif event.get("type") == "confirm_required":
                # The user's approval, given at once.
                post(f"http://127.0.0.1:{self.port}/api/agent/confirm",
                     {"id": event["id"], "approved": True, "remember": False}, self.token).read()
            elif event.get("type") == "tool_call":
                calls.append(event.get("tool"))
            elif "error" in event:
                error = event["error"]
        total = time.time() - started
        time.sleep(0.05)
        log = self.new_log()
        row = {"first_word_s": first, "total_s": total, "reply": reply, "calls": calls,
               "error": error}
        if m := self.REPLY.search(log):
            row.update(prompt_tokens=int(m[1]), kept=int(m[2]), read=int(m[3]),
                       read_s=float(m[4]), written=int(m[5]), write_s=float(m[6]))
        if m := self.LOADED.search(log):
            row.update(loaded_tokens=int(m[1]), load_ms=int(m[2]))
        return row

    def cache_bytes(self):
        return sum(f.stat().st_size for f in self.cache.rglob("*") if f.is_file())

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log.close()


class Llama:
    name = "llama"

    def __init__(self, model, context, max_tokens, extra):
        self.max_tokens = max_tokens
        self.context = context or 32768
        self.dir = Path(tempfile.mkdtemp(prefix="llama-long-run-"))
        self.log_path = self.dir / "server.log"
        self.log = open(self.log_path, "w")
        folder = HERE / "reference/llama-b12228"
        self.port = 18451
        env = dict(os.environ, DYLD_LIBRARY_PATH=str(folder))
        self.process = subprocess.Popen(
            [str(folder / "llama-server"), "-m", model, "-c", str(self.context), "-ngl", "99",
             "-fa", "on", "--port", str(self.port), "--parallel", "1", "--reasoning-budget", "0"]
            + extra, stdout=self.log, stderr=self.log, env=env)
        for _ in range(600):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health").read():
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"llama-server didn't start; see {self.log_path}")
        # How many leading turns (a user message and its reply) have been dropped.
        self.dropped = 0
        self.last_prompt = 0

    def ask(self, messages):
        own_chat = True
        while True:
            kept = messages[2 * self.dropped:] if own_chat else messages
            started = time.time()
            body = {"messages": [{"role": "system", "content": SYSTEM}] + kept,
                    "max_tokens": self.max_tokens, "temperature": 0, "stream": True,
                    "cache_prompt": True, "timings_per_token": False}
            first, reply, timings, error = None, "", None, None
            try:
                for event in events(post(f"http://127.0.0.1:{self.port}/v1/chat/completions", body)):
                    if "error" in event:
                        error = json.dumps(event["error"])[:300]
                    for choice in event.get("choices", []):
                        if text := (choice.get("delta") or {}).get("content"):
                            first = first or time.time() - started
                            reply += text
                    timings = event.get("timings") or timings
            except urllib.error.HTTPError as e:
                error = e.read().decode("utf-8", "replace")[:300]
            # Too long for the context: drop the oldest turn and ask again.
            if error and own_chat and "context" in error.lower() and 2 * (self.dropped + 1) < len(messages):
                self.dropped += 1
                continue
            break
        row = {"first_word_s": first, "total_s": time.time() - started, "reply": reply,
               "calls": [], "error": error, "dropped_turns": self.dropped if own_chat else 0}
        if timings:
            read = timings.get("prompt_n", 0)
            cached = timings.get("cache_n", 0)
            row.update(prompt_tokens=read + cached, kept=cached, read=read,
                       read_s=timings.get("prompt_ms", 0) / 1000,
                       written=timings.get("predicted_n", 0),
                       write_s=timings.get("predicted_ms", 0) / 1000)
            # Leave room for the next turn: a plain client drops the oldest turn once the
            # chat and a reply no longer fit.
            if own_chat and read + cached + 2 * self.max_tokens + 1200 > self.context:
                self.dropped += 1
        return row

    def cache_bytes(self):
        return 0

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log.close()


def steps_for(args, with_extras):
    """The run's steps in order; with the engine's extras, the remember turn too."""
    steps = plan(args.turns, args.seed)
    if with_extras and args.remember_at:
        steps.insert(args.remember_at - 1, {
            "kind": "remember", "expect": None,
            "text": "Please remember that my manager is Dana Whitfield and that I work from the Graz office.",
        })
    return steps


SIDE_CHAT = ["Different topic: give me three names for a bakery, one line.",
             "Pick the best one and say why in one sentence."]


def export_script(args):
    side = {int(t) for t in args.side_chats.split(",") if t}
    steps = []
    for n, step in enumerate(steps_for(args, True), 1):
        steps.append({"prompt": step["text"], "title": step["kind"],
                      **({"expect": [step["expect"]]} if step["expect"] else {})})
        if n in side:
            steps += [{"prompt": text, "title": "side chat", "chat": f"side chat after {n}"}
                      for text in SIDE_CHAT]
    out = Path(args.export_script)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"name": f"Long run, {args.turns} rounds", "systemPrompt": SYSTEM,
                               "steps": steps}, indent=1))
    print(f"{len(steps)} steps written to {out}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", choices=["vireo", "llama"])
    parser.add_argument("--model")
    parser.add_argument("--out")
    parser.add_argument("--export-script", metavar="FILE",
                        help="write the chat as a test script for the prototype app "
                             "(Test > Run Test Script, or --run-script) and stop")
    parser.add_argument("--turns", type=int, default=56, help="notes and recall rounds (default 56)")
    parser.add_argument("--context", type=int, default=0, help="tokens (default: the server's own)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--remember-at", type=int, default=12, help="vireo: the turn that asks to remember a fact (0 for none)")
    parser.add_argument("--side-chats", default="20,40", help="vireo: turns after which a short other chat is had")
    parser.add_argument("--keep", action="store_true", help="keep the server's temporary folder")
    parser.add_argument("extra", nargs="*", help="more arguments for the server, after --")
    args = parser.parse_args()
    if args.export_script:
        export_script(args)
        return
    if not (args.engine and args.model and args.out):
        parser.error("--engine, --model and --out are needed (or --export-script)")

    server = (Vireo if args.engine == "vireo" else Llama)(
        args.model, args.context, args.max_tokens, args.extra)
    side = {int(t) for t in args.side_chats.split(",") if t} if args.engine == "vireo" else set()
    rows, messages = [], []
    started = time.time()
    try:
        steps = steps_for(args, args.engine == "vireo")
        for n, step in enumerate(steps, 1):
            messages.append({"role": "user", "content": step["text"]})
            row = server.ask(messages)
            row.update(turn=n, kind=step["kind"])
            if step["expect"]:
                digits = re.sub(r"(?<=\d)[,.](?=\d{3})", "", row["reply"])
                row["right"] = step["expect"] in digits
            messages.append({"role": "assistant", "content": row["reply"]})
            rows.append(row)
            print(
                f"{n:3} {step['kind']:13} prompt {row.get('prompt_tokens', 0):6} "
                f"kept {row.get('kept', 0):6} read {row.get('read', 0):6} in {row.get('read_s', 0):6.2f}s  "
                f"first word {row['first_word_s'] or 0:6.2f}s  "
                f"{(row.get('written', 0) / row['write_s']) if row.get('write_s') else 0:5.1f} tok/s  "
                f"{'' if 'right' not in row else ('right' if row['right'] else 'WRONG')}"
                f"{' ERROR ' + row['error'] if row['error'] else ''}",
                flush=True,
            )
            if n in side:
                # Another chat, two turns, then back to the long one on the next turn.
                other = [{"role": "user", "content": SIDE_CHAT[0]}]
                a = server.ask(other)
                other += [{"role": "assistant", "content": a["reply"]},
                          {"role": "user", "content": SIDE_CHAT[1]}]
                b = server.ask(other)
                time.sleep(2)  # the save is written on a thread of its own
                rows.append({"turn": n, "kind": "side-chat", "first_word_s": a["first_word_s"],
                             "total_s": a["total_s"] + b["total_s"],
                             "cache_bytes": server.cache_bytes()})
                print(f"    side chat: {a['total_s'] + b['total_s']:.1f}s, chat cache on disk {server.cache_bytes() / 1e9:.2f} GB", flush=True)
    finally:
        cache_bytes = server.cache_bytes()
        server.stop()
        if not args.keep:
            shutil.rmtree(server.dir, ignore_errors=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "engine": args.engine, "model": Path(args.model).name, "context": server.context,
        "max_tokens": args.max_tokens, "seed": args.seed, "turns": len(steps),
        "seconds": round(time.time() - started, 1), "cache_bytes_at_end": cache_bytes,
        "rows": rows,
    }, indent=1))
    turns = [r for r in rows if r["kind"] != "side-chat"]
    asked = [r for r in turns if "right" in r]
    print(
        f"\n{len(turns)} turns in {time.time() - started:.0f}s. "
        f"Read {sum(r.get('read', 0) for r in turns)} of {sum(r.get('prompt_tokens', 0) for r in turns)} prompt tokens "
        f"({sum(r.get('read_s', 0) for r in turns):.0f}s reading, {sum(r.get('write_s', 0) for r in turns):.0f}s writing). "
        f"Right: {sum(r['right'] for r in asked)}/{len(asked)}. Saved to {out}."
    )


if __name__ == "__main__":
    main()
