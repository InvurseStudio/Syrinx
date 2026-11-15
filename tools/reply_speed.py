#!/usr/bin/env python3
"""How fast vireo-server writes four kinds of reply, one way of running it against another.

Each way is a set of server arguments (say "--mtp off" against none, to see what a draft
model adds). A server is started for each, and the ways take turns at each prompt, so
both are timed in the same sitting:

  list     a document rewritten as a list (copied text: the chat's own guesses)
  code     a code file returned with one edit (copied text)
  story    a new story (nothing to copy: only a draft model can guess)
  explain  an explanation with a code example (nothing to copy)

  python3 tools/reply_speed.py --model models/gemma-4-E4B_q4_0-it.gguf \\
      --way "chat=--mtp off" --way "draft=" --out evals/speed/draft-e4b.json

Greedy (temperature 0) and at the model's own sampling, --rounds times each. Each row
has the writing speed, the guesses made and taken by their source, and how fast the
prompt was read. Standard
library only; the server's data goes in a temporary folder (see long_run.py).
"""

import argparse
import json
import re
import shlex
import shutil
import time
from pathlib import Path

import long_run

DOCUMENT = """Notes from the harbour office, week 14.

The north quay crane was out of service from Monday to Wednesday while its slewing ring was replaced; the work was done by Halvorsen Marine and came in two days under the estimate. During that time the container ship Astrid Lund was moved to berth 4, which meant the ferry to Skarvik left from the old pier and passengers were told by text message and by a sign at the terminal. Fourteen passengers still went to the wrong pier on Monday morning and the ferry waited eleven minutes for them.

The dredging of the inner basin starts on the 22nd and will take about three weeks. Small boats moored along the east wall have to move to the visitors' pontoon before then; owners have been written to, and the harbour master will tow any boat still there on the 21st at the owner's cost, which is 450 kroner. The fuel dock stays open throughout but only from 07:00 to 15:00.

The winter storage price goes up by four percent from the first of October. Members of the sailing club keep the old price for one more season under the agreement signed in March. The new price list is on the notice board and on the website.

Finally, the office is closed on Friday for the staff outing. The duty phone is answered as usual, and the lock keeper has the spare keys to the fuel dock and the crane cabin."""

CODE = '''import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Reading:
    station: str
    day: str
    rain_mm: float
    wind_ms: float


def load_readings(path: Path) -> list[Reading]:
    """Reads a CSV of daily readings: station, day, rain_mm, wind_ms."""
    readings = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            readings.append(
                Reading(
                    station=row["station"].strip(),
                    day=row["day"],
                    rain_mm=float(row["rain_mm"] or 0),
                    wind_ms=float(row["wind_ms"] or 0),
                )
            )
    return readings


def total_rain(readings: list[Reading], station: str) -> float:
    """The rain a station measured over all its days."""
    return sum(r.rain_mm for r in readings if r.station == station)


def windiest_day(readings: list[Reading], station: str) -> str | None:
    """The day a station measured its strongest wind, if it has any readings."""
    own = [r for r in readings if r.station == station]
    if not own:
        return None
    return max(own, key=lambda r: r.wind_ms).day


def report(path: Path) -> str:
    readings = load_readings(path)
    lines = []
    for station in sorted({r.station for r in readings}):
        lines.append(
            f"{station}: {total_rain(readings, station):.1f} mm, "
            f"windiest on {windiest_day(readings, station)}"
        )
    return "\\n".join(lines)


if __name__ == "__main__":
    print(report(Path("readings.csv")))
'''

PROMPTS = {
    "list": DOCUMENT + "\n\nRewrite these notes as a bulleted list. Keep every fact and figure.",
    "code": "```python\n" + CODE + "```\n\nReturn the whole file with `total_rain` renamed to "
            "`rain_total` everywhere. Change nothing else, and say nothing besides the file.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a hash map works, with a small example in Python.",
}

GUESSES = re.compile(r"(\d+) of (\d+) guesses from the ([a-z\- ]+?) taken")


def ask(server, prompt, max_tokens, greedy):
    server.new_log()
    generation = {"max_tokens": max_tokens}
    if greedy:
        generation["temperature"] = 0
    response = long_run.post(f"http://127.0.0.1:{server.port}/api/chat/completions", {
        "messages": [{"role": "user", "content": prompt}], "generation": generation,
    }, server.token)
    reply = ""
    for event in long_run.events(response):
        if "content" in event:
            reply += event["content"]
    time.sleep(0.05)
    log = server.new_log()
    row = {"reply": reply}
    if m := server.REPLY.search(log):
        row.update(read=int(m[3]), read_s=float(m[4]), written=int(m[5]), write_s=float(m[6]))
    row["guesses"] = {m[3]: [int(m[1]), int(m[2])] for m in GUESSES.finditer(log)}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--way", action="append", required=True, metavar="NAME=ARGS",
                        help="a name and the server's arguments for it (may be none)")
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--sampling", default="greedy,own", help="greedy, own or both (default)")
    parser.add_argument("--out")
    args = parser.parse_args()
    ways = [w.split("=", 1) for w in args.way]
    servers = {name: long_run.Vireo(args.model, 0, args.max_tokens, shlex.split(extra))
               for name, extra in ways}
    rows = []
    try:
        # Each server's first reply is slower (graphs recorded, caches cold): not timed.
        for server in servers.values():
            ask(server, "Say hello.", 16, True)
        for sampling in args.sampling.split(","):
            for prompt in args.prompts.split(","):
                cells = []
                for name, server in servers.items():
                    tokens, seconds, guesses, read, read_s = 0, 0.0, {}, 0, 0.0
                    for _ in range(args.rounds):
                        row = ask(server, PROMPTS[prompt], args.max_tokens, sampling == "greedy")
                        tokens += row.get("written", 0)
                        seconds += row.get("write_s", 0.0)
                        read += row.get("read", 0)
                        read_s += row.get("read_s", 0.0)
                        for source, (taken, guessed) in row["guesses"].items():
                            have = guesses.setdefault(source, [0, 0])
                            have[0] += taken
                            have[1] += guessed
                    rate = tokens / seconds if seconds else 0.0
                    rows.append({"sampling": sampling, "prompt": prompt, "way": name,
                                 "tokens": tokens, "seconds": round(seconds, 3),
                                 "tokens_per_s": round(rate, 1), "guesses": guesses,
                                 "read_tokens": read, "read_seconds": round(read_s, 3),
                                 "read_tokens_per_s": round(read / read_s) if read_s else 0})
                    taken = sum(g[0] for g in guesses.values())
                    cells.append(f"{name} {rate:6.1f} tokens/s ({100 * taken // max(tokens, 1)}% guessed)")
                print(f"{sampling:6} {prompt:8} " + "   ".join(cells), flush=True)
    finally:
        for server in servers.values():
            server.stop()
            shutil.rmtree(server.dir, ignore_errors=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"model": Path(args.model).name, "ways": dict(ways),
                                              "max_tokens": args.max_tokens, "rounds": args.rounds,
                                              "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
