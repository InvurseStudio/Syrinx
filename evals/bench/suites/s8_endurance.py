"""S8 Endurance, on the chat server.

  soak-<N>  N minutes (30 in the full tier) of back-to-back mixed requests: a short
            question, a ~2.5K document, a whole reply, the multi-step tool task. Writing
            speed and memory in the first and last 5 minutes, errors.
  heat-<N>  N minutes (10) of continuous writing, 1,024 tokens a request: writing speed per
            minute. Only on the platform's heat models (the largest), where the Mac throttles.
"""
import time

import common
from desk import Desk

STORY = "Write a long, detailed story about a bicycle courier crossing a city in a storm."


def soak(ctx, srv, kind, minutes):
    doc = common.fixture_text("medium.txt")
    replies = list(common.fixture_json("replies.json")["prompts"].values())
    mixes = [
        lambda: srv.chat([{"role": "user", "content": common.nonce() + "Name three bike tools, one line."}], 64),
        lambda: srv.chat([{"role": "user", "content": common.nonce() + doc}], 128),
        lambda: srv.chat([{"role": "user", "content": common.nonce() + replies[int(time.time()) % len(replies)]}], 400),
        None,  # the tool task, below
    ]
    mem = common.watch(srv.pid)
    rows, t0, i = [], time.time(), 0
    try:
        mem_start = mem.reset()
        while time.time() - t0 < minutes * 60:
            f = mixes[i % len(mixes)]
            if f is None:
                d = Desk()
                r = srv.chat([{"role": "user", "content": common.nonce() + (
                    f"Using the quarterly sales files in {d.posix}/sales, give the year's total revenue "
                    "(units times unit price).")}], 1024, tools=True, desk=d)
                d.close()
            else:
                r = f()
            if common.WIN:
                mem.sample()
            rows.append({"t": round(time.time() - t0, 1), "kind": i % len(mixes), "gen_tps": r["gen_tps"],
                         "ttft": r["ttft"], "error": r["error"]})
            i += 1
        mem_end = mem.sample()
    finally:
        mem.stop()

    def avg(xs):
        xs = [x["gen_tps"] for x in xs if x["gen_tps"]]
        return round(sum(xs) / len(xs), 2) if xs else None
    first = [x for x in rows if x["t"] < 300]
    last = [x for x in rows if x["t"] > minutes * 60 - 300]
    ctx.rec("S8", f"soak-{minutes}", kind, requests=len(rows), errors=sum(bool(x["error"]) for x in rows),
            gen_tps_first5=avg(first), gen_tps_last5=avg(last), mem_start_gb=common.gb(mem_start),
            mem_end_gb=common.gb(mem_end), mem_peak_gb=common.gb(mem.peak), rows=rows)


def heat(ctx, srv, kind, minutes):
    rows, t0 = [], time.time()
    while time.time() - t0 < minutes * 60:
        r = srv.chat([{"role": "user", "content": common.nonce() + STORY}], 1024)
        rows.append({"t": round(time.time() - t0, 1), "gen_tps": r["gen_tps"], "gen": r["gen_tokens"], "error": r["error"]})
    per_min = {}
    for x in rows:
        if x["gen_tps"]:
            per_min.setdefault(int(x["t"] // 60), []).append(x["gen_tps"])
    curve = [round(sum(v) / len(v), 2) for _, v in sorted(per_min.items())]
    ctx.rec("S8", f"heat-{minutes}", kind, curve=curve, first=curve[0] if curve else None,
            last=curve[-1] if curve else None,
            held_pct=round(100 * curve[-1] / curve[0], 1) if len(curve) > 1 and curve[0] else None, rows=rows)


def run(ctx, srv, kind):
    o = ctx.opts("S8")
    soak(ctx, srv, kind, o.get("soak_minutes", 30))
    if ctx.key in o.get("heat_models", ctx.pcfg.get("heat_models", [])):
        heat(ctx, srv, kind, o.get("heat_minutes", 10))
