"""S3 Context ladder, with S7's memory readings.

For each rung (4K ... 256K) the server is started with that context, so its memory is what
that context costs. The prompt is the frozen corpus filling ~92% of it (less room for
Vireo's system turn), five facts planted at 5/25/50/75/95% depth, all asked for, then a
~200-word answer, so there's writing to time at that depth.

S3 row: read tok/s, time to first token, write tok/s, recall (0-5).
S7 row: memory idle after load, peak during the request, 5 s after it ("settle").
Windows reads the whole GPU's memory between steps, never during one.
"""
import time

import common
import engines
import prompts


def run(ctx, kind):
    rungs = ctx.opts("S3").get("rungs") or list(ctx.ladder)
    for rung in rungs:
        size = ctx.ladder[rung]
        if size > ctx.max_context:
            continue
        if kind == "mlx" and size > 131072 and not ctx.opts("S3").get("mlx_256k", True):
            continue
        text, body_tokens = prompts.ladder(ctx.key, rung)
        try:
            srv = engines.start(kind, ctx.model, size)
        except Exception as e:
            ctx.rec("S3", rung, kind, error=f"start: {e!r}"[:300])
            continue
        mem = common.watch(srv.pid)
        try:
            srv.chat([{"role": "user", "content": common.nonce() + "Say hello."}], 8, log=False)
            time.sleep(2)
            idle = mem.reset()
            r = srv.chat([{"role": "user", "content": common.nonce() + text}], 400)
            mem.sample()  # right away: a short request can fall between the 2-second samples
            peak = mem.peak
            time.sleep(5)
            settle = mem.sample()
            ctx.rec("S3", rung, kind, context=size, prompt_tokens=r["prompt_tokens"], read_tokens=r["read_tokens"],
                    read_tps=r["read_tps"], ttft=r["ttft"], gen_tokens=r["gen_tokens"], gen_tps=r["gen_tps"],
                    recall=prompts.recall(r["text"]), text=r["text"][:1500], error=r["error"])
            ctx.rec("S7", rung, kind, context=size, prompt_tokens=r["prompt_tokens"], idle_gb=common.gb(idle),
                    peak_gb=common.gb(peak), settle_gb=common.gb(settle))
        finally:
            mem.stop()
            srv.stop()
