"""S9 Several at once: 2 and 4 requests sent at the same moment (a whole 256-token reply
each, different prompts). Combined writing speed (all tokens / wall time), the slowest
user's time to first token, errors. llama-server gets that many slots (-np), each with the
chat context; Vireo serves one reply at a time today, so this is its baseline.
"""
import threading
import time

import common
import engines


def run(ctx, kind):
    counts = ctx.opts("S9").get("together", [2, 4])
    srv = engines.start(kind, ctx.model, ctx.chat_context, slots=max(counts) if kind == "llama" else 1)
    texts = list(common.fixture_json("replies.json")["prompts"].values())
    try:
        srv.chat([{"role": "user", "content": "Say hello."}], 8, log=False)
        for n in counts:
            out = [None] * n

            def go(i):
                out[i] = srv.chat([{"role": "user", "content": common.nonce() + texts[i % len(texts)]}], 256, log=False)
            t0 = time.time()
            ths = [threading.Thread(target=go, args=(i,)) for i in range(n)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.time() - t0
            # Tokens written: the server's own count where it gives one, else ~4 characters a token.
            gen = sum((r["gen_tokens"] or 0) or len(r["text"]) // 4 for r in out)
            ttfts = [r["ttft"] for r in out if r["ttft"] is not None]
            ctx.rec("S9", f"together-{n}", kind, wall=round(wall, 2), gen_tokens=gen,
                    combined_tps=round(gen / wall, 2) if wall else None,
                    slowest_ttft=round(max(ttfts), 3) if ttfts else None,
                    fastest_ttft=round(min(ttfts), 3) if ttfts else None,
                    errors=sum(bool(r["error"]) for r in out))
    finally:
        srv.stop()
