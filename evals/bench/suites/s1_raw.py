"""S1 Raw speed: llama-bench's shapes, so the numbers sit beside anyone's posted results.

  pp512, pp4096           reading a 512- / 4,096-token prompt (tokens/s)
  tg128                   writing 128 tokens after a short prompt
  tg128@d8K, tg128@d32K   writing 128 tokens with 8K / 32K already in the context

Vireo: `vireo-cli generate --gpu` (greedy, one token at a time: no drafting). llama.cpp:
`llama-bench -fa 1 -ngl 99`, one repetition per bench rep. A Vireo or mlx-vlm row that
wrote fewer than 128 tokens (the model ended its turn) is marked `short` and its speed is
less reliable. mlx-vlm has no bench tool:
its server, a single user message (its chat template adds ~20 tokens), speeds from the
response's timings.
"""
import common
import engines
import prompts

# A chat model answers a bare story opening briefly; through a server it's asked to go on.
CHAT_ASK = "\n\nContinue this story at length: at least 600 words, no title, no preamble."
DEPTHS = [("tg128@d8K", "d8192", 8192), ("tg128@d32K", "d32768", 32768)]


def depth_prompt(key, name):
    """The corpus to that depth, then the short story opening, so the model keeps writing
    instead of ending its turn in the middle of raw code (~35 tokens past the depth)."""
    text, _ = prompts.body(key, name)
    # A break and a heading first: straight after the code the model takes the story for
    # more code and closes the block (8 tokens on Gemma 4 E2B at 32K).
    return text + "\n\n=====\n\nChapter 1\n\n" + common.fixture_text("short.txt")


def run(ctx, kind):
    m, key = ctx.model, ctx.key
    cap = ctx.max_context
    depths = [d for d in DEPTHS if d[2] + 256 <= cap]
    if kind == "vireo":
        for test, name in [("pp512", "pp512"), ("pp4096", "pp4096")]:
            text, n = prompts.body(key, name)
            r = engines.cli_generate(m, text, 8)
            ctx.rec("S1", test, kind, tps=r["pp_tps"], tokens=r["prompt_tokens"])
        r = engines.cli_generate(m, common.fixture_text("short.txt"), 128)
        ctx.rec("S1", "tg128", kind, tps=r["tg_tps"], tokens=r["gen_tokens"], short=r["gen_tokens"] < 128 or None)
        for test, name, _ in depths:
            r = engines.cli_generate(m, depth_prompt(key, name), 128)
            ctx.rec("S1", test, kind, tps=r["tg_tps"], tokens=r["gen_tokens"], depth=r["prompt_tokens"],
                    pp_tps=r["pp_tps"], short=r["gen_tokens"] < 128 or None)
    elif kind == "llama":
        rows = engines.llama_bench(m, ["-p", "512,4096", "-n", "128", "-r", "1"])
        if depths:
            rows += engines.llama_bench(m, ["-p", "0", "-n", "128", "-r", "1", "-d", ",".join(str(d[2]) for d in depths)])
        for r in rows:
            p, g, d = r.get("n_prompt", 0), r.get("n_gen", 0), r.get("n_depth", 0)
            if p and not g:
                test = f"pp{p}"
            elif g and not p:
                test = "tg128" if not d else f"tg128@d{d // 1024}K"
            else:
                continue
            ctx.rec("S1", test, kind, tps=round(r["avg_ts"], 2), depth=d or None)
    elif kind == "mlx":
        srv = engines.start("mlx", m, ctx.chat_context)
        try:
            srv.chat([{"role": "user", "content": "Say hello."}], 8, log=False)
            for test, name in [("pp512", "pp512"), ("pp4096", "pp4096")]:
                text, n = prompts.body(key, name)
                r = srv.chat([{"role": "user", "content": common.nonce() + text}], 1)
                ctx.rec("S1", test, kind, tps=r["read_tps"], tokens=r["read_tokens"], error=r["error"])
            r = srv.chat([{"role": "user", "content": common.nonce() + common.fixture_text("short.txt") + CHAT_ASK}], 128)
            ctx.rec("S1", "tg128", kind, tps=r["gen_tps"], tokens=r["gen_tokens"], short=(r["gen_tokens"] or 0) < 128 or None,
                    error=r["error"])
            for test, name, _ in depths:
                r = srv.chat([{"role": "user", "content": common.nonce() + depth_prompt(key, name) + CHAT_ASK}], 128)
                ctx.rec("S1", test, kind, tps=r["gen_tps"], tokens=r["gen_tokens"], depth=r["prompt_tokens"],
                        pp_tps=r["read_tps"], short=(r["gen_tokens"] or 0) < 128 or None, error=r["error"])
        finally:
            srv.stop()
