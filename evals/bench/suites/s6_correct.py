"""S6 Correctness of the engine itself: kernel and quantisation bugs, not model mistakes.

  ppl           perplexity on fixtures/ppl.txt (512-token chunks, 40 of them), Vireo
                (`vireo-cli perplexity --gpu`) and llama.cpp (`llama-perplexity`), scored the
                same way. Should match within ~0.5%.
  greedy-match  20 raw prompts continued greedily for 256 tokens by Vireo (vireo-cli) and
                llama.cpp (llama-server /completion): words that agree before the first
                difference. Run from Vireo's side, llama.cpp's text made alongside, so the
                quick tier has it too.
  qa-50         50 short exact-answer questions through the chat server
  grammar       20 requests for strict JSON through the chat server

mlx-vlm has no perplexity tool or raw continuation here; it takes part in qa-50 and grammar.
"""
import statistics

import common
import engines
import prompts


def run_cli(ctx, kind):
    tests = ctx.opts("S6").get("tests")
    if kind in ("vireo", "llama") and (not tests or "ppl" in tests):
        try:
            r = engines.perplexity(kind, ctx.model, common.FIX / "ppl.txt")
            ctx.rec("S6", "ppl", kind, **r)
        except Exception as e:
            ctx.rec("S6", "ppl", kind, error=repr(e)[:300])
    if kind == "vireo" and (not tests or "greedy-match" in tests):
        greedy(ctx)


def greedy(ctx):
    g = common.fixture_json("greedy_prompts.json")
    ours = []
    for p in g["prompts"]:
        try:
            ours.append(engines.cli_generate(ctx.model, p, g["tokens"])["text"])
        except Exception as e:
            ours.append(None)
            ctx.rec("S6", "greedy-match", "vireo", error=repr(e)[:300])
    srv = engines.start("llama", ctx.model, 4096)
    try:
        ref = [srv.complete(p, g["tokens"]) for p in g["prompts"]]
    finally:
        srv.stop()
    agree = [prompts.words_agree(a, b) for a, b in zip(ours, ref) if a is not None]
    full = sum(1 for a, b in zip(ours, ref) if a is not None and a.split() == b.split())
    ctx.rec("S6", "greedy-match", "vireo", against="llama", median_words=statistics.median(agree) if agree else None,
            min_words=min(agree) if agree else None, identical=full, prompts=len(g["prompts"]), agree=agree,
            texts=[{"vireo": a, "llama": b} for a, b in zip(ours, ref)])


def run_chat(ctx, srv, kind):
    tests = ctx.opts("S6").get("tests")
    if not tests or "qa-50" in tests:
        qa = common.fixture_json("qa50.json")
        right, wrong = 0, []
        for i, q in enumerate(qa["questions"]):
            r = srv.chat([{"role": "user", "content": common.nonce() + q["q"]}], 48, system=qa["system"])
            if prompts.has_any(r["text"], q["accept"]):
                right += 1
            else:
                wrong.append({"i": i, "reply": r["text"][:120], "error": r["error"]})
        ctx.rec("S6", "qa-50", kind, right=right, of=len(qa["questions"]), wrong=wrong)
    if not tests or "grammar" in tests:
        jt = common.fixture_json("json_tasks.json")
        strict = parses = right = 0
        bad = []
        for i, t in enumerate(jt["tasks"]):
            r = srv.chat([{"role": "user", "content": common.nonce() + t["q"] + jt["suffix"]}], 256)
            s, v = prompts.parse_json(r["text"])
            strict += s
            parses += v is not None
            ok = v is not None and prompts.same(v, t["expect"])
            right += ok
            if not ok:
                bad.append({"i": i, "reply": r["text"][:160], "error": r["error"]})
        ctx.rec("S6", "grammar", kind, strict=strict, parses=parses, right=right, of=len(jt["tasks"]), bad=bad)
