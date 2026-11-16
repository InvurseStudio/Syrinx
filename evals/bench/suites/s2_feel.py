"""S2 Feel: what a person waits for. Through each engine's server, streaming, client-side.

  load-cold   start to first answer after the disk cache is dropped (`sudo purge`, Mac only)
  load-warm   the same straight after (the chat server's own start)
  ttft-short  time to the first token, a 40-token question
  ttft-3k     the same with a ~2.5K-token document pasted in
  reply-400   four whole replies (list rewrite, code edit, story, explanation), 400 tokens at most
"""
import subprocess

import common
import engines

SHORT = "Why does a dry bike chain wear out faster than a lubricated one? Two sentences."


def ready(ctx, srv, kind, test):
    """Records the server's start plus its first tiny reply: when a user could first talk."""
    r = srv.chat([{"role": "user", "content": common.nonce() + "Say hello in five words."}], 16, log=False)
    ctx.rec("S2", test, kind, load_s=round(srv.load_s, 2), first_reply_s=round(r["total"], 2),
            ready_s=round(srv.load_s + r["total"], 2), error=r["error"])


def cold(ctx, kind):
    if common.WIN or not ctx.sudo:
        ctx.rec("S2", "load-cold", kind, skipped="needs sudo (run `sudo -v` first) on the Mac; not on Windows")
        return
    subprocess.run(["sudo", "-n", "purge"], check=False)
    srv = engines.start(kind, ctx.model, ctx.chat_context)
    try:
        ready(ctx, srv, kind, "load-cold")
    finally:
        srv.stop()


def run(ctx, srv, kind):
    for i in range(3):
        r = srv.chat([{"role": "user", "content": common.nonce() + SHORT}], 128)
        ctx.rec("S2", "ttft-short", kind, i=i, ttft=r["ttft"], total=r["total"], gen_tps=r["gen_tps"], error=r["error"])
    doc = common.fixture_text("medium.txt")
    for i in range(2):
        r = srv.chat([{"role": "user", "content": common.nonce() + doc}], 128)
        ctx.rec("S2", "ttft-3k", kind, i=i, ttft=r["ttft"], total=r["total"], prompt_tokens=r["prompt_tokens"],
                read_tps=r["read_tps"], error=r["error"])
    for name, text in common.fixture_json("replies.json")["prompts"].items():
        r = srv.chat([{"role": "user", "content": common.nonce() + text}], 400)
        ctx.rec("S2", "reply-400", kind, prompt=name, ttft=r["ttft"], total=r["total"], gen_tokens=r["gen_tokens"],
                gen_tps=r["gen_tps"], error=r["error"])
