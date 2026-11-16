"""S4 Conversation, on the chat server (the platform's chat context, 32K).

  chat-16     16 turns, each pasting a ~600-token shift report and asking about it: per
              turn, time to first token, tokens read vs kept from the cache, right answer
  regenerate  the 16th turn sent again: should read almost nothing
  edit-turn   the first 12 turns with turn 6's report changed: how much is read again
  chat-64     64 turns in the 32K context, so it overflows; every 8th turn asks for an
              earlier report's till total (a recent one, then the second report). Vireo trims
              the chat itself; llama.cpp and mlx-vlm get a plain client that drops the oldest
              turn pair until the chat fits, as tools/long_run.py does.
"""
import time

import common
import prompts


def turn_text(note):
    return note["text"] + "\n\n" + note["question"]


def run(ctx, srv, kind):
    notes = common.fixture_json("chat_notes.json")["notes"]
    tag = common.nonce()
    msgs, rows = [], []
    for i, note in enumerate(notes[:16]):
        msgs.append({"role": "user", "content": (tag if i == 0 else "") + turn_text(note)})
        r = srv.chat(list(msgs), 64)
        msgs.append({"role": "assistant", "content": r["text"]})
        rows.append({"turn": i + 1, "ttft": r["ttft"], "prompt": r["prompt_tokens"], "cached": r["cached_tokens"],
                     "read": r["read_tokens"], "right": prompts.has_any(r["text"], [note["expect"]]),
                     "error": r["error"]})
    ok = [x for x in rows if not x["error"]]
    ctx.rec("S4", "chat-16", kind, turns=rows, right=sum(x["right"] for x in rows),
            ttft_mean=round(sum(x["ttft"] or 0 for x in ok) / max(1, len(ok)), 3),
            ttft_last=rows[-1]["ttft"], read_total=sum(x["read"] or 0 for x in rows),
            prompt_total=sum(x["prompt"] or 0 for x in rows))

    # The same last turn again, as a "regenerate" button would send it.
    r = srv.chat(msgs[:-1], 64)
    ctx.rec("S4", "regenerate", kind, prompt_tokens=r["prompt_tokens"], read_tokens=r["read_tokens"],
            ttft=r["ttft"], error=r["error"])

    # Turn 6 edited, the chat resent up to turn 12's question.
    edited = [dict(x) for x in msgs[:23]]
    n6 = notes[5]
    edited[10]["content"] = edited[10]["content"].replace(f"did {n6['count']} ", f"did {n6['count'] + 5} ", 1)
    r = srv.chat(edited, 64)
    ctx.rec("S4", "edit-turn", kind, prompt_tokens=r["prompt_tokens"], read_tokens=r["read_tokens"],
            ttft=r["ttft"], error=r["error"])

    chat64(ctx, srv, kind, notes)


def chat64(ctx, srv, kind, notes):
    budget = int(ctx.chat_context * 0.88)
    tag = common.nonce()
    pairs, rows = [], []          # pairs: [user message, assistant message, estimated tokens]
    last_prompt, dropped = 0, 0
    t0 = time.time()
    for t in range(1, 65):
        if t % 8 == 0:
            k = 2 if (t // 8) % 2 == 0 else t - 6
            note = notes[k - 1]
            text = f"Going back to shift report {k}: what did the till close at? Answer with the amount only."
            expect = str(note["till"])
        else:
            note = notes[t - 1]
            text, expect = turn_text(note), note["expect"]
        user = {"role": "user", "content": (tag if t == 1 else "") + text}
        est_new = len(text) // 3 + 100
        if kind not in ("vireo", "vireo-plain"):
            while pairs and last_prompt + est_new > budget:
                last_prompt -= pairs.pop(0)[2]
                dropped += 1
        msgs = [m for p in pairs for m in p[:2]] + [user]
        r = srv.chat(msgs, 64)
        right = prompts.has_any(r["text"], [expect])
        rows.append({"turn": t, "ttft": r["ttft"], "prompt": r["prompt_tokens"], "read": r["read_tokens"],
                     "read_s": round(sum(s.get("read_s") or 0 for s in r["steps"]), 2),
                     "probe": t % 8 == 0, "right": right, "error": r["error"]})
        if r["prompt_tokens"]:
            pair_tokens = max(1, r["prompt_tokens"] - last_prompt) + (r["gen_tokens"] or 0)
            last_prompt = r["prompt_tokens"] + (r["gen_tokens"] or 0)
        else:
            pair_tokens = est_new
            last_prompt += est_new
        pairs.append([user, {"role": "assistant", "content": r["text"]}, pair_tokens])
    probes = [x for x in rows if x["probe"]]
    ctx.rec("S4", "chat-64", kind, seconds=round(time.time() - t0, 1), turns=rows,
            read_total=sum(x["read"] or 0 for x in rows), read_s_total=round(sum(x["read_s"] for x in rows), 1),
            probes_right=sum(x["right"] for x in probes), probes=len(probes),
            notes_right=sum(x["right"] for x in rows if not x["probe"]), dropped_pairs=dropped,
            errors=sum(bool(x["error"]) for x in rows))
