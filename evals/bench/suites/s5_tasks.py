"""S5 Agent tasks: 12 tasks on the harbor-bikes sandbox, each checked without a person.
Same tools for every engine (Vireo's catalog), run by the harness; Vireo asks for them as
an app's (`clientTools`, calculate built in), the others call them as OpenAI tools.

Each row: right (true/false), why (when wrong), wall time, tools called, tokens written.
Answers were worked out from the sandbox: revenue 4,455.00 / 6,923.50 / 7,497.75 /
7,866.00, total 26,742.25 (Q4 best); workshop hours 78; Kestrel's terms 14 days.
"""
import time

import common
import engines
import prompts
from desk import Desk

SKILL = "weekly-report"
NOT_FOUND = ["not found", "doesn't exist", "does not exist", "no such", "couldn't find", "could not find",
             "can't find", "cannot find", "isn't there", "is not there", "there is no", "no file", "not exist",
             "unable to find", "wasn't found", "was not found", "missing", "unable to read", "couldn't read",
             "could not read", "check the path", "double-check", "no such file"]
LARGE = ("t-longlog",)  # needs more than the chat context: runs on deep()'s server
REFUSED = ["declin", "not approved", "wasn't", "was not", "didn't", "did not", "couldn't", "could not",
           "not saved", "not written", "not created", "cancel", "denied", "unable", "wasn't able"]


def skill_text():
    return (common.FIX / "skills" / SKILL / "SKILL.md").read_text().split("---", 2)[2].strip()


def incident_log(ctx):
    text = common.fixture_text("incident-log.txt")
    share = ctx.model.get("log_share")
    return text[:text.rindex("\n", 0, int(len(text) * share))] if share else text


def tasks(ctx, kind):
    """name -> (desk options, prompt builder(desk), max tokens, checker(r, desk) -> (ok, why))."""
    vireo = kind.startswith("vireo")

    def skill_prompt(d):
        q = f"Use the {SKILL} skill for this: the notes are in {d.posix}/notes."
        return q if vireo else q + "\n\nThe skill's instructions:\n\n" + skill_text()

    def written_has(d, s):
        return any(s in v for v in d.written.values())

    return {
        "t-single": ({}, lambda d: f"List the files in the folder {d.posix}/sales.", 256,
                     lambda r, d: (("list_directory" in r["calls"] and prompts.has_any(r["text"], ["2026-q1"])
                                    and prompts.has_any(r["text"], ["2026-q4"])), "listed the folder, named the files")),
        "t-parallel": ({}, lambda d: (f"Read {d.posix}/README.md, {d.posix}/suppliers.md and {d.posix}/notes/2026-11-10.md, "
                                      "then tell me in one line each what each file is about."), 512,
                       lambda r, d: ({"README.md", "suppliers.md", "2026-11-10.md"} <= set(d.read), "read all three files")),
        "t-multistep": ({}, lambda d: (f"Our back-office folder is {d.posix}. Using the quarterly sales files in its sales "
                                       "folder, work out each quarter's revenue for 2026 (units times unit price), say which "
                                       "quarter was best, and give the year's total."), 1024,
                        lambda r, d: (prompts.has_any(r["text"], ["26742.25", "26742.2"]), "total 26,742.25")),
        "t-skill": ({}, skill_prompt, 1536,
                    lambda r, d: (prompts.has_any(r["text"], ["78"]) and written_has(d, "78"),
                                  "report written with 78 hours, total in the reply")),
        "t-longlog": ({}, lambda d: ("Here is yesterday's point-of-sale incident log:\n\n<log>\n" + incident_log(ctx) +
                                     f"\n</log>\n\nWhich component logged the most ERROR lines? Then check our folder {d.posix} "
                                     "(the README and suppliers.md) and tell me whether any supplier is involved with that "
                                     "component. Keep the answer short."), 512,
                      lambda r, d: (prompts.has_any(r["text"], ["inventory-sync"]), "named inventory-sync")),
        "t-notfound": ({}, lambda d: f"What was the total revenue in {d.posix}/sales/2025-q4.csv?", 512,
                       lambda r, d: (prompts.has_any(r["text"], NOT_FOUND), "said the file isn't there")),
        "t-notool": ({}, lambda d: "In one sentence: what does a bicycle derailleur do?", 128,
                     lambda r, d: (not r["calls"] and prompts.has_any(r["text"], ["gear", "sprocket", "chainring", "cog", "shift"]), "answered with no tool call")),
        "t-json": ({}, lambda d: (f"Using the sales files in {d.posix}/sales, reply with only a JSON object with keys "
                                  "\"best_quarter\" (like \"Q1\") and \"total\" (the year's revenue, a number). No other text."), 1024,
                   lambda r, d: (lambda v: (isinstance(v, dict) and "4" in str(v.get("best_quarter", ""))
                                            and prompts.same(v.get("total"), 26742.25), "JSON with Q4 and 26742.25"))(
                       prompts.parse_json(r["text"])[1])),
        "t-approval": ({"deny_writes": True}, lambda d: f"Save a file {d.posix}/todo.md with the line 'Order more inner tubes'.", 256,
                       lambda r, d: (not d.written and prompts.has_any(r["text"], REFUSED),
                                     "wrote nothing and said the write was refused")),
        "t-recover": ({"fail_first_read": True},
                      lambda d: f"Read {d.posix}/suppliers.md and tell me the payment terms for Kestrel Components.", 512,
                      lambda r, d: (prompts.has_any(r["text"], ["14 days"]), "14 days, after a failed first read")),
        "t-calc": ({}, lambda d: ("Add up these invoices and add 13% tax: $1,284.37, $2,019.88, $377.05, $4,410.10, $96.42. "
                                  "Give the total."), 512,
                   lambda r, d: (prompts.has_any(r["text"], ["9252.24"]) and "calculate" in r["calls"],
                                 "9,252.24 through calculate")),
    }


def run_one(ctx, srv, kind, name, spec, thinking):
    opts, build, max_tokens, check = spec
    d = Desk(**opts)
    try:
        t = time.time()
        # Thinking needs room of its own, or it uses up the reply's budget.
        budget = max_tokens * 4 if thinking else max_tokens
        r = srv.chat([{"role": "user", "content": common.nonce() + build(d)}], budget, tools=True, desk=d)
        ok, why = check(r, d)
        ctx.rec("S5", name, kind, thinking=thinking, right=bool(ok and not r["error"]), checks=why,
                wall=round(time.time() - t, 2), calls=r["calls"], gen_tokens=r["gen_tokens"], gen_tps=r["gen_tps"],
                steps=len(r["steps"]), text=r["text"][:800], error=r["error"])
    finally:
        d.close()


def run(ctx, srv, kind, thinking=False):
    only = ctx.opts("S5").get("tasks")
    for name, spec in tasks(ctx, kind).items():
        if (only and name not in only) or name in LARGE:
            continue
        run_one(ctx, srv, kind, name, spec, thinking)


def deep(ctx, kind, thinking=False):
    """The tasks that need a large context, on one server at min(128K, the model's largest):
    t-longlog (a ~55K-token log pasted in) and t-deep-tool (~70% of that context filled
    with the corpus, then one tool call)."""
    only = ctx.opts("S5").get("tasks")
    if only and not ({"t-deep-tool"} | set(LARGE)) & set(only):
        return
    size = min(131072, ctx.max_context)
    name = "deep" if size >= 131072 else f"deep{size}"
    text, n = prompts.body(ctx.key, name)
    srv = engines.start(kind, ctx.model, size, thinking=thinking)
    try:
        srv.chat([{"role": "user", "content": "Say hello."}], 8, log=False)
        for name in LARGE:
            if not only or name in only:
                run_one(ctx, srv, kind, name, tasks(ctx, kind)[name], thinking)
        if only and "t-deep-tool" not in only:
            return
        spec = ({}, lambda d: ("<document>\n" + text + "\n</document>\n\nThat was a repository dump; ignore it for this "
                               f"question. List the folder {d.posix}/sales and tell me how many files it holds."), 256,
                lambda r, d: ("list_directory" in r["calls"] and prompts.has_any(r["text"], ["4", "four"]),
                              f"listed the folder after ~{n // 1000}K tokens, said 4"))
        run_one(ctx, srv, kind, "t-deep-tool", spec, thinking)
    finally:
        srv.stop()
