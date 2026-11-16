"""Sampled-behaviour and cache tests for PR #176. Usage:
python3 -I sampled.py <server-bin> <model.gguf> <suite[,suite]> <out.json> [builtin-skills-dir]
Suites: off, tools, think, edge, off4 (Qwen 9B short chat).
No temperature is sent except in 'edge' (temperature 0)."""
import json, os, re, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

BENCH = Path("/Users/me/Documents/Invurse Studio/Vireo - MacOS/VireoPrototype/evals/speed/bench-2026-11-13")
sys.path.insert(0, str(BENCH))
import bench  # noqa: E402
from bench import Desk, post, sse, SKILLS, VIREO_CLIENT_TOOLS  # noqa: E402

TOKEN = "bench"
REPLY = re.compile(r"Reply: (\d+) prompt tokens \((\d+) kept, (\d+) read in ([\d.]+)s, \d+/s\), "
                   r"(\d+) tokens in ([\d.]+)s \([\d.]+/s\), (\d+) tool calls(?:, (\d+) of them reasoning)?")


class Server:
    def __init__(self, exe, gguf, builtin, logdir, ctx=32768):
        self.dir = Path(tempfile.mkdtemp(prefix="pr176-", dir=logdir))
        self.log_path = self.dir / "server.log"
        self.log = open(self.log_path, "w")
        args = [exe, gguf, "--port", "0", "--token", TOKEN, "--chat-cache", str(self.dir / "chat-cache"),
                "--context", str(ctx)]
        env = dict(os.environ, INVURSE_SKILLS_DIR=str(SKILLS), VIREO_MEMORY_FILE=str(self.dir / "mem.txt"),
                   VIREO_APPROVALS_FILE=str(self.dir / "appr.txt"))
        if builtin:
            env["VIREO_BUILTIN_SKILLS"] = builtin
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=self.log, env=env, text=True)
        self.port = None
        for line in self.proc.stdout:
            if line.startswith("Listening on http://127.0.0.1:"):
                self.port = int(line.strip().rsplit(":", 1)[1])
                break
        if not self.port:
            raise RuntimeError(f"no server; see {self.log_path}")
        threading.Thread(target=lambda: [None for _ in self.proc.stdout], daemon=True).start()
        self.read_to = 0

    def new_log(self):
        self.log.flush()
        text = self.log_path.read_text(errors="replace")
        new, self.read_to = text[self.read_to:], len(text)
        return new

    def chat(self, messages, max_tokens=1024, thinking=False, tools=False, temperature=None, desk=None,
             system="You are a helpful assistant.", gen_extra=None):
        self.new_log()
        gen = {"max_tokens": max_tokens, **(gen_extra or {})}
        if temperature is not None:
            gen["temperature"] = temperature
        body = {"systemPrompt": system, "messages": messages, "thinking": thinking, "generation": gen}
        if tools:
            body["clientTools"] = VIREO_CLIENT_TOOLS
        t0 = time.time()
        first, text, think, calls, err, record = None, "", "", [], None, None
        for ev in sse(post(f"http://127.0.0.1:{self.port}/api/chat/completions", body, TOKEN)):
            typ = ev.get("type")
            if "content" in ev and typ is None:
                if ev["content"]:
                    first = first or time.time() - t0
                text += ev["content"]
            elif typ == "thinking":
                first = first or time.time() - t0
                think += ev.get("content", "") or ""
            elif typ == "tool_call":
                first = first or time.time() - t0
                calls.append({"tool": ev.get("tool"), "args": ev.get("args")})
            elif typ == "turn_record":
                record = ev.get("record")
            elif typ == "client_tool_request":
                ok, out = desk.run(ev.get("tool", ""), ev.get("args") or {})
                ans = {"id": ev["id"], "ok": ok, ("output" if ok else "error"): out}
                post(f"http://127.0.0.1:{self.port}/api/agent/client-tool-result", ans, TOKEN).read()
            elif typ == "confirm_required":
                post(f"http://127.0.0.1:{self.port}/api/agent/confirm",
                     {"id": ev["id"], "approved": True, "remember": False}, TOKEN).read()
            elif "error" in ev and typ is None:
                err = str(ev["error"])[:300]
        total = time.time() - t0
        time.sleep(0.2)
        steps = [{"prompt": int(m[1]), "kept": int(m[2]), "read": int(m[3]), "read_s": float(m[4]),
                  "gen": int(m[5]), "gen_s": float(m[6]), "calls": int(m[7]), "reasoning": int(m[8] or 0)}
                 for m in REPLY.finditer(self.new_log())]
        return {"ttft": first, "total": total, "text": text, "thinking": think, "calls": calls,
                "error": err, "steps": steps, "record": record}

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        time.sleep(3)


OFF_CHATS = {
    "merkle": [
        "Explain in a short paragraph what a Merkle tree is.",
        "Give me a numbered list of 5 places they are used in practice.",
        "Rewrite your first answer for a 10-year-old, in at most 3 sentences.",
        "Which item number in your list was about Git? Answer with the number and the item.",
        "Make a Markdown table with columns System | What gets hashed, one row for each item in your list.",
        "Summarise this whole conversation in one sentence.",
    ],
    "code": [
        "Write a Python function `parse_duration(s)` that turns strings like '1h30m', '45s', '2h5s' into a number of seconds. Include a docstring.",
        "Add tests for it using pytest, at least 6 cases.",
        "Now make it also accept days, like '1d2h'. Show the full updated function only.",
        "What does parse_duration('1d1h1m1s') return with your new version? Just the number.",
        "Rewrite the function in JavaScript with the same behaviour.",
    ],
    "offsite": [
        "I'm planning a team offsite for 14 people: Ana, Bilal, Chen, Dara, Emeka, Freya, Goran, Hana, Ivo, Jun, Kemal, Lena, Mika, Noor. Split them into 3 groups as evenly as possible and list the groups.",
        "Budget is 4,200 euros total; the venue is 1,750 and lunch is 23.50 per person. How much is left for activities?",
        "Which group is Freya in, and who else is in her group?",
        "Write a short friendly invitation email to the team for 12 March, mentioning the venue name 'Lakehouse Brenna' and the remaining activity budget.",
        "Translate the subject line only into German.",
        "List the names in alphabetical order from the largest group.",
    ],
}
OFF4 = OFF_CHATS["offsite"][:4]


def run_chat(srv, turns, **kw):
    msgs, rows = [], []
    for t in turns:
        msgs.append({"role": "user", "content": t})
        r = srv.chat(list(msgs), **kw)
        r["user"] = t
        msgs.append({"role": "assistant", "content": r["text"]})
        rows.append(r)
    return rows


def tool_tasks(root):
    return [
        ("revenue", f"Our back-office folder is {root}. Using the quarterly sales files in its sales folder, work out each quarter's revenue for 2026 (units times unit price), say which quarter was best, and give the year's total.", 1024),
        ("skill", f"Use the weekly-report skill for this: the notes are in {root}/notes.", 1536),
        ("terms", f"Read {root}/suppliers.md and tell me the payment terms of whoever supplies brake pads.", 512),
        ("notes", f"List the folder {root}/notes and tell me how many daily notes there are and their dates.", 512),
        ("search", f"Search {root}/notes for the words 'card reader' and tell me which day it was mentioned and what happened.", 512),
        ("helmets", f"Read {root}/sales/2026-q3.csv and {root}/sales/2026-q4.csv and tell me which quarter sold more helmet units, and how many each.", 512),
        ("copy", f"Make a copy of {root}/suppliers.md at {root}/backup/suppliers-copy.md with exactly the same text.", 1024),
        ("write", f"Create the file {root}/orders/reorder-2026-11-16.txt with these three lines: 'Order 40 tubes from Northwind Cycle Parts', 'Order 12 brake pads from Kestrel Components', 'Order 6 helmets from Alder & Finch'.", 512),
    ]


def run_tools(srv, kw):
    rows = []
    for name, q, n in tool_tasks("ROOT"):
        desk = Desk()
        root = desk.root.as_posix()
        r = srv.chat([{"role": "user", "content": q.replace("ROOT", root)}], max_tokens=n, tools=True, desk=desk, **kw)
        r["task"], r["root"] = name, root
        r["files"] = {}
        for p in list(desk.root.rglob("*")):
            rel = p.relative_to(desk.root).as_posix()
            if p.is_file() and (rel.startswith("backup") or rel.startswith("orders") or rel == "weekly-report.md"):
                r["files"][rel] = p.read_text()
        # anything written outside the expected places
        r["written"] = list(desk.written)
        r["copy_exact"] = r["files"].get("backup/suppliers-copy.md") == (desk.root / "suppliers.md").read_text() if name == "copy" else None
        shutil.rmtree(desk.root.parent, ignore_errors=True)
        rows.append(r)
    return rows


THINK_QS = [
    ("train", "A train leaves at 14:35 and the trip takes 3 h 50 min. After a 25-minute transfer I take a bus that takes 1 h 20 min. What time do I arrive?", "20:10"),
    ("letters", "How many times does the letter r appear in the two words 'strawberry raspberry' together?", "6"),
    ("code", "Write a Python function second_largest(xs) that returns the second-largest distinct integer in a list, or None if there isn't one. Only the code.", None),
    ("weekday", "If today is Sunday 15 November 2026, what weekday is 1 January 2027?", "Friday"),
    ("sum", "What is the sum of all integers from 1 to 200 that are divisible by 3 or by 5?", "9368"),
]
THINK_CHAT = [
    "I have three boxes holding 12, 7 and 19 apples. How many apples in total?",
    "I give away a third of the largest box, rounded down. How many apples do I have now?",
    "I split what's left evenly among 4 people. How many does each get, and how many are left over?",
]


def run_think(srv, kw):
    rows = []
    for name, q, want in THINK_QS:
        r = srv.chat([{"role": "user", "content": q}], max_tokens=12000, thinking=True, **kw)
        r["task"], r["want"] = name, want
        rows.append(r)
    chat = run_chat(srv, THINK_CHAT, max_tokens=12000, thinking=True, **kw)
    for r in chat:
        r["task"] = "chat"
    return rows + chat


def run_edge(srv):
    kw = {"temperature": 0, "max_tokens": 300}
    out = {}
    # (a) six turns
    a = run_chat(srv, [
        "My name is Tamsin and I keep bees. Say hello in one sentence.",
        "How many legs does a bee have? One sentence.",
        "Name three flowers bees like, as a short list.",
        "What did I say my name was?",
        "What is 17 times 23? Just the number.",
        "What hobby did I mention at the start? One word.",
    ], **kw)
    out["a_six_turns"] = a
    # (b) edit an earlier user message and resend
    b1 = run_chat(srv, ["My favourite colour is green. Remember it.", "What's my favourite colour? One word."], **kw)
    edited = [{"role": "user", "content": "My favourite colour is purple. Remember it."},
              {"role": "assistant", "content": b1[0]["text"]},
              {"role": "user", "content": "What's my favourite colour? One word."}]
    b2 = srv.chat(edited, **kw)
    # edit the first message and drop the rest (resend as a new first turn)
    b3 = srv.chat([{"role": "user", "content": "My favourite colour is orange. What is my favourite colour? One word."}], **kw)
    out["b_edit"] = b1 + [b2, b3]
    # (c) retry the same last message
    c_msgs = [{"role": "user", "content": "Give me a one-line fun fact about octopuses."}]
    c1 = srv.chat(c_msgs, **kw)
    c2 = srv.chat(c_msgs, **kw)
    out["c_retry"] = [c1, c2]
    # (d) two chats alternated
    A, B, rows = [], [], []
    for who, chat, text in [("A", A, "Chat A: I'm cooking a risotto with mushrooms. What wine goes with it? One sentence."),
                            ("B", B, "Chat B: I'm learning to juggle three balls. Give me one tip. One sentence."),
                            ("A", A, "What was I cooking, and with what? One sentence."),
                            ("B", B, "How many balls am I learning with? Just the number.")]:
        chat.append({"role": "user", "content": text})
        r = srv.chat(list(chat), **kw)
        r["chat"], r["user"] = who, text
        chat.append({"role": "assistant", "content": r["text"]})
        rows.append(r)
    out["d_alternate"] = rows
    # (e) tool call in an earlier turn then two plain turns
    desk = Desk()
    root = desk.root.as_posix()
    msgs = [{"role": "user", "content": f"Read {root}/suppliers.md and tell me who supplies helmets."}]
    e1 = srv.chat(list(msgs), tools=True, desk=desk, **kw)
    asst = {"role": "assistant", "content": e1["text"]}
    if e1["record"]:
        asst["record"] = e1["record"]
    msgs.append(asst)
    msgs.append({"role": "user", "content": "And what are that supplier's payment terms?"})
    e2 = srv.chat(list(msgs), tools=True, desk=desk, **kw)
    msgs.append({"role": "assistant", "content": e2["text"], **({"record": e2["record"]} if e2["record"] else {})})
    msgs.append({"role": "user", "content": "Which supplier has the shortest terms? One sentence."})
    e3 = srv.chat(list(msgs), tools=True, desk=desk, **kw)
    out["e_tool_then_plain"] = [e1, e2, e3]
    shutil.rmtree(desk.root.parent, ignore_errors=True)
    # (f) user message ending with the date-note text, with tools on (so the server adds its own note)
    desk = Desk()
    f_msgs = [{"role": "user", "content": "I'll be on leave next week. (Today is Monday.)"}]
    f1 = srv.chat(list(f_msgs), tools=True, desk=desk, **kw)
    f_msgs += [{"role": "assistant", "content": f1["text"]},
               {"role": "user", "content": "What day did I say it is today? One word."}]
    f2 = srv.chat(list(f_msgs), tools=True, desk=desk, **kw)
    f_msgs += [{"role": "assistant", "content": f2["text"]},
               {"role": "user", "content": "And when am I on leave? One short sentence."}]
    f3 = srv.chat(list(f_msgs), tools=True, desk=desk, **kw)
    out["f_date_text"] = [f1, f2, f3]
    shutil.rmtree(desk.root.parent, ignore_errors=True)
    # (a') six turns with tools offered (dated notes in play), thinking off, temp 0
    desk = Desk()
    out["a_six_turns_tools"] = run_chat(srv, [
        "My name is Tamsin and I keep bees. Say hello in one sentence.",
        "How many legs does a bee have? One sentence.",
        "Name three flowers bees like, as a short list.",
        "What did I say my name was?",
        "What is 17 times 23? Just the number.",
        "What hobby did I mention at the start? One word.",
    ], tools=True, desk=desk, **kw)
    shutil.rmtree(desk.root.parent, ignore_errors=True)
    return out


def run_chat_rec(srv, turns, **kw):
    """As run_chat, but sends each reply's turn_record back, as the app does."""
    msgs, rows = [], []
    for t in turns:
        msgs.append({"role": "user", "content": t})
        r = srv.chat(list(msgs), **kw)
        r["user"] = t
        m = {"role": "assistant", "content": r["text"]}
        if r.get("record"):
            m["record"] = r["record"]
        msgs.append(m)
        rows.append(r)
    return rows


def run_edge2(srv):
    kw = {"temperature": 0, "max_tokens": 300}
    out = {}
    desk = Desk()
    out["a_six_turns_records_tools"] = run_chat_rec(srv, [
        "My name is Tamsin and I keep bees. Say hello in one sentence.",
        "How many legs does a bee have? One sentence.",
        "Name three flowers bees like, as a short list.",
        "What did I say my name was?",
        "What is 17 times 23? Just the number.",
        "What hobby did I mention at the start? One word.",
    ], tools=True, desk=desk, **kw)
    # (b) edit: Lisbon -> Porto, keeping the neutral reply, then resending turn 2
    b = run_chat_rec(srv, ["I'm planning a trip to Lisbon in May. Reply with just: OK.",
                           "Which city am I going to? One word."], tools=True, desk=desk, **kw)
    edited = [{"role": "user", "content": "I'm planning a trip to Porto in May. Reply with just: OK."},
              {"role": "assistant", "content": b[0]["text"]},
              {"role": "user", "content": "Which city am I going to? One word."}]
    b2 = srv.chat(edited, tools=True, desk=desk, **kw)
    # edit the second message instead (the app's usual edit: the last user message)
    b3 = srv.chat([{"role": "user", "content": "I'm planning a trip to Lisbon in May. Reply with just: OK."},
                   {"role": "assistant", "content": b[0]["text"]},
                   {"role": "user", "content": "Which month am I travelling in? One word."}], tools=True, desk=desk, **kw)
    out["b_edit_clean"] = b + [b2, b3]
    # (f) user text ending with the note text, records sent
    f = run_chat_rec(srv, ["Please note this for the chat: I'll be on leave next week. (Today is Monday.)",
                           "What day did I say it is today? One word.",
                           "And when am I on leave? One short sentence.",
                           "Thanks. Reply with just: OK. (Today is Monday.)",
                           "Repeat the last word you said."], tools=True, desk=desk, **kw)
    out["f_date_text_records"] = f
    shutil.rmtree(desk.root.parent, ignore_errors=True)
    return out


def main():
    exe, gguf, suites, out = sys.argv[1:5]
    builtin = sys.argv[5] if len(sys.argv) > 5 else None
    logdir = Path(out).parent
    srv = Server(exe, gguf, builtin, str(logdir))
    res = {"exe": exe, "model": gguf, "log": str(srv.log_path)}
    try:
        for s in suites.split(","):
            t = time.time()
            if s == "off":
                res["off"] = {k: run_chat(srv, v, max_tokens=1200) for k, v in OFF_CHATS.items()}
            elif s == "offtools":
                desk = Desk()
                res["offtools"] = {k: run_chat(srv, v, max_tokens=1200, tools=True, desk=desk) for k, v in OFF_CHATS.items()}
            elif s == "off4":
                res["off4"] = run_chat(srv, OFF4, max_tokens=1200)
            elif s == "tools":
                res["tools"] = run_tools(srv, {})
            elif s == "think":
                res["think"] = run_think(srv, {})
            elif s == "leak":
                rows = []
                gen_extra = json.loads(os.environ.get("GEN_EXTRA", "{}"))
                n = int(os.environ.get("LEAK_N", "8"))
                for name, turns in OFF_CHATS.items():
                    for i in range(n):
                        desk = Desk()
                        r = srv.chat([{"role": "user", "content": turns[0]}], max_tokens=1200, tools=True, desk=desk,
                                     **({"gen_extra": gen_extra} if gen_extra else {}))
                        shutil.rmtree(desk.root.parent, ignore_errors=True)
                        r["task"], r["rep"] = name, i
                        r["leak"] = ("</think>" in r["text"]) or ("<think>" in r["text"])
                        rows.append(r)
                res["leak"] = rows
            elif s == "edge":
                res["edge"] = run_edge(srv)
            elif s == "edge2":
                res["edge"] = run_edge2(srv)
            res[s + "_seconds"] = time.time() - t
            Path(out).write_text(json.dumps(res, indent=1))
            print(s, "done", round(time.time() - t), "s", flush=True)
    finally:
        srv.stop()
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
