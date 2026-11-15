"""The Mac's 14 November benchmark, rebuilt for Windows from its report: Vireo against
llama.cpp through their servers, on the same GGUF files. (The Mac's own harness,
evals/speed/bench-2026-11-13, was not in the repository; the workloads here follow what
its report says they were, with prompts and tasks written again.)

A line a workload in <out>/results.jsonl, for each model and engine:

  short       a ~40-token question: writing speed, time to the first token
  medium      a ~2.5K-token document to answer from: reading speed
  long        as much of prompts-mac/long.txt (a repository dump) as the model's context
              on this GPU holds, with two facts planted in it and asked for: reading
              speed, writing after it, time to the first token, GPU memory, recall (2/2)
  multiturn   three turns of one chat, each answer checked
  tool        list a folder, read four files, add them up, answer (wall time, right or not)
  skill       a weekly-report skill: list, read, add up, write a file
  longtool    a long log pasted in, then a tool call for the answer

The Mac ran 32K, 96K and 128K prompts in a 131,072-token context; an 8 GB GPU holds
about 32K, so `long` is this run's largest and there is no 96K or 128K row. The tools are
the same for both engines and run here, in a folder made for the run: vireo-server asks
for them as an app's (`clientTools`), llama-server calls them as OpenAI tools.

Usage, from VireoPrototype (one model and engine loaded at a time, one run each):
    python evals/speed/bench_tasks.py --out evals/speed/bench-<date>-windows-tasks [name filter...]
Standard library only. Measure on mains in the fastest power mode: Gemma 4 E4B should read
prompts-mac/medium.txt at ~6,000 tokens/s and write at ~88 with `vireo-cli generate --gpu`.
"""
import argparse
import ast
import json
import operator
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import long_run  # noqa: E402

PROMPTS = REPO / "evals/speed/prompts-mac"
LLAMA = REPO / "reference/llama.cpp-b12262/llama-server.exe"
MODELS = [
    ("Gemma 4 E2B", "gemma-4-E2B_q4_0-it.gguf"),
    ("Gemma 4 E4B", "gemma-4-E4B_q4_0-it.gguf"),
    ("Qwen 3.5 4B", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen 3.5 9B", "Qwen3.5-9B-Q4_K_M.gguf"),
]
PORT = 8090
# The most a reply may write, by workload (the Mac's caps: 256, 128, 96 and 512 for tasks).
CAPS = {"short": 256, "medium": 128, "long": 96, "multiturn": 128, "task": 512}

SHORT = ("I'm planning a three-day trip to Lisbon in early spring with my sister, who loves old "
         "bookshops and trams. Suggest how we might spend the three days, a short paragraph for each day.")
MEDIUM_ASK = "\n\nIn two or three sentences, what are the most important changes described above?"
FACTS = [("The access code for the Halvorsen depot is 7391-KESTREL.", "What is the access code for the Halvorsen depot?", "7391"),
         ("The container ship Astrid Lund was moved to berth 14 on the north quay.", "Which berth was the Astrid Lund moved to?", "14")]
TURNS = [("My order number is 48213 and it ships to Lisbon. Reply with just: Noted.", ["noted"]),
         ("What is 17 times 23? Reply with just the number.", ["391"]),
         ("What was my order number, and where does it ship to?", ["48213", "lisbon"])]
QUARTERS = [("q1.txt", "6,120.50"), ("q2.txt", "7,304.25"), ("q3.txt", "5,890.00"), ("q4.txt", "7,427.50")]  # 26,742.25
HOURS = [("monday.txt", "16"), ("tuesday.txt", "15.5"), ("wednesday.txt", "17"), ("thursday.txt", "14.5"),
         ("friday.txt", "15")]  # 78
SKILL = """---
name: weekly-report
description: Write the week's report from the timesheets folder. Use when asked for the weekly report.
---

# Weekly report

1. Call `list_directory` on the `timesheets` folder.
2. Call `read_file` for every file in it (all the paths in one call, one a line).
3. Call `calculate` to add up the hours of every day.
4. Call `write_file` to save `weekly-report.md` in the home folder: a line for each day with its hours, then the week's total.
5. Tell the user the total hours and that the report is saved.
"""
SYSTEM = ("You are an AI agent working on the user's PC. Use the tools to look at files rather than guessing; "
          "paths are relative to the user's home folder. Use calculate for every sum. Answer briefly once you have what you need.")
TOOLS = [
    ("list_directory", "List the files and folders in a folder on this PC.",
     {"path": "The folder's path, relative to the home folder."}),
    ("read_file", "Read a text file on this PC. To read several files, give all their paths in one call, one a line.",
     {"path": "The file's path, relative to the home folder. Several files: one path a line."}),
    ("calculate", "Work out arithmetic exactly. Use it for every total or sum, e.g. \"412.50 + 129.30 + 400\".",
     {"expression": "Numbers with + - * / and brackets, or sum([...]). For several results, one expression a line."}),
    ("write_file", "Create a text file on this PC, or replace one's whole text.",
     {"path": "The file's path, relative to the home folder.", "content": "The file's whole new text."}),
]


def post(url, body, token=None, timeout=1800):
    headers = {"content-type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), headers), timeout=timeout)


def gpu_memory_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else None


# ---- the PC the tools see ---------------------------------------------------------------

class Pc:
    """A folder standing in for the user's home, and the four tools over it."""

    def __init__(self):
        self.home = Path(tempfile.mkdtemp(prefix="vireo-bench-home-"))
        self.reset()

    def reset(self):
        for child in self.home.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        (self.home / "sales").mkdir()
        for i, (name, amount) in enumerate(QUARTERS, 1):
            (self.home / "sales" / name).write_text(f"Quarter {i} revenue: ${amount}\nInvoices sent: {40 + 7 * i}\n")
        (self.home / "timesheets").mkdir()
        for name, hours in HOURS:
            (self.home / "timesheets" / name).write_text(f"{name[:-4].title()}: {hours} hours logged by the workshop team.\n")
        (self.home / "services").mkdir()
        for service, retries in (("inventory-sync", 7), ("billing-api", 3), ("mail-relay", 5), ("search-index", 4)):
            (self.home / "services" / f"{service}.conf").write_text(
                f"[service]\nname = {service}\ntimeout_seconds = 30\nretry_limit = {retries}\n")

    def path(self, given):
        given = str(given).strip().strip('"').replace("\\", "/")
        if given.startswith("~/"):
            given = given[2:]
        p = Path(given)
        p = p if p.is_absolute() else self.home / given
        p = p.resolve()
        if self.home.resolve() not in [p, *p.parents]:
            raise ValueError(f"{given} is outside the home folder")
        return p

    def run(self, tool, args):
        """The tool's output, or an error's text (as a tool's result either way)."""
        try:
            if tool == "list_directory":
                p = self.path(args.get("path", "."))
                return "\n".join(sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir())) or "(empty)"
            if tool == "read_file":
                paths = [l for l in str(args.get("path", "")).splitlines() if l.strip()]
                texts = [f"--- {g.strip()} ---\n{self.path(g).read_text()}" if len(paths) > 1 else self.path(g).read_text()
                         for g in paths]
                return "\n".join(texts)
            if tool == "calculate":
                return "\n".join(f"{line.strip()} = {calculate(line)}"
                                 for line in str(args.get("expression", "")).splitlines() if line.strip())
            if tool == "write_file":
                p = self.path(args.get("path", ""))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(args.get("content", "")))
                return f"Saved {p.name}."
            return f"There is no tool named {tool}."
        except Exception as e:  # a model's bad path or expression: told to it, as an app would
            return f"Error: {e}"


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression):
    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](value(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [value(e) for e in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("sum", "min", "max", "round", "abs"):
            return {"sum": sum, "min": min, "max": max, "round": round, "abs": abs}[node.func.id](*[value(a) for a in node.args])
        raise ValueError("only numbers, + - * / and sum([...])")
    cleaned = re.sub(r"(?<=\d),(?=\d{3})", "", expression).replace("$", "")
    result = value(ast.parse(cleaned.strip(), mode="eval").body)
    return f"{result:,.2f}" if isinstance(result, float) and not result.is_integer() else f"{result:,.0f}" if isinstance(result, (int, float)) else str(result)


# ---- the workloads ----------------------------------------------------------------------

def long_text(context):
    """As much of long.txt as a chat of `context` tokens holds beside the engine's own
    instructions and the reply, with the two facts planted a quarter and three quarters in."""
    text = (PROMPTS / "long.txt").read_text(encoding="utf-8")
    text = text[: int(max(context - 3500, 2000) * 3.6)]  # ~3.8 characters a token for Gemma here, more for Qwen
    lines = text.splitlines()
    for share, (fact, _, _) in zip((0.25, 0.75), FACTS):
        lines.insert(int(len(lines) * share), f"// NOTE FOR THE HARBOUR OFFICE: {fact}")
    return "\n".join(lines) + "\n\nFrom the notes for the harbour office above: " + " ".join(q for _, q, _ in FACTS)


def log_text(tokens):
    """A day's log of about `tokens` tokens; inventory-sync logs the most errors."""
    rng = random.Random(7)
    services = ["billing-api", "mail-relay", "search-index", "inventory-sync"]
    lines, minute = [], 0
    # Timestamps and numbers: ~1.9 characters a token for Gemma 4 and Qwen 3.5 (3.2 made the
    # log nearly the whole context, too long for Qwen 3.5 on vireo-server; 15 Nov).
    while sum(len(l) for l in lines) < tokens * 1.9:
        minute += rng.randint(1, 3)
        service = rng.choice(services)
        error = rng.random() < (0.30 if service == "inventory-sync" else 0.06)
        stamp = f"2026-11-14T{(minute // 60) % 24:02d}:{minute % 60:02d}:{rng.randint(0, 59):02d}Z"
        lines.append(f"{stamp} {'ERROR' if error else 'INFO '} {service} "
                     + (f"request {rng.randint(1000, 9999)} failed after retry: upstream timed out" if error
                        else f"request {rng.randint(1000, 9999)} completed in {rng.randint(12, 480)} ms"))
    return ("Here is today's log from our servers.\n\n" + "\n".join(lines) + "\n\nWhich service logged the most ERROR lines? "
            "Then read that service's config file in the services folder (services/<name>.conf) and tell me its retry_limit.")


TOOL_ASK = ("The sales folder has a revenue file for each quarter. List the folder, read every file, "
            "and tell me each quarter's revenue and the total for the year.")
SKILL_ASK = "Use the weekly-report skill: write this week's report from the timesheets folder."


def has_all(answer, needles):
    flat = answer.lower().replace(",", "")
    return all(n.lower().replace(",", "") in flat for n in needles)


# ---- Vireo ------------------------------------------------------------------------------

class Vireo:
    name = "vireo"

    def __init__(self, gguf, pc, skills, out):
        os.environ["INVURSE_SKILLS_DIR"] = str(skills)
        self.pc = pc
        self.server = long_run.Vireo(str(gguf), 0, 512, [])
        self.context = self.server.context
        self.out = out
        self.label = Path(gguf).stem
        self.kept_log = self.server.new_log()  # vireo-server's log, kept in <out> when it stops

    def ask(self, messages, max_tokens, tools=False):
        s = self.server
        s.new_log()
        started = time.time()
        body = {"messages": messages, "generation": {"max_tokens": max_tokens, "temperature": 0}}
        if tools:
            body["clientTools"] = ["read_file", "list_directory", "write_file"]
        response = post(f"http://127.0.0.1:{s.port}/api/chat/completions", body, s.token)
        first, reply, calls, steps, error = None, "", [], [], None
        for event in long_run.events(response):
            kind = event.get("type")
            if "content" in event:
                first = first or time.time() - started
                reply += event["content"]
            elif kind is None and "error" in event:
                error = str(event["error"])
            elif kind == "tool_result":
                steps.append({"tool": event.get("tool"), "result": str(event.get("result", ""))[:200]})
            elif kind == "client_tool_request":
                result = self.pc.run(event.get("tool", ""), event.get("args") or {})
                post(f"http://127.0.0.1:{s.port}/api/agent/client-tool-result",
                     {"id": event["id"], "ok": not result.startswith("Error:"),
                      "output" if not result.startswith("Error:") else "error": result}, s.token).read()
            elif kind == "confirm_required":
                # write_file is the task's own step: approved. Anything else (a model's
                # unasked `remember`) is refused at once.
                post(f"http://127.0.0.1:{s.port}/api/agent/confirm",
                     {"id": event["id"], "approved": event.get("tool") == "write_file", "remember": False}, s.token).read()
            elif kind == "tool_call":
                calls.append(event.get("tool"))
                steps.append({"tool": event.get("tool"), "args": event.get("args")})
        wall = time.time() - started
        time.sleep(0.05)
        log = s.new_log()
        self.kept_log += log
        row = {"answer": reply, "wall_s": round(wall, 2), "ttft_s": round(first, 2) if first else None, "calls": calls}
        if steps:
            row["steps"] = steps
        if error:
            row["stream_error"] = error
        # One "Reply:" line for the whole reply, its tool calls' steps included.
        if m := s.REPLY.search(log):
            read, read_s, written, write_s = int(m[3]), float(m[4]), int(m[5]), float(m[6])
            row.update(prompt_n=int(m[1]), kept_n=int(m[2]), read_n=read,
                       prompt_tps=round(read / read_s, 1) if read_s else None,
                       gen_n=written, gen_tps=round(written / write_s, 1) if write_s else None)
        if m := re.search(r"(\d+) of (\d+) guesses", log):
            row["guesses"] = [sum(int(a) for a, _ in re.findall(r"(\d+) of (\d+) guesses", log)),
                              sum(int(b) for _, b in re.findall(r"(\d+) of (\d+) guesses", log))]
        return row

    def stop(self):
        self.server.stop()
        self.kept_log += self.server.log_path.read_text(errors="replace")[self.server.read_to:]
        with open(Path(self.out) / "vireo-server.log", "a", encoding="utf-8") as f:
            f.write(f"==== {self.label} · {time.strftime('%H:%M:%S')} ====\n{self.kept_log}\n")
        shutil.rmtree(self.server.dir, ignore_errors=True)


# ---- llama.cpp --------------------------------------------------------------------------

class Llama:
    name = "llama"

    def __init__(self, gguf, pc, skills, out, context):
        self.pc = pc
        self.context = context
        self.skill = (skills / "weekly-report" / "SKILL.md").read_text().split("---", 2)[2].strip()
        log = open(Path(out) / "llama-server.log", "a")
        # Thinking off, as vireo-server's default is; the context the long prompt needs.
        self.proc = subprocess.Popen([str(LLAMA), "-m", str(gguf), "-ngl", "99", "-fa", "on", "-c", str(context + 2048),
                                      "--port", str(PORT), "--host", "127.0.0.1", "-np", "1", "--no-webui",
                                      "--reasoning", "off"], stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while True:
            if self.proc.poll() is not None or time.time() - started > 300:
                raise RuntimeError("llama-server didn't come up; see llama-server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)

    def turn(self, messages, max_tokens, tools):
        """One request, streamed: the message, its tool calls, timings and first-token time."""
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True,
                "stream": True, "timings_per_token": True}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": name, "description": about,
                "parameters": {"type": "object", "required": list(params),
                               "properties": {p: {"type": "string", "description": d} for p, d in params.items()}}}}
                for name, about, params in TOOLS]
        started = time.time()
        first, text, calls, timings = None, "", {}, {}
        for raw in post(f"http://127.0.0.1:{PORT}/v1/chat/completions", body):
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    first = first or time.time() - started
                    text += delta["content"]
                for call in delta.get("tool_calls") or []:
                    first = first or time.time() - started
                    slot = calls.setdefault(call.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = call.get("id") or slot["id"]
                    function = call.get("function", {})
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
        return text, [calls[i] for i in sorted(calls)], timings, first

    def ask(self, messages, max_tokens, tools=False):
        started = time.time()
        messages = ([{"role": "system", "content": SYSTEM}] if tools else []) + list(messages)
        first, read, read_ms, written, write_ms, names, answer, steps = None, 0, 0.0, 0, 0.0, [], "", []
        for _ in range(12):  # a task's round trips
            sent = time.time() - started
            text, calls, t, at = self.turn(messages, max_tokens, tools)
            first = first or (sent + at if at else None)  # `at` counts from this request's start
            read += t.get("prompt_n", 0)
            read_ms += t.get("prompt_ms", 0.0)
            written += t.get("predicted_n", 0)
            write_ms += t.get("predicted_ms", 0.0)
            answer = text
            if not calls:
                break
            messages.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for i, c in enumerate(calls)]})
            for i, c in enumerate(calls):
                names.append(c["name"])
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.pc.run(c["name"], args)
                steps.append({"tool": c["name"], "args": args, "result": result[:200]})
                messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}", "content": result})
        return {"answer": answer, "wall_s": round(time.time() - started, 2), "ttft_s": round(first, 2) if first else None,
                "calls": names, **({"steps": steps} if steps else {}), "read_n": read, "prompt_n": read,
                "prompt_tps": round(read / read_ms * 1000, 1) if read_ms else None,
                "gen_n": written, "gen_tps": round(written / write_ms * 1000, 1) if write_ms else None}

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        time.sleep(3)


# ---- the run ----------------------------------------------------------------------------

def workloads(engine, pc, record):
    def go(work, messages, cap, tools=False, check=None, **more):
        try:
            row = engine.ask(messages, cap, tools)
        except Exception as e:
            record(work=work, error=repr(e), **more)
            return {"answer": ""}
        if check is not None:
            row["right"] = bool(check(row))
        row["answer"] = row["answer"][:1500]
        record(work=work, **row, **more)
        return row

    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    go("warm-up", user("Say hello."), 16)
    go("short", user(SHORT), CAPS["short"])
    go("medium", user((PROMPTS / "medium.txt").read_text(encoding="utf-8") + MEDIUM_ASK), CAPS["medium"])
    go("long", user(long_text(engine.context)), CAPS["long"],
       check=lambda r: has_all(r["answer"], [FACTS[0][2], FACTS[1][2]]))
    record(work="memory", after="long", gpu_mb=gpu_memory_mb(), context=engine.context)
    # Three turns of one chat; every turn must be right.
    chat, rights, wall = [], [], 0.0
    for text, needles in TURNS:
        chat.append({"role": "user", "content": text})
        row = engine.ask(chat, CAPS["multiturn"])
        chat.append({"role": "assistant", "content": row["answer"]})
        rights.append(has_all(row["answer"], needles))
        wall += row["wall_s"]
    record(work="multiturn", wall_s=round(wall, 2), right=all(rights), turns_right=rights,
           answer=" | ".join(m["content"] for m in chat if m["role"] == "assistant")[:600])
    pc.reset()
    go("tool", user(TOOL_ASK), CAPS["task"], tools=True, check=lambda r: has_all(r["answer"], ["26742.25"]))
    pc.reset()
    skill_ask = SKILL_ASK if engine.name == "vireo" else f"{SKILL_ASK}\n\nThe weekly-report skill's instructions:\n\n{engine.skill}"
    report = pc.home / "weekly-report.md"
    go("skill", user(skill_ask), CAPS["task"], tools=True,
       check=lambda r: has_all(r["answer"], ["78"]) and report.exists() and "78" in report.read_text())
    record(work="skill-file", written=report.exists(), text=report.read_text()[:400] if report.exists() else None)
    pc.reset()
    go("longtool", user(log_text(min(20000, int(engine.context * 0.55)))), CAPS["task"], tools=True,
       check=lambda r: has_all(r["answer"], ["inventory-sync", "7"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--engines", default="vireo,llama")
    parser.add_argument("only", nargs="*", help="model name filters")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pc = Pc()
    skills = Path(tempfile.mkdtemp(prefix="vireo-bench-skills-"))
    (skills / "weekly-report").mkdir()
    (skills / "weekly-report" / "SKILL.md").write_text(SKILL)
    try:
        for name, file in MODELS:
            if args.only and not any(o.lower() in name.lower() for o in args.only):
                continue
            gguf = REPO / "models" / file
            context = None
            for engine_name in args.engines.split(","):
                def record(**row):
                    row = {"engine": engine_name, "model": name, **row, "time": time.strftime("%H:%M:%S")}
                    print(json.dumps(row)[:400], flush=True)
                    with open(out / "results.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")
                try:
                    started = time.time()
                    if engine_name == "vireo":
                        engine = Vireo(gguf, pc, skills, out)
                        context = engine.context
                    else:
                        # The context vireo-server gave this model on this GPU, so both read the same long prompt.
                        engine = Llama(gguf, pc, skills, out, context or 32768)
                    record(work="load", wall_s=round(time.time() - started, 2), context=engine.context)
                except Exception as e:
                    record(work="load", error=repr(e))
                    continue
                try:
                    workloads(engine, pc, record)
                finally:
                    engine.stop()
    finally:
        shutil.rmtree(pc.home, ignore_errors=True)
        shutil.rmtree(skills, ignore_errors=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
