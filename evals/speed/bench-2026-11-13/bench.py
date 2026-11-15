"""Nov 13 Mac benchmark: Vireo's engine (vireo-server) vs llama.cpp (llama-server) vs
mlx-vlm (its own server, latest from PyPI), all through their servers, streaming, so the
time to the first token is what a client sees.

Workloads (fixtures from make_fixtures.py):
  short       a 40-token question, up to 256 tokens of answer
  medium      ~2.5K-token document to summarise
  long        ~31K-token repository dump, recall two hidden facts
  large       ~90K-token repository dump, the same question
  xlarge      ~120K-token dump (large.txt and more of the plan after it), the same question:
              the 128K-context test, so every model's context is 131,072 tokens
  multiturn   eight turns, each pasting a ~600-token report (prefix reuse across turns)
  tools       multi-step task over a folder: list, read several files, calculate, answer
  skill       the weekly-report skill: list, read, calculate, write a file, answer
  longtools   a ~20K-token log pasted in, then tool calls over the folder

The same tools (Vireo's catalog names and descriptions) are offered to every engine and
run here, by this script, on the sandbox folder, with Vireo's 8,000-character cap on a
tool result applied for all. Thinking is off everywhere.

Recorded per request: client-side time to first token (any streamed token: text, a tool
call or its start), total time, the server's own prompt / cached / generated token
counts and speeds, and peak memory of the server process (RSS and phys_footprint,
sampled every 2 s).

  ENGINES=vireo,vireo-plain,llama,mlx MODELS=e2b,e4b REPS=2 python3 -I bench.py
Results: $OUT (default ./results.jsonl); server logs in $LOGS (default $TMPDIR).
"""
import ast
import json
import math
import operator
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROMPTS = HERE / "prompts"
SANDBOX = HERE / "sandbox" / "harbor-bikes"
SKILLS = HERE / "skills"
OUT = Path(os.environ.get("OUT", HERE / "results.jsonl"))
LOGS = Path(os.environ.get("LOGS", tempfile.gettempdir())) / "bench-logs"
LOGS.mkdir(parents=True, exist_ok=True)
GG = Path.home() / "Library/Application Support/InvurseChat/Models"
SM = REPO / "models"
MX = REPO / "models/mlx"
VENV_PY = Path.home() / ".venvs/mlxbench/bin/python"
LLAMA_DIR = REPO / "reference/llama-b12228"

# name, key, gguf, mlx folder, context (tokens), large workload?
MODELS = [
    ("Gemma 4 E2B", "e2b", SM / "gemma-4-E2B-it-Q6_K.gguf", MX / "gemma-4-e2b-it-6bit", 131072, True),
    ("Gemma 4 E4B", "e4b", SM / "gemma-4-E4B-it-Q6_K.gguf", MX / "gemma-4-e4b-it-6bit", 131072, True),
    ("Qwen 3.5 4B", "q4b", SM / "Qwen3.5-4B-Q6_K.gguf", MX / "Qwen3.5-4B-6bit", 131072, True),
    ("Qwen 3.5 9B", "q9b", SM / "Qwen3.5-9B-Q6_K.gguf", MX / "Qwen3.5-9B-6bit", 131072, True),
    ("Gemma 4 26B-A4B", "g26", GG / "unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K.gguf",
     MX / "gemma-4-26b-a4b-it-6bit", 131072, True),
    ("Gemma 4 31B", "g31", GG / "gemma-4-31B-it-GGUF/gemma-4-31B-it-Q6_K.gguf", GG / "gemma-4-31B-it-MLX-6bit", 131072, True),
    ("Qwen 3.8 27B", "q27", GG / "Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf", GG / "Qwen3.8-27B-MLX-6bit", 131072, True),
    ("Muse Glimmer 30B", "muse", GG / "Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-Q6_K.gguf", GG / "Muse-Glimmer-30B-6bit", 131072, True),
]
WORKLOADS = ["short", "medium", "long", "large", "xlarge", "multiturn", "tools", "skill", "longtools"]
SYSTEM = "You are a helpful assistant for Harbor Bikes, a small bike shop. Be concise and accurate."
TOKEN = "bench"

# ---- Tools, as Vireo's catalog describes them -------------------------------------------

TOOL_SPECS = [
    ("read_file", "Read a text file, PDF, or Word, Excel or PowerPoint document on this PC. To read several files, give all their paths in one call, one a line, rather than a call for each.",
     {"path": "The file's path: a full path, or one relative to the home folder (Documents/report.pdf). Several files: one path a line. Don't make one up."}),
    ("list_directory", "List the files and folders in a folder on this PC.",
     {"path": "The folder's path: a full path, or one relative to the home folder (Documents)."}),
    ("search_files", "Find files and folders by name, or files containing some text, in a folder on this PC and its subfolders.",
     {"pattern": "A name or part of one, or the text to look for.",
      "directory": "The folder to search: a full path, or one relative to the home folder."}),
    ("calculate", "Work out arithmetic exactly. Use it for every total, sum, difference, percentage or average instead of doing the maths yourself, e.g. \"412.50 + 129.30 + 400\" or \"sum([12.40, 8.99, 31.00]) * 1.13\". Dollar signs, thousands commas and % are fine. Work out everything you need in one call, one expression a line, rather than a call for each.",
     {"expression": "Numbers with + - * / % ** and brackets, or round, abs, min, max, sum. For several results, one expression a line."}),
    ("write_file", "Create a text file on this PC, or replace one's whole text. Asks the user for approval.",
     {"path": "The file's path: a full path, or one relative to the home folder.",
      "content": "The file's whole new text."}),
]
OPENAI_TOOLS = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in p.items()},
                   "required": list(p)}}} for n, d, p in TOOL_SPECS]
VIREO_CLIENT_TOOLS = ["read_file", "list_directory", "search_files", "write_file"]


class Desk:
    """Runs the tools on a fresh copy of the sandbox folder."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="bench-desk-")) / "harbor-bikes"
        shutil.copytree(SANDBOX, self.root)
        self.written = {}

    def path(self, p):
        p = (p or "").strip().strip("'\"")
        q = Path(p) if p.startswith("/") else self.root.parent / p
        q = q.resolve()
        if not str(q).startswith(str(self.root.resolve().parent)):
            raise ValueError(f"{p}: no such file or folder")
        return q

    def run(self, tool, args):
        try:
            if tool == "read_file":
                paths = [x for x in str(args.get("path", "")).splitlines() if x.strip()]
                if len(paths) > 1:
                    share = 8000 * min(3, len(paths)) // len(paths) - 120
                    out = []
                    for p in paths[:10]:
                        try:
                            t = self.path(p).read_text()
                        except Exception as e:
                            t = f"Error: {e}"
                        out.append(f"=== {p} ===\n" + (t[:share] + f"\n[... {len(t) - share} more characters not shown]" if len(t) > share else t.rstrip()))
                    return True, "\n\n".join(out)
                return True, self.path(paths[0] if paths else "").read_text()[:8000]
            if tool == "list_directory":
                d = self.path(args.get("path", ""))
                return True, "\n".join(f"{x.name}/" if x.is_dir() else f"{x.name} ({x.stat().st_size} bytes)"
                                       for x in sorted(d.iterdir()))
            if tool == "search_files":
                d, pat = self.path(args.get("directory", "")), str(args.get("pattern", "")).lower()
                hits = []
                for f in sorted(d.rglob("*")):
                    if pat in f.name.lower():
                        hits.append(str(f))
                    elif f.is_file():
                        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                            if pat in line.lower():
                                hits.append(f"{f}:{i}: {line.strip()}")
                return True, "\n".join(hits[:60]) or "No matches."
            if tool == "calculate":
                return True, calc(str(args.get("expression", "")))
            if tool == "write_file":
                p = self.path(args.get("path", ""))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(args.get("content", "")))
                self.written[str(p)] = len(str(args.get("content", "")))
                return True, f"Wrote {p}"
            return False, f"Unknown tool {tool}"
        except Exception as e:
            return False, str(e)


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sum": sum}


def calc(expr):
    out = []
    for line in [x for x in expr.splitlines() if x.strip()]:
        e = line.split("=")[-1] if "=" in line and "==" not in line else line
        e = e.replace("$", "").replace("%", "/100").replace("×", "*")
        e = re.sub(r"(?<=\d),(?=\d{3})", "", e)

        def ev(n):
            if isinstance(n, ast.Expression):
                return ev(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp):
                return OPS[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp):
                return OPS[type(n.op)](ev(n.operand))
            if isinstance(n, (ast.List, ast.Tuple)):
                return [ev(x) for x in n.elts]
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS:
                return FUNCS[n.func.id](*[ev(a) for a in n.args])
            raise ValueError("unsupported")
        try:
            v = ev(ast.parse(e.strip(), mode="eval"))
            v = round(v, 6) if isinstance(v, float) else v
            out.append(f"{line.strip()} = {v}")
        except Exception:
            out.append(f"{line.strip()} = error: can't work that out")
    return "\n".join(out) or "error: empty expression"


# ---- Memory sampling --------------------------------------------------------------------

class MemWatch:
    def __init__(self, pid):
        self.pid, self.peak_rss, self.peak_fp, self.peak_total, self.stop_ = pid, 0, 0, 0, False
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def sample(self):
        try:
            rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True).stdout.strip() or 0) * 1024
            fp_out = subprocess.run(["footprint", "-p", str(self.pid)], capture_output=True, text=True).stdout
            units = {"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}
            m = re.search(r"Footprint: ([\d.]+) (B|KB|MB|GB)", fp_out)
            fp = float(m[1]) * units[m[2]] if m else 0
            # Model weights mapped from their file (llama.cpp, Vireo) are clean pages, left
            # out of the footprint: add what's resident of them.
            mf = re.search(r"^\s*\S+ \S+\s+([\d.]+) (B|KB|MB|GB)\s+\S+ \S+\s+\d+\s+mapped file$", fp_out, re.M)
            mapped = float(mf[1]) * units[mf[2]] if mf else 0
            self.peak_rss, self.peak_fp = max(self.peak_rss, rss), max(self.peak_fp, fp)
            self.peak_total = max(self.peak_total, fp + mapped)
            return rss, fp, fp + mapped
        except Exception:
            return 0, 0, 0

    def loop(self):
        while not self.stop_:
            self.sample()
            time.sleep(2)

    def reset(self):
        self.peak_rss, self.peak_fp, self.peak_total = 0, 0, 0
        return self.sample()

    def stop(self):
        self.stop_ = True


# ---- HTTP -------------------------------------------------------------------------------

def post(url, body, token=None, timeout=3600):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), h), timeout=timeout)


def sse(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                pass


def wait_health(url, proc, secs=900):
    t = time.time()
    while time.time() - t < secs:
        if proc.poll() is not None:
            raise RuntimeError("server exited during load")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return time.time() - t
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not come up")


def stop_proc(p):
    p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(4)


# ---- Engines ----------------------------------------------------------------------------

class Vireo:
    REPLY = re.compile(r"Reply: (\d+) prompt tokens \((\d+) kept, (\d+) read in ([\d.]+)s, \d+/s\), "
                       r"(\d+) tokens in ([\d.]+)s")

    def __init__(self, model, plain):
        name, key, gguf, mlx, ctx, _ = model
        self.dir = Path(tempfile.mkdtemp(prefix="bench-vireo-"))
        (self.dir / "builtin").mkdir()
        self.log_path = LOGS / f"vireo{'-plain' if plain else ''}-{key}-{int(time.time())}.log"
        self.log = open(self.log_path, "w")
        args = [str(REPO / "target/release/vireo-server"), str(gguf), "--port", "0", "--token", TOKEN,
                "--chat-cache", str(self.dir / "chat-cache"), "--context", str(ctx), "--thinking", "off"]
        if plain:
            args += ["--mtp", "off", "--lookup", "0"]
        env = dict(os.environ, INVURSE_SKILLS_DIR=str(SKILLS), VIREO_BUILTIN_SKILLS=str(self.dir / "builtin"),
                   VIREO_MEMORY_FILE=str(self.dir / "memories.txt"),
                   VIREO_APPROVALS_FILE=str(self.dir / "approvals.txt"))
        t = time.time()
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=self.log, env=env, text=True)
        self.port = None
        for line in self.proc.stdout:
            if line.startswith("Listening on http://127.0.0.1:"):
                self.port = int(line.strip().rsplit(":", 1)[1])
                break
        if not self.port:
            raise RuntimeError(f"vireo-server didn't start; see {self.log_path}")
        threading.Thread(target=lambda: [None for _ in self.proc.stdout], daemon=True).start()
        self.load_s = time.time() - t
        self.pid = self.proc.pid
        self.read_to = 0

    def new_log(self):
        self.log.flush()
        text = self.log_path.read_text(errors="replace")
        new, self.read_to = text[self.read_to:], len(text)
        return new

    def chat(self, messages, max_tokens, tools, desk):
        self.new_log()
        body = {"systemPrompt": SYSTEM, "messages": messages, "thinking": False,
                "generation": {"max_tokens": max_tokens, "temperature": 0}}
        if tools:
            body["clientTools"] = VIREO_CLIENT_TOOLS
        t0 = time.time()
        first, text, calls, err = None, "", [], None
        for ev in sse(post(f"http://127.0.0.1:{self.port}/api/chat/completions", body, TOKEN)):
            typ = ev.get("type")
            if "content" in ev and typ is None:
                if ev["content"]:
                    first = first or time.time() - t0
                text += ev["content"]
            elif typ in ("thinking",):
                first = first or time.time() - t0
            elif typ == "tool_call":
                first = first or time.time() - t0
                calls.append(ev.get("tool"))
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
        time.sleep(0.1)
        steps = []
        for m in self.REPLY.finditer(self.new_log()):
            steps.append({"prompt": int(m[1]), "cached": int(m[2]), "read": int(m[3]), "read_s": float(m[4]),
                          "gen": int(m[5]), "gen_s": float(m[6])})
        return {"ttft": first, "total": total, "text": text, "calls": calls, "error": err, "steps": steps}

    def stop(self):
        stop_proc(self.proc)
        self.log.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class OpenAIish:
    """llama-server or mlx-vlm's server: the tool loop runs here, a request per step."""
    kind = ""
    port = 0

    def request_extra(self):
        return {}

    def one(self, msgs, max_tokens, tools):
        body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, **self.request_extra()}
        if tools:
            body["tools"] = OPENAI_TOOLS
        t0 = time.time()
        first, text, tcs, timings, usage, err = None, "", {}, None, None, None
        try:
            for ev in sse(post(f"http://127.0.0.1:{self.port}/v1/chat/completions", body)):
                if "error" in ev:
                    err = json.dumps(ev["error"])[:300]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                        first = first or time.time() - t0
                    text += d.get("content") or ""
                    for tc in d.get("tool_calls") or []:
                        slot = tcs.setdefault(tc.get("index", len(tcs)), {"id": tc.get("id"), "name": "", "args": ""})
                        f = tc.get("function") or {}
                        slot["id"] = slot["id"] or tc.get("id")
                        slot["name"] += f.get("name") or ""
                        slot["args"] += f.get("arguments") or ""
                timings = ev.get("timings") or timings
                usage = ev.get("usage") or usage
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")[:300]
        total = time.time() - t0
        step = {"ttft": first, "total": total}
        if timings:
            step.update(read=timings.get("prompt_n"), cached=timings.get("cache_n"),
                        read_s=(timings.get("prompt_ms") or 0) / 1000, gen=timings.get("predicted_n"),
                        gen_s=(timings.get("predicted_ms") or 0) / 1000,
                        gen_tps_server=timings.get("predicted_per_second"),
                        read_tps_server=timings.get("prompt_per_second"))
            step["prompt"] = (step["read"] or 0) + (step["cached"] or 0)
        if usage:
            step["usage"] = usage
        return text, [tcs[k] for k in sorted(tcs)], step, err

    def chat(self, messages, max_tokens, tools, desk):
        msgs = [{"role": "system", "content": SYSTEM}] + messages
        t0 = time.time()
        first, calls, steps, err, text = None, [], [], None, ""
        for _ in range(12):
            text, tcs, step, err = self.one(msgs, max_tokens, tools)
            steps.append(step)
            if first is None and step["ttft"] is not None:
                first = (time.time() - t0) - step["total"] + step["ttft"]
            if err or not tcs:
                break
            msgs.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": tc["id"] or f"call_{i}", "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["args"] or "{}"}} for i, tc in enumerate(tcs)]})
            for i, tc in enumerate(tcs):
                calls.append(tc["name"])
                try:
                    args = json.loads(tc["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                ok, out = desk.run(tc["name"], args)
                msgs.append({"role": "tool", "tool_call_id": tc["id"] or f"call_{i}", "content": out if ok else f"Error: {out}"})
        return {"ttft": first, "total": time.time() - t0, "text": text, "calls": calls, "error": err, "steps": steps}


class Llama(OpenAIish):
    kind = "llama"

    def __init__(self, model):
        name, key, gguf, mlx, ctx, _ = model
        self.port = 18452
        self.log_path = LOGS / f"llama-{key}-{int(time.time())}.log"
        self.log = open(self.log_path, "w")
        env = dict(os.environ, DYLD_LIBRARY_PATH=str(LLAMA_DIR))
        t = time.time()
        self.proc = subprocess.Popen(
            [str(LLAMA_DIR / "llama-server"), "-m", str(gguf), "-c", str(ctx), "-ngl", "99", "-fa", "on",
             "--port", str(self.port), "--host", "127.0.0.1", "-np", "1", "--jinja", "--no-webui",
             "--reasoning-budget", "0"], stdout=self.log, stderr=self.log, env=env)
        wait_health(f"http://127.0.0.1:{self.port}/health", self.proc)
        self.load_s = time.time() - t
        self.pid = self.proc.pid

    def request_extra(self):
        return {"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}

    def stop(self):
        stop_proc(self.proc)
        self.log.close()


class Mlx(OpenAIish):
    kind = "mlx"

    def __init__(self, model):
        name, key, gguf, mlx, ctx, _ = model
        self.port = 18453
        self.model = str(mlx)
        self.log_path = LOGS / f"mlx-{key}-{int(time.time())}.log"
        self.log = open(self.log_path, "w")
        # Its automatic prefix cache on (off by default), as shipped Vireo 0.9 ran it.
        env = dict(os.environ, APC_ENABLED="1")
        t = time.time()
        self.proc = subprocess.Popen([str(VENV_PY), "-m", "mlx_vlm.server", "--model", self.model,
                                      "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "WARNING"],
                                     stdout=self.log, stderr=self.log, env=env, cwd=tempfile.gettempdir())
        wait_health(f"http://127.0.0.1:{self.port}/health", self.proc)
        self.load_s = time.time() - t
        self.pid = self.proc.pid

    def request_extra(self):
        return {"model": self.model, "enable_thinking": False}

    def stop(self):
        stop_proc(self.proc)
        self.log.close()


# ---- Workloads --------------------------------------------------------------------------

def nonce():
    return f"[Request {uuid.uuid4().hex[:8]}]\n"


def prompt(name):
    return (PROMPTS / f"{name}.txt").read_text()


def run_workload(eng, w, mem):
    desk = Desk()
    rows = []
    if w in ("short", "medium", "long", "large", "xlarge"):
        n = {"short": 256, "medium": 256, "long": 128, "large": 128, "xlarge": 128}[w]
        r = eng.chat([{"role": "user", "content": nonce() + prompt(w)}], n, False, desk)
        if w in ("long", "large", "xlarge"):
            r["recall"] = sum(x.lower() in r["text"].lower() for x in ["OSPREY-41", "Dagny Vell"])
        rows.append(r)
    elif w == "multiturn":
        turns = json.loads((PROMPTS / "multiturn.json").read_text())
        msgs, tag = [], nonce()
        for i, t in enumerate(turns):
            msgs.append({"role": "user", "content": (tag if i == 0 else "") + t["text"]})
            r = eng.chat(list(msgs), 96, False, desk)
            r["turn"] = i + 1
            r["right"] = t["expect"] in r["text"]
            msgs.append({"role": "assistant", "content": r["text"]})
            rows.append(r)
    elif w == "tools":
        q = (nonce() + f"Our back-office folder is {desk.root}. Using the quarterly sales files in its "
             "sales folder, work out each quarter's revenue for 2026 (units times unit price), say which "
             "quarter was best, and give the year's total.")
        rows.append(eng.chat([{"role": "user", "content": q}], 1024, True, desk))
    elif w == "skill":
        q = (nonce() + f"Use the weekly-report skill for this: the notes are in {desk.root}/notes.")
        msgs = [{"role": "user", "content": q}]
        if not isinstance(eng, Vireo):
            # Vireo puts the named skill's instructions in for itself; the others get the same text.
            skill = (SKILLS / "weekly-report/SKILL.md").read_text().split("---", 2)[2].strip()
            msgs = [{"role": "user", "content": q + "\n\nThe skill's instructions:\n\n" + skill}]
        r = eng.chat(msgs, 1536, True, desk)
        r["wrote"] = list(desk.written.values())
        rows.append(r)
    elif w == "longtools":
        q = (nonce() + "Here is yesterday's point-of-sale incident log:\n\n<log>\n" + prompt("incident-log") +
             f"\n</log>\n\nWhich component logged the most ERROR lines? Then check our folder {desk.root} "
             "(the README and suppliers.md) and tell me whether any supplier is involved with that "
             "component. Keep the answer short.")
        rows.append(eng.chat([{"role": "user", "content": q}], 512, True, desk))
    shutil.rmtree(desk.root.parent, ignore_errors=True)
    return rows


def record(**kw):
    kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    line = json.dumps(kw)
    print(line[:400], flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def start(kind, model):
    if kind == "vireo":
        return Vireo(model, plain=False)
    if kind == "vireo-plain":
        return Vireo(model, plain=True)
    if kind == "llama":
        return Llama(model)
    return Mlx(model)


def block(kind, model, rep, workloads):
    name, key = model[0], model[1]
    try:
        eng = start(kind, model)
    except Exception as e:
        record(engine=kind, model=name, rep=rep, error=f"start: {e!r}")
        return
    mem = MemWatch(eng.pid)
    try:
        time.sleep(2)
        idle_rss, idle_fp, idle_total = mem.reset()
        record(engine=kind, model=name, rep=rep, work="load", load_s=round(eng.load_s, 2),
               idle_rss=idle_rss, idle_fp=idle_fp, idle_total=idle_total)
        # Warm-up: kernels compiled, weights paged in.
        try:
            eng.chat([{"role": "user", "content": nonce() + "Say hello in five words."}], 16, False, Desk())
        except Exception as e:
            record(engine=kind, model=name, rep=rep, work="warmup", error=repr(e))
        for w in workloads:
            if w in ("large", "xlarge") and not model[5]:
                continue
            mem.reset()
            t = time.time()
            try:
                rows = run_workload(eng, w, mem)
                for r in rows:
                    r.pop("text_full", None)
                    r["text"] = (r.get("text") or "")[:600]
                time.sleep(2.2)
                record(engine=kind, model=name, rep=rep, work=w, wall=round(time.time() - t, 2),
                       peak_rss=mem.peak_rss, peak_fp=mem.peak_fp, peak_total=mem.peak_total, rows=rows)
            except Exception as e:
                record(engine=kind, model=name, rep=rep, work=w, error=repr(e)[:500])
                if eng.proc.poll() is not None:
                    break
    finally:
        mem.stop()
        eng.stop()


def main():
    engines = os.environ.get("ENGINES", "vireo,vireo-plain,llama,mlx").split(",")
    keys = os.environ.get("MODELS", ",".join(m[1] for m in MODELS)).split(",")
    reps = int(os.environ.get("REPS", "2"))
    workloads = os.environ.get("WORK", ",".join(WORKLOADS)).split(",")
    for key in keys:
        model = next(m for m in MODELS if m[1] == key)
        for rep in range(reps):
            order = engines if rep % 2 == 0 else engines[::-1]
            for kind in order:
                block(kind, model, rep, workloads)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
