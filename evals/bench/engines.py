"""The three engines behind one interface, through their own servers (streaming, so the
time to the first token is what a client sees), and their command-line tools for the raw
tests. Based on evals/speed/bench-2026-11-13/bench.py's drivers.

  start(kind, model, ctx, thinking=False, slots=1) -> a server with .chat(), .complete(), .stop()
    kind: vireo | vireo-plain | llama | mlx
  cli_generate / llama_bench / perplexity: S1 and S6, no server.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
from pathlib import Path

import common
from common import TOKEN, post, sse, stop_proc, wait_health
from desk import OPENAI_TOOLS, VIREO_CLIENT_TOOLS

SYSTEM = "You are a helpful assistant for Harbor Bikes, a small bike shop. Be concise and accurate."
LOGS = Path(tempfile.gettempdir()) / "vireo-bench-logs"
LOGS.mkdir(parents=True, exist_ok=True)
PORTS = {"llama": 18462, "mlx": 18463}


def engine_cfg(name):
    return common.config()["platforms"][common.PLATFORM]["engines"].get(name)


def totals(r):
    """A reply's reading and writing, summed over its steps (a tool loop has several)."""
    st = r.get("steps") or []
    read = sum(s.get("read") or 0 for s in st)
    read_s = sum(s.get("read_s") or 0 for s in st)
    gen = sum(s.get("gen") or 0 for s in st)
    gen_s = sum(s.get("gen_s") or 0 for s in st)
    r["prompt_tokens"] = st[-1].get("prompt") if st else None
    r["cached_tokens"] = st[-1].get("cached") if st else None
    r["read_tokens"], r["gen_tokens"] = read, gen
    r["read_tps"] = round(read / read_s, 1) if read and read_s else None
    r["gen_tps"] = round(gen / gen_s, 2) if gen and gen_s else None
    return r


class Vireo:
    REPLY = re.compile(r"Reply: (\d+) prompt tokens \((\d+) kept, (\d+) read in ([\d.]+)s, \d+/s\), "
                       r"(\d+) tokens in ([\d.]+)s")

    def __init__(self, model, ctx, plain=False, thinking=False, extra=()):
        cfg = engine_cfg("vireo")
        self.kind = "vireo-plain" if plain else "vireo"
        self.thinking = thinking
        self.dir = Path(tempfile.mkdtemp(prefix="vbench-vireo-"))
        (self.dir / "builtin").mkdir()
        self.log_path = LOGS / f"{self.kind}-{model['key']}-{int(time.time())}.log"
        self.log = open(self.log_path, "w", encoding="utf-8")
        args = [str(common.exe(common.path(cfg["server"]))), str(common.path(model["gguf"])), "--port", "0",
                "--token", TOKEN, "--chat-cache", str(self.dir / "chat-cache"), "--context", str(ctx),
                "--thinking", "on" if thinking else "off"]
        if plain:
            args += ["--mtp", "off", "--lookup", "0"]
        args += list(extra)
        env = dict(os.environ, INVURSE_SKILLS_DIR=str(common.FIX / "skills"),
                   VIREO_BUILTIN_SKILLS=str(self.dir / "builtin"),
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
        self.lock = threading.Lock()

    def new_log(self):
        self.log.flush()
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        new, self.read_to = text[self.read_to:], len(text)
        return new

    def chat(self, messages, max_tokens, tools=False, desk=None, system=SYSTEM, log=True, chat_id=None):
        if log:
            self.new_log()
        body = {"systemPrompt": system, "messages": messages, "thinking": self.thinking,
                "generation": {"max_tokens": max_tokens, "temperature": 0}}
        if chat_id:
            body["chatId"] = chat_id
        if tools:
            body["clientTools"] = VIREO_CLIENT_TOOLS
        t0 = time.time()
        first, text, calls, err = None, "", [], None
        try:
            for ev in sse(post(f"http://127.0.0.1:{self.port}/api/chat/completions", body, TOKEN)):
                typ = ev.get("type")
                if "content" in ev and typ is None:
                    if ev["content"]:
                        first = first or time.time() - t0
                    text += ev["content"]
                elif typ == "thinking":
                    first = first or time.time() - t0
                elif typ == "tool_call":
                    first = first or time.time() - t0
                    calls.append(ev.get("tool"))
                elif typ == "client_tool_request":
                    ok, out = desk.run(ev.get("tool", ""), ev.get("args") or {})
                    if ev.get("tool") not in calls[-1:]:
                        calls.append(ev.get("tool"))
                    ans = {"id": ev["id"], "ok": ok, ("output" if ok else "error"): out}
                    post(f"http://127.0.0.1:{self.port}/api/agent/client-tool-result", ans, TOKEN).read()
                elif typ == "confirm_required":
                    approve = not (desk and desk.deny_writes)
                    post(f"http://127.0.0.1:{self.port}/api/agent/confirm",
                         {"id": ev["id"], "approved": approve, "remember": False}, TOKEN).read()
                elif "error" in ev and typ is None:
                    err = str(ev["error"])[:300]
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")[:300]
        except Exception as e:
            err = repr(e)[:300]
        total = time.time() - t0
        steps = []
        if log:
            time.sleep(0.1)
            for m in self.REPLY.finditer(self.new_log()):
                steps.append({"prompt": int(m[1]), "cached": int(m[2]), "read": int(m[3]), "read_s": float(m[4]),
                              "gen": int(m[5]), "gen_s": float(m[6])})
        return totals({"ttft": first, "total": total, "text": text, "calls": calls, "error": err, "steps": steps})

    def stop(self):
        stop_proc(self.proc)
        self.log.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class OpenAIish:
    """llama-server or mlx-vlm's server: the tool loop runs here, a request per step."""
    kind = ""
    port = 0
    thinking = False

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
        except Exception as e:
            err = repr(e)[:300]
        step = {"ttft": first, "total": time.time() - t0}
        if timings:
            step.update(read=timings.get("prompt_n"), cached=timings.get("cache_n"),
                        read_s=(timings.get("prompt_ms") or 0) / 1000, gen=timings.get("predicted_n"),
                        gen_s=(timings.get("predicted_ms") or 0) / 1000)
            step["prompt"] = (step["read"] or 0) + (step["cached"] or 0)
        elif usage:
            step["prompt"] = usage.get("prompt_tokens")
            step["gen"] = usage.get("completion_tokens")
        return text, [tcs[k] for k in sorted(tcs)], step, err

    def chat(self, messages, max_tokens, tools=False, desk=None, system=SYSTEM, log=True):
        msgs = [{"role": "system", "content": system}] + messages
        t0 = time.time()
        first, calls, steps, err, text = None, [], [], None, ""
        for _ in range(20):
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
                ok, out = desk.run(tc["name"], args) if desk else (False, "no tools here")
                msgs.append({"role": "tool", "tool_call_id": tc["id"] or f"call_{i}", "content": out if ok else f"Error: {out}"})
        return totals({"ttft": first, "total": time.time() - t0, "text": text, "calls": calls, "error": err, "steps": steps})


class Llama(OpenAIish):
    kind = "llama"

    def __init__(self, model, ctx, thinking=False, slots=1):
        folder = common.path(engine_cfg("llama")["dir"])
        self.thinking = thinking
        self.port = PORTS["llama"]
        self.log_path = LOGS / f"llama-{model['key']}-{int(time.time())}.log"
        self.log = open(self.log_path, "w")
        args = [str(common.exe(folder / "llama-server")), "-m", str(common.path(model["gguf"])),
                "-c", str(ctx * slots), "-ngl", "99", "-fa", "on", "--port", str(self.port), "--host", "127.0.0.1",
                "-np", str(slots), "--jinja", "--no-webui"]
        if not thinking:
            args += ["--reasoning-budget", "0"]
        t = time.time()
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=self.log, env=common.llama_env(folder))
        wait_health(f"http://127.0.0.1:{self.port}/health", self.proc)
        self.load_s = time.time() - t
        self.pid = self.proc.pid

    def request_extra(self):
        return {"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": self.thinking}}

    def complete(self, prompt, n):
        """Raw continuation, greedy (S6 greedy-match)."""
        body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False}
        r = json.loads(post(f"http://127.0.0.1:{self.port}/completion", body).read())
        return r.get("content", "")

    def stop(self):
        stop_proc(self.proc)
        self.log.close()


class Mlx(OpenAIish):
    kind = "mlx"

    def __init__(self, model, ctx=None, thinking=False):
        cfg = engine_cfg("mlx")
        if not cfg or not model.get("mlx"):
            raise RuntimeError("no mlx-vlm on this platform / for this model")
        self.thinking = thinking
        self.port = PORTS["mlx"]
        self.model = str(common.path(model["mlx"]))
        self.log_path = LOGS / f"mlx-{model['key']}-{int(time.time())}.log"
        self.log = open(self.log_path, "w")
        # Its automatic prefix cache on (off by default), as shipped Vireo 0.9 ran it; a long
        # read needs more than the 600 s token-queue default.
        env = dict(os.environ, APC_ENABLED="1", MLX_VLM_TOKEN_QUEUE_TIMEOUT="7200")
        t = time.time()
        self.proc = subprocess.Popen([str(common.path(cfg["python"])), "-m", "mlx_vlm.server", "--model", self.model,
                                      "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "WARNING"],
                                     stdout=self.log, stderr=self.log, env=env, cwd=tempfile.gettempdir())
        wait_health(f"http://127.0.0.1:{self.port}/health", self.proc)
        self.load_s = time.time() - t
        self.pid = self.proc.pid

    def request_extra(self):
        return {"model": self.model, "enable_thinking": self.thinking}

    def stop(self):
        stop_proc(self.proc)
        self.log.close()


def start(kind, model, ctx, thinking=False, slots=1):
    if kind in ("vireo", "vireo-plain"):
        return Vireo(model, ctx, plain=kind == "vireo-plain", thinking=thinking)
    if kind == "llama":
        return Llama(model, ctx, thinking, slots)
    if kind == "mlx":
        return Mlx(model, ctx, thinking)
    raise ValueError(kind)


def available(kind, model):
    if kind == "mlx":
        return bool(engine_cfg("mlx")) and bool(model.get("mlx")) and common.path(model["mlx"]).exists()
    return engine_cfg(kind.replace("-plain", "")) is not None and common.path(model["gguf"]).exists()


# ---- Command-line tools (no server) -----------------------------------------------------

GEN = re.compile(r"Prompt: (\d+) tokens in .*?\(([\d.]+) tokens/s\)\. Generated: (\d+) tokens in .*?\(([\d.]+) tokens/s\)")


def cli_generate(model, prompt_text, n):
    """vireo-cli generate --gpu: greedy, one token at a time. Returns tokens and speeds."""
    cli = common.exe(common.path(engine_cfg("vireo")["cli"]))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(prompt_text)
    try:
        p = subprocess.run([str(cli), "generate", str(common.path(model["gguf"])), f.name, "--file", "--gpu", "-n", str(n)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    finally:
        os.unlink(f.name)
    m = GEN.search(p.stderr)
    if not m:
        raise RuntimeError(f"vireo-cli: {p.stderr[-400:]}")
    return {"prompt_tokens": int(m[1]), "pp_tps": float(m[2]), "gen_tokens": int(m[3]), "tg_tps": float(m[4]),
            "text": p.stdout.rstrip("\n")}


BATCHES = re.compile(r"(\d+): ([\d.]+)")


def cli_batches(model, text_file, depth=4000):
    """vireo-cli batches: milliseconds a batch of 1 to 64 tokens, `depth` tokens into a chat."""
    cli = common.exe(common.path(engine_cfg("vireo")["cli"]))
    p = subprocess.run([str(cli), "batches", str(common.path(model["gguf"])), str(text_file), "--depth", str(depth)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("Batches at")), None)
    if not line:
        raise RuntimeError(f"vireo-cli batches: {p.stderr[-400:]}")
    return {int(n): float(ms) for n, ms in BATCHES.findall(line.split("ms:", 1)[1])}


def llama_bench(model, args):
    folder = common.path(engine_cfg("llama")["dir"])
    p = subprocess.run([str(common.exe(folder / "llama-bench")), "-m", str(common.path(model["gguf"])), "-ngl", "99",
                        "-fa", "1", "-o", "json"] + args, capture_output=True, text=True, env=common.llama_env(folder),
                       timeout=7200)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"llama-bench: {p.stderr[-400:]}")


PPL = re.compile(r"Final estimate: PPL = ([\d.]+)")


def perplexity(kind, model, text_file, ctx=512, chunks=40):
    if kind == "vireo":
        args = [str(common.exe(common.path(engine_cfg("vireo")["cli"]))), "perplexity", str(common.path(model["gguf"])),
                str(text_file), "-c", str(ctx), "--chunks", str(chunks), "--gpu"]
        env = None
    else:
        folder = common.path(engine_cfg("llama")["dir"])
        args = [str(common.exe(folder / "llama-perplexity")), "-m", str(common.path(model["gguf"])), "-f", str(text_file),
                "-c", str(ctx), "--chunks", str(chunks), "-ngl", "99", "-fa", "on"]
        env = common.llama_env(folder)
    t = time.time()
    p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=7200)
    m = PPL.search(p.stdout + p.stderr)
    if not m:
        raise RuntimeError(f"{kind} perplexity: {(p.stdout + p.stderr)[-400:]}")
    return {"ppl": float(m[1]), "seconds": round(time.time() - t, 1)}
