"""Windows speed run: Vireo's engine against llama.cpp (CUDA build) on the same GGUF files,
the way prompts-mac/bench_mac.py does it on the Mac, with the same three prompt files.

Two parts, both written to <out>/results.jsonl a line a measurement:

  raw      Reading and writing with no drafting, one token at a time, at three context
           sizes: prompts-mac/short.txt (34 tokens, 128 written), medium.txt (~2,300, 64)
           and long.txt (27-34K by tokenizer, 64). Vireo: `vireo-cli generate --gpu`.
           llama.cpp: `llama-server -ngl 99 -fa on`, the prompt to /completion with
           cache_prompt off. One engine and model loaded at a time; with more than one
           round (--reps) the engines' order is reversed each round.
  replies  Whole replies through each engine's server, as an app gets them: vireo-server
           at its defaults (the chat's guesses, a draft model where it has one) against
           llama-server, the same user message to each (tools/reply_speed.py's four), at
           temperature 0, 400 tokens at most. Each server's GPU memory is read
           once its rounds are done (never during one: it moves the clocks).

Usage, from VireoPrototype:
    python evals/speed/bench_windows.py --out evals/speed/bench-2026-11-15-windows [name filter...]
Standard library only. Measure on mains in the fastest power mode, and say which."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import long_run  # noqa: E402
import reply_speed  # noqa: E402

PROMPTS = REPO / "evals/speed/prompts-mac"
VIREO = REPO / "target/release/vireo-cli.exe"
LLAMA = REPO / "reference/llama.cpp-b12262/llama-server.exe"
# The Gemma and Qwen models, one round each, unless asked otherwise (--reps, --with-llama).
LLAMA_MODEL = ("Llama 3.2 1B", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
MODELS = [
    ("Qwen 3.5 0.8B", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    ("Gemma 4 E2B", "gemma-4-E2B_q4_0-it.gguf"),
    ("Gemma 4 E4B", "gemma-4-E4B_q4_0-it.gguf"),
    ("Qwen 3.5 4B", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen 3.5 9B", "Qwen3.5-9B-Q4_K_M.gguf"),
]
# The models an app would chat with: the replies part.
CHAT = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen 3.5 4B", "Qwen 3.5 9B"]
WORK = [("short", 128), ("medium", 64), ("long", 64)]
PORT = 8090
CLI_LINE = re.compile(r"Prompt: (\d+) tokens in .*?\(([\d.]+) tokens/s\)\. "
                      r"Generated: (\d+) tokens in .*?\(([\d.]+) tokens/s\)")


def post(url, body, timeout=1800):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def gpu_memory_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else None


class Run:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.results = self.out / "results.jsonl"

    def record(self, **row):
        row["time"] = time.strftime("%H:%M:%S")
        print(json.dumps(row), flush=True)
        with open(self.results, "a") as f:
            f.write(json.dumps(row) + "\n")

    def llama_server(self, gguf, context):
        log = open(self.out / "llama-server.log", "a")
        proc = subprocess.Popen([str(LLAMA), "-m", str(gguf), "-ngl", "99", "-fa", "on", "-c", str(context),
                                 "--port", str(PORT), "--host", "127.0.0.1", "-np", "1", "--no-webui"],
                                stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while time.time() - started < 300:
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited; see llama-server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        return proc
            except Exception:
                pass
            time.sleep(1)
        proc.kill()
        raise RuntimeError("llama-server didn't come up")

    # ---- raw: no drafting, three context sizes ---------------------------------------

    def raw_vireo(self, name, gguf, rep):
        subprocess.run([str(VIREO), "generate", str(gguf), str(PROMPTS / "medium.txt"), "--file", "--gpu", "-n", "8"],
                       capture_output=True)
        for work, n in WORK:
            p = subprocess.run([str(VIREO), "generate", str(gguf), str(PROMPTS / f"{work}.txt"), "--file", "--gpu",
                                "-n", str(n)], capture_output=True, text=True, errors="replace")
            m = CLI_LINE.search(p.stderr + p.stdout)
            if not m:
                self.record(part="raw", engine="vireo", model=name, work=work, rep=rep,
                            error=(p.stderr + p.stdout)[-600:])
                continue
            self.record(part="raw", engine="vireo", model=name, work=work, rep=rep, prompt_n=int(m[1]),
                        prompt_tps=float(m[2]), gen_n=int(m[3]), gen_tps=float(m[4]))

    def raw_llama(self, name, gguf, rep):
        proc = self.llama_server(gguf, 36000)
        try:
            url = f"http://127.0.0.1:{PORT}/completion"
            post(url, {"prompt": (PROMPTS / "medium.txt").read_text(encoding="utf-8"), "n_predict": 8,
                       "cache_prompt": False})
            for work, n in WORK:
                try:
                    r = post(url, {"prompt": (PROMPTS / f"{work}.txt").read_text(encoding="utf-8"), "n_predict": n,
                                   "temperature": 0, "top_k": 1, "cache_prompt": False})
                    t = r["timings"]
                    self.record(part="raw", engine="llama", model=name, work=work, rep=rep, prompt_n=t["prompt_n"],
                                prompt_tps=t["prompt_per_second"], gen_n=t["predicted_n"],
                                gen_tps=t["predicted_per_second"])
                except Exception as e:
                    self.record(part="raw", engine="llama", model=name, work=work, rep=rep, error=repr(e))
            self.record(part="memory", engine="llama", model=name, rep=rep, after="raw, 36,000-token context",
                        gpu_mb=gpu_memory_mb())
        finally:
            proc.terminate()
            proc.wait()
            time.sleep(3)

    # ---- replies: through the servers, as shipped ------------------------------------

    def replies_vireo(self, name, gguf, reps):
        server = long_run.Vireo(str(gguf), 0, 400, [])
        try:
            reply_speed.ask(server, "Say hello.", 16, True)
            for rep in range(reps):
                for prompt, text in reply_speed.PROMPTS.items():
                    row = reply_speed.ask(server, text, 400, True)
                    written, seconds = row.get("written", 0), row.get("write_s", 0.0)
                    read, read_s = row.get("read", 0), row.get("read_s", 0.0)
                    taken = sum(g[0] for g in row["guesses"].values())
                    self.record(part="replies", engine="vireo", model=name, work=prompt, rep=rep,
                                prompt_n=read, prompt_tps=round(read / read_s, 1) if read_s else None,
                                gen_n=written, gen_tps=round(written / seconds, 1) if seconds else None,
                                guessed_share=round(taken / max(written, 1), 3), guesses=row["guesses"])
            self.record(part="memory", engine="vireo", model=name, after="replies, the server's own context",
                        context=server.context, gpu_mb=gpu_memory_mb())
        finally:
            server.stop()
            time.sleep(3)

    def replies_llama(self, name, gguf, reps):
        proc = self.llama_server(gguf, 8192)
        try:
            url = f"http://127.0.0.1:{PORT}/v1/chat/completions"

            def ask(text, n):
                return post(url, {"messages": [{"role": "user", "content": text}], "max_tokens": n,
                                  "temperature": 0, "cache_prompt": False})

            ask("Say hello.", 16)
            for rep in range(reps):
                for prompt, text in reply_speed.PROMPTS.items():
                    t = ask(text, 400)["timings"]
                    self.record(part="replies", engine="llama", model=name, work=prompt, rep=rep,
                                prompt_n=t["prompt_n"], prompt_tps=round(t["prompt_per_second"], 1),
                                gen_n=t["predicted_n"], gen_tps=round(t["predicted_per_second"], 1))
            self.record(part="memory", engine="llama", model=name, after="replies, 8,192-token context",
                        gpu_mb=gpu_memory_mb())
        finally:
            proc.terminate()
            proc.wait()
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--parts", default="raw,replies")
    parser.add_argument("--reps", type=int, default=int(os.environ.get("REPS", "1")))
    parser.add_argument("--with-llama", action="store_true", help="Llama 3.2 1B too")
    parser.add_argument("only", nargs="*", help="model name filters")
    args = parser.parse_args()
    run = Run(args.out)
    parts = args.parts.split(",")
    for name, file in ([LLAMA_MODEL] if args.with_llama else []) + MODELS:
        if args.only and not any(o.lower() in name.lower() for o in args.only):
            continue
        gguf = REPO / "models" / file
        if "raw" in parts:
            blocks = [("vireo", run.raw_vireo), ("llama", run.raw_llama)]
            for rep in range(args.reps):
                for engine, block in (blocks if rep % 2 == 0 else blocks[::-1]):
                    try:
                        block(name, gguf, rep)
                    except Exception as e:
                        run.record(part="raw", engine=engine, model=name, rep=rep, error=repr(e))
        if "replies" in parts and name in CHAT:
            for engine, block in (("vireo", run.replies_vireo), ("llama", run.replies_llama)):
                try:
                    block(name, gguf, args.reps)
                except Exception as e:
                    run.record(part="replies", engine=engine, model=name, error=repr(e))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
