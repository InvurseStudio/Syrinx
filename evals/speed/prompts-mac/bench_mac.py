"""Mac speed run: Vireo core (vireo-cli) vs llama.cpp b12228 (llama-server) vs shipped Vireo 0.9.0
(its bundled mlx_vlm server). Same prompt files; one token at a time (no drafting).

Usage (from anywhere): mount the 0.9.0 DMG read-only, then
    VIREO09_MOUNT=/path/to/mount OUT_DIR=/some/scratch/dir python3 -I bench_mac.py [model name filter...]
Results go to $OUT_DIR/results.jsonl, server logs beside them. See ../README.md."""
import json, os, re, subprocess, sys, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRATCH = os.environ["OUT_DIR"]
RES = f"{os.environ['VIREO09_MOUNT']}/Vireo.app/Contents/Resources"
MODELS = os.path.expanduser("~/Library/Application Support/InvurseChat/Models")
PROMPTS = f"{REPO}/evals/speed/prompts-mac"
OUT = f"{SCRATCH}/results.jsonl"
VIREO = f"{REPO}/target/release/vireo-cli"
LLAMA = f"{REPO}/reference/llama-b12228/llama-server"

SETS = [
    ("Gemma 4 31B", "gemma-4-31B-it-GGUF/gemma-4-31B-it-Q6_K.gguf", "gemma-4-31B-it-MLX-6bit"),
    ("Qwen 3.8 27B", "Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf", "Qwen3.8-27B-MLX-6bit"),
    ("Muse Glimmer 30B", "Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-Q6_K.gguf", "Muse-Glimmer-30B-8bit"),
]
WORK = [("short", 128), ("medium", 64), ("long", 64)]
REPS = int(os.environ.get("REPS", "2"))
only = sys.argv[1:]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait(url, proc, secs=600):
    t = time.time()
    while time.time() - t < secs:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited: {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"timeout {url}")


def record(**kw):
    kw["time"] = time.strftime("%H:%M:%S")
    print(json.dumps(kw), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(kw) + "\n")


def run_vireo(name, gguf, work, n):
    p = subprocess.run([VIREO, "generate", gguf, f"{PROMPTS}/{work}.txt", "--file", "--gpu", "-n", str(n)],
                       capture_output=True, text=True)
    m = re.search(r"Prompt: (\d+) tokens in .*?\(([\d.]+) tokens/s\)\. Generated: (\d+) tokens in .*?\(([\d.]+) tokens/s\)", p.stderr)
    if not m:
        record(engine="vireo", model=name, work=work, error=p.stderr[-800:])
        return
    record(engine="vireo", model=name, work=work, prompt_n=int(m[1]), prompt_tps=float(m[2]),
           gen_n=int(m[3]), gen_tps=float(m[4]))


def run_llama(name, work, n):
    text = open(f"{PROMPTS}/{work}.txt").read()
    r = post("http://127.0.0.1:8090/completion", {"prompt": text, "n_predict": n, "temperature": 0,
                                                   "top_k": 1, "cache_prompt": False})
    t = r["timings"]
    record(engine="llama", model=name, work=work, prompt_n=t["prompt_n"], prompt_tps=t["prompt_per_second"],
           gen_n=t["predicted_n"], gen_tps=t["predicted_per_second"])


def run_shipped(name, mlx, work, n):
    text = fresh(open(f"{PROMPTS}/{work}.txt").read())
    r = post("http://127.0.0.1:8091/v1/chat/completions",
             {"model": f"{MODELS}/{mlx}", "messages": [{"role": "user", "content": text}],
              "max_tokens": n, "temperature": 0})
    t = r["timings"]
    record(engine="shipped", model=name, work=work, prompt_n=t["prompt_n"], cache_n=t.get("cache_n"),
           prompt_tps=t["prompt_per_second"], gen_n=t["predicted_n"], gen_tps=t["predicted_per_second"],
           reasoning=bool(r["choices"][0]["message"].get("reasoning_content")))


import uuid


def fresh(text):
    # A first line no earlier request had, so the shipped server's prompt cache can't help.
    return f"Request {uuid.uuid4().hex[:8]}.\n\n{text}"


def stop(proc):
    proc.terminate(); proc.wait(); time.sleep(5)


def block_vireo(name, gguf, mlx):
    subprocess.run([VIREO, "generate", gguf, f"{PROMPTS}/medium.txt", "--file", "--gpu", "-n", "8"], capture_output=True)
    for work, n in WORK:
        run_vireo(name, gguf, work, n)


def block_llama(name, gguf, mlx):
    proc = subprocess.Popen([LLAMA, "-m", gguf, "-ngl", "99", "-fa", "on", "-c", "34000", "--port", "8090",
                             "--host", "127.0.0.1", "-np", "1", "--no-webui"],
                            stdout=open(f"{SCRATCH}/llama-{mlx}.log", "a"), stderr=subprocess.STDOUT)
    try:
        wait("http://127.0.0.1:8090/health", proc)
        post("http://127.0.0.1:8090/completion", {"prompt": open(f"{PROMPTS}/medium.txt").read(), "n_predict": 8, "cache_prompt": False})
        for work, n in WORK:
            run_llama(name, work, n)
    finally:
        stop(proc)


def block_shipped(name, gguf, mlx):
    env = dict(os.environ, APC_ENABLED="1", APC_NUM_BLOCKS="8192", APC_CHECKPOINT_INTERVAL_TOKENS="256",
               APC_CHECKPOINT_ENTRIES="8", MLX_VLM_MAX_TOKENS="4096")
    proc = subprocess.Popen([f"{RES}/python/bin/python3", f"{RES}/server/mlx_server.py", "--model", f"{MODELS}/{mlx}",
                             "--port", "8091", "--host", "127.0.0.1"], env=env,
                            stdout=open(f"{SCRATCH}/mlx-{mlx}.log", "a"), stderr=subprocess.STDOUT)
    try:
        wait("http://127.0.0.1:8091/health", proc)
        post("http://127.0.0.1:8091/v1/chat/completions", {"model": f"{MODELS}/{mlx}", "messages": [{"role": "user", "content": fresh(open(f"{PROMPTS}/medium.txt").read())}], "max_tokens": 8, "temperature": 0})
        for work, n in WORK:
            run_shipped(name, mlx, work, n)
    finally:
        stop(proc)


BLOCKS = [("vireo", block_vireo), ("llama", block_llama), ("shipped", block_shipped)]
for name, gguf_rel, mlx in SETS:
    if only and not any(o.lower() in name.lower() for o in only):
        continue
    gguf = f"{MODELS}/{gguf_rel}"
    for rep in range(REPS):
        for eng, fn in (BLOCKS if rep % 2 == 0 else BLOCKS[::-1]):
            try:
                fn(name, gguf, mlx)
            except Exception as e:
                record(engine=eng, model=name, rep=rep, error=repr(e))
print("DONE", flush=True)
