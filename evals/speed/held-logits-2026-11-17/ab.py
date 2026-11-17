"""Alternate two vireo-server builds on the same prompts. usage: ab.py <repo> <model> <rounds> <prompts,comma> <name=exe>... [-- server args]"""
import sys, subprocess, time, os, tempfile, urllib.request, re, json, zlib
repo, model, rounds, prompts, *rest = sys.argv[1:]
extra = []
if "--" in rest:
    i = rest.index("--"); rest, extra = rest[:i], rest[i + 1:]
ways = [w.split("=", 1) for w in rest]
sys.path.insert(0, os.path.join(repo, "tools"))
import long_run, reply_speed
os.chdir(repo)
GREEDY = os.environ.get("AB_SAMPLING", "greedy") == "greedy"
MAXTOK = int(os.environ.get("AB_TOKENS", "400"))
class Server:
    REPLY = long_run.Vireo.REPLY
    def __init__(self, exe, port):
        self.dir = tempfile.mkdtemp(prefix="vireo-ab-")
        self.log_path = os.path.join(self.dir, "server.log"); self.log = open(self.log_path, "w")
        self.port, self.token, self.read_to = port, "ab", 0
        env = dict(os.environ, VIREO_MEMORY_FILE=os.path.join(self.dir, "memories.txt"))
        self.p = subprocess.Popen([exe, model, "--port", str(port), "--token", "ab", "--chat-cache",
                                   os.path.join(self.dir, "cache")] + extra, stdout=subprocess.DEVNULL, stderr=self.log, env=env)
        for _ in range(240):
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/metrics",
                                       headers={"Authorization": "Bearer ab"}), timeout=2).read(); return
            except Exception: time.sleep(0.25)
        raise SystemExit("no server")
    def new_log(self):
        self.log.flush(); text = open(self.log_path, errors="replace").read()
        new, self.read_to = text[self.read_to:], len(text); return new
    def stop(self):
        self.p.terminate(); self.p.wait(timeout=60)
results = {}
for r in range(int(rounds)):
    order = ways if r % 2 == 0 else ways[::-1]
    for name, exe in order:
        s = Server(exe, 18560 + r)
        reply_speed.ask(s, "Say hello in five words.", 16, True)  # warm up: graphs, pages
        for prompt in prompts.split(","):
            row = reply_speed.ask(s, reply_speed.PROMPTS[prompt], MAXTOK, GREEDY)
            speed = row["written"] / row["write_s"]
            g = row.get("guesses", {})
            results.setdefault((prompt, name), []).append(speed)
            print(f"round {r} {name:5s} {prompt:8s} {speed:7.1f} tok/s  {row['written']} tokens  guesses {g}  reply {zlib.crc32(row["reply"].encode()) & 0xffff:04x}", flush=True)
        s.stop(); time.sleep(1)
for prompt in prompts.split(","):
    line = f"{prompt:8s}"
    vals = {}
    for name, _ in ways:
        v = results[(prompt, name)]; vals[name] = sum(v) / len(v)
        line += f"  {name} {vals[name]:7.1f} ({', '.join('%.1f' % x for x in v)})"
    names = [n for n, _ in ways]
    if len(names) == 2: line += f"  {100 * (vals[names[1]] / vals[names[0]] - 1):+.1f}%"
    print(line)
