"""Profile vireo-server answering one prompt under Nsight Systems.
usage: prof_server.py <repo> <model> <out.nsys-rep> <prompt-name> <max_tokens> [server args...]"""
import sys, subprocess, time, json, os, tempfile, urllib.request
repo, model, out, prompt, max_tokens, *extra = sys.argv[1:]
sys.path.insert(0, os.path.join(repo, "tools"))
import long_run, reply_speed
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.2\target-windows-x64\nsys.exe"
os.chdir(repo)
d = tempfile.mkdtemp(prefix="vireo-prof-")
port = 18555
log_path = os.path.join(d, "server.log")
log = open(log_path, "w")
env = dict(os.environ, VIREO_MEMORY_FILE=os.path.join(d, "memories.txt"))
p = subprocess.Popen([NSYS, "profile", "--trace=cuda", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
                      "--duration=" + os.environ.get("PROF_SECONDS", "30"), "--kill=true", "-o", out,
                      "--force-overwrite", "true", os.path.join(repo, "target", "release", "vireo-server.exe"), model,
                      "--port", str(port), "--token", "prof", "--chat-cache", os.path.join(d, "cache")] + extra,
                     stdout=log, stderr=log, env=env)
class S: pass
server = S(); server.port = port; server.token = "prof"; server.REPLY = long_run.Vireo.REPLY
server.new_log = lambda: open(log_path, errors="replace").read()
for _ in range(120):
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/metrics",
                               headers={"Authorization": "Bearer prof"}), timeout=2).read()
        break
    except Exception:
        time.sleep(0.25)
else:
    raise SystemExit("no server: " + open(log_path, errors="replace").read()[-500:])
for i in range(int(os.environ.get("PROF_ASKS", "1"))):
    t = time.time()
    row = reply_speed.ask(server, reply_speed.PROMPTS[prompt], int(max_tokens), True)
    print("asked in %.2fs" % (time.time() - t), len(row["reply"]))
print("waiting for nsys")
p.wait(timeout=300)
print(open(log_path, errors="replace").read()[-900:])
