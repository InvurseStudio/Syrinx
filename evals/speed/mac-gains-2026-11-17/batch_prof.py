import subprocess, sys, re, os
model=sys.argv[1]; counts=sys.argv[2:]
def run(c):
    env=dict(os.environ,VIREO_CHECK_MODEL=model,VIREO_CHECK_TEXT=os.getcwd()+"/evals/speed/prompts-mac/long.txt",VIREO_CHECK_DEPTHS="4000",VIREO_CHECK_COUNTS=c,VIREO_METAL_PROFILE="1")
    out=subprocess.run(["cargo","test","--release","-p","vireo-engine","--test","check_speed","--","--ignored","--nocapture"],capture_output=True,text=True,env=env)
    d={}
    for l in (out.stdout+out.stderr).splitlines():
        m=re.match(r"\s+([\d.]+) ms\s+[\d.]+%\s+(\d+) ×\s+[\d.]+ µs\s+(\S+)",l)
        if m: d[m.group(3)]=(float(m.group(1)),int(m.group(2)))
    if not d: print(out.stdout[-2000:],out.stderr[-2000:])
    return d
base=run("")
for c in counts:
    b=run(c); rows=[]; tot=0
    for name in b:
        ms=(b[name][0]-base.get(name,(0,0))[0])/6; n=(b[name][1]-base.get(name,(0,0))[1])/6
        if n>0.01: rows.append((ms,n,name)); tot+=ms
    print(f"--- batch of {c}: {tot:.1f} ms of kernels")
    for ms,n,name in sorted(rows,reverse=True)[:14]: print(f"{ms:8.2f} ms {n:7.1f} × {ms*1000/n:8.1f} µs  {name}")
