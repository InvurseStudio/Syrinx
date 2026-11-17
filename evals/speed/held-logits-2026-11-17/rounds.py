import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
m = [(s, e, "memcpy%d:%d" % (ck, b)) for s, e, ck, b in db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
ev = sorted(k + m)
last_prompt = max((i for i, x in enumerate(ev) if x[2].startswith("matmul8_geglu_t") or x[2].startswith("matmul8_swiglu_t") or x[2].startswith("matmul8_t128")), default=0)
ev = ev[last_prompt + 1:]
start = int(sys.argv[2]); n = int(sys.argv[3])
t0 = ev[start][0]; run = None
out = []
i = start
prev_end = ev[start][0]
seg_start = ev[start][0]; seg_names = {}
def flush(end):
    global seg_names, seg_start
    if seg_names:
        top = sorted(seg_names.items(), key=lambda x: -x[1])[:3]
        out.append(f"{(seg_start-t0)/1e3:9.0f}us  busy {(end-seg_start)/1e3:8.0f}us  " + ", ".join(f"{a}x{b}" for a, b in top))
    seg_names = {}
for s, e, nm in ev[start:]:
    if s - prev_end > 20000:
        flush(prev_end)
        out.append(f"{'':9s}    GAP {(s-prev_end)/1e3:8.0f}us")
        seg_start = s
    if nm.startswith("memcpy"):
        flush(max(prev_end, s)); out.append(f"{(s-t0)/1e3:9.0f}us  {nm} {(e-s)/1e3:.0f}us"); seg_start = e
    else:
        seg_names[nm] = seg_names.get(nm, 0) + 1
    prev_end = max(prev_end, e)
    if len(out) > n: break
print("\n".join(out[:n]))
