import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
m = [(s, e, "memcpy%d:%d" % (ck, b)) for s, e, ck, b in db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
ev = sorted(k + m)
last_prompt = max((i for i, x in enumerate(ev) if x[2].startswith("matmul8_geglu_t") or x[2].startswith("matmul8_swiglu_t") or x[2].startswith("matmul8_t128")), default=0)
ev = ev[last_prompt + 1:]
t0 = ev[0][0]; pe = ev[0][1]; hist = collections.Counter(); tot = collections.Counter()
for i, (s, e, n) in enumerate(ev[1:], 1):
    g = s - pe
    if g > 20000:
        b = "<100us" if g < 1e5 else "<300us" if g < 3e5 else "<1ms" if g < 1e6 else ">=1ms"
        hist[b] += 1; tot[b] += g
        if g >= 1e6: print(f"at {(s-t0)/1e6:8.1f} ms gap {g/1e6:6.2f} ms  {ev[i-1][2]} -> {n}")
    pe = max(pe, e)
for b in hist: print(b, hist[b], "%.1f ms" % (tot[b] / 1e6))
