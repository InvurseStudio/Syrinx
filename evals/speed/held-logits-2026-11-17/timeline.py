import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
k = [(s, e, names[n]) for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
m = [(s, e, "memcpy%d:%d" % (ck, b)) for s, e, ck, b in db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
ev = sorted(k + m)
# the writing phase: after the last big prompt kernel (matmul8_t*), take everything
last_prompt = max((i for i, x in enumerate(ev) if x[2].startswith("matmul8_geglu_t") or x[2].startswith("matmul8_swiglu_t") or x[2].startswith("matmul8_t128")), default=0)
ev = ev[last_prompt + 1:]
print("events in writing phase", len(ev), "span ms", (ev[-1][1] - ev[0][0]) / 1e6)
busy = collections.Counter(); cnt = collections.Counter()
gap_total = 0; gaps = []
for i, (s, e, n) in enumerate(ev):
    key = n if not n.startswith("memcpy") else n.split(":")[0] + (":big" if int(n.split(":")[1]) > 100000 else ":small")
    busy[key] += e - s; cnt[key] += 1
    if i:
        g = s - max(x[1] for x in ev[max(0, i - 3):i])
        if g > 20000:
            gaps.append((g, ev[i - 1][2], n)); gap_total += g
span = ev[-1][1] - ev[0][0]
print("kernel+memcpy busy ms", sum(busy.values()) / 1e6, "gaps>20us total ms", gap_total / 1e6, "count", len(gaps))
for n, t in busy.most_common(28):
    print(f"{t/1e6:9.2f} ms {100*t/span:5.1f}%  n={cnt[n]:6d} avg {t/cnt[n]/1e3:8.1f}us  {n}")
gc = collections.Counter(); gt = collections.Counter()
for g, a, b in gaps:
    key = (a.split(":")[0], b.split(":")[0]); gc[key] += 1; gt[key] += g
print("gaps by (before, after):")
for key, t in gt.most_common(12):
    print(f"{t/1e6:9.2f} ms n={gc[key]:5d} avg {t/gc[key]/1e3:7.1f}us  {key}")
if len(sys.argv) > 2:
    a = int(sys.argv[2]); t0 = ev[a][0]
    for s, e, n in ev[a:a + int(sys.argv[3])]:
        print(f"{(s-t0)/1e3:10.1f} +{(e-s)/1e3:8.1f}us {n}")
