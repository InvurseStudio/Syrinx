import sys, math
def load(p): return [(float(a), int(b)) for a, b in (l.split() for l in open(p))]
ref = load(sys.argv[1])
for p in sys.argv[2:]:
    o = load(p); n = len(ref)
    d = [a[0] - b[0] for a, b in zip(o, ref)]
    rms = math.sqrt(sum(x * x for x in d) / n); mean = sum(d) / n
    agree = sum(a[1] == b[1] for a, b in zip(o, ref)) / n
    print(f"{p.split('/')[-1]:28} rms dlogp {rms:.4f}  mean {mean:+.5f}  worst {max(abs(x) for x in d):.3f}  same top token {agree*100:.2f}%")
