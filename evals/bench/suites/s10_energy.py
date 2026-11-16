"""S10 Energy, its own tier: nothing else runs while power is read.

  read   the pp4096 prompt read 5 times (1 token written each)
  write  a story written 3 times, 512 tokens each

Joules per 1,000 tokens and average watts over each. Mac: `powermetrics` (CPU + GPU + ANE,
needs `sudo -v` first), sampled every 500 ms while the work runs. Windows: the GPU's own
energy counter (nvidia-smi total_energy_consumption) read before and after; GPU only.
Every engine through its server, so the numbers include what a user's request costs.
"""
import re
import subprocess
import tempfile
import time
from pathlib import Path

import common
import engines
import prompts

STORY = "Write a long story about a lighthouse keeper who finds a message in a bottle."


class Meter:
    def __enter__(self):
        self.t0 = time.time()
        if common.WIN:
            self.e0 = self.gpu_mj()
        else:
            self.out = Path(tempfile.mkstemp(suffix=".txt")[1])
            self.p = subprocess.Popen(["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power", "-i", "500",
                                       "-o", str(self.out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0
        if common.WIN:
            self.joules = (self.gpu_mj() - self.e0) / 1000
        else:
            subprocess.run(["sudo", "-n", "kill", "-INT", str(self.p.pid)], check=False)
            self.p.wait(timeout=30)
            mw = [int(x) for x in re.findall(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", self.out.read_text())]
            self.out.unlink()
            self.joules = (sum(mw) / len(mw) / 1000) * self.seconds if mw else None

    @staticmethod
    def gpu_mj():
        out = subprocess.run(["nvidia-smi", "--query-gpu=total_energy_consumption", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return float(out.split()[0])


def run(ctx, kind):
    if not common.WIN and not ctx.sudo:
        ctx.rec("S10", "read", kind, skipped="needs `sudo -v` first (powermetrics)")
        return
    srv = engines.start(kind, ctx.model, ctx.chat_context)
    try:
        srv.chat([{"role": "user", "content": "Say hello."}], 8, log=False)
        text, _ = prompts.body(ctx.key, "pp4096")
        for test, work in [
            ("read", lambda: [srv.chat([{"role": "user", "content": common.nonce() + text}], 1) for _ in range(5)]),
            ("write", lambda: [srv.chat([{"role": "user", "content": common.nonce() + STORY}], 512) for _ in range(3)]),
        ]:
            with Meter() as m:
                rs = work()
            tokens = sum((r["read_tokens"] if test == "read" else r["gen_tokens"]) or 0 for r in rs)
            ctx.rec("S10", test, kind, tokens=tokens, seconds=round(m.seconds, 2),
                    joules=round(m.joules, 1) if m.joules else None,
                    j_per_1k=round(m.joules / tokens * 1000, 2) if m.joules and tokens else None,
                    watts=round(m.joules / m.seconds, 1) if m.joules else None,
                    errors=sum(bool(r["error"]) for r in rs))
    finally:
        srv.stop()
