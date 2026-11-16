"""Vireo Bench v1: runs a tier of the suite (see BENCH-SUITE.md) and writes one results file.

    python3 -I run.py --tier quick                    after a tuning change (~15 min)
    python3 -I run.py --tier standard                 before an engine PR merges (~4-5 h)
    python3 -I run.py --tier full                     overnight, before a release
    python3 -I run.py --tier energy                   S10 alone (Mac: `sudo -v` first)
    python3 -I run.py --tier standard --models e4b,g31 --engines vireo,llama --suites S1,S3
    python3 -I run.py --tier quick --dry-run          what it would run, and nothing else

On Windows the same, as `python -X utf8 -I run.py ...` from VireoPrototype's folder or
this one: the platform is picked from the OS, and suite.json's "windows" block names the
laptop's models, contexts and llama.cpp build.

Results: results/<date>-<platform>-<tier>-<commit>.jsonl, one row per test per model per
engine per rep, then report.py builds the page beside it. Server logs go to the system's
temporary folder (vireo-bench-logs). One engine and model is loaded at a time; the
engines' order is reversed on every other rep.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import engines  # noqa: E402
from suites import (s1_raw, s2_feel, s3_ladder, s4_chat, s5_tasks, s6_correct, s8_endurance,  # noqa: E402
                    s9_together, s10_energy)

ALL_SUITES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]


class Ctx:
    def __init__(self, cfg, tier, suites, rec, sudo):
        self.cfg, self.tier, self.suites, self.rec_, self.sudo = cfg, tier, suites, rec, sudo
        self.pcfg = cfg["platforms"][common.PLATFORM]
        self.chat_context = self.pcfg["chat_context"]
        self.ladder = cfg["ladder"]
        self.model, self.rep = None, 0

    @property
    def key(self):
        return self.model["key"]

    @property
    def max_context(self):
        return self.model["max_context"]

    def selected(self, s):
        return s in self.suites

    def opts(self, s):
        v = self.tier["suites"].get(s)
        return v if isinstance(v, dict) else {}

    def rec(self, suite, test, kind, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return self.rec_(suite=suite, test=test, engine=kind, model=self.key, model_name=self.model["name"],
                         rep=self.rep, **kw)

    def guard(self, suite, kind, f, *a, **kw):
        """Runs one suite's piece; an error is recorded and the run goes on."""
        try:
            return f(*a, **kw)
        except Exception as e:
            self.rec(suite, "error", kind, error=repr(e)[:400], trace=traceback.format_exc()[-1200:])
            print(traceback.format_exc(), file=sys.stderr, flush=True)


def chat_block(ctx, kind):
    """The suites that share one server at the chat context: S2, S4, S5, S6's chat tests, S8."""
    s6_chat = ctx.selected("S6") and (not ctx.opts("S6").get("tests") or
                                      {"qa-50", "grammar"} & set(ctx.opts("S6")["tests"]))
    if not any([ctx.selected("S2"), ctx.selected("S4"), ctx.selected("S5"), s6_chat, ctx.selected("S8")]):
        return
    if ctx.selected("S2"):
        ctx.guard("S2", kind, s2_feel.cold, ctx, kind)
    try:
        srv = engines.start(kind, ctx.model, ctx.chat_context)
    except Exception as e:
        ctx.rec("run", "start", kind, error=f"chat server: {e!r}"[:400])
        return
    try:
        if ctx.selected("S2"):
            ctx.guard("S2", kind, s2_feel.ready, ctx, srv, kind, "load-warm")
        else:
            srv.chat([{"role": "user", "content": "Say hello."}], 8, log=False)
        mem = common.watch(srv.pid)
        time.sleep(2)
        if ctx.selected("S7"):
            ctx.rec("S7", "idle", kind, context=ctx.chat_context, idle_gb=common.gb(mem.reset()))
        mem.stop()
        if ctx.selected("S2"):
            ctx.guard("S2", kind, s2_feel.run, ctx, srv, kind)
        if ctx.selected("S4"):
            ctx.guard("S4", kind, s4_chat.run, ctx, srv, kind)
        if ctx.selected("S5"):
            ctx.guard("S5", kind, s5_tasks.run, ctx, srv, kind, False)
        if s6_chat:
            ctx.guard("S6", kind, s6_correct.run_chat, ctx, srv, kind)
        if ctx.selected("S8"):
            ctx.guard("S8", kind, s8_endurance.run, ctx, srv, kind)
    finally:
        srv.stop()


def thinking_pass(ctx, kind):
    if True not in ctx.opts("S5").get("thinking", [False]):
        return
    srv = engines.start(kind, ctx.model, ctx.chat_context, thinking=True)
    try:
        srv.chat([{"role": "user", "content": "Say hello."}], 64, log=False)
        s5_tasks.run(ctx, srv, kind, True)
    finally:
        srv.stop()


def model_block(ctx, kinds, reps):
    n = max(reps["speed"], reps["other"])
    for rep in range(n):
        ctx.rep = rep
        order = kinds if rep % 2 == 0 else kinds[::-1]
        for kind in order:
            if not engines.available(kind, ctx.model):
                if rep == 0:
                    ctx.rec("run", "skip", kind, skipped="engine or model files not on this machine")
                continue
            if rep < reps["speed"]:
                if ctx.selected("S1") and kind != "vireo-plain":
                    ctx.guard("S1", kind, s1_raw.run, ctx, kind)
                if ctx.selected("S3") or (ctx.selected("S7") and not ctx.selected("S3")):
                    ctx.guard("S3", kind, s3_ladder.run, ctx, kind)
            elif kind == "vireo-plain":
                continue
            if rep >= reps["other"] or kind == "vireo-plain":
                continue
            if ctx.selected("S6"):
                ctx.guard("S6", kind, s6_correct.run_cli, ctx, kind)
            chat_block(ctx, kind)
            if ctx.selected("S5"):
                ctx.guard("S5", kind, s5_tasks.deep, ctx, kind)
                ctx.guard("S5", kind, thinking_pass, ctx, kind)
            if ctx.selected("S9"):
                ctx.guard("S9", kind, s9_together.run, ctx, kind)
            if ctx.selected("S10"):
                ctx.guard("S10", kind, s10_energy.run, ctx, kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", required=True, choices=["quick", "standard", "full", "energy"])
    ap.add_argument("--models", help="model keys, comma-separated (default: the tier's)")
    ap.add_argument("--engines", help="vireo,vireo-plain,llama,mlx (default: the tier's)")
    ap.add_argument("--suites", help="only these of the tier's suites, e.g. S1,S3")
    ap.add_argument("--out", help="results file (default: results/<date>-<platform>-<tier>-<commit>.jsonl)")
    ap.add_argument("--note", default="", help="a line saved with the run (what changed, why it ran)")
    ap.add_argument("--reps", type=int, help="override the tier's reps (speed and other), e.g. 1 for a smoke test")
    ap.add_argument("--rungs", help="override S3's rungs, e.g. 4k,16k")
    ap.add_argument("--set", action="append", default=[], metavar="SUITE.KEY=VALUE",
                    help="override a suite option, e.g. S8.soak_minutes=2 (value is JSON)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()

    cfg = common.config()
    tier = cfg["tiers"][a.tier]
    if a.reps:
        tier["reps"] = {"speed": a.reps, "other": a.reps}
    if a.rungs:
        s3 = tier["suites"].get("S3")
        tier["suites"]["S3"] = dict(s3 if isinstance(s3, dict) else {}, rungs=a.rungs.split(","))
    for s in a.set:
        name, value = s.split("=", 1)
        suite, k = name.split(".", 1)
        cur = tier["suites"].get(suite)
        tier["suites"][suite] = dict(cur if isinstance(cur, dict) else {}, **{k: json.loads(value)})
    pcfg = cfg["platforms"][common.PLATFORM]
    roster = {m["key"]: m for m in pcfg["models"]}
    tm = tier["models"]
    keys = (a.models.split(",") if a.models else
            list(roster) if tm == "all" else tm.get(common.PLATFORM, []) if isinstance(tm, dict) else tm)
    unknown = [k for k in keys if k not in roster]
    if unknown:
        ap.error(f"not in the {common.PLATFORM} roster: {', '.join(unknown)} (have {', '.join(roster)})")
    kinds = a.engines.split(",") if a.engines else list(tier["engines"])
    kinds = [k for k in kinds if pcfg["engines"].get(k.replace("-plain", ""))]
    suites = [s for s in ALL_SUITES if s in tier["suites"]]
    if a.suites:
        suites = [s for s in suites if s in a.suites.split(",")]

    state = common.machine_state()
    print(f"Vireo Bench {cfg['version']} | {common.PLATFORM} | tier {a.tier} | models {','.join(keys)} | "
          f"engines {','.join(kinds)} | suites {','.join(suites)} | reps {tier['reps']}")
    for k in keys:
        m = roster[k]
        print(f"  {k:5} {m['name']:18} " + "  ".join(f"{e}:{'ok' if engines.available(e, m) else 'MISSING'}" for e in kinds))
    if not state.get("on_mains", True):
        print("WARNING: on battery. Measure on mains in the fastest power mode (rules, item 7).")
    if not common.WIN and not state.get("sudo") and ({"S10"} & set(suites) or "S2" in suites):
        print("Note: no cached sudo, so S2 load-cold and S10 are skipped. Run `sudo -v` first to include them.")
    if a.dry_run:
        return

    out = Path(a.out) if a.out else HERE / "results" / (
        f"{time.strftime('%Y-%m-%d-%H%M')}-{common.PLATFORM}-{a.tier}-{state['commit'] or 'nocommit'}"
        f"{'-dirty' if state.get('dirty') else ''}.jsonl")
    base = {"run": out.stem, "suite_version": cfg["version"], "platform": common.PLATFORM, "tier": a.tier,
            "commit": state["commit"]}
    rec = common.Recorder(out, base)
    rec(suite="run", test="start", engine="", model="", machine=state, note=a.note, models=keys, engines=kinds,
        suites=suites, pinned={k: v.get("version") for k, v in pcfg["engines"].items()})
    ctx = Ctx(cfg, tier, suites, rec, state.get("sudo", False))
    t0 = time.time()
    for k in keys:
        ctx.model = roster[k]
        model_block(ctx, kinds, tier["reps"])
    rec(suite="run", test="end", engine="", model="", minutes=round((time.time() - t0) / 60, 1))
    print(f"\nResults: {out}")
    if not a.no_report:
        import report
        page = report.build(out)
        print(f"Report:  {page}")


if __name__ == "__main__":
    main()
