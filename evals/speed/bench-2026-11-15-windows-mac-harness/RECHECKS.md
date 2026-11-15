# Re-checks (15 Nov), not part of the results

- `recheck-e2b-context.jsonl`: Gemma 4 E2B on Vireo, short and medium only, at a 32,768 and a
  65,536-token context (`CONTEXT=`), to see whether the forced context slows writing. It doesn't:
  short reply 189.7 and 197.6 tokens/s, medium read 17,017 tokens/s both. The main run's short
  reply (144.7 tokens/s) was one reply that took fewer of the draft model's guesses.
- `../bench-2026-11-15-windows-tasks-after-merge/recheck-e2b.jsonl`: bench_tasks.py again for
  Gemma 4 E2B only, after its after-merge row read medium at 10,287 tokens/s (15,773 before the
  merge): the re-run read it at 15,773 again and wrote the short reply at 200.0 (175.0 in the
  run). The run's rows are kept as measured; the report notes both.

# Context probe (15 Nov, before the run)

`context-probe.jsonl`: each model and engine at a forced context, the Mac's long and
longtools prompts sent whole, then Windows' per-process GPU counters read while the server
still held its memory (`mem` and `load`: dedicated GB, shared GB). "Shared" growing means the
GPU's memory spilled into the PC's. Gemma 4 E2B, E4B and Qwen 3.5 4B held 65,536 tokens on
both engines (shared 0.07 GB on Vireo). Qwen 3.5 9B on Vireo at 59,392 spilled (shared 0.51 GB;
it read the long tool prompt at 145 tokens/s, against 1,651 for the long prompt at 32,768), so
it ran at 32,768 with its long tool log cut to the first half (bench.py, WIN_LOG_SHARE).
