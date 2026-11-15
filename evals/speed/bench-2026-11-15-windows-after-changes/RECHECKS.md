# Re-checks and extra rows (15 Nov, afternoon), not part of the main results

The main run is `results.jsonl` (64 rows, no errors): bench-2026-11-13/bench.py with
`ENGINES=vireo,llama MODELS=e2b,e4b,q4b,q9b REPS=1`, the contexts of the morning's run
(65,536; 32,768 for Qwen 3.5 9B with its long tool log cut to half), on 7e00a19 with the
three uncommitted changes (Qwen's saved state at the system turn's end, sliding-window
rings on CUDA, the prompt attention kernel's f16 queries). Server logs in `bench-logs/`.

Before it, the full-speed check: `vireo-cli generate` Gemma 4 E4B on medium.txt read
5,549 then 6,308 tokens/s and wrote 82.8 then 88.5 (the first run warming up).

## recheck-gemma-medium.jsonl

The main run's ~3K read on Vireo was lower than the morning's on both Gemmas (E2B 13,927
against 17,022; E4B 5,888 against 6,249), and E4B's short reply 104 tokens/s against 118.
Re-run for the Gemmas on Vireo alone (`WORK=short,medium,short,medium`): E2B read medium at
17,017 both times, E4B at 6,249 and 6,253, and E4B wrote the short reply at 115.1 and
115.2, the same as the morning. The main run's rows were one-offs; they are kept as measured.

The same rows show the one real change in what Vireo keeps: after a prompt longer than a
Gemma sliding-window layer's ring, the next chat keeps 0 tokens and reads its ~1,300
tokens of instructions and tools again (short after medium: 0 kept). This is what the
Mac's rows show too (results-q6-128k.jsonl: long, multiturn, tools, skill, longtools all
0 kept on Gemma); a ring that has gone round can't go back to an earlier position.

## probe-131k.jsonl

`probe_131k.py`: the Gemmas at a 131,072-token context on both engines, the Mac's `large`
prompt (~95K tokens on Vireo) through bench.py's own server classes, with Windows'
per-process GPU counters (dedicated GB, shared GB) read after loading and again after the
prompt, while the server still held its memory. Shared didn't grow on any of the four
(Vireo 0.07 GB, llama.cpp 0.23–0.24 GB, the same as at load), so nothing spilled. Server
logs in `probe-logs/`. Not part of the main comparison.
