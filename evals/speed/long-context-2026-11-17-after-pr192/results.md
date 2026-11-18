# Long-context bench, Gemma 4 31B, 2026-11-17 after PR #192

The 15 Nov long-context bench (`../long-context-2026-11-15/`, the engine deck's cards 2, 3 and 8)
rerun on the current engine only: `main` at d3bedf5 (PRs #188 faster short-batch kernels and
chat-text guessing, #189 prompt-cache starts, #190 a fourth draft guess a round for Gemma,
#192 that fourth guess only while a chat is under 48,000 tokens), `cargo build --release`.
MacBook Pro M5 Max, 128 GB, Q6_K GGUF, Google's 31B draft model as shipped. Same harness,
same prompts, untouched (`../long-context-2026-11-15/long_bench.py`, `prompts/`).

**llama.cpp and mlx-vlm were not run**: their columns are the stored 15 Nov rows. "Deck" is
the 15 Nov run of the new engine (the deck's numbers). "PR #188" is this afternoon's run
(`../mac-after-pr188-2026-11-17/`, 128K = its cool-down rerun, as that write-up used; no
256K). 32K, 64K and 128K are two runs each (mean; peak memory = the higher), 256K one run.
Every run recalled both planted facts; no errors, no out-of-memory, no heat-affected numbers,
nothing rerun.

## Headlines

| Context | Read tok/s | First token | Write tok/s | Peak memory | Facts | Draft guesses taken |
|---|---|---|---|---|---|---|
| 32K | 573 → 595 → **620** (1.18× mlx-vlm) | 52 → 50 → **48 s** | 29.7 → 29.9 → **31.6** (1.97× llama.cpp) | 32.3 → 33.3 → **33.3 GB** | 2/2, 2/2 | 52% → 52% → 45% |
| 64K | 460 → 478 → **486** (1.14×) | 133 → 128 → **126 s** | 24.1 → 26.6 → **29.2** (2.34×) | 35.5 → 36.6 → **36.6 GB** | 2/2, 2/2 | 52% → 53% → 53% |
| 128K | 346 → 345 → **361** (1.11×) | 360 → 361 → **345 s** | 20.3 → 19.8 → **20.8** (2.15×) | 42.0 → 43.1 → **43.1 GB** | 2/2, 2/2 | 49% → 50% → 51% |
| 256K | 217 → – → **230** (1.17×) | 19.5 → – → **18.3 min** | 16.2 → – → **15.8** (2.37×) | 56.0 → – → **57.1 GB** | 2/2 | 53% → – → 52% |

(deck → PR #188 → now; ratio = new engine ÷ the faster of llama.cpp and mlx-vlm.)

- **Reading is faster than the deck at every size**, +4% to +8%, and so is the first token
  (−4% to −8%). 256K now reads in 18.3 min instead of 19.5.
- **Writing: +6% at 32K, +21% at 64K, +2% at 128K, −2% at 256K.** The −2.7% at 128K that the
  PR #188 run found is gone (both runs here, 21.1 and 20.5, are above the deck's 20.3).
- **Memory: the ~1 GB offset is still there, and holds at 256K**: +1.1 GB peak at every size
  (32.3 → 33.3, 35.5 → 36.6, 42.0 → 43.1, 56.0 → 57.1), identical to the PR #188 run where one
  exists. Idle after load is unchanged (25.8 GB) and memory after the reply equals the deck's
  at 64K-256K (+1.1 GB at 32K, as at PR #188), so the extra gigabyte is working memory used
  during the request, not held cache. Still below mlx-vlm at every size (75.2 GB at 256K) and
  1.08-1.13× llama.cpp's peak.
- **256K still recalls both facts** (OSPREY-41, Dagny Vell) and writes a coherent answer.

## Tables

### Reading speed (tok/s)

| Context | Deck (15 Nov) | PR #188 (17 Nov pm) | Now (PR #192) | Now vs deck | llama.cpp | mlx-vlm | new ÷ faster other |
|---|---|---|---|---|---|---|---|
| 32K | 573 | 595 | **620** | +8.1% | 444 | 526 | 1.18× |
| 64K | 460 | 478 | **486** | +5.8% | 333 | 429 | 1.14× |
| 128K | 346 | 345 | **361** | +4.3% | 232 | 324 | 1.11× |
| 256K | 217 | - | **230** | +6.1% | 134 | 196 | 1.17× |

### Time to first token (s)

| Context | Deck (15 Nov) | PR #188 (17 Nov pm) | Now (PR #192) | Now vs deck | llama.cpp | mlx-vlm | faster other ÷ new |
|---|---|---|---|---|---|---|---|
| 32K | 52 | 50 | **48** | -7.6% | 67 | 57 | 1.18× |
| 64K | 133 | 128 | **126** | -5.5% | 184 | 143 | 1.14× |
| 128K | 360 | 361 | **345** | -4.1% | 537 | 385 | 1.12× |
| 256K | 1168 | - | **1101** | -5.7% | 1885 | 1293 | 1.17× |

### Writing speed (tok/s)

| Context | Deck (15 Nov) | PR #188 (17 Nov pm) | Now (PR #192) | Now vs deck | llama.cpp | mlx-vlm | new ÷ faster other |
|---|---|---|---|---|---|---|---|
| 32K | 29.7 | 29.9 | **31.6** | +6.3% | 16.0 | 12.7 | 1.97× |
| 64K | 24.1 | 26.6 | **29.2** | +21.2% | 12.5 | 9.1 | 2.34× |
| 128K | 20.3 | 19.8 | **20.8** | +2.2% | 9.7 | 6.0 | 2.15× |
| 256K | 16.2 | - | **15.8** | -2.1% | 6.7 | 3.5 | 2.37× |

### Peak memory (GB)

| Context | Deck (15 Nov) | PR #188 (17 Nov pm) | Now (PR #192) | Now vs deck | llama.cpp | mlx-vlm | new ÷ lowest other |
|---|---|---|---|---|---|---|---|
| 32K | 32.3 | 33.3 | **33.3** | +3.3% | 30.8 | 37.6 | 1.08× |
| 64K | 35.5 | 36.6 | **36.6** | +3.0% | 33.6 | 41.9 | 1.09× |
| 128K | 42.0 | 43.1 | **43.1** | +2.6% | 39.7 | 54.8 | 1.08× |
| 256K | 56.0 | - | **57.1** | +1.9% | 50.5 | 75.2 | 1.13× |

### Facts recalled (of 2) and draft guesses taken

| Context | Facts deck → PR #188 → now (each run) | Guesses deck | Guesses PR #188 | Guesses now |
|---|---|---|---|---|
| 32K | 2, 2 → 2, 2 → **2, 2** | draft 604/1167 (52%) | draft 607/1159 (52%), chat 0/0 | draft 633/1408 (45%), chat 0/0 |
| 64K | 2, 2 → 2, 2 → **2, 2** | draft 604/1161 (52%) | draft 613/1149 (53%), chat 0/0 | draft 617/1173 (53%), chat 0/0 |
| 128K | 2 → 2, 2 → **2, 2** | draft 300/609 (49%) | draft 604/1212 (50%), chat 0/0 | draft 605/1197 (51%), chat 0/0 |
| 256K | 2 → - → **2** | draft 313/589 (53%) | - | draft 311/596 (52%), chat 0/0 |

### Each run now

| Context | Run | Finished | Read | Read tok/s | First token (s) | Written | Write tok/s | Peak GB | Facts | Guesses |
|---|---|---|---|---|---|---|---|---|---|---|
| 32K | 1 | 18:09:28 | 29,746 | 623 | 47.8 | 492 | 32.0 | 33.3 | 2/2 | draft 320/688 (47%), chat 0/0 |
| 32K | 2 | 18:12:51 | 29,747 | 616 | 48.3 | 493 | 31.2 | 33.3 | 2/2 | draft 313/720 (43%), chat 0/0 |
| 64K | 1 | 18:18:03 | 61,112 | 486 | 125.7 | 504 | 29.0 | 36.6 | 2/2 | draft 307/591 (52%), chat 0/0 |
| 64K | 2 | 18:23:14 | 61,114 | 487 | 125.6 | 504 | 29.5 | 36.6 | 2/2 | draft 310/582 (53%), chat 0/0 |
| 128K | 1 | 18:33:42 | 124,525 | 361 | 345.1 | 509 | 21.1 | 43.1 | 2/2 | draft 309/600 (52%), chat 0/0 |
| 128K | 2 | 18:45:11 | 124,525 | 360 | 345.8 | 495 | 20.5 | 43.1 | 2/2 | draft 296/597 (50%), chat 0/0 |
| 256K | 1 | 19:11:23 | 253,018 | 230 | 1100.8 | 512 | 15.8 | 57.1 | 2/2 | draft 311/596 (52%), chat 0/0 |

### Memory (GB): footprint + resident model file, sampled every 2 s

Each run's value, in run order (PR #188 at 128K: main run / cool-down rerun).

| Context | What | Deck (15 Nov) | PR #188 (17 Nov pm) | Now (PR #192) | Now − deck (max peak) | llama.cpp | mlx-vlm |
|---|---|---|---|---|---|---|---|
| 32K | Idle after load | 25.8 / 25.8 | 25.8 / 25.8 | **25.8 / 25.8** |  | 28.8 / 28.8 | 26.8 / 26.8 |
|  | Peak, reading + writing | 32.3 / 32.3 | 33.3 / 33.3 | **33.3 / 33.3** | +1.1 | 30.8 / 30.8 | 34.4 / 37.6 |
|  | After the reply | 31.2 / 31.2 | 32.3 / 32.3 | **32.3 / 32.3** |  | 30.7 / 30.7 | 30.1 / 30.1 |
| 64K | Idle after load | 25.8 / 25.8 | 25.8 / 25.8 | **25.8 / 25.8** |  | 31.4 / 31.4 | 26.8 / 26.8 |
|  | Peak, reading + writing | 35.5 / 35.5 | 36.6 / 36.6 | **36.6 / 36.6** | +1.1 | 33.6 / 33.6 | 41.9 / 41.9 |
|  | After the reply | 34.4 / 34.4 | 34.4 / 34.4 | **34.4 / 34.4** |  | 33.4 / 33.4 | 33.3 / 33.3 |
| 128K | Idle after load | 25.8 | 25.8 / 25.8 | **25.8 / 25.8** |  | 36.5 | 26.8 |
|  | Peak, reading + writing | 42.0 | 43.1 / 43.1 | **43.1 / 43.1** | +1.1 | 39.7 | 54.8 |
|  | After the reply | 39.9 | 39.9 / 39.9 | **39.9 / 39.9** |  | 39.7 | 37.6 |
| 256K | Idle after load | 25.8 | - | **25.8** |  | 47.2 | 26.8 / 26.8 |
|  | Peak, reading + writing | 56.0 | - | **57.1** | +1.1 | 50.5 | 75.2 |
|  | After the reply | 50.6 | - | **50.6** |  | 50.5 | 48.3 |

## Notes

- **What got slower than the deck: 256K writing, −2.1%** (16.2 → 15.8 tok/s): 512 tokens in
  32.37 s vs 31.70 s, with the same draft work (311 of 596 guesses taken vs 313 of 589). One
  run; 0.67 s is inside the run-to-run spread seen here at 32K (32.0 vs 31.2) and 128K (21.1
  vs 20.5). Not rerun: the Mac was cool (load average 0.33 at the start of a 7-minute
  idle) and reading in the same run was +6%, so it does not look like heat. PR #192 turns the
  fourth draft guess off above 48,000 tokens, so 128K and 256K write as at PR #188. Peak memory
  is also above the deck at every size (+1.1 GB, above). Nothing came back wrong.
- **32K draft guesses taken fell 52% → 45%** (633 of 1,408): the fourth guess a round below
  48,000 tokens makes more guesses (1,408 vs 1,167), takes slightly more, and writing is still
  +6%. 64K (61K tokens) is above the 48,000 cut, yet writes +21% over the deck and +10% over
  PR #188 with the same share of guesses taken. Chat-text guesses: 0 made at every size (none
  apply to this prompt).
- **Memory fields:** every load row has `idle_total`, every long row `peak_fp`, `peak_rss`,
  `peak_total` and `after_total` (checked for all 7 runs). Repeat runs gave identical memory
  to 0.01 GB.
- **Heat:** 2-7 minutes idle before each run (2 before 32K #2, 2.5 before each 64K, 4 and 5
  before the 128K runs, 7 before 256K); 1-minute load averages 0.3-1.8 at the start of each idle (1.8 before the first run, from the build). The repeats
  agree within 1-3%, so nothing was rerun.
- One server at a time: the chain waited for `pgrep -f vireo-server` to come back empty before
  each run, and none was left at the end.

## Commands

    cd VireoPrototype && git log --oneline -1          # d3bedf5
    export PATH="$(dirname $(rustup which cargo)):$PATH" && cargo build --release
    # each run separately so the Mac idles between them (the harness runs reps back to back):
    D=evals/speed; N=$D/long-context-2026-11-17-after-pr192
    export OUT=$N/results.jsonl LOGS=$N/logs ENGINES=vireo REPS=1
    SIZES=32k  caffeinate -i python3 -I $D/long-context-2026-11-15/long_bench.py   # x2, 120 s idle between
    SIZES=64k  caffeinate -i python3 -I $D/long-context-2026-11-15/long_bench.py   # x2, 150 s idle before each
    SIZES=128k caffeinate -i python3 -I $D/long-context-2026-11-15/long_bench.py   # x2, 240 / 300 s idle before
    SIZES=256k caffeinate -i python3 -I $D/long-context-2026-11-15/long_bench.py   # x1, 420 s idle before
    python3 -I $N/compare.py > $N/tables.md

Run 18:08-19:11. Because each run was its own invocation with `REPS=1`, every row in
`results.jsonl` has `rep: 0`; run order is the row order (and `time`). Files: `results.jsonl`,
`logs/bench-logs/` (server logs with the "Reply:" lines), `run.log`, `compare.py`, `tables.md`.
