# Mac benchmarks after PR #188, 2026-11-17

The engine on `main` at 1262b83 (PR #188 merged: faster 8/16-token product kernels, chat-text
guessing on Qwen 3.5 4B/9B, longer guesses on a running copy, long chat runs for models with a
draft model/layer), run through the same Mac benchmarks behind the engine slide deck
(`docs/dev/engine-social-2026-11-15/`, cards 2-9; card 1 is the cover, card 10 is Windows).
MacBook Pro M5 Max, 128 GB, Q6_K GGUF. **Only the new engine was run**; the llama.cpp and
mlx-vlm columns are the stored rows the deck was built from (`long-context-2026-11-15/results.jsonl`,
`bench-2026-11-13/results-q6-128k.jsonl`, `evals/long-run/gemma-4-31B-llama.json`). No 256K run.

"Deck" is the deck's number for the new engine (14-15 Nov builds). "Main run" is this
afternoon's pass; "Rerun after cool-down" is the same test again after 2 minutes idle, run
where the main number looked heat-affected (see the notes). The change uses the rerun where
there is one. Speeds in tokens/s, times in seconds, memory in GB.

## Headlines

- **64-turn chat (card 6): 320 → 260 s (−19%), now 4.7× faster than llama.cpp (was 3.8×)**;
  one full re-read instead of two; 60/63 answers right, as before.
- **Short replies on the large models (card 9), after a cool-down:** Gemma 4 31B 31.6 → 33.7
  (+7%), Qwen 3.8 27B 30.8 → 34.5 (+12%), Muse 25.3 → 25.7. In the hot main run they read
  4-12% *lower*: heat, not the engine (same draft guesses taken).
- **Writing after a ~125K-token prompt (card 4):** Gemma 4 31B 18.8 → 20.1 (+7%), Qwen 3.5 9B
  46.0 → 61.3 (+33%, chat-text guessing, new for it in PR #188: 56% of guesses taken); the
  others level within noise; still ahead of the faster other engine on all 8 (1.02-1.98×).
- **Long context, Gemma 4 31B (cards 2, 3):** reading +4% at 32K and 64K, level at 128K;
  writing level at 32K, +10% at 64K, **−2.7% at 128K** (reproduced after a cool-down).
- **Agent tasks (card 5):** the six card models within −10% to +8% of the deck (the small
  ones' 4-7 s totals move by tenths); Muse wrong → right and 83 → 47 s.
- **Same answers (card 7):** perplexity 120.78 vs llama.cpp 120.76 and 19/20 greedy
  continuations identical, both exactly as the deck; planted facts 5/5 at 32K and 128K.
- **Memory (card 8):** ~1 GB more at every size (still below mlx-vlm, above llama.cpp).
- **Wrong now that was right:** Gemma 4 E2B long tool task (names `card-reader`). No engine
  errors anywhere.

## Cards 2, 3, 8: Gemma 4 31B at long context

### Card 2: reading speed (tok/s)

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 573 | 595 |  | +3.7% | 444 | 526 | 2/2, 2/2 |
| 64K | 460 | 478 |  | +3.9% | 333 | 429 | 2/2, 2/2 |
| 128K | 346 | 340 | 345 | -0.2% | 232 | 324 | 2/2, 2/2 |

### Card 3: writing speed after the prompt (tok/s)

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 29.7 | 29.9 |  | +0.4% | 16.0 | 12.7 | 2/2, 2/2 |
| 64K | 24.1 | 26.6 |  | +10.4% | 12.5 | 9.1 | 2/2, 2/2 |
| 128K | 20.3 | 20.0 | 19.8 | -2.7% | 9.7 | 6.0 | 2/2, 2/2 |

### Card 8: peak memory (GB, lower is better)

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 32.3 | 33.3 |  | +3.3% | 30.8 | 37.6 | 2/2, 2/2 |
| 64K | 35.5 | 36.6 |  | +3.0% | 33.6 | 41.9 | 2/2, 2/2 |
| 128K | 42.0 | 43.1 | 43.1 | +2.6% | 39.7 | 54.8 | 2/2, 2/2 |

### Time to first token (s), not on a card

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 52 | 50 |  | -3.6% | 67 | 57 | 2/2, 2/2 |
| 64K | 133 | 128 |  | -3.8% | 184 | 143 | 2/2, 2/2 |
| 128K | 360 | 367 | 361 | +0.2% | 537 | 385 | 2/2, 2/2 |

## Cards 4, 5, 9: eight models, 131,072-token context

### Card 4: writing after the ~125K-token prompt (tok/s)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Facts (of 2) deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 18.8 | 20.1 |  | +7.1% | 10.1 | 9.2 | 1.98× | 2 → 2 (llama 2, mlx 2) |
| Qwen 3.8 27B | 30.4 | 30.2 |  | -0.8% | 14.6 | 16.4 | 1.84× | 2 → 2 (llama 2, mlx 2) |
| Muse Glimmer 30B | 23.0 | 21.7 | 22.3 | -3.2% | 17.7 | 17.8 | 1.25× | 2 → 2 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 56.7 | 53.1 | 56.7 | +0.0% | 42.5 | 43.6 | 1.30× | 2 → 2 (llama 2, mlx 2) |
| Gemma 4 E4B | 77.9 | 77.9 |  | +0.0% | 50.4 | 39.3 | 1.55× | 2 → 2 (llama 1, mlx 1) |
| Gemma 4 E2B | 119.6 | 109.8 |  | -8.2% | 67.2 | 99.8 | 1.10× | 2 → 2 (llama 2, mlx 2) |
| Qwen 3.5 4B | 63.4 | 59.7 |  | -5.8% | 58.3 | 58.5 | 1.02× | 2 → 2 (llama 2, mlx 2) |
| Qwen 3.5 9B | 46.0 | 61.3 |  | +33.1% | 43.0 | 43.7 | 1.40× | 2 → 2 (llama 2, mlx 2) |

### Card 5: tool task, total time (s, lower is better)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Right deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 30.1 | 30.1 |  | -0.2% | 57.9 | 57.6 | 1.92× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.8 27B | 31.5 | 31.0 |  | -1.6% | 53.8 | 56.8 | 1.74× | 1 → 1 (llama 1, mlx 1) |
| Muse Glimmer 30B | 83.3 | 47.3 |  | -43.2% | 121.8 | 129.3 | 2.57× | 0 → 1 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 8.9 | 8.7 |  | -2.8% | 12.7 | 13.2 | 1.47× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E4B | 7.4 | 7.3 |  | -1.7% | 10.7 | 15.2 | 1.47× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E2B | 4.0 | 4.3 |  | +8.1% | 6.9 | 8.1 | 1.59× | 0 → 1 (llama 1, mlx 1) |
| Qwen 3.5 4B | 7.1 | 7.3 |  | +3.0% | 10.3 | 11.9 | 1.41× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.5 9B | 11.1 | 10.0 |  | -9.8% | 18.3 | 18.6 | 1.84× | 1 → 1 (llama 1, mlx 1) |

### Skill task, total time (s), the other agent task

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Right deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 28.2 | 27.7 |  | -1.7% | 59.5 | 60.3 | 2.15× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.8 27B | 30.6 | 30.2 |  | -1.4% | 55.1 | 55.2 | 1.82× | 1 → 1 (llama 1, mlx 1) |
| Muse Glimmer 30B | 79.3 | 68.5 |  | -13.5% | 176.0 | 143.8 | 2.10× | 1 → 1 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 10.2 | 10.2 |  | -0.0% | 14.0 | 13.4 | 1.31× | 1 → 1 (llama 0, mlx 0) |
| Gemma 4 E4B | 7.5 | 7.3 |  | -1.8% | 12.8 | 12.2 | 1.66× | 0 → 1 (llama 1, mlx 1) |
| Gemma 4 E2B | 2.5 | 0.9 |  | -63.6% | 8.0 | 8.6 | 8.99× | 0 → 0 (llama 0, mlx 0) |
| Qwen 3.5 4B | 7.1 | 7.0 |  | -2.0% | 11.0 | 11.3 | 1.57× | 0 → 0 (llama 0, mlx 0) |
| Qwen 3.5 9B | 12.5 | 12.2 |  | -2.1% | 24.3 | 25.2 | 1.99× | 1 → 1 (llama 1, mlx 1) |

### Long tool task, total time (s)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Right deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 130.7 | 134.6 |  | +2.9% | 197.0 | 295.6 | 1.46× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.8 27B | 115.5 | 131.4 |  | +13.7% | 151.5 | 117.6 | 0.90× | 1 → 1 (llama 0, mlx 1) |
| Muse Glimmer 30B | 107.7 | 132.7 |  | +23.2% | 95.5 | 82.7 | 0.62× | 1 → 1 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 27.4 | 28.6 |  | +4.3% | 38.2 | 48.7 | 1.34× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E4B | 11.1 | 11.3 |  | +1.0% | 17.4 | 13.6 | 1.21× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E2B | 4.2 | 4.7 |  | +11.8% | 8.7 | 5.4 | 1.14× | 1 → 0 (llama 1, mlx 1) |
| Qwen 3.5 4B | 22.5 | 23.3 |  | +3.5% | 31.3 | 23.5 | 1.00× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.5 9B | 30.6 | 30.0 |  | -2.2% | 52.8 | 33.1 | 1.10× | 1 → 1 (llama 1, mlx 1) |

### Card 9: writing a short reply (tok/s)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other |
|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 31.6 | 27.9 | 33.7 | +6.6% | 17.9 | 18.4 | 1.84× |
| Qwen 3.8 27B | 30.8 | 27.2 | 34.5 | +12.0% | 20.1 | 22.6 | 1.53× |
| Muse Glimmer 30B | 25.3 | 24.3 | 25.7 | +1.8% | 21.2 | 21.0 | 1.21× |
| Gemma 4 26B-A4B | 92.1 | 87.6 |  | -4.9% | 94.8 | 112.7 | 0.78× |
| Gemma 4 E4B | 98.3 | 98.3 |  | +0.0% | 95.8 | 105.8 | 0.93× |
| Gemma 4 E2B | 176.2 | 176.2 |  | +0.0% | 153.0 | 177.4 | 0.99× |
| Qwen 3.5 4B | 119.4 | 116.7 |  | -2.3% | 110.9 | 109.3 | 1.05× |
| Qwen 3.5 9B | 67.4 | 65.8 |  | -2.3% | 65.0 | 71.3 | 0.92× |

### Card 6: 64-turn chat, Gemma 4 31B, 32K context

| | Deck (new engine) | Now | Change | llama.cpp (stored) |
|---|---|---|---|---|
| Whole chat (s) | 319.9 | 259.6 | -18.8% | 1212.0 |
| Full re-reads | 2 | 1 | | 12 (its client drops old turns; per the deck) |
| Answers right | 60/63 | 60/63 | | 62/63 |
| Turns / errors | 64 / 0 | 64 / 0 | | 63 / 0 |
| vs llama.cpp | 3.79× | 4.67× | | |

### Card 7: same answers (Gemma 4 E2B)

| Check | Deck | Now | Notes |
|---|---|---|---|
| Perplexity, Vireo vs llama.cpp | 120.78 vs 120.76 (0.01%) | 120.78 vs 120.76 (0.01%) | llama-perplexity rerun (b12228) |
| Greedy 256-token continuations identical | 19/20 | 19/20 | median words agreeing 109.5, min 7 |
| Planted facts, 32K context | 5/5 | 5/5 | read 19044 tok/s, write 172.9 tok/s |
| Planted facts, 128K context | 5/5 | 5/5 | read 7535 tok/s, write 118.0 tok/s |
| Gemma 4 31B facts at every size | 2/2 to 256K | 32K 2/2, 2/2, 64K 2/2, 2/2, 128K 2/2 (256K not run) | card 2's run |

## Notes on the numbers

- **Heat.** The 8-model bench ran for about an hour straight after 15 minutes of Gemma 4 31B
  long-context work. The short replies on the three large models came out 4-12% below the
  deck in that run; after a 2-minute cool-down they came out at or above it, with the same
  share of draft guesses taken (31B 46-49%, Qwen 27B 43-44%). The "Rerun after cool-down"
  column is the number to use where it exists (2 runs averaged for the short replies, 1 for
  128K and the ~125K-prompt writing rerun). Rows without a rerun that ran late in the hot
  stretch (26B-A4B, Muse, Qwen 3.5 9B) may read a few percent low.
- **Small models, writing after the ~125K prompt.** Their answers are 60-70 tokens written
  in 0.5-1 s, and the server rounds times to 0.01 s, so a ±5-8% move there is noise. Same
  for the exact repeats (E4B 77.9 → 77.9, short replies 98.3 → 98.3 and 176.2 → 176.2):
  greedy text, same tokens, same rounded time.
- **Drafting.** As shipped. Gemma 4 31B drafts with Google's 31B draft model (3 tokens a
  round), Qwen 3.8 27B with its multi-token-prediction layer, Muse with its own draft
  model plus chat-text guesses. The small models and 26B-A4B have no draft model; they use
  only the chat-text guesses (14-37% taken on the tool tasks). llama.cpp and mlx-vlm
  drafted nothing (as in the deck).
- **Muse's short reply** includes 127-141 tokens of reasoning even with thinking off; its
  deck run (392 tokens) did the same, so the two compare.

## What got slower or wrong

- **Gemma 4 31B writing at 128K: −2.7% (20.3 → 19.8 tok/s), real, not heat.** Rerun after a
  cool-down gave the same: 503-506 tokens in 25.6 s vs 24.7 s, with the same 49% of draft
  guesses taken. Reading at 128K is level (346 → 345). The PR #188 gain shows at 64K
  (+10%) and is gone by 128K.
- **Peak memory up ~1 GB at every size on Gemma 4 31B** (32K 32.3 → 33.3, 64K 35.5 → 36.6,
  128K 42.0 → 43.1 GB). Still below mlx-vlm at every size, still above llama.cpp. Not
  explained here (a guess: the second decode buffer, or bigger tiles' scratch).
- **Gemma 4 E2B, long tool task: right → wrong.** With the ~55K-token log pasted in it now
  answers `card-reader` instead of `inventory-sync`, with no tool calls either time
  (57,701 tokens read, 23 written). A small model's greedy choice flipping on a long read;
  perplexity, greedy match and planted facts are unchanged, so it is not an engine error,
  but it is the one answer that changed for the worse.
- **Muse long tool task 107.7 → 132.7 s (+23%)**: same three tool calls and the same right
  answer, but 2,436 tokens written instead of 1,464 (a longer answer), at 30.6 tok/s vs
  27.6. Not slower per token. **Qwen 3.8 27B long tool task 115.5 → 131.4 s (+14%)** the same
  way: same read (56K tokens in 102 s), but two tool calls instead of five and a 1,095-token
  answer instead of 435, written at 39.2 tok/s vs 34.5. Both still right; on this task Qwen
  27B is now 0.90× mlx-vlm's stored time and Muse 0.62× (Muse was behind both in the deck
  too, where the other two answered wrong).
- **Muse writing after the ~125K prompt: −3.2%** (23.0 → 22.3 tok/s after a cool-down, 21.7
  hot). Small; its reply is ~200 tokens with ~150 of reasoning, and chat guesses took 35-37%.
- **Still wrong, as in the deck:** Qwen 3.5 4B skill task (the same "67 hours" answer, should
  be 78); Gemma 4 E2B skill task (now stops after two reads and says it found no files;
  before it read them all and summed wrongly); Gemma 4 E2B and Qwen 3.5 4B skill were wrong
  in the deck too.
- **Better:** Muse's tool task wrong → right and 83.3 → 47.3 s; Gemma 4 E2B tool task
  wrong → right; Gemma 4 E4B skill task wrong → right.
- No engine errors or server failures in any run (0 error rows).

## Skipped

- **256K** (cards 1-3, 7, 8): left out on purpose, to keep the run short. Card 7's "0 errors
  from 32K to 256K" and "2 of 2 at every size to 256K" are only re-checked to 128K.
- **llama.cpp and mlx-vlm**: not rerun; their stored rows are the comparison. The one
  exception is card 7: no stored Vireo Bench run or `baselines.json` existed
  (`evals/bench/results/` was empty), so `llama-perplexity` and llama-server's greedy
  continuations (b12228, Gemma 4 E2B, about a minute) ran again. They gave the same
  120.76 as the deck.
- **Cold load and energy** (need `sudo`): not part of these cards; not run.
- **Card 10** (Windows laptop): not part of this run.
- No harness or engine code was changed. The only new code is `compare.py` here (reads the
  stored and new results files, prints the tables).

## Commands

All from `VireoPrototype/`, on `main` at 1262b83, after
`export PATH="$(dirname $(rustup which cargo)):$PATH"; cargo build --release`.
`N=evals/speed/mac-after-pr188-2026-11-17`.

```bash
# cards 2, 3, 8
cd evals/speed/long-context-2026-11-15 && OUT=$N/long-context.jsonl LOGS=$N/logs SIZES=32k,64k,128k ENGINES=vireo caffeinate -i python3 -I long_bench.py
# cards 4, 5, 9 (all 9 workloads, as results-q6-128k.jsonl; 131,072 context)
cd evals/speed/bench-2026-11-13 && OUT=$N/bench-8models.jsonl LOGS=$N/logs ENGINES=vireo REPS=1 caffeinate -i python3 -I bench.py
# card 6 (defaults give the 64 turns of the stored run: 56 rounds + remember + side chats; max_tokens 200, seed 7)
python3 -I tools/long_run.py --engine vireo --model ~/Library/Application\ Support/InvurseChat/Models/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q6_K.gguf --context 32768 --out $N/long-run-gemma-4-31B-vireo.json
# card 7
cd evals/bench && python3 -I run.py --tier standard --models e2b --engines vireo,llama --suites S6 --reps 1 --set 'S6.tests=["ppl","greedy-match"]' --out $N/bench-s6-e2b.jsonl --no-report
cd evals/bench && python3 -I run.py --tier standard --models e2b --engines vireo --suites S3 --rungs 32k,128k --reps 1 --out $N/bench-s3-e2b.jsonl --no-report
# reruns after a 120 s cool-down each
cd evals/speed/bench-2026-11-13 && OUT=$N/rerun-short.jsonl LOGS=$N/logs ENGINES=vireo REPS=2 MODELS=g31,q27,muse WORK=short python3 -I bench.py
cd evals/speed/long-context-2026-11-15 && OUT=$N/rerun-long-context-128k.jsonl LOGS=$N/logs SIZES=128k ENGINES=vireo python3 -I long_bench.py
cd evals/speed/bench-2026-11-13 && OUT=$N/rerun-xlarge.jsonl LOGS=$N/logs ENGINES=vireo REPS=1 MODELS=g26,muse WORK=xlarge python3 -I bench.py
# tables
python3 -I $N/compare.py
```

Files here: `*.jsonl` / `long-run-*.json` raw rows, `*.log` harness output, `logs/bench-logs/`
every server's log (the `Reply:` lines carry read/write times and draft guesses taken).
