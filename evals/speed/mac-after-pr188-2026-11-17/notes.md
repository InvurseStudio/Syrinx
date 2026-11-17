
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
