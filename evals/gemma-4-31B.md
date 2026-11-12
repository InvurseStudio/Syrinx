# Gemma 4 31B on the Mac engine (2026-11-11)

The first large model through `vireo-server`: Gemma 4 31B, Q6_K GGUF (25.2 GB, from the
standard Vireo test's model folder), on an Apple M5 Max with 128 GB.

## What it needed

The engine was written on Gemma 4 E2B and E4B. The 31B differs in three ways:

- its key-value heads change by layer (16 on sliding-window layers, 4 on global ones);
  `gemma4.attention.head_count_kv` is a list, which the engine refused;
- it has no per-layer inputs (`embedding_length_per_layer_input` is 0, no
  `per_layer_token_embd`); the engine required them;
- its global layers have no `attn_v` and use their keys as values. The CPU path had that
  already; the GPU path refused it.

All three are handled now, in the CPU reference (`gemma4.rs`) and the GPU path
(`gemma4_cuda.rs`). Gemma 4 26B-A4B (mixture of experts) is still refused.

## Tuning

Three changes, each checked against the small models' eval (same score on every case,
same speed). Built on the Mac's own engine copy and carried into the one workspace; the
ring, the wide tiles and the small-batch kernel are the Metal backend's, drafting is the
server's (it builds for CUDA too, not yet run there):

1. **Sliding-window cache.** A sliding-window layer keeps its window and a batch's new
   positions in a ring (2,048 rows for 31B), not every position. Chat cache at 32K
   tokens: 29.5 GB → 4.4 GB. A test reads 1,842 tokens through a 576-row ring and picks
   the same next token as a full cache every time.
2. **Wider prompt tiles for long prompts.** Whole 1,024-token batches use a 256-token
   tile (the kernel decodes each weight tile half as often); shorter ones keep 128, which
   is faster for them (the eval's prompts and tool results read 10–20% slower with 256).
3. **Drafting (`--draft`, off by default).** Gemma 4 E2B guesses 2 tokens a round; 31B
   checks them with the token it picked last in one pass, and keeps each guess while its
   own sampler picks the same token, so every token is still 31B's. A new kernel does the
   2–8-token products with each weight read once (Q6_K). 61% of guesses were taken on
   prose, 70% on short answers.

## Speed

| | First run | Now | Now, drafting |
|---|---|---|---|
| Load | 12.0 s | 3.6 s (file cached) | +0.4 s for E2B |
| Writing, short chat | 19.2 tokens/s | 19.6 | **23.1** |
| Writing, 6K-token chat | 17.5 | 18.1 | **21.2** |
| Writing, 20K-token chat | — | 16.2 | 17.3 |
| Reading a 6K prompt | 587 tokens/s | **640** | 613 (E2B reads it too) |
| Reading a 20K prompt | — | **556** | 544 |
| Chat cache at 32K | 29.5 GB | **4.4 GB** | 4.4 GB + E2B's |

Writing is close to what the Mac's memory delivers: each token reads the whole 25 GB
file, and 19.6 tokens/s is about 490 GB/s. Drafting gets past that, but by less than it
could: checking 2 tokens takes 61 ms and 3 take 69, against 54 for one, mostly the
2–8-token kernel (well short of the weights' bandwidth) and the prompt attention kernel,
which small batches use.

## Tools eval (`vireo-eval`, 35 cases, `--context 32768`)

Run after the two engines became one, beside the Mac's run of the eval models
(`mac.json`):

| Model | Passed | Time for all 35 | A case on average | Tool calls |
|---|---|---|---|---|
| Gemma 4 E2B | 28 | 20.5 s | 0.59 s | 40 |
| Gemma 4 E4B | 32 | 41.7 s | 1.19 s | 39 |
| Qwen 3.5 4B | 34 | 62.0 s | 1.77 s | 50 |
| Qwen 3.5 9B | 33 | 104.0 s | 2.97 s | 40 |
| **Gemma 4 31B** | **35** | **348.4 s** | **9.95 s** | **40** |
| Gemma 4 31B, drafting | 35 | 349.7 s | 9.99 s | 40 |

Gemma 31B is the only model to pass every case, including the ones the small models
miss (`find-then-read`, `three-steps`, `which-report`, `recover`, and the change and skill
cases). It is 3 to 17 times slower a case than the small ones: the change and skill cases
add scoring questions and skills' longer answers, and those are where its time goes.
Drafting doesn't help here: most of a case is reading prompts, tool results and scoring
questions, and writing calls under the grammar, which drafting doesn't do. On the first
24 cases alone (before the change and skill cases), it took 3.96 s a case.
Results: `gemma-4-31B.json`, `gemma-4-31B-draft.json`.

## Short reads (2026-11-11, PLAN.md "Seventh pass")

An agent case is mostly short reads and writing. Nine eval cases (datetime, read, list,
no-tool-writing, find-then-read, git, three-steps, skill-summarize, skill-review),
`--context 32768`, 9/9 before and after:

| | Before | After |
|---|---|---|
| A case on average | 9.0 s | **8.4 s** |
| Reads of 9-64 tokens (78) | 15.5 s | **10.4 s** |
| Reads of 65-256 tokens (58) | 12.8 s | 13.7 s (run-to-run noise) |
| Writing, one token | 53.8 ms | 54.1 ms |
| Checking 2-8 tokens of guesses | 68.8 ms | 70.4 ms |

Short batches get tiles of their own size, and matrices with few rows and long ones
(ffn_down, attn_output) split along the columns. `cargo test --release -p vireo-engine
--test big_matmul -- --ignored --nocapture` with `VIREO_BIG_MODEL` set times a large
model's products.

## The other large models, for a fair comparison

The same quantization as this model's (plain Q6_K, not a "UD" mix), from one publisher,
downloaded 2026-11-11 into the app's model folder beside it and checked:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf` (bartowski) | 23,860,565,728 | `539041d2cf400b02f53a53cae54339ea65103d276645a34a68eac1c8d69a9db3` |
| `Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-Q6_K.gguf` (bartowski) | 23,414,418,720 | `7746e3dafcece07236a3826c8c3ee149a682d6f7fc9f0da9e24a4e481ccdb0bc` |

## Next

- Writing is 37% of a case and at memory speed: drafting (E2B for long answers, tool
  calls under the grammar) is the way past it.
- Qwen 3.8 27B uses the Qwen 3.5 design; its GGUF is here now, not yet run. Drafting
  needs a rewind Qwen's DeltaNet doesn't have.
- 26B-A4B needs mixture-of-experts support; Muse Glimmer 30B needs its own model (its
  GGUF is here now).
