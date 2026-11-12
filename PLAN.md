# Vireo's own engine: plan

Written 2026-11-10. A prototype plan, not a commitment: each phase ends in a decision
point, and the work continues only if the numbers hold up.

**The first version runs on Windows with NVIDIA GPUs (CUDA), tested with small models.**
The Mac (Metal) and other Windows GPUs (Vulkan) come later.

## The idea

Replace the Python server (`server/`) and the inference servers it starts (mlx_vlm,
llama-server) with one Rust program that Vireo owns from top to bottom: the agent,
the tools, and the inference engine that runs the model.

Language alone won't make Vireo faster. A turn's time is spent in the model, and
mlx_vlm and llama.cpp already run near the hardware's limit. The gains come from three
other places:

- **Doing less work.** An engine built only for Vireo can cache, schedule and constrain
  generation in ways a general-purpose server, reached over an OpenAI-style HTTP API,
  can't.
- **Shipping less.** One binary of roughly 20–40 MB instead of a bundled CPython with
  its packages. It starts in about a second, uses less memory, and there's less to sign
  and notarize. Because the kernels are our own, there's no cuBLAS or other large CUDA
  library to ship: the NVIDIA driver is enough.
- **One server on Mac and Windows.** This makes the "This PC" plan possible
  (`docs/dev/windows-native-plan.md`) without bundling Python a second time. Starting on
  Windows means "This PC" is the first thing the prototype delivers.

## Decisions so far

| Decision | Choice | Why |
|---|---|---|
| Language | Rust | One cross-platform binary, no garbage collector, errors caught by the compiler |
| Model format | **GGUF only**, read natively, never converted | One format on every platform, GGUF builds exist for nearly every open model, and the format is simple and documented |
| How much to own | **As much as possible.** Borrow only the commodity layers (below) | The engine, tools and skills designed together are what set Vireo apart |
| First platform | **Windows, NVIDIA GPUs** | The development PC has one, CUDA's tools are the most mature, and llama.cpp's CUDA build is a strong reference |
| First GPU backend | **CUDA**, through our own kernels | Fastest on NVIDIA; Vulkan (other Windows GPUs) and Metal (Mac) follow |
| Test models | **Small models (about 0.5–4B)** | Fast to iterate on, fit easily in 8 GB of VRAM, and cheap enough to run in CI |
| Client contract | Keep today's HTTP + SSE API unchanged | The Mac, Windows and iPhone apps keep working, and the existing tests can check the new server |
| Unsupported models | llama-server stays as a permanent fallback | Users can run any model on day one, and the engine grows only where it wins |

### What this means for MLX

MLX models leave the main path. Featured models and **Discover** point to GGUF
repositories instead of `mlx-community`. For people who already have MLX models
downloaded, Vireo offers the matching GGUF build. Whether mlx_vlm stays as a temporary
fallback on the Mac is still open (see Open questions). This doesn't affect the
Windows-first work.

## The Mac (Metal)

The Mac's own record: the Metal backend (`crates/vireo-engine/src/metal`,
`kernels/*.metal`), its tuning and its measurements. Since 2026-11-11 the Mac builds the
same workspace as Windows (it had its own copy under `Mac/Engine/` for a day); only the
backend and its tuning are the Mac's, everything else is shared.

**Machine:** MacBook Pro, Apple M5 Max (40 GPU cores, neural accelerators in the GPU),
128 GB, macOS 27. Rust 1.99.1 (Homebrew's `rustup`). Reference: llama.cpp b12228's macOS
arm64 build (Metal, with Metal 4 tensor operations on the M5) in `reference/`.

**How it's built.** `src/metal/` is a Metal backend with the CUDA backend's types and
methods; on macOS `crate::cuda` is that module (see `lib.rs`), so the models' GPU passes,
the CLI, the server and the tests are the same code as on Windows. The kernels are
`kernels/vireo.metal` (a port of `vireo.cu`, plus SIMD-group-matrix kernels) and
`kernels/vireo_tensor.metal` (matrix products on the neural accelerators, compiled only
where Metal has them), compiled from source when the GPU opens (~0.4 s), with precise
math. Kernels take raw GPU addresses as on CUDA (buffers resident through a residency
set); dispatches run concurrently, with a barrier only where one touches what an earlier
one wrote (each kernel's written arguments are listed). `VIREO_METAL_PROFILE=1` adds up
GPU time by kernel; `VIREO_METAL_TIMING=1` reports command-buffer GPU time.

**Modes.** Apple GPUs have no dp4a or CUDA tensor cores, so there are no 8-bit kernels.
The default mode's counterpart is f16: a prompt's activations are rounded to f16 and its
matrix products run on the neural accelerators with f16 weights (f32 sums), and prompt
attention runs on SIMD-group matrices over the f16 cache. Exact mode is f32 throughout,
the CPU's numbers. Generating is the same in both (matrix-vector products in f32 on the
exact weights).

**Checked (2026-11-11):** all 95 engine tests pass (each kernel against the CPU; Llama,
Qwen 3.5 and Gemma 4 against llama.cpp; batches against token by token; saved sessions).
Perplexity on 4 × 2,048 tokens (PLAN.md and gguf.rs as text), default / exact /
llama.cpp: Llama 33.636 / 33.635 / 33.558, Qwen 18.297 / 18.297 / 18.316, Gemma 35.369 /
35.378 / 35.385. `vireo-server` runs all three, streaming replies to the app's API.

**Speed** (tokens/s, default mode, against `llama-bench -p 2048 -n 128 -ngl 99`, and
`-d 2048` for generating with a context, run back to back with Vireo; Vireo generating
128 tokens from a short prompt and after a prompt of 2,600–2,700 tokens, and reading
that prompt; 2026-11-11, fourth tuning pass). Llama is no longer tuned or measured (the
Windows side dropped it from its tests as an old model); its code and tests remain.

| Model | Generating: Vireo / llama.cpp | … with ~2,600 tokens before it | Reading a prompt: Vireo / llama.cpp | Vireo CPU |
|---|---|---|---|---|
| Qwen 3.5 0.8B Q4_K_M | ~412 / 353 (117%) | ~383 / 341 (112%) | ~19,700 / 16,700 (118%) | ~40 |
| Gemma 4 E2B Q4_0 | ~204 / 181 (113%) | ~194 / 174 (111%) | ~21,050 / 20,660 (102%) | ~22 |

Runs move by ±3% (the laptop warms up: compare within a sitting). Exact mode generates at
the same speed and reads prompts at ~7,400 / 6,800.

How prompt reading got there (Qwen / Gemma): the straight port 3,750 / 3,850;
SIMD-group matrices for the products and tiled attention over the f16 cache 5,100 /
6,500; the matrix products on the neural accelerators 7,600 / 11,300; flash attention on
SIMD-group matrices and the chunked DeltaNet (chunks of 32, so it fits Metal's 32 KB of
threadgroup memory) 11,600 / 11,900; f16 activations 12,600 / 12,700; concurrent
dispatch ~12,900 / 12,800; tiles of 32 rows × 128 tokens (searched 16–128 × 64–256,
2–8 SIMD groups) and batches of 1,024 ~15,100 / 14,400; attention on the neural
accelerators (`attention_tc`) ~16,700 / 19,600; the DeltaNet scan on SIMD-group matrices
(`delta_chunk_scan_sg`) ~19,100 / –; 128 columns a step in the products ~19,100 / ~19,300;
committing every 128 dispatches (the GPU starting while the CPU encodes the rest)
~19,450 / ~20,300; decoding the weights 32 at a time (`decode32`: a Q4_0 block or a
K-quant sub-block, its scale worked out once and its bits read 16 bytes at a time)
~19,700 / ~21,050. Scoring a text (every token's logits, `vireo-cli perplexity`) also got
faster with it: Qwen 4 × 2,048 tokens in 3.4 → 2.6 s.

And generating (short prompt): the port 338 / 169; concurrent dispatch 351 / 166; one
token's attention in one dispatch (`attention_decode`) 364 / 176; matrix-vector rows
split over 4 SIMD groups (`matvec_split_*`) 380 / 180; gate, up and the activation in
one dispatch (`matvec_glu_*`) ~+1% each; `residual_norm` on 512 threads ~+2% / +4.5%;
wide heads with a SIMD group a position (`attention_decode_wide`) ~394 / ~197;
committing every 128 dispatches ~407 / ~202; Qwen's one-token DeltaNet with 8 threads
to a column (`delta_net_p<8>`) ~412 / ~204.

Tried and left out (slower on the M5 Max): `residual_norm` with the row in registers;
the o and down projections adding into the residual and finishing the norm in the same
dispatch (the last threadgroup's norm plus every threadgroup's extra barriers cost more
than the step saved); one token's attention for a key/value head's whole query group at
once (2–8 heads; the repeated reads come from the cache); bigger query tiles for flash
attention at 64- and 256-wide heads (kept for 512 where there are no neural
accelerators); the DeltaNet scan with 16 or 32 state columns a threadgroup; the tensor
products' weight tile double-buffered (1–3% faster reading prompts, but scoring a text
~30% slower, probably fewer threadgroups a core with 16 KB each, and Gemma's perplexity
not the same to the digit); 32-bit loads for Q6_K's words in the matrix-vector kernels
(kept, no measurable change).

A bug found on the way: the first version of `delta_chunk_scan_sg` zeroed its diagonal
matrix and wrote γ on it without a barrier between, so some first chunks scaled rows of
the state by 0; the model tests caught it (logits off by up to 1.5 in batches of 7), a
new test (`delta_net_in_chunks_matches_token_by_token`) checks the chunked DeltaNet on
its own.

**Fifth pass: agent work (2026-11-12).** Timing a reply in the server showed reading,
not writing, is 80-90% of an eval case: the scoring questions (the change question and
one per skill, ~280 tokens each) were read from empty every time, a new chat read its
whole system prompt again on Qwen, and a batch of 2-128 tokens cost about as much as 128.
Changes, each checked case by case against main (same scores to two decimals, same
choices, same pass or fail):

- `matvec_few` (Metal): 1-4 tokens' products with each weight decoded once for all of
  them, any type, 2 rows a threadgroup split over its 4 SIMD groups; 5-8 tokens as two
  side by side. Used for every 2-8-token input (Q6_K's own kernel stays for 2-3 tokens,
  where it's faster). Per matrix, 2 / 4 tokens against the prompt kernels: Qwen 3.5 4B's
  Q4_K ffn_gate 38 / 51 µs against 123, Gemma 4 E4B's Q4_0 ffn_down 31 / 42 against 197,
  its 32-row ssm_alpha 4 against 38. A whole batch (ms, before → after): E2B 2 tokens
  8.4 → 4.6, 8 tokens 8.6 → 6.4; E4B 19.8 → 7.9, 20.0 → 13.4; Qwen 4B 25.3 → 12.5,
  33.4 → 23.1. Tried: 8 tokens in one dispatch (slower than two of 4 at any row count:
  each SIMD group reads the 8 tokens' activations, ~14× the weights' bytes); 1, 4 and 8
  rows a threadgroup.
- The server (both platforms): up to 3 picker sessions, one a kind of question, each
  going on from the start its last question shares with the next (Gemma rewinds; Qwen
  restores a DeltaNet checkpoint saved where the last two parted), so a skill question
  reads ~50 tokens, not ~280. And a second Qwen checkpoint where a prompt parts from the
  one before (a new chat: the end of the system prompt and tools it shares with the
  last), so a new chat no longer reads ~1,600 tokens again (330 ms on 4B).
- The sampler (both): one pass for the likeliest candidates, the repetition penalty
  applied to the recent tokens alone. A pick over 262K tokens: Gemma's settings 315 →
  35 µs, Qwen's 293 → 24, greedy 243 → 19, with a repetition penalty 2,700 → 41.
- Smaller, Metal: the per-layer input's GELU gate in its matrix product
  (`matvec_split_gelu_*`, Gemma one token, +1-1.5%); `residual_norm`'s two sums with one
  barrier each (6.9 → 6.4 µs, same bits); the first command buffer committed after 16
  dispatches when the GPU is idle (+0.5-1%). `vireo-cli generate`'s argmax was a scalar
  pass slow enough to show in its numbers (~0.3 ms a token).

Six eval cases (datetime, read, list, no-tool-writing, find-then-read, git), main →
now, seconds a case, all 6/6 both: Gemma 4 E2B 0.46 → 0.37, E4B 1.06 → 0.72, Qwen 3.5 4B
1.95 → 1.05, 9B 3.35 → 1.72. Generating, 256 tokens from a short prompt (`vireo-cli`):
Qwen 0.8B ~479, E2B ~223, E4B ~128 (llama.cpp 115), Qwen 4B ~145 (128), 9B ~90 (81).
Reading prompts once warm is ahead of llama.cpp at every length measured (E2B 256 tokens
~20K tokens/s against 13.6K; Qwen 4B ~4.8K against 3.4K); a fresh process's first read
costs ~20-35 ms more (first use of the buffers). Kept out: fewer attention parts at short
contexts (+0.5% there, much slower at 7K tokens); 4 or 8 rows a threadgroup in the
one-token kernels (2 is fastest). Not measured yet: Gemma 4 31B, and anything on CUDA
(the server changes are shared; the kernel changes are Metal's).

**Sixth pass: generating (2026-11-12).** Where a token's time goes: E2B reads 1.41 GB a
token (a floor of ~391 tokens/s at ~550 GB/s), E4B 2.82 GB (~195), Qwen 3.5 4B 2.73 GB
(~201), 9B 5.67 GB (~97): they ran at 57%, 66%, 72% and 92% of that. With every barrier
dropped (wrong numbers, real timing) E2B ran at 354 tokens/s, E4B 183, Qwen 4B 175: the
dependent steps cost 37%, 30% and 17%. A dependent dispatch costs ~2.2 µs against ~0.4
for an independent one; the rest of a step is the GPU draining one kernel and filling
the next. Skipping one kind of barrier at a time bounds what fusing it could give (E2B):
the three norms of a layer +7%, qkv_prep → attention +4.5%, the products → qkv_prep
+3.5%, the GELU gate +2.3%; on Qwen, none. Tried and left out: a serial encoder (E2B +1%,
Qwen -3%); grid-stride matrix-vector kernels (no change, Qwen slower with fewer
threadgroups); the plain layout for E2B's short rows (+10% on one matrix, -5% on
others).

So, rather than shave steps, make tokens cheaper to check and guess more of them:

- `--lookup <n>` (server, on by default, 3): with no draft model, guesses from the chat
  itself, what followed the latest earlier place its last 4 (3, 2) tokens appeared, 3 (2,
  1) of them, checked as a draft model's are, so every token is still the model's own
  pick. Gemma 4 and Llama (Qwen's DeltaNet can't go back). Through the server, 200-400-
  token replies, tokens/s off → on (guesses taken): E2B rewriting a document as a list 217
  → 342-358 (78-80%), returning an edited code file 217 → 445 (94%), a summary 217 → 212-216,
  a new story 219 → 216-217 (6-11%); E4B 125 → 165-196, 125 → 209-269, 125 → 128-131,
  125 → 122-124.
- One token's attention kernels take up to 8 tokens (a grid layer each, its own causal
  length and scratch), not the prompt kernels: a 2-token check on E2B at 500 tokens 6.6
  → 4.9 ms (one token 4.5), E4B 10.2 → 8.2 (7.9); Gemma 4 31B 61 → 53 ms (one 51).
- Qwen's DeltaNet steps through up to 8 tokens with the one-token kernel rather than the
  chunked prompt kernels (~13 µs a layer against ~77): Qwen 4B's 2-token batch 12.2 →
  10.8 ms. Its numbers are now exactly token by token's for short batches.
- `matvec_few` for 3 tokens (they were padded to 4): E2B's 3-token check 6.1 → 5.6 ms.
  Tried and removed: gate, up and the activation in one few-token kernel (level on Q4_0,
  slower on Q4_K, and on Q6_K 31B's 2-token check 53 → 71 ms).

Eight eval cases (the six, three-steps, skill-summarize, skill-review), main → now,
seconds for all eight, the same pass or fail and the same tokens in every case: E2B 6.00
→ 5.34, E4B 12.76 → 11.10, Qwen 4B 18.31 → 16.29, 9B 30.12 → 26.92 (11-13% less; Qwen
from the short batches alone). Next: the K-quants' few-token products (Qwen's are
slower than Q4_0's: decode8 reads a byte at a time), a way back for Qwen so it can
draft (a checkpoint a round), and the per-layer fusions above for Gemma.

**Seventh pass: the large models (2026-11-11).** Gemma 4 31B (Q6_K, 25.2 GB) on the M5
Max. Writing is at the memory's speed (~20 tokens/s is ~500 GB/s) and a long prompt's
products run at 50-55 TFLOPS, against ~62 for F16 weights read straight from device
memory (the accelerators' ceiling here; F16 through the same tile is no faster than
Q6_K, so decoding isn't the limit). Where an agent case's time goes (`VIREO_TIME_LOG`,
nine eval cases): writing 37%, checking lookup guesses 20%, reads of 9-64 tokens 19%
and 65-256 tokens 16% (tool results, new turns, scoring questions), at 2.8-4.7 ms a
token against 1.8 for long prompts: a 42-token read paid for a 128-token tile. Built:

- 32- and 64-token tiles (`SMALL_TILES`) for batches of up to 32 and 64 tokens, the
  same sums to the bit: Q6_K ffn_up at 16 / 42 / 64 tokens 0.59 → ~0.40 / 0.60 → 0.46 /
  0.62 → 0.51 ms.
- Short batches (up to 128 tokens) split along the columns until there are ~1,280
  threadgroups (`SPLIT_TARGET`, `sum_parts` adds the parts in order): ffn_down and
  attn_output have few rows and long ones, 168 threadgroups for 40 cores; ffn_down at
  20 / 64 / 100 tokens 0.77 → 0.33 / 0.91 → 0.45 / 1.02 → 0.63 ms. The numbers move by
  f32 rounding (31B: one of nine replies worded differently, same passes; the small
  models' replies all the same).
- `tests/big_matmul.rs` (ignored; `VIREO_BIG_MODEL`): a large model's products against
  F16 of the same shape, and 1-8 tokens' products.

Nine cases on 31B, 9/9 both: 9.0 → 8.4 s a case, 9-64-token reads 15.5 → 10.4 s; a
31-token prompt 322 → 274 ms; long prompts and writing unchanged. Eight cases on the
small models, the same replies: 9-64-token reads E2B 0.69 → 0.51 s, E4B 1.57 → 1.12,
Qwen 4B 1.22 → 0.97. Tried and left out: two weight tiles in turn, the next step's bytes
loaded ahead, 16-128 rows and 4-8 SIMD groups a tile (all within ±3%: the product alone
takes 0.22 of a 32-token tile's 0.42 ms); Q6_K's own few-token kernel for 4-8 tokens,
reordered to hold fewer registers (checks ~4% cheaper, ~1% of a case); more parts for
one token's 512-wide attention (level at 8.5K tokens). Next for the large models:
writing (drafting with E2B for long answers, and tool calls), and running Qwen 3.8 27B
and Muse Glimmer 30B, whose Q6_K GGUFs are now in the app's model folder (see
`evals/gemma-4-31B.md`).

**Eighth pass: Qwen 3.8 27B (2026-11-12, night).** Its Q6_K GGUF (bartowski, 23.9 GB,
mostly Q6_K and Q8_0) is the Qwen 3.5 design with 48 DeltaNet and 16 attention layers,
3 value heads a key head, 6 query heads a key/value head, untied output, and one
multi-token-prediction layer (`blk.64.nextn.*`) counted in `block_count`, now left out.
`vireo-cli perplexity` put BOS on every chunk whenever a model names one; llama.cpp only
when the model adds one (this model names it and doesn't), which made 27B look wrong
(14.30 against 10.39). Fixed: 10.3902 against llama.cpp's 10.3898 (PLAN.md, 8 chunks of
512), chunk by chunk; the other models' numbers unchanged. Speed, M5 Max, against
llama.cpp b12228 (llama-bench): reading an 8.5K prompt 700-714 tokens/s (llama.cpp
561 at 8,192), writing 19.1-20.4 tokens/s (19.7), a 43-token prompt 159-161 tokens/s.
Nine eval cases: 9/9.

- A batch of 5-8 tokens on a matrix of 64M+ weights takes the short batches' tile
  (`Gpu::tile_beats_few`): matvec_few reads its weights twice for 5-8 (two dispatches
  of up to 4), and the tile costs the same from 2 to 32 tokens. Qwen 27B's reads of
  5 / 6 / 8 tokens in the eval: 118 → 82, 128 → 90, 155 → 91 ms.
- Qwen can go back now, so it drafts from the chat (`--lookup`) as Gemma does: the
  few-token DeltaNet kernel (`delta_net_8`) writes the state before a batch of up to 8
  and after each of its tokens but the last, and the session keeps each layer's
  convolution history and the batch's inputs (`Qwen35Cuda::keep_steps`, ~1.2 GB for
  27B); `rewind` copies back the step wanted and drops the attention layers' later
  positions. A GPU test reads 5 guesses, keeps 0, 2, 4 or 5, and goes on as a session
  that read only those (exact: to 8e-6, 0 for 0 and 5). The CUDA kernel doesn't keep
  steps yet (`can_rewind` is false there). Nine cases, the same replies to the token:
  writing 1,210 tokens 58.4 → 52.9 s (20.7 → 22.9 tokens/s; a check of up to 4 tokens
  56 ms against 48 for one), 10.5 → 9.9 s a case.

Seen, not yet looked into: the first read in each of the first two cases took ~2 s
(7 tokens; a new session's buffers touched for the first time?), and the chat's read
right after the scoring questions costs twice a plain read of its size. Next for Qwen:
its own multi-token-prediction layer as a draft (one layer and the output head a guess,
~1.3 GB against 23.9), which guesses prose too, where the chat's text doesn't.

**Ninth pass: Muse Glimmer 30B on the engine (2026-11-12, night).** Its Q6_K GGUF
(bartowski, 23.4 GB; `muse-glimmer`) runs as a variant of Gemma 4's GPU path
(`Config::muse`), since in GGUF form it is Gemma 4's layer with other choices: the
converter folds the centered norms' 1 + w into the weights and the 3.87 query scale
into `attn_q_norm` (k_norm is all 1s). What differs from Gemma: the embedding
RMS-normed (no weight) rather than scaled; values not normed; attention scaled by
1/sqrt(128); RoPE on the sliding-window layers only (every 4th layer global, NoPE), on
neighbouring pairs (the converter permuted q and k as for Llama; rotating halves gave
perplexity 496); SiLU; a sigmoid gate on each head's output from its own projection
(`attn_gate`, new Metal kernel `sigmoid_mul`, so CUDA refuses Muse for now); a separate
output matrix; logits times `logit_scale` 0.196 (folded into the final norm's weight)
before the softcap of 20. 52 layers, 6,656 wide, 32 query heads to 2 key/value heads,
128-wide heads, a 2,048-token window. Its tokenizer splits words at case changes
(`llama4`, GPT-4o style): a new pre-tokenizer, and tables of upper/titlecase and
lowercase letters (`unicode_case_tables.rs`, `gen_unicode_tables.py --case`; from
Unicode 14, Python 3.11's: letters added since count as caseless).

Checked: `tools/compare_tokenizer.py` (now finds the Mac's binaries) 6,453 of 6,458
runs identical to llama.cpp; of the 5, two are llama-tokenize failing (error -10) and
three are llama.cpp reading `<|start|>` as its control token in raw mode, which it does
for this vocabulary's control tokens even with --no-parse-special (Qwen's and Gemma's
it doesn't); ours reads them as text there, kept. Perplexity (PLAN.md, 8 chunks of 512):
16.913 against llama.cpp's 16.862 on Metal, the same in exact mode, so not our
rounding; llama.cpp's CPU gives 10.81 / 12.84 on the first two chunks against its
Metal's 10.89 / 12.93 (its CPU rounds activations to 8 bits for Q6_K), ours 10.94 /
12.98. Gemma 4 31B, checked before, is 0.3% from llama.cpp's Metal the same way (6,057
against 6,039). It answers a question sensibly, reasoning first (`to=self`) as its
template has it. Speed against llama.cpp b12228: reading a 7.8K prompt 766-855 tokens/s
(llama.cpp 635 at 8,192), writing 22.2-22.8 tokens/s (21.4). Not in vireo-server yet:
its chat format (`<|start|>role<|message|>…<|eot|>`, recipients `to=user` / `to=self`)
and tool calls (`<atem:function_calls>` XML) need a family of their own there.

| Q6_K, M5 Max | Read 8K prompt (ours / llama.cpp) | Write (ours / llama.cpp) |
|---|---|---|
| Gemma 4 31B | 652-675 / 519 | 19.7-20.0 / 19.3 |
| Qwen 3.8 27B | 700-714 / 561 | 19.1-20.4 / 19.7 |
| Muse Glimmer 30B | 766-855 / 635 | 22.2-22.8 / 21.4 |

**Tenth pass: Qwen 3.8 27B drafts with its own multi-token-prediction layer
(2026-11-12, night).** Writing on all three large models runs at 90-95% of what the
matrix kernels reach from memory (Gemma ~476 GB/s, Qwen ~475, Muse ~507, against
~530): one token at a time is at its ceiling, so the way on is guessing. Qwen 3.8 27B
carries a guesser: `blk.64.nextn.*`, mlx-vlm's `qwen3_5_mtp`. Entry i pairs the model's
final normed hidden state after token i with token i + 1, as `fc([RMSNorm(embed) ×
enorm, RMSNorm(h) × hnorm])` (eh_proj split by columns at load, a byte copy, so no
concatenation), then one full-attention layer of the model's kind with its own cache,
`shared_head_norm` and the model's output matrix: a guess at token i + 2.

- The session keeps the layer's cache up with every token it reads (`use_mtp`): after
  each batch, each hidden state pairs with the next token and the last waits for the
  token picked next (the last layer then runs for every token of a batch; about one
  layer more a token). `mtp_draft` pairs the waiting state with the pick (the first
  guess, from the model's own state), then chains the layer's output for the rest;
  the model checks them in one pass as it does the chat's guesses, and `rewind` (the
  DeltaNet steps) cuts both back. Checkpoints keep the layer's count and waiting state;
  an imported chat starts its cache again.
- Measured on its own (`tests/mtp.rs`, three prompts, greedy, 354 rounds of 3): the
  first guess right 86% of the time, 1.93 of 3 taken a round, 8.3 ms drafting a round.
- Nine eval cases, 9/9, every reply the same to the token: 10.5 s a case with no
  drafting (9.9 with the chat's guesses) → 8.3; writing 1,210 tokens ~58.4 s → ~38.4
  (20.7 → ~31.5 tokens/s with the drafting). `--mtp off` turns it off.
- The chat's guesses (`--lookup`) for Qwen only on a large model (5,120 wide or more):
  on Qwen 3.5 4B a few-token check costs 1.5 times a token and writing went 5.5 → 5.9 s
  over eight cases.

**Eleventh pass: tiles for 65-192 tokens, fewer guesses (2026-11-12, early
morning).** Reads of 65-256 tokens (tool results, new turns) were the largest reading
share left on the large models. A 96-token tile for 65-96 tokens and a 192-token one
for 129-192 (two 128-token tiles before), and column splits up to 256 tokens (128
before, so ffn_down kept its 168 threadgroups at 150 tokens): Q6_K ffn_up at 80 tokens
0.64 → 0.56 ms, at 150 1.13 → 0.89; ffn_down at 150 1.19 → 0.89. Nine cases: Gemma
31B's 65-256-token reads 13.7 → 11.0 s (7.7 s a case); Qwen 27B's, alternating with
the old tiles, 21.1 → 19.3 and 21.8 → 20.6 s; small models level or a little faster,
the same passes. Qwen's multi-token-prediction drafting at 2 guesses a round, not 3: a
check of 4 tokens costs 1.8 times one, so the third guess (right 43% of the time)
didn't pay: 8.6 / 8.9 s a case against 9.1 / 9.3, alternating. Gemma 31B with Gemma 4
E2B as its draft (`--draft`) instead of the chat's guesses: 7.9 s a case against 7.8 /
8.4 for two plain runs, every reply the same; long answers a little faster, short ones
a little slower, 3.3 GB more: left off.

Measuring at night: after hours of steady GPU work this Mac ran 5-10% slower than at
the start (every setting, single tokens 47.8 → 50 ms on Qwen 27B), so the numbers above
compare runs made side by side, alternating.

Where the three large models stand (Q6_K, M5 Max; tokens/s; ours against llama.cpp
b12228's llama-bench):

| | Read an 8K prompt | Write, one at a time | Nine eval cases, a case |
|---|---|---|---|
| Gemma 4 31B | 652-675 / 519 | 19.7-20.0 / 19.3 | 9.0 → 7.7 s (lookup drafting) |
| Qwen 3.8 27B | 700-714 / 561 | 19.1-20.4 / 19.7 | 10.5 → 8.6 s (its own drafting layer) |
| Muse Glimmer 30B | 766-855 / 635 | 22.2-22.8 / 21.4 | 10.6 s (five other cases, lookup drafting; see below) |

Writing one token at a time is within 5-10% of what the matrix kernels reach from
memory on all three; long prompts' products run at ~85% of the accelerators' F16
ceiling. What's left is in guessing (a cheaper 4-token check would let Qwen take a
third guess and help Gemma's chat guesses), and the short reads' fixed costs.

**Muse Glimmer 30B in vireo-server (2026-11-12).** A chat family of its own (`Family::Muse`,
`muse-glimmer`), run by Gemma 4's model code. Its format is the template's: turns are
`<|start|>role<|message|>…<|eot|>`; the system turn carries Vireo's system prompt (with the
Mac's note: knowledge cutoff, keep the reply short), `Reasoning strength: low.` (the Mac's
setting for thinking off), the tools' declarations and the valid recipients; results come
back as `<|start|>tool NAME<|message|><tool_output name="NAME">…</tool_output><|eot|>`. Each
assistant message names its recipient first, so the reply loop has the model choose there,
under the call grammar, each option scored whole: `to=user` (plain text from there),
`to=self` (its reasoning, not shown), or a tool and its `<atem:function_calls>` block.
`<|eom|>` opens another message. Muse can't do without reasoning: held to the user (no
`to=self`), it wanted `self` at every turn and called no tool on any case (0 of 5). Given
`self` it reasons briefly (18-144 tokens a reply) and passes all five. Reasoning is capped
at 2,048 tokens a reply, then ended for it. Eval, `--cases datetime,read,list,git,three-steps
--server-arg --context --server-arg 32768`: **5/5**, 10.6 s a case, 10 tool calls (three-steps
read all three READMEs, then the right project's Git). Reading the 1,878-token first prompt
849 tokens/s; writing 26.5-35.2 tokens/s with lookup drafting (53-72% of guesses taken),
against 22-23 one at a time; each call takes ~0.9-1.4 s to write. Results:
`evals/muse-glimmer-30B.json`.

**What limits speed now, and what's next:**

- **Generating** reads the big matrices at memory speed, but a token is still a chain of
  ~8 dependent steps a layer at ~2–3 µs each (a dependent dispatch costs 2–3 µs, an
  independent one 0.2–0.6), and the small matrices (a layer's q, k and o) are bound by
  the latency of their few steps. Next: q, k and v in one dispatch, and the norms folded
  into the projections' inputs.
- **Reading prompts:** Gemma's Q4_0 matrix products are ~60% of its prompt time; its
  small matrices (256 rows: k, v and the per-layer projections) leave the GPU part idle
  and might want a smaller tile of their own. Qwen's DeltaNet is ~28% (the scan 18%, the
  prep 10%, the prep's forward substitution a 32-step chain in each column).
- **Measuring:** compare within a sitting, alternating A and B (`VIREO_*` switches
  removed once settled), and time scoring a text too: it reads every token's logits, a
  prompt-sized product over the whole vocabulary, which a prompt benchmark misses.
- On Macs without neural accelerators (before the M5) the default mode uses the
  SIMD-group kernels (`matmul_sg`, `attention_flash`, `attention_flash2`), untested here.

## Development machine

The PC this starts on (checked 2026-11-10):

| Part | What it is | Used for |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8 GB, compute capability 12.0 (Blackwell), driver 621.14 | The CUDA backend: phase 1 onward |
| Integrated GPU | AMD Radeon 880M | Testing the Vulkan backend later, without a second PC |
| NPU | AMD XDNA (Ryzen AI 9 465) | NPU experiments (see NPUs) |
| Memory | 31 GB | Room for llama.cpp, our engine and the reference tools side by side |

All three kinds of accelerator Vireo might use are on one machine.

**Installed 2026-11-10:**

- Rust stable (`rustup`, the MSVC toolchain)
- CUDA Toolkit 13.5. A test kernel built for compute capability 12.0 (`sm_120`) runs
  correctly on the RTX 5060.
- Visual Studio 2026's **Desktop development with C++** workload and the Windows SDK
  10.0.26100 (Rust and `nvcc` both need them)

**Reference:** llama.cpp b12262's Windows CUDA 13.5 build, in `reference/` (not in
git), for checking our output and speed.

## Progress

| Done | What | Checked by |
|---|---|---|
| 2026-11-10 | GGUF reader (our own memory-mapping, all 32 tensor types) | Unit tests; Llama 3.2 1B Q4_K_M: every tensor lines up with the file |
| 2026-11-10 | Tokenizer: byte-level BPE, Llama 3 pre-tokenizer, our own Unicode tables | `tools/compare_tokenizer.py`: identical ids to llama.cpp on all 3,484 runs (every text file in the repo plus 400 random texts, as prompts and raw) |
| 2026-11-10 | CPU reference: F16/BF16/Q8_0/Q4_K/Q6_K decoding, our own thread pool, the Llama forward pass, `vireo-cli generate` and `perplexity` | Llama 3.2 1B: perplexity 38.3242, the same as llama.cpp to every digit (F32 copy of the model, CPU, f32 cache); next-token probabilities equal to 4 decimals. ~25 tokens/s on the CPU |
| 2026-11-10 | CUDA backend: our own driver bindings (loads `nvcuda.dll` at runtime; no CUDA library linked or shipped), kernels in `kernels/vireo.cu` built to PTX by `build.rs`, the whole Llama forward pass on the GPU (`--gpu`) | Matrix-vector kernels equal the CPU reference to f32 rounding (~1e-6) on Llama's real Q4_K and Q6_K weights; perplexity 38.3242 on the GPU, the same as the CPU and llama.cpp |
| 2026-11-10 | Qwen 3.5 on the CPU: the `qwen35` pre-tokenizer (combining marks in words; our own Unicode marks table), Q5_K and Q4_0 decoding, Gated DeltaNet layers (causal conv1d, L2-normalized q/k, the delta-rule state update, gated RMSNorm), gated attention with partial RoPE in halves | Tokenizer: identical ids to llama.cpp on all 4,310 runs. Qwen 3.5 0.8B: perplexity 25.1818 against llama.cpp's exact 25.1817 (llama.cpp runs a prompt's DeltaNet in chunks: the same arithmetic in another order); next-token probabilities within 0.0005 points |
| 2026-11-10 | Gemma 4 on the CPU: its tokenizer (UTF-8 BPE with "▁" spaces, newline-only splits, `<0xXX>` byte fallback; end-of-turn tokens made control tokens as llama.cpp does), per-layer embeddings, keys and values shared from earlier layers, sliding-window and global layers with their own head sizes and RoPE bases, the extra norms, layer output scale and logit softcap | Tokenizer: identical ids to llama.cpp on all 4,312 runs (and Llama's and Qwen's re-checked after the end-of-turn change). Gemma 4 E2B: next-token probabilities within 0.0002 points; perplexity 30.8273 against llama.cpp's 30.8287 (0.005%) |
| 2026-11-10 | Qwen 3.5 and Gemma 4 on the GPU: Q4_0 and Q5_K matrix-vector kernels, RMSNorm per head (with or without a weight), RoPE in halves and on part of a head, sliding-window attention, the DeltaNet convolution, L2 norm and a whole DeltaNet head step in one kernel, gated attention, GELU, softcap. Gemma's two embedding tables stay in system memory (looked up per token) | Every quantized kernel equals the CPU to f32 rounding on real Llama, Qwen and Gemma weights. Perplexity on the GPU: Qwen 25.1818 and Gemma 30.8273, the CPU's numbers to every digit; GPU tests check llama.cpp's probabilities and the answer for all three models |
| 2026-11-10 | First round of GPU speed work: prompts in batches (matrix-matrix kernels, tiled attention, batched DeltaNet), one CUDA graph per generated token, fused residual/norm and query/key/value steps, faster matrix-vector kernels (see "Speed so far") | Perplexity identical to every digit before and after on all three models; the matrix-matrix kernels equal the CPU to f32 rounding; a prompt read in batches of 7 gives every token's logits within 2×10⁻⁵ of their range of reading it token by token, the same next token every time (Gemma across its sliding window); llama.cpp's probabilities still match with batched prompts. 72 engine tests pass |
| 2026-11-10 | 8-bit activations on the GPU by default (`Precision::Int8`; `--exact` keeps f32): dp4a matrix-vector and tensor-core matrix-matrix kernels for Q4_0, Q8_0, Q4_K, Q5_K and Q6_K, a second PTX for compute 8.0, Q4_0 repacked on upload (see "Second round") | Perplexity within 0.1% of exact on all three models; tensor-core and dp4a kernels agree to ~1e-6 and stay within the rounding of x of the CPU; the France answers unchanged; exact-mode tests still match llama.cpp's exact numbers. 74 engine tests pass |
| 2026-11-10 | An f16 key/value cache in 8-bit mode, prompt attention on the tensor cores, one token's attention split along the context (see "Third round") | Tensor-core and per-token attention over an f16 cache against f64 for head sizes 64–512 and a sliding window; perplexity within 0.13% of exact; exact mode unchanged (still llama.cpp's exact numbers); batches and token-by-token pick the same next token (Gemma: 553 of 554). 75 engine tests pass |
| 2026-11-10 | Prompt speed: deeper tensor-core matrix steps with scales read once, Qwen's DeltaNet split into parallel norms and a column-split state update, the convolution over every token at once (see "Fourth round") | Exact perplexity unchanged to every digit (Qwen 25.0418); 8-bit perplexity within 0.13% of exact; the tensor-core and dp4a kernels still agree to ~1e-6. 75 engine tests pass |
| 2026-11-11 | Prompt speed again: m16n8k32 products, unpacking Q4_K/Q5_K by sub-block pairs and Q6_K by run pairs, Q6_K padded to aligned 224-byte blocks (see "Fifth round") | Perplexity unchanged to every digit, exact and 8-bit; the exact Q6_K kernels still equal the CPU on the padded layout; tensor-core and dp4a kernels agree to ~1e-6. 75 engine tests pass |
| 2026-11-11 | Nsight Compute profiling; `__vsubss4` replaced by cheap sign extension everywhere; Gemma's shared-KV layers run only for tokens whose logits are wanted (see "Sixth round") | Perplexity unchanged to every digit, exact and 8-bit (scoring runs every layer); the Gemma GPU test still matches llama.cpp's exact probabilities with a prompt read through the new path. 75 engine tests pass |
| 2026-11-11 | Where a prompt's time goes (Nsight Systems), then prompt attention rewritten: cp.async loads overlapping the arithmetic, values read with `ldmatrix.trans`, S worked out once for 512-wide heads (see "Seventh round" and "Eighth round") | The f16-cache attention test against f64 for every head size and a window; 8-bit perplexity unchanged to every digit on Llama and Qwen; Gemma's moves by up to 0.1% per chunk either way (the 512-wide scores are added in two halves), the same 0.27% from exact on average. 75 engine tests pass |
| 2026-11-11 | The tensor-core matrix kernels read the next step's bytes ahead into registers (one block per SM, all but Q6_K); Llama's and Qwen's last layer runs past its keys and values only for the token whose logits are wanted (see "Ninth round") | 8-bit perplexity unchanged to every digit on all three (the same integer sums; scoring runs every layer); the same generated tokens. 75 engine tests pass |
| 2026-11-11 | The small kernels: Qwen's convolution in registers (it had gone to local memory), `qkv_prep` with a thread per number and 8 tokens a block, Gemma's F16 per-layer projection as Q8_0 in 8-bit mode (see "Tenth round") | Llama and Qwen: 8-bit perplexity and generated tokens unchanged to every digit. Gemma: the same tokens; perplexity per chunk within 0.3% of before and nearer exact on average (0.14% from it, 0.27% before); next-token probabilities nearer exact too (20.04% against exact 20.67%, before 22.12%). 75 engine tests pass |
| 2026-11-11 | Qwen's DeltaNet for a prompt in chunks of 64 tokens, the WY form (`delta_chunk_prep`, `delta_chunk_scan`; see "Eleventh round") | Exact perplexity the same as token by token to every digit at 2,048 tokens (chunk 1: 16.2601 against 16.2602 on the first build, then identical); 8-bit within 0.25% per chunk, nearer exact on average. A new test case reads Qwen in batches of 200 (across chunks, ending in part of one) against one token at a time: logits within 1e-4 of 31 exact. 75 engine tests pass |
| 2026-11-11 | Phase 2 begins: vireo-server keeps one session between replies and reads only what's new. Llama and Gemma cut their cache back to the start a new prompt shares with it (`truncate`); Qwen's DeltaNet can't be cut back, so its state is saved where each prompt's history ends (`Qwen35Cuda::save`, `restore`, `reset`), and the next turn goes on from there | A new GPU test: a next turn read on top of a cut-back or restored session gives a fresh read's logits (exact: within 3e-5; 8-bit: the same next token, within the batches' rounding). A five-turn chat about a 4,000-token document: the first word in 0.05-0.1 s from turn 2 on, against 1.0-1.5 s reading the whole chat again (Llama 1.52 → 0.07 s, Qwen 1.33 → 0.10 s, Gemma 1.06 → 0.05 s at turn 5). 76 engine tests pass |
| 2026-11-11 | A saved cache per chat: a session's keys and values (and Qwen's DeltaNet at its checkpoint) copied to system memory and back (`export`, `import`, `SavedState`); vireo-server saves the chat in use when a prompt would throw much of it away, or when the app closes, to `%LOCALAPPDATA%\VireoPrototype\chat-cache\<model>\` (a file per chat, written on a thread of its own; 4 GB per model, the oldest go first), and loads the saved chat that shares the most of a new prompt. Llama and Gemma can load part of one (a shared start); Qwen's loads whole | A new GPU test: an imported session goes on to the bit as the one it came from, whole and in part, exact and 8-bit. Two chats about 3,500- and 4,400-token documents taking turns, then a restart: going back to a chat or reopening it loads its cache in 17-66 ms, the first word in 0.06-0.13 s, against 0.8-1.4 s reading it again, on all three models. 77 engine tests pass |
| 2026-11-11 | Tools, step 1: our own grammar matcher (`grammar`: a schema to a byte-level matcher, the vocabulary's tokens in a byte trie for the tokens that may come next, forced text written ahead) | Unit tests, and on all three vocabularies the trie's tokens equal trying each of 128K-262K tokens at every byte of a call; 0.04-0.06 ms a step on average, 0.5 ms at most. On the GPU each model asked about a file writes the expected `read_file` call: 23-30 tokens in 10-12 passes (forced text saves over half). Llama 3.2 1B also filled the optional `max_lines` with 1: a valid call, a poor choice, which the grammar can't catch. 80 engine tests pass |
| 2026-11-11 | Tools, step 2: vireo-server's tool loop. Each family's own form, from its template: declarations in the system turn, Qwen's `<tool_call><function=...><parameter=...>`, Llama's JSON (opened with `<|python_tag|>` or `{`), Gemma 4's `<|tool_call>call:name{key:<|"|>value<|"|>}`; the grammar gained raw text up to an end mark and other punctuation for objects. The model writes freely until a call starts, the grammar writes the call, the app runs the PC's tools (`client_tool_request`, `/api/agent/client-tool-result`) and the server its own (the date), and the result goes back in the family's form, all in the kept cache. A repeated call isn't run again; used-up calls leave the model only an answer. Read-only tools so far | Unit tests for each family's calls and declarations. A first baseline, five chats (read a file, list a folder, the date, 17 × 23 with no tool, find then read): Gemma 4 E2B 4/5, Qwen 3.5 0.8B 2/5, Llama 3.2 1B 1/5. Every call parses and reaches its tool; what's left is the small models choosing the tool (Qwen reaches for `git_*` when unsure, Llama calls one more than needed). In the installed app, Gemma reads a README through the app's own `read_file` and sums it up |
| 2026-11-11 | Tools, step 3: `vireo-eval`, our own eval. It starts vireo-server with each model and stands in for the app: a PC of a few files, folders and a Git project in memory (`pc.rs`), answering the app's tools as `PcTools` does, so any sensible route gets real answers and every run the same ones. Twelve cases (`cases.rs`): the date, reading, listing, finding then reading, whether a file is there, Git status, two without tools, a missing file, adding up a file's numbers, two files, and an instruction planted in a file; each checks the tools, their arguments and the answer. `--server-arg` compares a server feature on and off. Baseline in `evals/baseline.json` | Gemma 4 E2B 12/12 (one call per step needed), Qwen 3.5 0.8B 8/12 (it reaches for `git_diff` when unsure: the date, listing, finding), Llama 3.2 1B 1/12 (24 calls for 12 cases: extra and wrong ones, and answers that drop what the tool said). 0.8 s a case on average. The fake PC's own tests pass |
| 2026-11-11 | Tools, step 4, first round, both behind switches and measured: `--score-choices on` scores each tool's whole name (the grammar offers a branch's options; each is read and rolled back, Qwen through a scratch DeltaNet checkpoint) instead of a token at a time; `--tools-shown scored` asks the model, in a small session of its own, which groups a request needs (files, Git, the date, none: a multiple-choice question with eight worked examples, its letters scored in one pass) and declares only those (no tools only when "none" is 0.7 likely or more) | Eval (baseline / names scored / fewer tools): Gemma 4 E2B 12 / 12 / 12, Qwen 3.5 0.8B 8 / 8 / 8, Llama 3.2 1B 1 / 2 / 4. Scored names change nothing for Qwen: it really does rank `git_status` above `get_current_datetime` for the date (log-probability -0.44 against -2.26), so the tool's name isn't the problem; scoring costs 0.4-0.6 s a case. The picker is sharp on Gemma (0.99-1.00, all right, after the examples: without them it took "What does minutes.txt say?" as needing nothing) and near chance on Qwen 0.8B and Llama 1B, yet Llama gains from the shorter menu. Both stay off by default; fewer tools is the first per-model setting worth having (Llama 3.2 1B) |
| 2026-11-11 | More test models, and token healing. Added Gemma 4 E4B (Google's Q4_0), Qwen 3.5 4B and 9B (Q4_K_M); Llama 3.2 1B left the eval's default set (2024, superseded; still the engine tests' reference). Qwen 3.5 9B scored 4/12, worse than the 0.8B, calling `git_diff` to read a file: forced text was written as its own tokens, so Qwen saw `\n<function=` end in a lone `=`, where its training merges `=` with the name (`=read`). Token healing (on by default, `--heal off`): forced text's last token is left to the model, under the grammar | Perplexity against llama.cpp, 2,048 tokens: E4B 22.81 / 22.85, 4B 11.274 / 11.272, 9B 9.948 / 9.947. The 9B's score for `read_file` went from -2.66 to -0.000 (`git_diff` -12). Eval (healing off / on / + names scored / + fewer tools): Gemma E2B 12/12/12/12, E4B 11/11/11/12, Qwen 0.8B 8/10/10/9, 4B 8/11/11/12, 9B 4/12/12/12 (and 11.7 to 5.0 s a case). Most of what looked like small models choosing badly was this. Scored names help nowhere and stay off; fewer tools is +1 on E4B and 4B, -1 on 0.8B, so a per-model setting, off for now |
| 2026-11-11 | Twelve harder cases for `vireo-eval` (`hard_cases.rs`), counted apart from the easy twelve: three steps (find the project whose README mentions invoices, then its Git status), one line in a 6,000-character log, which of eight logs holds a code, "my 2026 report" beside 2025's, a path relative to home, counting PDFs among other files, two follow-ups on an earlier turn (one needs a file, one needs nothing), explaining `git status` without running it, recovering from a wrong path, a deadline against today's date, and what a diff changed. Cases can carry earlier turns now. Reading every failure by hand found three that were the eval's: the stand-in PC searched for the whole search text in a name where the app's `SearchFiles` looks for each word ("*.pdf" found nothing; it now searches as the app does, matching lines included), `find-then-read` insisted on opening the file though a search's line shows the code word, and `long-log` didn't take "the disk on drive D: was full" | Eval, all 24 (easy / hard), after the fixes: Gemma 4 E2B 20 (12 / 8), E4B 22 (11 / 11), Qwen 3.5 0.8B 16 (10 / 6), 4B 23 (11 / 12), 9B 23 (12 / 11). `recover` fails on four of five: told a path that's wrong, each says the file is missing and stops, never looking around; `which-report` fails on two, which guess a path instead of listing the folder. Scored names (20 / 22 / 16 / 23 / 23) and fewer tools (20 / 22 / 15 / 23 / 22) help nowhere and stay off. A report of every case and score: `evals/eval-report.pdf` |
| 2026-11-11 | A missing path's error says what is there (the app's `PcTools.NotFound`, and the eval's stand-in PC, which gives the same text): the nearest folder above it the chat can reach, what that holds (folders first, the first 30), up to three names there sharing the most words with the missing part (letters and digits apart, the extension left out: "2026_report.pdf" finds report-2026.txt before report-2025.txt), as full paths, and files of the same name a few folders down. Facts only: a last line telling the model to go on with one of them was tried two ways and scored lower | Eval, all 24, four versions against a fresh baseline (20 / 22 / 16 / 23 / 23, the same as before): names alone 21 / 22 / 17 / 23 / 23; full paths and "use it and say so; if not, tell the user what's there" 20 / 22 / 17 / 23 / 23; **full paths alone 21 / 22 / 18 / 23 / 23, kept**; full paths and "Go on with the one that fits the request" 21 / 22 / 17 / 23 / 23. `which-report` now passes on all five (3 before), and in two calls where 4B and 9B took three; `recover` on two (one before: Qwen 0.8B joins 4B). No other case moved, `missing` included. The three still failing `recover` (both Gemmas, Qwen 9B) now name meeting-notes.txt and ask the user whether to read it: a choice to stop, not a wrong path. Results: `evals/not-found-hints.json`. App tests 258 pass |
| 2026-11-11 | Tools, step 2 finished: the tools that change files, with approvals. vireo-server offers `write_file`, `rename_file`, `move_file` and `delete_file` (the app's PcTools runs them, its undo journal recording each reply's changes), and before one runs it asks the user as the Mac's server does: `confirm_required` with a one-line preview ("Create …", "Move … to …"), answered at `/api/agent/confirm`; "Don't ask again" kept in `%LOCALAPPDATA%\VireoPrototype\approvals.txt` (deleting always asks); declined or unanswered in two minutes, the model is told it wasn't done and not to try another way. Before asking, the app checks the change could be made (`check_change`, not a tool the model sees: the file is there, the name free, the folder there), so nobody approves a guess; an app without it is asked as before. One scored question (worked examples, " yes"/" no") decides whether a request asks for a change; only then are the change tools declared and callable (`--changes declared`). `vireo-eval` gained six change cases (`change_cases.rs`: write a note, add a line keeping the rest, rename, find then move, a declined delete, advice without a change), a stand-in PC that changes and a stand-in user who approves or declines, and a rule for the other 24: a reply that changes a file fails | All 30 (older 24 / changes 6), against the older 24's last run (21 / 22 / 18 / 23 / 23): changes always offered (`offered`) Gemma 4 E2B 20 / 4, E4B 21 / 5, Qwen 3.5 0.8B 15 / 3, 4B 23 / 6, 9B 23 / 5; declared always but callable only when asked (`scored`) the same; **declared only when asked (`declared`, kept) 21 / 5, 22 / 5, 16 / 4, 23 / 6, 23 / 5**. The longer tool list was the cost on the older cases (Qwen 0.8B saved a poem to a file, and with `scored` was pushed into `git_status` instead); `declared` keeps those prompts as they were. The check works: on the guessed "Downloads\Acme invoice" no model was asked to approve anything. Still failing: `move-found` on four models (each guesses the name, is told the real one, then asks the user instead of moving it, as in `recover`), and Qwen 0.8B saying a declined file was deleted. Cost: a chat whose next request asks for a change reads its history again once (the declarations change). Results: `evals/changes.json` (`changes-offered.json`, `changes-scored.json` scored `advice-only` before its check took a name without ".docx"). App tests 259 pass |
| 2026-11-11 | Skills, first round. vireo-server reads SKILL.md files as the Mac's `skill_library.py` does (`skills.rs`: the Mac's ten built-in skills from `server/builtin-skills`, the user's from %APPDATA%\InvurseChat\Skills, a user's replacing a built-in of the same name), and offers one when the first tool it names is one the PC has: `summarize-document`, `translate-document` and `code-review` (the calendar, web and spreadsheet skills wait for those tools). `/api/skill-library/commands` lists them, so the app's slash commands appear. A skill is picked by its slash command ("Use the X skill for this:") or by scoring: one short yes/no question per skill, with worked examples of made-up skills. The pick is written into the reply as the model's own `use_skill` call and its result (the skill's text, as the Mac's `use_skill` returns it), after the request: the model reads it just as on the Mac, and the chat's cached history is kept (a skill in the system prompt would change the whole prompt). Qwen 3.5 0.8B said yes to nearly every skill for every request (a skill on 25 of 30 other cases), so each skill is also scored once per model against a request that fits nothing ("What is 17 times 23?") and is picked only 0.4 above that: a model whose yes means nothing picks none, as before skills. `vireo-eval` gained five skill cases (`skill_cases.rs`: summarizing minutes, translating a file into a new one, reviewing a diff with a bug, a slash command, and a plain question about a file that takes no skill) and shows the scores behind each choice with `VIREO_DEBUG_CALLS` | All 35. Older 30, before and after: Gemma 4 E2B 26 / 26, E4B 27 / 27, Qwen 3.5 0.8B 20 / 20, 4B 29 / 29, 9B 28 / 29. Skill cases: 3, 5, 2, 5, 4 of 5. Skills taken on other cases: 2, 3, 0, 5, 4, all fitting ones (`summarize-document` for "sum up this email") and none failing. Misses: Gemma 4 E2B scores summarizing low (0.04 for "Can you summarize board-minutes.txt?", after an example that names a file and fits was added; it was lower before) and misses the review's bug; Qwen 0.8B picks no skill (its slash command still works); Qwen 9B takes `summarize-document` for "how long does the bread bake, going by bread.txt". Cost: 1.03 to 1.26 s a case on Gemma 4 E2B, 6.7 to 9.1 s on Qwen 9B, mostly the scoring passes (a change question and three skill questions of ~350 tokens each, read from scratch every time), with longer answers where a skill applies. Next: the questions' shared start kept in the picker's session, so each reads only its request. Results: `evals/skills.json` |
| 2026-11-11 | The Mac gets the tools: vireo-server's tool loop (`tools.rs`, each family's own call format, token healing), the grammar's scored names, and `vireo-eval` with its 24 cases, ported from the Windows engine (PR #118). On the Mac the server reports `macOS` and answers `get_current_datetime` in local time (`localtime_r`), not UTC | 107 engine tests pass. Eval on the M-series GPU (f16), all five models' checksums as in `evals/MODELS.md` (easy / hard, against the Windows baseline): Gemma 4 E2B 19 (11 / 8; Windows 20), E4B 22 (11 / 11; 22), Qwen 3.5 0.8B 15 (9 / 6; 16), 4B 23 (11 / 12; 23), 9B 23 (12 / 11; 23); the same tool calls in 115 of 120 cases. The two lost points are both `find-then-read`, a close first choice (Windows E4B fails it too) that also fails here in `--exact`, so not the f16 rounding. 0.3-1.3 s a case, against 0.9-5.8 s on the RTX 5060 Laptop. Results in `evals/baseline.json` and `run-baseline.txt` |
| 2026-11-11 | One workspace for both platforms: the Mac's copy of the engine (`Mac/Engine/`, a day apart) folded back in. The Metal backend comes in as the GPU backend on macOS (`src/metal`, standing in for `cuda` there; its kernels in `kernels/*.metal`), with the Mac's fused gate-up-activation for one token and its prompt batch of 1,024 (512 stays on CUDA) behind `cfg(target_os = "macos")`. vireo-server gains the Mac's server-run file and Git tools (`files.rs`, used when an app doesn't run them itself) and attachments (`attach/`), and keeps approvals, the change tools (app-run only until `files.rs` makes changes), the change check and skills. The Mac app builds and finds `vireo-server` here (`target/release`); `Mac/Engine/` is gone. See HANDOFF.md for where things stand | On Windows: every test passes (engine 70, GPU 11 on the RTX 5060, server 54, eval 9; `files.rs`'s three macOS-only tests run on the Mac), and all 35 eval cases pass or fail exactly as before on every model (Gemma 4 E2B 29, E4B 32, Qwen 3.5 4B 34, 9B 33; `evals/one-engine.json`). Qwen 3.5 0.8B leaves the eval's default models, too small to use the tools well (22 of 35; on request still). Not run here: the Metal build, the Mac's tests and the app's Swift tests |
| 2026-11-11 | More tools, the first ones the gauntlet needs. `calculate`: the Mac's calculator ported to Rust (`calc.rs`: exact decimals, "$1,234.50", "18%", round/abs/min/max/sum, several a line; the Mac's own answers, 12 places cut). The Windows app's `read_spreadsheet`, `create_docx` (asks for approval, checked first; the server words its result) and `run_tests`, `run_build`, `run_linter` offered when the app runs them; the server runs only what `files.rs` has, so the Mac app doesn't get them yet. With `read_spreadsheet` and `create_docx` the PC can follow two more skills, `analyze-data` and `write-document`. The skills' control request became "What's the capital of France?" (Gemma 4 E2B said yes to `analyze-data` for "What is 17 times 23?"). After a tool's result the model must say something: Gemma 4 E4B, with the longer tool list, ended its turn on the result three times, answering nothing, so the end-of-turn tokens wait for its first token. `files.rs` runs on Windows too: drive letters and `\\`, the home folder from USERPROFILE without AppData, Git on the PATH, PDFs and Word read with `attach/`'s readers off the Mac (Windows tests of their own; the Mac's run there). `vireo-eval`: four cases for the new tools (`more_cases.rs`: a tip, a spreadsheet with its skill, a memo saved as Word with its skill, a failing test named); `no-tool` takes `calculate` as well; a plain run is two models (Gemma 4 E2B, Qwen 3.5 4B), `--full` all four; `--cases` takes a category (easy, hard, changes, skills, more). The gauntlet's generator (`docs/dev/test-gauntlet/make_aurelia.py`) wraps table cells in their columns: text ran over the next column in two invoices and the prospectus, which PDF readers then gave the models garbled on 2026-11-10 | All 39, four models, against `evals/one-engine.json`: older 35 Gemma 4 E2B 29 / 29, E4B 32 / 32, Qwen 3.5 4B 34 / 35, 9B 33 / 34; the four new cases pass on all four. Time a case 1.65 to 1.82 s (E2B), 10.6 to 13.5 s (9B): five more tool declarations read at every new chat. The regenerated Aurelia files were checked against the answer key (identical key; every invoice, conversion, tracker, guest list, nomination and email as the key has it) and all 15 PDF and Word files read cleanly with the Rust readers. Results: `evals/more-tools.json` |
| 2026-11-11 | What the quick gauntlet still needed. **Lists and objects as tool arguments:** the grammar gained lists in any punctuation (Gemma 4's `[a,b]`, and objects with keys the model names) and values of several kinds told apart by their first byte (a spreadsheet's cell: text, a number, true/false, null); each family writes them as its template does (read from the GGUF files): Gemma 4's `format_argument` (bare keys in alphabetical order, `<|"|>` strings), Qwen 3.5's `tojson` inside `<parameter=...>`, Llama's JSON. That unlocks the app's `write_spreadsheet` (rows, with `start_cell` "append"), `fill_docx_template` (values) and `create_pptx` (slides), each approved and checked first (`check_change`). **Memory:** `remember` and `forget`, run by the server (`memory.rs`, one fact a line in `memories.txt`), the facts added to each chat's instructions in the Mac's words; offered in every chat, apart from the change question (remembering isn't a change to a file). **Chat search:** `search_chats`, run by the app over its own chats (`ChatSearch`, the asking chat left out). **Skills:** `fill-template`, `apply-policy` and `process-invoices` in `server/builtin-skills`, general procedures. A reply may make 20 calls (6 before), and `max_tokens` counts the model's writing, not its calls. **Scripted gauntlet:** `run-gauntlet.ps1` and the app's `--gauntlet` mode (`GauntletRunner`): each prompt a new chat through the app's own ChatStore and PcTools, approvals answered by the script's rule, a report with the answer key's hints after each prompt; the answer key's folder locked in every checkout for the run | Server 55 tests, grammar 11, app 295 (the runner end to end against a stand-in server, chat search, the new checks). Eval, plain run, against `evals/more-tools.json`: memory tools offered everywhere Gemma 4 E2B 33/39 and Qwen 3.5 4B 39/39 (the same), Qwen 8.0 to 8.9 s a case; with the three skills too, 33 and 39 again, 2.2 and 9.6 s a case (three more scoring passes), the new skills taken on three other cases, all still passing (`evals/gauntlet-tools.json`). In the installed app: a model started from Settings › Models, a note written after its approval and undone, a PDF read. Found there: a second copy of the app can start and stops the first one's engine; the Installed tab lists no model kept outside its folders |
| 2026-11-11 | **The quick Aurelia gauntlet through the Windows app on Vireo's own engine**, small models against the Mac's large ones (write-up: `docs/dev/test-gauntlet/runs/Aurelia-Quick-2026-11-11-small-models-notes.md`). Qwen 3.5 4B and 9B ran 12 of its 13 prompts (Q11 left out: no folder tasks yet), one model at a time, with fixes between runs, each measured with `vireo-eval` and its four new office cases (`office_cases.rs`). Fixed: a line after a skill's text to start on step 1 (Gemma 4 E2B described the steps instead); a second control request for skills (E2B took `fill-template` for an introduction); the tools preamble says to remember what people say about themselves; a repeated call refused on its own instead of ending all calls (Qwen 4B's Q5); `fill-template` only for a named template; each run's own chats, memory and approvals (`VIREO_CHATS_DIR` and the like: Claude's packaged app sees a layered `%LOCALAPPDATA%`) | Pass / Partial / Fail on the same 12 tasks: Qwen 3.5 4B 1 / 5 / 6 in 17.6 min, 9B 0 / 7 / 5 in 25.8 min, against the Nov 10 full run's Qwen 3.8 27B 11 / 1 / 0, Gemma 4 31B 9 / 3 / 0, Muse Glimmer 30B 9 / 3 / 0. Every tool call was well formed and reached its tool (memory, PDFs, register rows, Word templates filled with exact values, approvals; up to 21 calls a reply); 9B's card form matched Gemma 31B's. What the small models lack is judgment: both named the contractor the safety winner, and on approving over the limit 4B claimed it had approved and 9B tried to remember the approval. 9B's Q2 analysis was right throughout but ran out of the 2,048-token writing budget before the register. Eval after the fixes: Gemma 4 E2B 33/43, Qwen 3.5 4B 43/43; office cases E2B 0 → 2/4, Qwen 4B 3 → 4/4. Next: thinking in the engine, a per-model writing budget, skill picks on mixed requests |
| 2026-11-12 | Windows tuning for the larger models (Qwen 3.5 4B and 9B, Gemma 4 E4B; see "Twelfth round"): the tensor-core matrix kernels' scales laid out by group so a warp reads them in pairs, their scaling as two fused multiply-adds, and a prompt batch of 1,024 on CUDA as on the Mac | Reading a prompt, against `main` (~2,000 / ~9,000 tokens): Qwen 4B +13% / +15%, 9B +11% / +8%, Gemma E4B +1.5% / +1.5%; generating unchanged. Perplexity (2 × 2,048 tokens of PLAN.md, 8-bit): Gemma E4B the same to every digit; Qwen 4B 11.7149 → 11.7239 and 9B 9.6416 → 9.6389 (exact 11.6965 and 9.6339: the fused sums round differently, within 8-bit mode's spread). Engine tests 70 and GPU 11 pass |
| 2026-11-12 | Windows tuning, second pass (see "Thirteenth round"): one token's attention over an f16 cache rewritten (`attention_part_kv_*`: a block per key/value head and its query heads, a warp per position with the lanes splitting the head, parts sized from the kernel's real occupancy), and `residual_norm` reading a row into registers at once | Generating ~30K tokens deep, against `main`: Qwen 3.5 4B 19.4 → 32.9, 9B 14.5 → 20.8, Gemma 4 E4B 22.3 → 39.9 tokens/s overnight (llama.cpp 21.9, 16.4, 30.1); at full clocks the next morning 46.2 → 62.4, 33.5 → 43.0, 44.6 → 59.2 (llama.cpp 63.8, 41.9, 61.7: level); short chats +1–12%. `residual_norm` gives the same numbers; the new attention is within 1e-8 of f64 (the f16 test gains two cases 6,000 positions deep). Engine 70, GPU 11, server and eval tests pass; the full eval scores as `evals/more-tools.json` but for one borderline case on one model (`find-then-read`, failing as in earlier runs; `evals/windows-tuning.json`), 12.2 s a case against 14.0 |

There's no small Qwen 3.8, but Qwen3.8-27B is the same `qwen3_5` architecture as Qwen 3.5
0.8B, so the 0.8B tests the code the 27B runs.

**Open question, Gemma 4:** a 0.005% perplexity difference from llama.cpp remains
(Llama matches to every digit, Qwen 3.5 to 0.0001); our CPU and GPU agree with each other
exactly. GELU is part of it but not all: exact GELU gives 30.8273, GELU through an f16
table 30.8275, llama.cpp 30.8287. Worth tracking down with a per-layer comparison
(llama.cpp's `llama-eval-callback`-style tensor dumps).

**Speed so far** (RTX 5060 Laptop on mains power, tokens/s, Vireo in 8-bit mode against
llama.cpp b12262 by `llama-bench`, measured back to back on 2026-11-11; generating 128
tokens from a short prompt, reading a prompt of 1,600–1,800 tokens). The laptop's clocks
drift with temperature and power state (both engines moved ~10% between runs that day),
so compare within a row:

| Model | Generating: Vireo / llama.cpp | Reading a prompt: Vireo / llama.cpp | Vireo CPU |
|---|---|---|---|
| Llama 3.2 1B Q4_K_M | ~103 / 83 (125%) | ~2,110 / 3,590 (59%) | ~25 |
| Qwen 3.5 0.8B Q4_K_M | ~127 / 106 (119%) | ~2,600 / 4,100 (63%) | ~28 |
| Gemma 4 E2B Q4_0 | ~67 / 55 (122%) | ~3,040 / 4,710 (65%) | ~6.5 |

After the eighth round (prompt attention; prompts of 1,900–2,200 tokens against
`llama-bench -p 2048`, same day): Llama ~2,350 / 3,860 (61%), Qwen ~2,880 / 4,320 (67%),
Gemma ~3,920 / 4,930 (79%). Generating is unchanged (it uses the per-token kernels).

**Against Ollama too (2026-11-11):** Ollama 0.40.16 (its `llamacpp` runner, 100% on the
GPU, default settings) with the same GGUF files imported by `ollama create` from a
`FROM <file>` Modelfile, timed from its API's own `prompt_eval_duration` and
`eval_duration`. Each request starts with a different number, so its prompt cache can't
reuse the last one. Run back to back with Vireo and `llama-bench -p 2048 -n 128`, one
model loaded at a time, while the GPU ran faster than earlier that day (tokens/s):

| Model | Reading ~2,000 tokens: Vireo / Ollama / llama.cpp | Generating 128: Vireo / Ollama / llama.cpp |
|---|---|---|
| Llama 3.2 1B Q4_K_M | ~2,480 / ~2,600 / ~4,240 | ~131 / ~71 / ~99 |
| Qwen 3.5 0.8B Q4_K_M | ~3,050 / ~2,880 / ~4,770 | ~146 / ~88 / ~123 |
| Gemma 4 E2B Q4_0 | ~4,060 / ~2,930 / ~5,470 | ~77 / ~46 / ~63 |

- Generating, Vireo is ~1.7–1.85× Ollama on all three; Ollama is ~25–30% behind the
  llama.cpp it runs (per-token overhead outside the kernels).
- Reading a prompt, Vireo is ~5% behind Ollama on Llama, ~6% ahead on Qwen and ~39%
  ahead on Gemma. Ollama reads prompts at ~55–60% of `llama-bench` (its default batch of
  512 tokens, and its server around the runner).
- Not quite like for like: prompts are raw text with no chat template on all three, but
  Ollama tokenized Gemma's prompt to 2,076 tokens against 2,168 for Vireo and llama.cpp.
  Gemma's generation was timed from a raw story opening on Ollama and from the chat
  template on Vireo (Vireo's raw run ended after one token); both are short prompts.

Exact mode (`--exact`) is ~20% slower generating and ~2–3× slower reading prompts.
Before the speed work below it was ~40 / 53 / 23 generating and ~40 / 53 / 17 reading a
prompt. Exact perplexity on the GPU is unchanged to every digit.

**First round of speed work (2026-11-10):**

- **Prompts in batches** (`forward_batch`): matrix-matrix kernels (`matmul_*`, tiles of
  128 rows × 64 tokens, weights decoded once per tile in f32), a tiled attention kernel
  per head size (`attention_tiled_64` to `_512`: a block of queries shares each chunk of
  keys and values, online softmax), Qwen's convolution and DeltaNet running a whole batch
  per launch with each head's state in registers. `vireo-cli perplexity` and `generate`
  use it. 4–12× faster prompts; scoring a text is 5–12× faster.
- **One graph per token**: kernels read the position from GPU memory, so one token's
  launches (300–600) are recorded once per session as a CUDA graph and replayed. Fused
  steps: the residual add with the next RMSNorm (`residual_norm`, Gemma's extra norms and
  layer scale included), and each head's norm, RoPE and cache write (`qkv_prep`).
- **Matrix-vector kernels**: several bytes per lane per load instead of one, two rows per
  warp, Q4_K's scales read as words, and small whole numbers made into floats without
  I2F. 35–60 GB/s became 90–120 GB/s.

**Second round: 8-bit activations (2026-11-10, decided: 8-bit by default, exact kept):**

- `Precision::Int8` (the default; `--exact` or `Precision::Exact` for f32): before a matrix
  product, the activations are rounded to 8 bits per block of 32 (`quantize_q8`: a scale
  `max|x| / 127`, plus the scaled sum of each 16 for the K-quants' minimums), once per input
  and shared by the products that read it (q/k/v, gate/up). Generating uses dp4a kernels
  (`matvec8_*`, 4 multiply-adds per instruction); prompts the tensor cores
  (`matmul8_*`: `mma.m16n8k16` on 8-bit numbers, 32-bit sums scaled into f32 per 32 or, for
  Q6_K, per 16). The tensor-core kernels need compute capability 8.0 (RTX 30 series on), so
  `build.rs` makes a second PTX for `compute_80`; older GPUs run the dp4a kernel per token.
  F16 and F32 weights, attention, norms and DeltaNet stay f32.
- Q4_0 is repacked when uploaded (`Gpu::upload_weights`): each row's 16-byte runs of 4-bit
  numbers, then its scales, so the kernels read 8 and 16 bytes at a time instead of 2.
- Accuracy: perplexity 48.45 against exact 48.42 (Llama), 25.07 against 25.04 (Qwen), 46.25
  against 46.23 (Gemma), all within 0.1% (llama.cpp's own 8-bit gap on Llama is 0.26%); the
  France prompt keeps its answer and likeliest first token on all three (Qwen's near-tie
  moves 3 points). The tensor-core and dp4a kernels agree to ~1e-6. Reading a prompt in
  batches or token by token picks the same next token every time; logits drift by up to 8%
  of their range on Gemma, as small differences tip numbers across 8-bit steps.

**Third round: attention (2026-11-10):**

- In 8-bit mode the key/value cache is f16 (half the memory: a 32K-token context of
  Llama 3.2 1B takes 1 GB, not 2; it matters most for the large models), written by
  `qkv_prep`. Exact mode keeps f32.
- A prompt's attention runs on the tensor cores (`attention_mma_64` to `_512`, compute 8.0
  on; flash attention: S = Q Kᵀ and O += P V as f16 products with f32 sums, the online
  softmax in f32 between them, P passed from one product to the next in registers;
  Gemma's 512-wide heads in two halves of the output). Dynamic shared memory (up to ~87 KB)
  through `cuFuncSetAttribute`. Attention went from 36% of Llama's prompt time to 6%.
  Checked against f64 for every head size and a sliding window: within 0.06–0.08% of the
  output's size (the queries and softmax weights rounded to f16, as llama.cpp's flash
  attention does); perplexity 48.48 / 25.07 / 46.28 against exact 48.42 / 25.04 / 46.23.
- One generated token's attention is split along the context (`attention_part_*`, then
  `attention_combine`): about four blocks per SM instead of one per head, the split worked
  out on the GPU from the position so the recorded graph fits every length. Generating
  after a long prompt went from ~78 to ~107 tokens/s on Llama and ~33 to ~57 on Gemma.

**Fourth round: prompts (2026-11-10):**

- The tensor-core matrix kernels take 128 columns per step (64 tensor-core products per
  warp between syncs, not 16) and read each group's scales once; two blocks per SM for
  Q4_0, Q8_0 and Q6_K (Q4_K and Q5_K would spill registers). ~5.5 → ~6–6.7 TOPS on
  Q4_K at 8192 × 2048 × 512 tokens. Measured by switching parts off in a standalone
  benchmark: the per-32 scaling is about half the time (each result needs about as many
  float operations to scale as multiply-adds on the tensor cores: inherent to a scale per
  32 columns on both sides, which llama.cpp uses too), unpacking the weights most of the
  rest. Neither I2F nor shared-memory loads were the limit.
- Qwen's DeltaNet for a prompt in three steps (`delta_l2`, `delta_scan`, `delta_out`):
  the q/k norms and the gated output norm, which don't depend on the state, run for every
  token at once; the state update is split over blocks of 16 columns (the columns are
  independent), with 16 tokens of q and k staged in shared memory at a time and no
  block-wide sums left in the loop. 5 ms → 2 ms per layer for 512 tokens; the same
  arithmetic (exact perplexity 25.0418, unchanged). The convolution runs over every
  token at once too (`conv_silu_parallel`, then `conv_history`).
- Prompts: Llama ~1,900 → ~2,130, Qwen ~2,000 → ~2,690, Gemma ~1,030 → ~1,200 tokens/s.

**Fifth round: prompts again (2026-11-11):**

- The tensor-core products take a whole group of 32 columns at once (`mma.m16n8k32`),
  halving the products and fragment loads for every type but Q6_K (scales per 16).
- Unpacking by bigger units: Q4_K and Q5_K a pair of sub-blocks per thread (two 16-byte
  loads and one scale header for 64 numbers, instead of four headers); Q6_K a pair of runs
  that share their bytes. Q6_K's 210-byte blocks are padded to 224 when uploaded so they
  start 16-byte aligned (`Q6K_BYTES`), which all Q6_K kernels use. Q4_K 6.3 → 7.9 TOPS at
  8192 × 2048 × 512 tokens; Q5_K and Q6_K +13–40%.
- Tried and dropped: `ldmatrix` for the fragments (2% slower than plain loads here);
  f16 tensor-core products with the scales folded into f16 weights (no per-group scaling,
  but f16 products top out at ~8.3 TOPS here against ~12.6 for int8); folding Q4_K's
  integer sub-block scales into the sums (3% at most).
- Measured, not adopted: activation scales per 128 or 256 numbers instead of 32 would
  let the sums stay integers across groups (about half the tensor-core kernels' time is
  the per-32 scaling), at 2–5× the perplexity cost of 8-bit mode today: +0.27–0.74% from
  exact against +0.03–0.13%. A quality decision, left open.
- Prompts: Llama ~2,130 → ~2,540, Qwen ~2,690 → ~3,020, Gemma ~1,200 → ~1,250 tokens/s.
  Exact and 8-bit perplexity unchanged to every digit (the same integer sums).
- The GPU ran at ~2.8 GHz this round (~1.5 GHz in earlier ones: a power setting); llama.cpp
  was re-measured under the same conditions.

**Sixth round: Nsight Compute, and Gemma's shared layers (2026-11-11):**

- With the GPU's performance counters open (an NVIDIA Control Panel setting), Nsight
  Compute on the Q4_0 tensor-core kernel at Gemma's shape: the tensor cores busy ~30% of
  the time, ~28 instructions issued per tensor-core instruction (12 of them the per-32
  scaling, ~5 unpacking, the rest addresses, loops and zeroing), issue half busy with ~1
  eligible warp per scheduler (127 registers: 4 warps each). The SM clock fell to ~700 MHz
  under this load, so fewer instructions also means less power.
- `__vsubss4` (a byte-wise saturating subtract) has no instruction on this GPU and
  compiles to ~8; replaced everywhere by a 3-instruction sign extension (`sext4`,
  `sext6`), the same bytes. Q4_0's dp4a generation kernel went from ~135 to ~200 GB/s
  (Gemma generates ~10% faster).
- Tried in a standalone benchmark: cp.async double buffering with Q4_0's raw 4-bit
  numbers read straight into the fragments (+6% with the cheaper sign extension, not yet
  in the engine); 128-token tiles (too much shared memory and registers).
- The big one came from profiling llama.cpp itself: it runs Gemma's shared-KV layers
  (15–34 of 35 in E2B) only for the tokens whose logits are wanted. Those layers keep no
  keys or values, so a token's pass through them only makes its own logits. Vireo now
  does the same in `forward_batch`: a prompt runs layers 0–14 for every token and 15–34
  for the last token alone (none at all for chunks without logits; every layer for
  every token when all logits are asked for, as in scoring). Gemma reads prompts 2.6×
  faster (~1,185 → ~3,100 tokens/s); its llama.cpp test still matches.

**Seventh round: where a prompt's time goes now (2026-11-11, Nsight Systems):**

Reading a ~1,900–2,200-token prompt (the first 7,000 characters of this file) with
`vireo-cli generate --gpu -n 1`, kernel time by kind:

| Model (tokens/s) | Matrix products | Attention | Other |
|---|---|---|---|
| Llama 3.2 1B (~2,300) | 81% (`matmul8_q4_k` 70%, `_q6_k` 12%) | 8% (`attention_mma_64`) | `quantize_q8` 4%, `qkv_prep` 3%, `swiglu` 2% |
| Qwen 3.5 0.8B (~2,750) | 53% | 11% (`attention_mma_256`) | `delta_scan` 24%, other DeltaNet 6%, `quantize_q8` 3% |
| Gemma 4 E2B (~2,900) | 43% (`matmul8_q4_0`) + 5% `matmul_f16` | **41%** (`_512` 30%, `_256` 11%) | `quantize_q8` 4%, `qkv_prep` 3% |

- The GPU is busy 95–99% of the prompt's span and copies take under 1%: launch gaps and
  Gemma's per-layer embedding uploads don't matter.
- The attention kernels for wide heads are slow: ~3.4 TFLOPS at head size 64 (Llama),
  ~1.2–1.6 at 256 and ~0.4–0.5 at 512 (Gemma's three global layers in 0–14: 4–29 ms per
  512-token chunk). Likely causes, from the code: values transposed into shared memory
  one f16 at a time (bank conflicts; `ldmatrix.trans` would read them untransposed);
  key/value chunks of 32 positions loaded synchronously with no overlap (each load
  waited on before the next); 2 warps per block at 512 and ~87 KB of shared memory, so
  one block (2 warps) per SM; S = Q Kᵀ computed twice at 512 (one block per half of the
  output); and every query head of a group loading the same keys and values again.
- `matmul_f16` (Gemma's per-layer embedding projection, f32 arithmetic) is ~1.7 TFLOPS.

**Eighth round: prompt attention (2026-11-11):**

- `attention_mma` loads keys and values with cp.async, as they lie in the cache: a
  chunk's values load while its S = Q Kᵀ runs and the next chunk's keys while P V runs,
  so one buffer of each is enough. Values are read for the second product with
  `ldmatrix.x4.trans` instead of being transposed into shared memory. The output's
  rescaling is skipped when no row's largest score moved (multiplying by 1: the same
  numbers). The longest rows' blocks start first.
- 512-wide heads: two warps share each 16 rows, each working out S over half the head;
  the halves are added through shared memory (in the same order by both, so they agree)
  and each warp makes half the output. S is worked out once instead of twice. Shapes
  measured (rows × warps per row × positions per chunk): 2 × 2 × 16 best at 512
  (1 × 4 × 32, 3 × 2 × 16 and 2 × 4 × 16 within noise or behind); the old 4 × 1 × 32 stays
  best at 64, 128 and 256 (fewer rows, more rows, 16-position chunks and split warps all
  measured slower).
- Attention per ~2,000-token prompt: Gemma 305 → ~75 ms (512-wide 225 → ~41, 256-wide
  80 → ~34), Qwen 88 → ~30 ms, Llama 68 → ~44 ms. Prompts: Gemma ~2,970 → ~3,920 tokens/s
  (+32%), Qwen ~2,720 → ~2,880 (+5%), Llama unchanged within noise. On a ~10,000-token
  prompt, where attention is a bigger share: Gemma 1,600 → 3,190 (2.0×), Qwen 2,080 →
  2,500 (+20%), Llama 1,870 → 1,960 (+5%).
- Not done: one block for all the query heads that share keys and values (Gemma E2B's
  8 heads read one set): it would cut loads, not arithmetic, and loads no longer wait.

**Ninth round: matrix kernels' reads, and the last layer (2026-11-11):**

- `matmul8_*` read a step's weights and x into registers, unpacked them into shared
  memory, synced, and multiplied: the reads' wait was never hidden (Q4_K and Q5_K ran one
  block per SM). Now each step's raw bytes are read before the previous step's products
  (`fetch`) and unpacked after them (`store`). That needs one block per SM: Q4_0 and
  Q8_0 move from two blocks to one with the read-ahead and are faster for it; Q6_K
  stays at two blocks without it (one with it measured 3% slower). No spills (ptxas on
  sm_120: 252–255 registers). Kernel time on a ~2,000-token prompt: Q4_K −9%, Q5_K −3%,
  Q4_0 −6%, Q8_0 −25%. The same integer sums, so the same results.
- The last layer leaves nothing for later tokens but its keys and values (Llama) or its
  keys, values or DeltaNet state (Qwen). So it runs its attention input (Llama) or its
  whole mixer (Qwen) for every token, and the rest only for the token whose logits are
  wanted: not at all for chunks without logits, every token when all are asked for
  (scoring). Gemma did this already for its shared layers. Llama's last-token attention
  then runs on the per-token kernel (no f16 rounding of the query and softmax), so its
  next-token probabilities move slightly (25.35% → 25.49% on the test prompt; the same
  tokens); Qwen's are unchanged to 4 digits.
- Prompts (median of 3, old and new builds alternating): ~2,000 tokens Llama 2,608 →
  2,928 (+12%), Qwen 3,194 → 3,362 (+5%), Gemma 4,330 → 4,530 (+5%); ~10,000 tokens +11%,
  +4%, +3%. Against `llama-bench -p 2048` right after: Llama ~2,910 / 4,240 (69%), Qwen
  ~3,350 / 4,780 (70%), Gemma ~4,530 / 5,420 (84%).

**Tenth round: the small kernels (2026-11-11):**

- A fresh profile (after the ninth round) showed kernels doing little work slowly:
  `conv_silu_parallel` ~0.4 ms a call (5% of Qwen's prompt time), `qkv_prep` 0.4 ms
  (4% of Llama's), and Gemma's `matmul_f16` (its per-layer embedding projection, F16
  weights in f32 arithmetic) 8% of Gemma's.
- `conv_silu_parallel`: its window arrays were in local memory (38 local stores in the
  SASS: the kernel size was known only at run time), so it ran at 200 instructions per
  output. The size is now a template constant (`CONV_DISPATCH`, 2–8), and a thread walks
  16 tokens of its channel with the weights and last inputs in registers: 27.9 → 6.3 ms.
  `conv_silu` (Qwen's per-token step) and `conv_history` had the same local memory and
  get the same fix. Same sums, same order: the same numbers.
- `qkv_prep`: a thread per number of a head (up to 256; Llama's 64-wide heads left 3 of 4
  threads idle), each RoPE pair rotated by one thread, and a block for 8 tokens of a head
  so each pair's `powf` is worked out once, not per token. Llama 25.8 → 9.6 ms, Gemma
  20.1 → 12.5. The same numbers (block_sum counts missing warps as zeros).
- Gemma's per-layer projection gets a Q8_0 copy on upload (`GpuMatrix::upload_q8_0`,
  rounded as llama.cpp's `quantize_row_q8_0`), used in 8-bit mode: 35 → ~8 ms on the
  tensor cores. Its next-token probabilities moved nearer exact, not away (20.04% against
  exact 20.67% and 22.12% before on the test prompt; 97.24% against 97.43% and 97.11% on
  another).
- Prompts against the last commit (`main`, before the ninth round too; median of 3,
  alternating): ~2,000 tokens Llama 2,583 → 2,962 (+15%), Qwen 3,157 → 3,438 (+9%), Gemma
  4,283 → 4,815 (+12%); ~10,000 tokens +13%, +8%, +9%. So this round gives ~2%, ~4% and
  ~7% over the ninth. Generating unchanged.

**Eleventh round: Qwen's DeltaNet in chunks (2026-11-11):**

- `delta_scan` (28% of Qwen's prompt time) ran each head's state 128 × 128 token by
  token: 512 dependent steps a chunk with ~10 warps an SM, 2.35 ms a call.
- Now the chunked (WY) form, as llama.cpp and flash-linear-attention: over 64 tokens from
  a state S0, the δ's solve a triangular system, so δ = U − W S0 with W and U independent
  of S0 (derivation at `delta_chunk_prep` in `vireo.cu`). `delta_chunk_prep` works out
  W, U, P (masked, decayed q·k), Q̂, K̂ and the chunk's decay for every chunk and head at
  once: k·k and q·k in 4 × 4 register tiles, then a forward substitution with a thread per
  column of [W | U] (a column needs only its own earlier rows: no syncs; the sum in 4
  parts). `delta_chunk_scan` runs the chunks in order, each head's columns split over
  blocks of 16: δ = U − W S0, O = Q̂ S0 + P δ, S = γ S0 + K̂ᵀ δ, all f32. Scratch: ~18 MB
  for a 512-token batch (Qwen 3.5 0.8B).
- Tuned with Nsight Compute: the scan was held by shared-memory instruction throughput
  (MIO throttle), so a thread takes 2 rows × 8 columns and reads 16 bytes at a time (99
  → 67 ms); the prep fit one block an SM until q was read from GPU memory and A took k's
  place (63 → 39 ms). Tried and dropped: blocks of 8 columns (twice the blocks, 99 ms).
- DeltaNet per ~2,000-token prompt: 182 → ~123 ms (state update 166 → ~106, the norms
  as before). Prompts against `main`: ~2,000 tokens 3,166 → 3,751 tokens/s (+18% with the
  ninth and tenth rounds; this round ~9%), ~10,000 tokens 2,731 → 3,162 (+16%).
  `llama-bench -p 2048` right after: 4,740, so ~79%.


**Twelfth round: the larger models on Windows (2026-11-12):**

Measured first with three prompts (a 27-token story opening generating 128, ~2,100 tokens
of this file generating 64, ~9,200 tokens generating 64) against `llama-bench -fa 1 -p
2048,8192 -n 128`, run back to back the same night (tokens/s):

| Model | Reading ~2,100: Vireo before → now / llama.cpp | Reading ~9,200: now / llama.cpp (8,192) | Generating 128: Vireo / llama.cpp | Generating ~9K deep: Vireo / llama.cpp |
|---|---|---|---|---|
| Qwen 3.5 4B Q4_K_M | 727 → 822 / 1,057 (69% → 78%) | ~795 / 1,009 (79%) | ~36.6 / 30.1 (122%) | ~29 / 27.7 |
| Qwen 3.5 9B Q4_K_M | 443 → 493 / 630 (70% → 78%) | ~458 / 611 (75%) | ~22.8 / 19.2 (119%) | ~18 / 18.1 |
| Gemma 4 E4B Q4_0 | 1,584 → 1,606 / 1,776 (89% → 90%) | ~1,489 / 1,688 (88%) | ~45.3 / 34.9 (130%) | ~34.5 / 33.1 |

(The deep-generation llama.cpp numbers are `-n 64 -d 9000` from earlier that night.)

- Where Qwen's prompt time goes (Nsight Systems, 4B, ~2,100 tokens): the matrix kernels
  80% (`matmul8_q4_k` 52%, `_q5_k` 18%, `_q6_k` 10%), DeltaNet ~10%, attention 3%. The
  9B the same, at ~6.3–7 TOPS for every type. Nsight Compute on `matmul8_q4_k` (9B):
  issue slots 34% busy, two warps a scheduler, and the top stall MIO throttle (1.4 per
  issue), with the load/store pipe the busiest (44%) ahead of the tensor cores (24%).
- **Kept:** the scales in shared memory by group, then row or token (`w_sm[group][row]`
  holding a row's scale and minimum together; `x_scale` and `x_sum` `[group][token]`),
  so a warp reads them as float2 pairs without bank conflicts, and the activations' sums
  per 32 columns added as they're stored (a shuffle with the neighbouring lane) instead
  of per 16 at every read: the same numbers. Qwen 4B +7%, 9B +7.5%.
- **Kept:** each group's scaling as `acc = fma(c, ws·xs, acc)` and `acc = fma(−wm, xsum,
  acc)` instead of `v = c·(ws·xs) − wm·xsum; acc += v` (I2F + 2 FMA a result instead of
  I2F, 2 FMUL, FFMA, FADD in the SASS). Qwen 4B +3.4%. It rounds differently (see
  Progress for the perplexity).
- **Kept:** the prompt batch at 1,024 on CUDA too. With 512, a 2,560-row matrix is 160
  tiles over 26 SMs (one block an SM), so its last wave is 4 blocks, and Q6_K's (two
  blocks an SM) 3.08 waves' work in 4. Qwen 4B and Gemma E4B +2.5%, the 9B unchanged.
- **Tried and dropped:** whole numbers made into floats with the 2²³ trick instead of
  I2F (−4%); each thread's two 4-byte fragment words side by side in shared memory, so
  one 8-byte read gets both (the same sums in another column order; −4%, the extra
  4-byte stores cost more than the reads saved, with rows padded to 144 or 160 bytes);
  `ldmatrix.x4` for the fragments (−5%, as in the fifth round); Q6_K at one block an SM
  with the read-ahead (no change).
- Next for prompts: the matrix kernels are still at ~7 TOPS against ~11 for llama.cpp's
  (estimated from its whole-model speed). Fewer shared-memory instructions remain the
  lever (the fragments are 16 4-byte reads a group a warp; the stores are 16-byte), or
  more warps a scheduler, which needs fewer than 255 registers.


**Thirteenth round: generating in long chats, and the small kernels (2026-11-12):**

Three prompts again (a 27-token story opening, ~2,100 and ~31,000 tokens of the
repository's docs, the last with the story opening after it so the 9B keeps writing),
`main` against this round, alternating, then `llama-bench -fa 1` (tokens/s):

| Model | Generating short: main → now / llama.cpp | ~2,100 deep | ~31K deep: main → now / llama.cpp | Reading ~2,100: now / llama.cpp | Reading ~31K: now / llama.cpp |
|---|---|---|---|---|---|
| Qwen 3.5 4B | 38.2 → 38.7 / 30.1 | 35.6 → 37.9 | 19.4 → **32.9** / 21.9 | 867 / 1,057 (82%) | 657 / 881 (75%) |
| Qwen 3.5 9B | 22.3 → 22.9 / 19.2 | 22.1 → 23.0 | 14.5 → **20.8** / 16.4 | 489 / 630 (78%) | 420 / 574 (73%) |
| Gemma 4 E4B | 45.0 → 47.7 / 34.9 | 41.0 → 45.9 | 22.3 → **39.9** / 30.1 | 1,592 / 1,776 (90%) | 1,144 / 1,465 (78%) |

- Where a deep token's time went (Gemma E4B, 31K tokens, Nsight Systems with
  `--cuda-graph-trace=node`): one token's attention 27 of 48 ms, nearly all in the 7
  global layers (3.5–5.8 ms each, where reading their cache once takes ~0.5 ms; the 35
  sliding layers 40 µs each). `attention_part` gave each thread a position and read
  its whole 1 KB key row (uncoalesced), and every query head read its key/value head
  again (4× on E4B and the Qwens).
- **`attention_part_kv_64` to `_512`** (f16 cache, so 8-bit mode; exact mode keeps
  `attention_part`): a block per key/value head and set of up to 4 of its query heads, so
  each key and value is read once for them all; a warp per position, its lanes splitting
  the head (each lane's numbers contiguous, so a warp reads a row whole); 4 positions a
  step (2 for 512-wide heads, by registers), all read before any is used, the 4 heads'
  scores summed across the warp together and one rescaling of the sums a step; an online
  softmax in base 2 per warp, the warps merged at the end into `attention_part`'s
  partial rows for `attention_combine`. Parts: as many as fill the GPU in one wave at the
  kernel's real occupancy (`cuOccupancyMaxActiveBlocksPerMultiprocessor`: 2 blocks an SM
  at 512, 221 registers; it first ran 104 blocks in two waves of 52), up to 64 (no more
  than a sliding window fills 32 positions of each). E4B's global layers 3.5–5.8 ms →
  ~0.53 ms (~245 GB/s); the sliding ones 40 → ~20 µs. In steps: the new kernel 21.1 →
  31.7 (E4B) and 18.3 → 28.3 (Qwen 4B); one wave and two positions a step 33.8 → 35.2;
  scores together and one rescaling 35.3 → 38.8 and 30.5 → 32.2; four positions a step
  for heads up to 256 +0.5% on Qwen. Within 1e-8 of f64 (the f16 attention test gains
  two cases 6,000 positions deep); greedy replies drift from `main`'s after tens of
  tokens, as any change of rounding does.
- **`residual_norm`**: one token's norm is one block of 256 threads, and its time was the
  reads' latency (16–26 µs a call, 127 calls a token on E4B: 3.3 ms, at the ~700 MHz the
  SM runs at between kernels). Rows of up to 24 numbers a thread are now read into
  registers all at once: the same sums in the same order, so the same numbers. Short
  chats: E4B +7%, Llama +3%, Qwen 4B +1%.
- **Tried and dropped:** `attention_combine` reading its parts' maxima in parallel (no
  change); the 8-bit matrix kernels with two sets of tiles (unpacking the next step
  while this one's products run, one barrier a step, ~66 KB of dynamic shared memory):
  Qwen −1%, Gemma's Q4_0 −8%; prompt attention at head size 512 with S summed in two
  halves (shorter chains of tensor-core steps): +0.7%, within noise.
- Measuring: `nvidia-smi` polling once a second while a benchmark runs makes the GPU's
  clock swing between ~180 MHz and ~2.6 GHz and cut prompt speed to a third. Don't
  sample clocks during a measured run.

**The same, at full speed (2026-11-12, morning):** every number above (and in the
twelfth round) was measured overnight with the laptop in a slower power state: by
morning, on the Performance power scheme, both engines ran ~2.5–3.5× faster. Measured
again the same way (tokens/s; `main` → this round / llama.cpp):

| Model | Generating short | ~31K deep | Reading ~2,100 | Reading ~31K |
|---|---|---|---|---|
| Qwen 3.5 4B | 86.6 → 90.8 / 77.0 (118%) | 46.2 → 62.4 / 63.8 (98%) | 2,757 → 3,076 / 3,526 (87%) | 2,072 → 2,235 / 2,699 (83%) |
| Qwen 3.5 9B | 50.4 → 53.7 / 47.1 (114%) | 33.5 → 43.0 / 41.9 (103%) | 1,446 → 1,626 / 2,232 (73%) | 1,308 → 1,417 / 1,819 (78%) |
| Gemma 4 E4B | 80.9 → 90.0 / 78.9 (114%) | 44.6 → 59.2 / 61.7 (96%) | 5,519 → 5,536 / 6,396 (87%) | 3,915 → 3,929 / 4,506 (87%) |

The gains over `main` hold (deep generation +28–35%, Qwen's prompts +8–12%), but the
comparison with llama.cpp shifts: at full clocks Vireo's per-token lead in short chats is
14–18% (not 19–37%), and deep in a long chat the two are level (96–103%), not 27–50%
apart. Slow clocks favour Vireo's one-graph-per-token path, so benchmark in the
Performance power scheme and say which one was used.

**Long chats (2026-11-11):** vireo-server's context is now automatic by default: as
many tokens as the GPU's free memory holds once the model is loaded (leaving 1 GB and a
tenth of what's free), up to the model's trained length and 32,768, the length checked
below. On the RTX 5060 all three test models get 32,768 (cache 1.1 GB for Llama, 0.6 GB
for Gemma, 0.4 GB for Qwen). The prototype app's Settings › Your Host has the choice
(Automatic, or 4K to 32K).

| Model | Perplexity, one 32K window: Vireo / llama.cpp | at 4K | Reading ~25K tokens: Vireo / llama.cpp (tokens/s) | Generating 25K deep: Vireo / llama.cpp |
|---|---|---|---|---|
| Llama 3.2 1B | 18.3595 / 18.3520 (+0.04%) | 24.2976 / 24.2865 (+0.05%) | 1,702 / 2,318 (73%) | ~31 / 49 (63%) |
| Qwen 3.5 0.8B | 12.6015 / 12.6022 (-0.01%) | 17.8125 / 17.7922 (+0.11%) | 2,371 / 3,511 (68%) | ~62 / 85 (73%) |
| Gemma 4 E2B | 23.7043 / 23.6904 (+0.06%) | 31.9485 / 31.9462 (+0.01%) | 2,596 / 3,635 (71%) | ~29 / 56 (52%) |

(Perplexity: the repository's own docs and Python, `vireo-cli perplexity -c 32768 --chunks 1
--gpu` against `llama-perplexity -c 32768 --chunks 1 -ngl 99`, both 8-bit. Speeds:
vireo-server's own log for a ~25K-token chat against `llama-bench -p 25000 -n 64 -d 0,25000
-fa 1`.) Accuracy holds at 32K: no further from llama.cpp than at 4K. Reading a long
prompt keeps the same share of llama.cpp's speed as a short one; the first reply of a
25K-token chat starts after 10-15 s, and the next ones in 0.1-0.2 s (the kept cache).
Generating deep in a long chat fell behind, though: short chats were 19-25% ahead of
llama.cpp, a 25K-token one 27-48% behind, in one token's attention (`attention_part_*`).
The thirteenth round's `attention_part_kv` fixed that: ~31K tokens deep the larger
models now generate level with llama.cpp at full clocks (96–103%).

**What limits speed now, and what's next:**

- **Reading a prompt** is at about 70% (Llama), 79% (Qwen) and 89% (Gemma) of llama.cpp.
  For Llama and Qwen it's the matrix products (Q4_K alone is 71% of Llama's time). Qwen's
  DeltaNet is now ~20%: the scan's parallelism is fixed by heads × columns (128 blocks
  on the 0.8B; the 27B has more heads), and its products could go to the tensor cores
  (TF32, at some cost in precision). What's left in the matrix products is
  the tensor-core kernels' instruction count (the per-32 scaling is half of it and is
  the same in llama.cpp, so it's the rest: unpacking, addresses, zeroing) and Q6_K's
  reads, still not overlapped (cp.async into a second buffer would avoid the
  registers). Then the smaller kernels: `quantize_q8` (~4%) could be folded into the
  norms and `swiglu` that make its input.
- **Generating**, at full clocks, is 14–18% ahead of llama.cpp in short chats and level
  (96–103%) ~31K tokens deep on the larger models (thirteenth round). What's left per
  token is mostly the weights' matrix-vector products (Gemma E4B, in the slower power
  state: 12.8 of 26 ms, at ~270 GB/s), then deep in a chat its attention.
- **Reading a prompt**, at full clocks, is at 73–87% of llama.cpp at ~2K tokens and
  78–87% at ~31K (the 9B furthest behind). For long ones
  prompt attention is 30–35% of the time there (`attention_mma_256` and `_512`, ~3–4
  TFLOPS), one block of 4 warps an SM (~70 KB of shared memory each), with "wait" the
  top stall. More warps an SM needs a different tiling (the queries out of shared
  memory, or smaller key/value chunks with more rows); the cheap changes measured flat.
- **The chat cache, for later passes** (alongside this tuning):
  - Saved files: smaller still. Keys and values saved in 8 bits (half of f16; lossy, so
    measured first), or a saved chat written as what it adds to its last save instead
    of the whole chat again each time. A size limit per file (a 32K-token Llama chat is
    1 GB, a quarter of the 4 GB budget).
  - Qwen: a DeltaNet checkpoint at the system prompt's end too, so a new chat with the
    same system prompt loads it (Llama and Gemma already do), and a checkpoint at a
    retry's start as well as the history's end.
  - Gemma: positions kept from a window before the history's end, so a saved chat loads
    in part (an edited earlier message) and not only whole.
  - More than one chat in GPU memory when there's room, so switching between two needs no
    copy at all; saved states in pinned memory for faster copies.
  - The keys and values on the GPU in 8 bits, which would double the context the same
    memory holds.
  - Measure what switching and saving cost in the app (copy, disk, first word), and
    how big the budget should be by default.
- **For larger models**: splitting a model between GPU and system memory (Muse Glimmer
  30B), mixture-of-experts layers (Gemma 4 26B-A4B, Qwen 3.6 35B-A3B), more than one GPU.
- Nsight Compute: the counters are open on this PC now (all users); set them back to
  administrators only when the tuning is done.

**NPU check (2026-11-10):** the AMD NPU's driver is installed and working
(32.0.203.329), with AMD's XRT runtime (`xrt_coreutil.dll`), but not Ryzen AI Software.
The Windows App SDK 2.x runtime the WinUI app uses is installed, and it includes Windows
ML, which fetches AMD's execution provider for ONNX models when asked. So the NPU is
reachable without installing anything by hand, but only for ONNX models through Windows
ML / ONNX Runtime, not for our own kernels or GGUF files directly. The first use is
still embeddings (already ONNX); running a chat model there means converting it to AMD's
format, which is real work. Left for later, as planned (phase 6).

**Comparing with llama.cpp, lessons learned (2026-11-10):**

- On a quantized model, llama.cpp rounds activations to 8 bits for its matrix products
  (CPU and CUDA alike), so its numbers are slightly off from exact arithmetic: on Llama
  3.2 1B Q4_K_M, perplexity 38.42 against the exact 38.32, and next-token probabilities
  a few percent apart. Our CPU reference does exact f32 arithmetic. To compare exactly,
  have llama.cpp make an F32 copy (`llama-quantize --allow-requantize <in> <out> F32`):
  llama.cpp's F32 path doesn't round, and our reference gives the same numbers on the
  copy and the original.
- Use `-dev none -ctk f32 -ctv f32 -fa off` for exact llama.cpp numbers. `-ngl 0` alone
  still uses the GPU.
- `llama-perplexity -f` reads the file in Windows text mode (`\r\n` becomes `\n`);
  `llama-tokenize -f` doesn't. Give perplexity texts `\n` line endings.
- llama-server always adds BOS, even with `add_special: false`. Send its requests from
  something that writes UTF-8 (PowerShell 5.1's `Invoke-RestMethod` doesn't).
- Greedy generations drift apart between any two engines (llama.cpp's own CPU and CUDA
  builds differ after 21 characters on a story prompt) at near ties. Compare
  probabilities or perplexity, not generated text.

## Test models

Updated 2026-11-10 to the families Vireo runs today, the ones with profiles in
`server/model_profiles.py`. For each family, the smallest current model with the same
architecture as the big ones: the forward pass is written once per architecture, so a
small model tests the same code the 27B or 30B will run.

| Order | Model | GGUF architecture | Stands in for | Size (GGUF) | Adds to the engine |
|---|---|---|---|---|---|
| 1 | **Llama 3.2 1B Instruct** (bartowski, Q4_K_M) | `llama` | (bring-up only) | 0.81 GB | The baseline: done, see Progress |
| 2 | **Qwen 3.5 0.8B** (unsloth, Q4_K_M), then 2B | `qwen35` | Qwen 3.5, 3.6 and 3.8 (Qwen3.8-27B is the same `qwen3_5` architecture) | 0.53 GB | Qwen's tokenizer, `<think>` thinking, and whatever layers `qwen35` adds (check with `vireo-cli inspect`) |
| 3 | **Gemma 4 E2B** (Google's own QAT Q4_0) | `gemma4` | Gemma 4 E4B, 12B, 26B-A4B, 31B | 3.35 GB (+ 0.99 GB vision) | SentencePiece-style tokenizer, Q4_0, sliding-window layers, Gemma's norms and its thinking channel |
| 4 | **Muse Glimmer 30B** (Meta, Q4_K_M) | `muse-glimmer` | itself: there's no smaller one | 16.8 GB | Splitting a model between the GPU and system memory: it doesn't fit in 8 GB of VRAM |
| 5 (2026-11-11) | **Gemma 4 E4B** (Google's own QAT Q4_0), **Qwen 3.5 4B and 9B** (unsloth, Q4_K_M) | `gemma4`, `qwen35` | the larger sizes of each, for the tools eval | 5.15, 2.74, 5.68 GB | Nothing new: the same code, larger shapes (perplexity matches llama.cpp, see Progress) |

The tools eval (`vireo-eval`) runs Gemma 4 E2B and E4B and Qwen 3.5 0.8B, 4B and 9B:
models people choose today. Llama 3.2 1B stays the engine tests' reference, but isn't in
the eval's default set: it's from 2024, and no one picks it over these.

There's no small Qwen 3.8: Qwen has released it at 27B, 2.4T-A95B and Flash-Next (180B)
only, and Qwen 3.6 at 27B and 35B-A3B. Qwen 3.5's 0.8B–9B are the smallest of the
architecture. Meta also publishes a 1.6 GB "dflash" draft model for Muse Glimmer, worth
trying for speculative decoding later.

Each model is checked against llama.cpp the same way as Llama 3.2: tokenizer ids
(`tools/compare_tokenizer.py`), then perplexity and next-token probabilities on an F32
copy.

### Profiles

`server/model_profiles.py` holds each family's settings: recommended sampling, the
tool-call note, the embedding model, thinking markers and thinking budget, reply length.
The Rust server (phase 3) carries these over as they are. Two of them reach into the
engine and come earlier, with each family's model:

- **Thinking markers and budget** (`<think>` for Qwen, `<|channel>` for Gemma 4): the
  engine's scheduler ends thinking at the budget itself, instead of relying on mlx_vlm's
  `thinking_budget`.
- **Sampling defaults** (Qwen: temperature 0.6, top-p 0.95, top-k 20; Gemma 4 and Muse:
  1.0, 0.95, 64): the engine's sampler needs temperature, top-k, top-p, min-p and
  repetition penalty, read from each profile.

## The shape of it

```
Vireo app (WinUI first; then Swift / iPhone)
        │  HTTP + SSE (the API the apps use today)
        ▼
vireo-server (one Rust binary)
├── orchestrator: agent loop, tools, retrieval, MCP, Office files
│        │  plain function calls, no HTTP hop
│        ▼
└── vireo-engine
    ├── gguf         reader: metadata, tensor table, memory-mapped weights
    ├── text         tokenizer (BPE / SentencePiece from the GGUF vocabulary), chat templates
    ├── models       forward pass per architecture: llama, qwen, gemma, …
    ├── backends     CUDA (first), Vulkan, Metal, CPU (fallback and reference); NPU later
    ├── kv-cache     prefix tree, kept on disk, saved per chat
    ├── sampler      grammar-constrained tool calls, speculative decoding
    └── scheduler    chat before background work, cancel, branch; one or more devices
        │
        └─► fallback: starts llama-server for models the engine can't run yet
```

## What we write and what we borrow

| Layer | Ours? | Notes |
|---|---|---|
| GGUF reader | Ours | A header, key-value metadata, a tensor table and memory-mapped data |
| Tokenizer | Ours | Built from the vocabulary inside each GGUF file. The tricky part is each model family's pre-tokenizer rules (named in `tokenizer.ggml.pre`); tested against reference output. |
| Chat templates | Ours | A Jinja subset is enough (llama.cpp wrote its own for the same reason) |
| Model forward passes | Ours | One per architecture, including its quirks: sliding-window layers, RoPE scaling, vision encoders |
| GPU kernels | Ours | The largest piece of work (see Kernels) |
| KV cache, sampler, grammar engine, scheduler | Ours | Where Vireo stands out |
| Agent loop, tools, skills, retrieval | Ours | Ported from `server/` |
| Office and PDF files | Mostly ours | PowerPoint writing has almost nothing to build on in Rust. Reading PDFs may borrow pdfium (`pdfium-render`). |
| GPU API bindings | Borrowed | `cudarc` (CUDA driver API and NVRTC); later `ash` (Vulkan) and `objc2-metal` (Metal). Thin wrappers, nothing to gain from rewriting. |
| HTTP, async, SQLite | Borrowed | `axum`, `tokio`, `rusqlite` |
| Embeddings | Borrowed at first | `ort` (ONNX Runtime) for today's embedding models; could move onto our engine later |
| MCP | Borrowed at first | `rmcp`, the official Rust SDK |

Check each borrowed crate's license. Most are MIT or Apache 2.0, which allow building a
proprietary product on top.

## Kernels

Each supported quantization type needs a dequantize step plus fast matrix-vector
(generation) and matrix-multiply (prompt processing) kernels, for each backend. The
block layouts must match GGUF bit for bit. ggml's source (MIT) is the reference for
the layouts.

**Quantization types, in order:**

1. F32 and F16 (bring-up and correctness), then Q8_0, then Q4_K and Q6_K. Q4_K_M files
   use Q4_K and Q6_K, and they cover most downloads.
2. Q5_K and Q4_0.
3. The IQ types.

A file that uses a type we don't support yet goes to llama-server.

**Backends, in order:**

1. **CUDA (NVIDIA).** Kernels written in CUDA C++, compiled to PTX when Vireo is built
   and loaded through the driver API (`cudarc`), so users need only the NVIDIA driver.
   Matrix-vector kernels first (they decide generation speed), then tensor-core
   matrix-multiply kernels for prompt processing. Profiled with Nsight Compute.
2. **Vulkan** for AMD and Intel GPUs on Windows, testable on this PC's Radeon 880M.
3. **Metal** for the Mac.

For Vulkan and Metal, the choice between writing each kernel natively and using one
shader language compiled to both (for example Slang, which can also output CUDA) is
still open. Deciding it before the CUDA kernels are written matters: if Slang works
well, the CUDA kernels could be written in it from the start.

The CPU backend (Rust SIMD) is the fallback and the reference for correctness checks.

## Why it's worth it: what Vireo-only features become possible

### Engine

- **The shared prompt start loaded for good.** The system prompt and tool list are
  computed once, saved to disk and loaded at launch. The 6–8 second wait before the
  first word in a new chat (fixed in `server/mlx_server.py` on 2026-11-10 by working
  around mlx_vlm's cache) can't come back.
- **A saved KV cache per chat.** Reopening an old chat continues immediately instead of
  rereading its history.
- **Tool results added without rereading the prompt.** Only the new tokens are
  processed, and the engine guarantees it instead of hoping a server's cache kept the
  right snapshot.
- **Speculative decoding** with a small draft model. Vireo's output (JSON, tool
  arguments, repeated phrasing) is predictable, which suits it.
- **Scheduling.** Chat always comes first. Indexing, rolling summaries and memory
  learning run when the GPU is free and pause when a message arrives. With more than one
  device (a GPU and an NPU), background work can run on the other one.
- **Cancel and branch** without losing the cache: Stop, edit a message, retry from a
  point.

### Tools

- **Tool calls that can't come out malformed.** Each tool's schema becomes a grammar the
  sampler enforces. Arguments can be limited further: a `path` to files that exist in the
  project folder, a cell reference to valid A1 references.
- **Tools that start before the model finishes.** A read-only tool starts once its
  arguments are complete, while the model is still writing its closing tokens.
- **Decisions in one forward pass.** Choosing a tool group (`server/tool_groups.py`),
  checking a trigger (`server/task_triggers.py`), or deciding whether a step needs
  approval: score a few labels instead of generating an answer.
- **Faster heavy tools.** Indexing watched folders, the vector store and chat search in
  parallel across cores, with less memory. Single everyday calls are already fast, so
  expect little gain there.

### Tools and skills for small models (design, 2026-11-11)

All our own, from the grammar up: no llguidance, xgrammar, outlines or JSON library. The
target is the smallest models (0.8–4B) working well, fast and right, and everything must
work just as well on large ones (Qwen3.8-27B, Muse Glimmer 30B, the mixture-of-experts
models). Small models fail in known ways, and each part of the design answers one of them:

| A small model... | So Vireo... |
|---|---|
| writes broken JSON, or a call in the wrong shape | **constrains the call.** Each tool's schema becomes a grammar the sampler enforces on bytes (our own matcher, token masks from the vocabulary's byte trie). The call's wrapper is each family's own (Qwen's `<tool_call>`, Gemma 4's and Llama 3's), so the model writes what it was trained on. |
| spends many forward passes on braces, quotes and key names | **writes forced text itself.** Wherever the grammar allows one continuation (`{"name": "`, the rest of an unambiguous tool name, `", "arguments": {"path": "`), those tokens are added without sampling and read in one batch. |
| retypes a path or link and drops a character (`_fix_retyped_links` in `tool_text.py`) | **copies instead of retyping.** Arguments of a path, link or id type are limited to values that exist: paths in the folders the chat may reach, links and ids that appeared in the chat or a tool's result. |
| picks worse from a long tool list, and each schema costs context | **shows few tools, briefly.** A short fixed core; a group's tools only when a step needs them, chosen by scoring labels in one forward pass (not keywords, as `tool_groups.py` does); one compact line per tool, in the form measured to work best per family. Each group's text is a cached branch of the KV prefix, so showing it costs nothing after the first time. |
| decides badly in free text (tool or answer, which skill, needs approval?) | **decides by scoring.** A few labels scored in one pass, constrained to the choices. |
| loses track in long results | **shapes results.** Short, structured tool output (counts, the first lines, "N more"), with the values it may need to copy kept exact. |
| follows long instructions poorly | **compiles skills** (below): a skill's instructions cached, its own few tools, a grammar for its output, its steps as branches. |

Read-only tools start as soon as their arguments close, while the model writes its
closing tokens. Approvals stay as the Mac's server asks them.

**Large models too.** None of this may cost a large model anything:

- Constraints only rule out what's invalid, so a model that writes valid calls writes the
  same calls; forced text and cached tool text make large models faster as well, and
  they gain the most from it (each pass costs more).
- How much guidance a model gets is a setting per model, not a guess from its size: a
  skill one step at a time or whole, a group's tools or all of them, optional arguments
  offered or held back. Each model's setting is what the eval measures best for it
  (kept with the model's profile, `model_profiles.py`'s successor), with a default for
  models never measured.
- The eval runs large models as well as small ones, and a feature that helps small
  models but costs large ones is made a setting, not left on for all.

### Skills (design, 2026-11-11)

Skills stay in the open SKILL.md format (`server/skill_library.py`: a name and description,
then steps in Markdown; 10 built in, the user's own, and drafts from skill learning). The
file is the source; Vireo compiles each one when it loads, without changing it:

- **Picked by scoring.** The request scores each skill's name and "none" in one pass, in
  place of the model choosing to call `use_skill`. (`tool_evals.py` has the cases:
  plan my day, research, and leaving skills alone.)
- **Instructions cached.** A skill's text is a branch of the KV prefix, read once.
- **One step at a time, where it helps.** A skill's numbered steps become a checklist the
  engine walks: the model sees the step it's on and that step's tools only (step 3 of
  `analyze-data` offers `calculate`), and the engine moves on when the step is done. A
  model that does better with the whole skill (the per-model setting above) gets it whole.
- **The answer's shape kept.** Where a skill sets an answer's layout (`analyze-data`'s
  **Answer:** / **Details:** / **Worth noting:**), the headings are written for the model
  (forced text) and it writes what goes under each. Free text itself is never constrained.
- **A fallback.** A skill without clear steps is used whole, as cached instructions.
- **Learned skills alike.** A skill drafted from a run compiles the same way.

The tool loop (step 2) is built for this from the start: the tools offered can change from
one step to the next, and a step can end on a forced heading as well as on a call.

**How we know:** our own eval, in Rust, against vireo-server: the 14 cases of
`server/tool_evals.py` first (Gemma 4 E2B passes 13 on today's stack), then cases from the
test gauntlet, each scored for the right tool, right arguments, using the result, steps
and time. Run on every test model, with each feature on and off, against today's Python
server on the same models. A feature stays only if the numbers say so.

A lesson from step 1: optional arguments are where small models go wrong once the shape
is guaranteed (Llama 3.2 1B asked for one line of a file). So each tool's optional
arguments are offered only where a model needs them, or chosen by scoring, and the eval
counts arguments as well as tools.

**Order:**
1. ~~The grammar matcher and token masks, with forced text written ahead (vireo-engine).~~
   Done 2026-11-11 (see Progress).
2. ~~vireo-server's tool loop, with the app's own PC tools (the protocol is there:
   `client_tool_request`, `/api/agent/client-tool-result`) and each family's call format.~~
   Done 2026-11-11 (see Progress): read-only tools, then the tools that change files,
   with approvals, a check before asking, and offered only when a request asks.
3. ~~The eval, and a baseline.~~ Done 2026-11-11 (`vireo-eval`, see Progress). Gemma 4 E2B
   passing all twelve means the cases are too easy for it: harder ones from the test
   gauntlet (several steps, long results, near-miss tool names) come with the next
   features, so there's room to measure them on capable models too.
4. Copying constraints, scored choices, compact tool text. First round done 2026-11-11
   (see Progress): token healing was the fix that mattered (Qwen 3.5 9B 4/12 to 12/12);
   scored names and fewer tools are switches, measured, off by default. Harder cases
   added 2026-11-11 (see Progress): four of five models give up on a wrong path, and two
   guess a path rather than look. So next, measured on them: results shaped to help the AI
   agent on (a missing file's error lists what is in its folder, and the nearest names:
   done 2026-11-11, see Progress, `which-report` passes on all five, `recover` on two),
   then worked examples of calls and the tools' text as the model sees them. Paths
   constrained to ones that exist (the copying constraint) wait: after the new errors
   every path the models wrote was right, so it has nothing to fix in these cases yet.
   What's left in `recover` is the model asking before it reads a file it wasn't named;
   whether it should go on is a product choice as much as a model one. The tools that change files have their
   own six cases (see Progress).
5. Compiled skills (above), then the per-model guidance settings from the eval. First round
   done 2026-11-11 (see Progress): loaded, picked by scoring with a per-model control, read
   as a `use_skill` result written into the reply. Next: the scoring questions' shared
   start cached, then a skill one step at a time and its answer's headings forced.

### Compiled skills

Today a skill (`server/skill_library.py`) is mostly text added to the prompt. In the new
engine a skill is a bundle:

- its instructions, already computed and cached as a branch of the KV prefix tree,
  so turning a skill on costs nothing
- its own short, stable set of tools
- grammars for its output (a calendar skill always produces a valid event)
- branch points, so a plan step (`server/agent_plan.py`) can be tried and kept or
  thrown away without rereading anything

Other apps can copy prompt text. They can't copy the engine behavior underneath it.

### Rules for these features

- Constrain arguments and structured output, not the model's free text: grammars that
  are too strict can make a model reason worse.
- Engine features are optional speedups. Tools and skills must still work on the
  llama-server fallback, just without the extras.
- Tool design (fewer, better-shaped tools, fewer steps) still matters more than any
  engine trick.

## NPUs

Possible, with limits worth knowing up front.

**What an NPU is good at.** A lot of int8 and int4 math at very low power. Generating
tokens is limited by memory bandwidth, and a laptop NPU shares system memory with the
CPU, so it won't generate faster than a discrete GPU like the RTX 5060. What it offers
is low power use and a second device, so the GPU stays free for chat. Prompt processing
is limited by compute, so NPUs can do well there.

**How NPUs are programmed.** Mostly through each vendor's compiler, not our own kernels:
we hand over a model graph and the vendor compiles it for the chip.

| NPU | Route |
|---|---|
| AMD XDNA (this PC) | Ryzen AI Software (ONNX Runtime with the VitisAI execution provider). AMD has also open-sourced a toolchain for writing kernels for XDNA directly (IRON / MLIR-AIE); whether it can be used from a Windows app needs checking. |
| Intel (Core Ultra) | OpenVINO |
| Qualcomm (Snapdragon X) | QNN SDK |
| Any Windows NPU | Windows ML, which reaches each vendor through ONNX Runtime execution providers |
| Apple Neural Engine | Core ML only; there's no public lower-level API |

So for most NPUs, "own everything" stops at the vendor's compiler. AMD's open toolchain
is the possible exception, and this PC has an AMD NPU to try it on.

**GGUF on an NPU.** NPUs don't run GGUF's quantization types directly. Vireo would
repack the weights into the NPU's layout and compile the graph the first time a model
is used on the NPU, then cache the result. People never convert anything themselves,
but that first preparation could take minutes.

**Where an NPU helps Vireo, in order:**

1. **Embeddings.** Vireo's embedding models are already ONNX models, so running them on
   the NPU through an ONNX Runtime execution provider is close to a drop-in change, and
   it can be tried any time, even before the engine exists.
2. **Background work.** Rolling summaries, memory learning, and routing and
   classification decisions, run by a small model on the NPU while the GPU serves chat.
3. **A draft model for speculative decoding**, on the NPU while the main model runs on
   the GPU. This depends on how quickly the two devices can hand tokens to each other;
   it needs measuring.
4. **Chat itself on laptops without a discrete GPU** (Copilot+ PCs and similar), with
   small models, at much lower power than the integrated GPU.

**What this means now.** The engine's backend interface should allow a device that
takes a whole compiled graph, not only individual kernels, and the scheduler should
handle more than one device. Neither costs much if it's planned in from the start.

## How we know it's right

- **Logits compared with llama.cpp.** For each supported model and quantization type,
  run the same prompts through both engines and check the logits match within a small
  tolerance, token by token. Run this in CI with the small test models: it catches
  tokenizer, kernel and forward-pass bugs.
- **Tokenizer tests** against reference tokenizations for each model family.
- **Benchmarks** against llama.cpp's CUDA build on the same PC and the same GGUF file:
  time to first token, prompt tokens per second, generated tokens per second, peak VRAM.
- **The existing server tests** (`server/tests/`) run against the Rust server through the
  unchanged HTTP API, wherever they test over HTTP.
- **The test gauntlet** (`docs/dev/test-gauntlet/`): steps per task and finished tasks,
  compared with the current stack.

## Phases

Each phase ends with a decision point. Nothing replaces the current server until it is
at least as good.

### 0. Set up and measure

- ~~Install the toolchain~~ (done 2026-11-10, see Development machine), then the
  llama.cpp CUDA build.
- Record llama.cpp's numbers for each test model on this PC: these are the targets.
- Add timing to one turn of today's stack (on the Mac): server startup, prompt
  processing, generation, tools, and Python's own overhead.

### 1. Engine spike: small models on CUDA

- GGUF reader, tokenizer and chat template for Llama 3.2 1B
- The CPU backend first, as the reference
- Our own CUDA kernels for F16, then Q8_0, then Q4_K and Q6_K
- A command-line program that generates text
- Then Qwen3 0.6B and 1.7B

**Done when:** logits match llama.cpp for each test model and quantization type, and
generation speed is within about 10% of llama.cpp's CUDA build on this PC. Expect weeks
of kernel tuning before matrix-vector speed catches up. That is the main risk of the
whole plan, and this phase finds out early.

**Decision:** continue, keep tuning, or reconsider (for example, borrowing ggml's
kernels while keeping everything above them ours).

### 2. Vireo-only engine features

Started 2026-11-11: the KV cache kept between turns, then a saved cache per chat that
lasts across restarts, and chats as long as the GPU has room for (see Progress and "Long
chats"). Still to do: grammar-constrained tool calls and the scheduler. Further cache work
(Qwen's shared prompt start, smaller saves, more than one chat on the GPU) is listed with
the tuning under "What limits speed now". A saved state only fits the model file and precision that
made it (the folder is per model), and a Llama 3.2 1B chat of 4,000 tokens takes 128 MB
on disk.

The shared prompt start loaded at launch, a saved KV cache per chat, grammar-constrained
tool calls, the scheduler.

**Done when:** time to first token and steps per task beat today's stack by a margin
you can see in the gauntlet.

### 3. Rust orchestrator, on Windows

Port `server/` behind the same HTTP + SSE API, one area at a time: chats and storage,
the agent loop, retrieval and the vector store, Office files, MCP, pairing. The engine
runs inside the same process, and llama-server stays as the fallback.

The Windows app connects to it on the PC the way it connects to the Mac today. This is
"This PC".

**Done when:** the Windows app runs whole tasks against the Rust server on the PC, and
the existing tests pass against it.

### 4. More models, compiled skills, faster tools

A small Gemma and a 4B model, more quantization types, compiled skills, and the tools
that gain from parallel work.

### 5. Other GPUs and the Mac

The Vulkan backend (tested on the Radeon 880M), then Metal. The same server replaces the
Python one on the Mac.

### 6. NPUs

Embeddings on the NPU first (this can start any time, see NPUs), then background work
with a small model, then a draft model for speculative decoding.

## Risks

| Risk | How we handle it |
|---|---|
| Our kernels can't match llama.cpp's speed | Phase 1 measures it before anything else depends on it |
| A laptop GPU's 8 GB limits what can be tested | Small test models; larger models tested on other hardware later |
| CUDA versions and GPU generations. CUDA 13 builds only for Turing (RTX 20 series, compute capability 7.5) and newer. | Build PTX for 7.5 and up, which covers RTX 20 series and later; decide whether older cards go to llama-server. Test on at least one older NVIDIA card before release. |
| New models arrive every few weeks and each needs work | llama-server fallback; support only featured models natively |
| Product work slows during the port | Port behind the unchanged API, one piece at a time; Vireo stays shippable |
| Tokenizer and template edge cases give subtly wrong output | Reference tokenizer tests and logit comparisons in CI |
| Giving up MLX costs speed on some models on the Mac | Benchmark featured models in phase 5; keep mlx_vlm as a fallback until settled |
| NPU tooling differs for each vendor and changes often | NPUs are optional speedups; start with embeddings, where the risk is small |
| Gaps in the Rust ecosystem (PowerPoint, some PDF work) | Write our own on top of OpenXML; borrow pdfium for PDF reading |

## Open questions

- CUDA kernels in CUDA C++, or in a shader language (such as Slang) that can also
  produce Vulkan and Metal later?
- Keep mlx_vlm as a fallback on the Mac for a while, or drop MLX entirely at the switch?
- What do people with MLX models downloaded see: an automatic offer of the GGUF build,
  or a note in **Models > Installed**?
- Can AMD's open XDNA toolchain be used from a Windows app, or only Ryzen AI Software
  through ONNX Runtime?
- Does this folder become its own Cargo workspace (for example `vireo-engine`,
  `vireo-server`), or join the main repository's layout once it works?
