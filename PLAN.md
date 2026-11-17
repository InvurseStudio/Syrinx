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

**Twelfth pass: attention for a group of heads, a few tokens at once, longer guesses
(2026-11-12, afternoon).** Where a 30K prompt's time went on Gemma 4 31B
(`VIREO_METAL_PROFILE`): the matrix products 59%, the ten global layers' attention 25%
(49 ms a batch, growing with the context), the sliding layers' 5%. The products run at
51-57 TFLOPS on their own (`big_matmul_speed`; Q6_K a little ahead of F16 through the same
tile), so they are at this design's ceiling; attention ran at ~20.

- `attention_tcg` (vireo_tensor.metal): a prompt's attention with a key/value head's query
  heads together. A threadgroup takes one key/value head and 16-32 rows of its queries
  (several tokens of the group's heads), read as f16 from device memory (`qkv_prep` writes
  them there too, regrouped; `q_group_half` for other callers), and each key and value
  chunk once for the group, 128 or 256 positions at a time (64 a head before). Tiles
  searched: 16-64 rows, 64-256 positions, 4 or 8 SIMD groups; the quickest leave a thread 32
  of the output's numbers (512-wide heads 16 × 256 × 8, 256-wide 32 × 128 × 8, 128-wide
  32 × 128 × 4). A batch of 1,024 at 30K positions, ms a layer: Gemma 31B's global layers
  91 → 58, Qwen 3.8 27B 39 → 26, Muse 30B's global 30.4 → 13.7; sliding layers Gemma
  1.86 → 1.44, Muse 2.14 → 1.11 (`tests/prompt_attention_speed.rs`). Rings are rounded to
  256 positions for it. With the old chunk of 64 its sums are the old kernel's to the bit.
- A short batch has few threadgroups, so its context is split into parts (about three
  threadgroups a GPU core) that `attention_tcg_merge` adds up: 40 tokens at 30K, Qwen 27B
  3.87 → 1.32 ms a layer, Muse 2.74 → 0.69.
- 1 to 8 tokens the same way where it's quicker (`Gpu::grouped_few`): the one-token
  kernels walk the context once a token and head, this reads it once. Gemma 31B's global
  layers at 30K, 2 / 4 / 8 tokens: 1.09 → 0.58, 2.29 → 0.62, 4.98 → 0.80 ms. One token
  only for Muse's global layers (0.26 → 0.08 ms at 30K); Gemma's and Qwen's read their
  cache at memory speed already.
- 8- and 16-token tiles for the matrix products (`matmul_tc_t8`, `_t16`, 256 columns a
  step), and `Gpu::tile_beats_few` by weight type: the tile from 4 tokens for K-quants and
  5 for the others, on any matrix over 4M weights (64M and 5 before). Gemma 31B's ffn_up,
  µs at 3 / 4 / 5 / 8 tokens: few-token kernels 221 / 296 / 437 / 583, the 32-token tile
  ~330, the 8-token one 250-260 (one token 194); ffn_down at 4: 400 → 252
  (`big_few_or_tile`). 256 columns a step did nothing for whole batches (4.69 → 4.63 ms).
  A slip found on the way: a tile whose step doesn't divide the columns dropped the last
  columns (none of the old tiles' did).
- `VIREO_METAL_ERRORS=1` says why the tensor kernels didn't compile: the backend falls
  back to the kernels without them silently, as it must before the M5.

A batch of n tokens after a chat of that length, ms, main → now (`tests/check_speed.rs`):

| | 1 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Gemma 4 31B at 2K | 52.5 | 67.6 → 67.0 | 96.6 → 71.7 | 120.4 → 74.3 | 101.0 → 80.0 |
| Gemma 4 31B at 27K | 56.4 | 82.0 → 71.5 | 111.1 → 80.4 | 146.2 → 86.9 | 130.1 → 95.4 |
| Qwen 3.8 27B at 2K | 46.5 | 52.4 → 52.5 | 67.5 → 57.7 | 96.7 → 69.0 | 96.4 → 73.8 |
| Qwen 3.8 27B at 27K | 50.0 | 63.8 → 61.9 | 87.7 → 68.0 | 137.9 → 83.3 | 150.6 → 94.3 |
| Muse 30B at 2K | 46.7 → 44.8 | 64.5 → 51.7 | 97.8 → 59.3 | 108.6 → 70.5 | 103.3 → 79.2 |
| Muse 30B at 27K | 51.5 → 49.0 | 73.5 → 61.7 | 107.5 → 66.5 | 143.5 → 76.3 | 128.4 → 79.7 |

Eight tokens now cost what four do, so the guessing takes more (vireo-server): guesses
from the chat run up to 7 after a match of 8 tokens (3 after 4 before; the Mac only, a
longer match guessing further), and Qwen's multi-token-prediction layer gives 3 a round
(2 / 3 / 4 guesses: a copied document at 44 / 53 / 59 tokens/s, a new story at 30.1 / 29.6
/ 26.5). Three replies through the server, tokens/s, main → now, every reply's text the
same: Gemma 31B rewriting a document as a list 28-33 → 43, returning an edited code file
30-35 → 62, a new story 18-19 → 18.4; Qwen 27B 42.5 → 53.2, 43.4 → 53.2, 29.5 → 29.6;
Muse 30B 31.9 → 42.0, 32.2 → 45.3, 20.4 → 20.3.

Whole reads and writes (`vireo-cli`, the speed report's prompts; the Mac ran hot and busy
all afternoon, runs moving by 10-15%, so these are alternating runs): Muse 30K read 593 →
662 tokens/s and writing after it 19.2 → 20.5; Gemma 31B's 30K read by kernel 63.8 / 65.1 s
of GPU time → 59.6 / 61.7 (attention 19.7-20.5 → 14.0-14.4 s; the products ran 3-5% slower
beside it, the GPU's heat budget); Qwen 27B 592 → 597 at 30K, 698-722 → 718-722 at 8K.
Perplexity old / new attention: Qwen 27B 10.3910 / 10.3902, Muse 16.9149 / 16.9134. Nine
eval cases on Gemma 4 E4B, Qwen 3.5 4B, Gemma 31B and Qwen 27B, main and now alternating:
every reply the same to the token, and the time level (13.3-15.0 / 18.7-18.8 / 80.5-90.8 /
82.5-93.6 s against 13.4-13.8 / 18.6-19.5 / 81.2-85.3 / 83.0-85.0): at the eval's ~2K
contexts these changes cost and save little. Muse's reasoning on the rewrite ran 406
tokens, not main's 41, with the same answer: exact mode gives 406 too, so main's was the
f16 rounding's path. One test's limit moved: re-reading 10 tokens after a rewind differs
by 0.02 in the logits in the default mode (1e-3 before), a short batch's parts each
rounding probabilities to f16 against their own largest.

`VIREO_TCG=off` and `VIREO_TCG_FEW=off` keep the old attention kernels, to compare. Not
done: the CUDA side of any of it; the speed report's page (its numbers are the morning's).

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
The other two large models on the same five cases and build, side by side: all three 5/5;
a case Gemma 4 31B 5.9 s, Qwen 3.8 27B 6.8 s, Muse 10.6 s. Muse's extra time is length, not
speed: 1.4-2.2x the tokens a reply (its answers restate the file or list more; reasoning is
18-144 of them), six calls on three-steps where the others made three (it read every
README), and a longer system turn (1,878 tokens against 1,292 and 1,655). Writing: Gemma
19.5-22.2 tokens/s (lookup), Qwen 27.2-39.5 (its drafting layer, 80-100% taken), Muse
26.5-35.2 (lookup). Results: `evals/large-models-five-cases.json`.
The twelve hard cases on all three: Gemma 4 31B 12/12 (6.6 s a case), Qwen 3.8 27B 12/12
(6.2 s), Muse 11/12 (10.2 s, 20 calls against the others' 15). Muse's miss, three runs of
three: `follow-up-file` ("And what does plan.md in the same folder say?", after an earlier
turn that answered from notes.txt with no call in the history) goes straight `to=user`
without reasoning and makes the file's contents up. `Reasoning strength: medium` doesn't
change that and slows every other case (it reasons 2-3x longer), so it stays `low`. Muse
is thorough where the others are quick: `recover` 4 calls in 18.7 s (the others 2 in ~6),
`three-steps` reads every README. Results: `evals/large-models-hard-cases.json`.

Where a Muse call's ~1 s goes (`VIREO_TIME_LOG`, `read`): the call's scaffolding is fed in
one ~100 ms read; the path is written a token at a time at 46 ms. Lookup guesses inside the
call (tried, the calls the same in all 15 cases) saved nothing: a path in the request is
tokenized after a space (" C") and in the call without, so the chat's runs rarely match.
Left out. The same log shows ~3.6 s a case in the skill-picking questions before every
reply, on every model (11 skills, ~330 ms each): the shared start of those questions kept
in the picker's session is the larger win, for all three.

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
| 2026-11-12 | Windows tuning, third pass (see "Fourteenth round"): the Mac's fifth to twelfth passes measured on CUDA, and their CUDA side built. Few-token product kernels (`matvec8_few<2-8>_*`: the weights read once for 2 to 8 tokens), one token's attention for up to 8 tokens (`attention_part_kv`, a grid layer each), the convolution and DeltaNet stepping through their one-token kernels with the states kept (`Qwen35Cuda::can_rewind` on CUDA), a recorded graph for every batch of 1 to 64 tokens, tiles of 16 and 32 tokens for the tensor-core products. In the server: 7 guesses a round from the chat on CUDA too, Qwen guessing there on every size where the GPU has room for its steps, and picker sessions read in batches of 256 on CUDA | In 8-bit mode a batch of up to 8 tokens gives the logits of the same tokens read one at a time to the bit (Llama, Qwen 3.5, Gemma 4: difference 0.0), so guesses can't change a reply; the small tiles give the 64-token tile's numbers to the bit; rewinding Qwen after 0, 2, 4 or 5 of 5 guesses goes on to the bit. Against `main`, back to back: a 4-token check Gemma 4 E4B 21.9 → 13.1 ms, Qwen 3.5 4B 26.1 → 14.7; the full eval (43 cases) scores the same on all four models with no case changed, in 70 → 44, 98 → 73, 107 → 101 and 179 → 172 s (`evals/windows-short-batches.json`); copied replies 2-3 times as fast. Engine 73, GPU 14, server 60 tests pass. Not compiled on the Mac |
| 2026-11-12 | The chat cache, three ways a chat's whole history was read again for nothing (vireo-server's `reply`; shared code, measured on the Mac). **The tools that change files** (`--changes declared`): once a chat's cache declares them they stay declared in that chat, callable only when a request asks (as `scored`), so the system turn changes once in a chat, not at every turn that does or doesn't ask for a change. **Long chats:** a chat too long for the context starts at a fixed cut (`chat::cut_points`: the first user message with a third of the context before it, then two thirds, and so on, counted in bytes), not one turn further on at every reply, so it reads everything again once in about a third of the context. **Remembered facts:** a chat keeps the facts its instructions began with (the list only grows between forgets, so that's the first few of today's), and the ones remembered since are told at the end of the request (`memory::since`), after the point the next turn's prompt shares (and Qwen's DeltaNet is saved); a new chat has every fact in its instructions as before, and a forgotten fact that was in a chat's instructions has it read again. How the prompt is chosen: written the usual way first; only if no cached chat (the session's or a saved one) starts as it does through its first message are the other ways tried, the first that a cache holds taken (up to 8 facts back) | Unit tests for the cut points, the closing text of each family's messages and the facts' two parts (server 74 pass). Scripted chats over HTTP, Gemma 4 E2B and Qwen 3.5 4B, from the "Reply:" log: question, change, question, change, question reads the chat again once (at the first change; before, at each of the four switches), 24–45 tokens a turn after; 16 turns of ~300 tokens in a 4,096-token context read everything again once in the four turns past the limit (before, at each of them); the turn after a `remember` reads 47–65 tokens of a 1,600–2,300-token chat (before, all of it), a new chat answers from both facts, the chat loads from its saved file the same way, and forgetting a fact its instructions held reads it again (Qwen; Gemma E2B didn't call `forget`). Eval, 14 cases (changes, memory, skills, follow-ups) on both models, against `main` built apart: every result, call and answer the same (11/14 and 14/14). Not measured: the large models, Windows, the app |
| 2026-11-12 | Mixture of experts: Gemma 4 26B-A4B (128 experts a layer, 8 used a token, beside a dense network). On the CPU reference first (`gemma4.rs`, the formula in its header, from llama.cpp's `gemma4.cpp` and mlx_vlm), then on Metal: four kernels (`moe_pick`: a token's 8 highest-scored experts and their shares; `moe_matvec_q6_k` / `_q8_0`: the matrix-vector kernels with the matrix found through the picked index, a threadgroup row a slot; `moe_geglu`; `moe_sum`), the same for one token and for a batch. CUDA still refuses the model | CPU against llama.cpp b12228 on an F32 copy (llama.cpp rounds activations on quantized files: ~4% of this model's perplexity): top token 74.29% against 74.15% (the dense 31B the same way: 0.03 points; E2B 0.0002). GPU exact against the CPU: every probability within 2e-4, the same tokens. All 43 eval cases pass through vireo-server, 4.6 s a case. Speed, first version, untuned: writing 60-70 tokens/s, reading ~500 tokens/s (llama.cpp: 96 and 3,326 at 2K). Reading is memory-bound: a batch reads each chosen expert's matrix once a token (8 × 5.3 MB a token a layer); grouping a batch's tokens by expert, so each matrix is read once a batch, is next. Writing spends 11 more dispatches a layer than a dense model |
| 2026-11-12 | Excel workbooks read in Rust (`attach/xlsx.rs`: sheets, shared strings, date styles, cells as Excel shows them, a sheet written as the Windows app's `Spreadsheet.cs` writes it), so the server runs `read_spreadsheet` itself where it runs the file tools (the Mac app), `read_file` gives a workbook's sheets and a workbook can be attached. More skills are then offered on the Mac, each a scoring question a turn, so skills are shortlisted first: one question lists every skill by letter (read once and kept), its letters are scored, and only the two likeliest get their own yes-or-no question (`--skill-shortlist`, on by default). `vireo-eval` now gives the server an empty skills folder: it had been reading the machine's own skills | Against openpyxl on the 13 workbooks in the Mac's test folders: 4,097 cells, none different. Eval, 43 cases, shortlist off / on: Gemma 4 E2B 33 / 33, E4B 39 / 40, Qwen 3.5 4B 43 / 43, 9B 39 / 39 (no case lost; 9B 2.7 → 2.5 s a case, the others level). Gemma 4 31B, a turn's time before its prompt is read: 1.65 → 1.28 s (what's left: the change question, the shortlist and two skill questions, ~0.29 s each). Server tests 80 pass. Not run on Windows: a fourth picker session there takes ~0.1 GB more of the GPU |
| 2026-11-12 | Gemma 4 26B-A4B, a batch's experts grouped by expert: `moe_group` sorts a batch's slots by their expert and cuts them into tiles of up to 32, and `moe_matmul` (the SIMD-group matrix kernel, its matrix and tokens found through the tiles) reads each expert's rows once for all the tokens that chose it, from 80 tokens up (below that a slot at a time is quicker) | The grouped path against one token at a time on the GPU: every probability within 5e-4. Reading ~800 new tokens 1.6 → 0.43 s, a cold ~2,800 tokens ~500 → 1,822 tokens/s (llama.cpp 3,326), the eval 4.6 → 2.6 s a case, still 43 of 43. The experts' products are ~15 TFLOPS and two thirds of a prompt's GPU time; the tensor kernels reach 50 |
| 2026-11-12 | The skill shortlist scores names, not letters. As merged it lettered the skills and scored " B", " C"...: fine on the four small models it was measured on, but Gemma 4 26B-A4B and 31B then lost two of five skill cases (the right skill outside the first two). Now the skills are listed by name and, after "Skill:" in the model's turn, each name's first token is scored | Skill cases back to 5 of 5 on both: 26B-A4B 43 of 43, 31B 19 of 19 (skills, changes, more tools). Small models as with the shortlist off: E2B 33, E4B 39, Qwen 3.5 4B 43, 9B 39. Lesson: a change to what the server asks the model is measured on the large models too before it merges |
| 2026-11-12 | The Mac server makes changes itself (HANDOFF to-do 1): `files.rs` writes, renames, moves and deletes files as the Windows app's PcTools does, with the same answers and the same check before the user is asked (`Files::check`, so nobody approves a change to a file that isn't there). Nothing is deleted for good: a deleted file, and the old version of one that's replaced, go to the Trash (NSFileManager). And `write_spreadsheet` in Rust (`attach/xlsx_write.rs`): rows from a cell or after the last row; an existing workbook keeps every other part as it is, a cell written over keeps its style and a new one takes the style of the cell above; a new workbook is made when there's none. The chat cache's disk budget is sized to the model (room for four full chats, 4-32 GB: 16 GB for Gemma 4 31B). No undo yet on the Mac (the Windows app has its journal) | Server tests 85 pass. A row added to copies of the 13 real workbooks in the Mac's test folders, read back with openpyxl: every original value, number format, fill and column width unchanged, the new row right. Not opened in Excel itself. Through the server: Gemma 4 26B-A4B wrote, renamed and deleted in a scratch folder (a missing file refused before any approval), and 31B added a correct row to a copy of the Aurelia invoice register. Eval unchanged: E2B 33, E4B 39, Qwen 3.5 4B 43, 9B 39, Gemma 4 26B-A4B 43 of 43, 31B 11 of 11 (changes, skills) |
| 2026-11-12 | Word and PowerPoint written in Rust, so the Mac server runs `create_docx`, `fill_docx_template` and `create_pptx` itself: `attach/markdown.rs` (the model's Markdown as blocks and styled runs, as the Windows app's `MarkdownBlocks`), `attach/docx_write.rs` (a document from a title and Markdown: headings, lists that restart, tables, quotes, code, links, a memo's header lines kept as lines; and a template's `{{placeholders}}` filled where they sit, each keeping its run's formatting, in the body, headers, footers and text boxes), `attach/pptx_write.rs` (a title slide and a slide an entry, in Vireo's theme). All as ZIPs without compression | Server tests 91 pass. Opened with python-docx, python-pptx and macOS's textutil and Quick Look; the Harbor Lane invoice and policy templates filled with no placeholder left and every paragraph style kept. Not opened in Word or PowerPoint themselves. Through the server, Gemma 4 31B filled the invoice template (40 × 25.00 as 1,000.00), wrote a memo and made a deck from it (the memo dated "October 24, 2023": the model didn't ask the date). Eval unchanged: E2B 33, E4B 39, Qwen 3.5 4B 43, 9B 39, Gemma 4 26B-A4B 43 of 43, 31B 19 of 19 (changes, skills, more tools) |
| 2026-11-13 | The Aurelia gauntlet on the prototype: `tools/aurelia_run.py` plays the 32 prompts in the Mac app (so a run can be watched) and grades each with the suite's own checks, ported from `docs/dev/test-gauntlet/run_full.py`. For it the app's test scripts gained file access for their chats (`access`, `folders`), an approval answered by what it names (`approveIf`), a pause after each step for a grader (`gate`), and each step's tool steps and approvals in its record. A run deletes the app's chats and chat cache first and keeps everything else of its own in a scratch folder (test files, memory, approvals, an empty skills folder). Nothing in the server was changed | Gemma 4 31B, one run, `docs/dev/test-gauntlet/runs/Aurelia-Proto-Run-2026-11-12-Gemma4-31B-notes.md` (every prompt, with the likely cause where worse). 5 prompts not runnable (calendar, tasks and triggers). On the other 27, by the hints alone: 9 pass, 11 partial, 7 fail, 100 of 147 hints, 24.5 min; Vireo 0.9's after-fixes run by the same rule: 16 / 11 / 0, 128 of 148, 44 min (by hand it was 23 pass, 4 partial). 15 prompts the same, 12 worse, none better. Likely causes, in order: thinking is off and there is no check pass (a careful step skipped: 2.1's sum error, 4.1's second file, 7.1's row count, 12.4's cap); earlier turns' tool results are not kept (7.3, 8.2, 9.2); 20 calls a reply (in a five-prompt trial 2.1 hit it and ended with no answer). App tests for the script's new parts pass (4) |
| 2026-11-13 | After the gauntlet's first run on the prototype, what it found wanting. **Thinking** (a request's `thinking`, else `--thinking`; the Mac app's Settings, Chats, on by default): Gemma 4's system turn opens with `<|think|>` and Qwen's reply with `<think>`, as their templates write them, and a thought is opened after each tool's result; at most 2,000 tokens a thought and 8,000 a reply, closed for the model past that; sent to the app as `thinking` events apart from the answer (Muse reasons at high strength). No check pass: it was off in the 0.9 runs. **Earlier turns' tool results kept**: a reply's calls and results go to the app as a `turn_record`, which it keeps with the reply and sends back; `chat::history` writes them where they were made, in the family's own form, read as plain text (`chat::encode`); whole while the chat fits, then the first 1,200 characters of each, then none, before the chat's start moves on. **Fewer calls**: `read_file` takes several paths, one a line (up to 10, sharing three results' room); `calculate` says to put every sum in one call; 30 calls a reply, and the last one's result says it was the last (a thought no longer counts as the answer that's due, which is how a reply ended with nothing said). **Rules**: the tools' preamble says to check a change someone waved through against the user's own rules and to name the rule; a request with tools ends with today's date. **New on the Mac**: `search_chats` over the app's chat files (`chats.rs`); `list_events`, `create_event`, `create_reminder` run by the app through EventKit (pretended in a test script unless it says `"calendar": "real"`); `schedule_task` (`tasks.rs`: a time of day, once, a folder's new files, a link) kept by the server and run by the app as a chat of its own (`TaskRunner`, Settings, Tasks). `aurelia_run.py` plays all 32 prompts now | Unit tests only: server 103 (thinking and kept results in each family's form, a record to the app and back, task schedules and what is due, chat search), Mac app 144 (reasoning apart from the answer, records saved with a reply, dates read, a dry calendar run). **Not run:** the eval, the gauntlet, or any model at all: every prompt changed (the preamble, the date, two tool descriptions, three more tools declared when the app offers them), so the eval needs running on the small and the large models before this is merged, with thinking on and off. **Not built:** the Windows app's side (it ignores the new events; it would have to keep and send back `record`, and say `run_tasks` to get tasks) |
| 2026-11-13 | The Mac against mlx-vlm after the 8-model benchmark (`evals/speed/bench-2026-11-13`). **Gemma 4 26B-A4B's experts**: `moe_pick` on a threadgroup (each thread places its experts among the scores; one thread picking a best at a time took 0.2 ms a token, a third of the GPU's time writing), `moe_group` on 1,024 threads (an expert and a part of the slots each; 1.4 → 0.2 ms for 8,192 slots), and a batch's products on the neural accelerators (`moe_matmul_tc`, vireo_tensor.metal): the slots' inputs gathered in their experts' order in f16 (`moe_gather_half`), the products left in that order and added up through `place` (`moe_sum`); the default mode only, exact mode keeps `moe_matmul`. **A model's file let go after loading** (`Mmap::release`, the file mapped again over its own mapping; `GgufFile::release_pages` after the model and a draft model load): its pages no longer sit beside the GPU's copy. **Qwen's first chat** saves its DeltaNet at the system turn's end (`continue_with`'s `head`), so the second chat no longer reads that turn again. **Prompt attention** (`attention_tcg`): the output rescaled only where a row's maximum moved, each thread's rows found once, 8-17% less time at 30K positions (`prompt_attention_speed`), the same numbers. | 26B-A4B (UD-Q6_K, M5 Max, one run through vireo-server, before → after, mlx-vlm 6-bit / llama.cpp b12228): short reply 61 → 95 tok/s (108 / 94); reading 3K 1,730 → 3,160 (4,210 / 2,960), 32K 1,670 → 2,940 (3,040 / 2,010), 96K 1,270 → 1,830 (1,850 / 1,270); writing after 96K 78 (49 / 48); peak memory 53 → 31 GB (34 / 38). Qwen 3.5 9B Q6_K: writing 63-68 (mlx 66-71), reading 3K / 32K / 96K 2,790 / 2,240 / 1,670 (3,180 / 2,520 / 1,890); a new chat's first reply keeps 1,477 of 1,563 tokens (Qwen 3.5 4B). Perplexity, 26B, 6 × 1,024 tokens: 27.44 (27.56 before, exact 28.04). Engine tests (75 + GPU 17, with the 26B's default-mode batch against exact mode's next token), server 110. Writing on the dense models is bound by reading the weights (Qwen 9B: 7.5 GB a token at 70 tok/s). Open: prompts read 10-25% behind mlx-vlm on Qwen and at 3K on the 26B; the 26B's default mode on a repeating text (next token 0.95 exact, 0.71 with the f16 cache alone); E2B and E4B write their reasoning as the answer with thinking off under the agent's system prompt |
| 2026-11-14 | What the all-Q6 comparison of Nov 13 found (eight models, three engines; HANDOFF.md's item 7). **Tool calls drafted** (`write_call`): where the model is free to write (a path, a sum, a file's text) the token is read with the draft layer's guesses, or the chat's, and each the model would have picked itself under the grammar is taken; where the grammar writes the text or scores a choice it goes on as before. Calls were 70% of the writing in Gemma 4 31B's tool task, a token at a time (54 ms); 95% of the guesses are taken there. The small Qwens, which don't guess from the chat in a reply's text on the Mac, do inside calls (`text_drafts`). **A skill that names a tool that changes files makes those tools callable** (the 31B was kept from `write_file` by the request's own words and read and summed in circles). **The same call again** is caught wherever it was made earlier in the reply, until something is changed, and counted once a round. **A call past its limit** (4,096 tokens; 768 for `calculate`) is taken back (`take_back`) and the model answers without more calls, instead of the reply failing. **Gemma with thinking off** is told so in its system turn (`GEMMA_NO_THINKING`). **Q6_K's blocks on the GPU** are 212 bytes, not 224 (`decode32` reads them as packed words). | One run each through vireo-server (M5 Max, Q6, before → after; mlx-vlm): tool task, Gemma 4 31B 56.2 → 28.5 s (57.9), Qwen 3.8 27B 56.0 → 30.4 (57.7), Muse 66.1 → 50.3 (145), 26B 11.6 → 9.2 (13.1), E4B 14.1 → 7.3 (14.8), Qwen 3.5 4B 9.5 → 6.5 (12.9), 9B 16.0 → 10.9 (18.6); skill task, 31B 113 → 27.1 (60.6) and right, Qwen 27B 59.5 → 28.7 (55.3), Muse 92.1 → 64.8 (193), E4B 15.8 → 7.4 (11.5), Qwen 4B 11.1 → 7.3 (11.5). Scored tasks right of 6: E4B 1 → 6, E2B 0 → 4. Qwen 27B's long tool task 365 s and an error → 209 s and right. Q6_K at 212 bytes: Qwen 3.5 9B writing 63 → 66 tokens/s, perplexity the same to every digit on Qwen 3.5 9B and Gemma 4 E4B, both modes. Server tests 111, engine 75 + GPU 17. Not run: vireo-eval, the gauntlet |
| 2026-11-15 | Windows, fourth pass (see "Fifteenth round"): Google's draft models for Gemma 4 E2B and E4B on CUDA. Their centroid head (2,048 clusters, the 32 best scored: `gemma4_assistant.rs`'s `Centroids`, on the CPU), 4 query heads of their own, a guess's layers replayed from a graph, guesses stopped where the draft gives under 0.7 (`mtp_draft_sure`; CUDA), and the chat's long runs still taken beside the draft's. `vireo-eval` saves each case's reading, writing and tool times; `tools/reply_speed.py` times four kinds of reply one way against another | Greedy in the engine the same text as one token at a time, 1.25× (E2B) and 1.41× (E4B) as fast; first guess right 63% and 69% with no floor. Through the server against the chat's guesses alone (`--mtp off`), 400-token replies: E2B a list 85 → 105 tokens/s, an explanation 78 → 97, a story 79 → 86, an edited code file 170 → 179; E4B 46 → 66, 46 → 62, 42 → 53, 104 → 109; much the same at the models' own sampling (`evals/speed/windows-2026-11-15/`). The full eval, draft off → on: E2B 34 and 34, E4B 39 and 39, all 43 answers identical on both, writing 76.5 → 85.0 and 46.1 → 53.3 tokens/s (`evals/windows-small-gemma-draft.json`, with each case's speeds). Qwen 3.5 4B 38 for 13 November's 39 on `main` too. Engine 77, GPU 14, server 98, eval 12 tests pass. The GPU was held to ~33 W (a token about twice 12 November's time): ratios, not absolutes. Not compiled on the Mac |
| 2026-11-15 | Windows against llama.cpp again, the Mac benchmark's way (`evals/speed/bench_windows.py`, after `prompts-mac/bench_mac.py`, with its three prompt files; `bench_report.py` builds the page): read and write speeds with no drafting at 33-34, ~2,300 and 29,166-33,915 tokens for Qwen 3.5 0.8B, 4B and 9B and Gemma 4 E2B and E4B, whole replies through both servers for the four chat models, GPU memory. `tools/reply_speed.py` now refuses an approval at once (Qwen 3.5 9B asked to remember something and the reply waited two minutes) | RTX 5060 Laptop at full speed (Turbo plan, after a restart; that morning it ran at a third to a half), llama.cpp b12262, two rounds, Vireo as a share of llama.cpp: writing 95-111% in a short chat and 93-103% after the long prompt; reading ~2,300 tokens 105% (0.8B), 109% (E2B), 102% (E4B), 87% (4B), 79% (9B); reading the long prompt 78-84%. Replies as shipped, tokens/s against llama.cpp's: E2B a list 265 / 154, an edited code file 500 / 154, a story 191 / 155, an explanation 226 / 156; E4B 169, 336, 118, 160 / 90; Qwen 3.5 4B 95, 98, 87, 87 / 91 (it called a tool for the code request instead of returning the file); 9B 104 and 175 / 57 when it has the memory to guess, 57 when not, 54 on new text. `evals/speed/windows-rtx5060-2026-11-15-llama.html` and `.pdf`, raw in `bench-2026-11-15-windows/` |
| 2026-11-15 | Windows against llama.cpp, the Mac's 14 November benchmark rebuilt from its report (`evals/speed/bench_tasks.py`; the Mac's harness was never pushed): both servers, Gemma 4 E2B and E4B and Qwen 3.5 4B and 9B at Q4, one run each, a ~40-token question, ~3K, and a long prompt as large as the GPU holds (26,306 tokens for the Gemmas, 24,736 and 15,310 for the Qwens) with two facts planted, a three-turn chat, tool, skill and long-log tool tasks, GPU memory, answers scored (5, the Mac's 6 less its 96K prompt). The harness fixed on the way: llama.cpp's first-token time had been its whole first request's, and the log was sized at 3.2 characters a token (it is ~1.9), too long for Qwen 3.5 9B's 21,504-token context; it now keeps vireo-server's log, stream errors and tool arguments. `bench_tasks_report.py` builds the page | RTX 5060 Laptop at full speed (Turbo plan), llama.cpp b12262, Vireo's difference (speeds Vireo ÷ llama.cpp − 1, times and memory the other way): writing a short reply +28% and +33% on the Gemmas (the server's guesses and draft models), −1% on both Qwens; after the long prompt +42%, +45%, −8%, +2%; reading ~3K +4%, +1%, −18%, −22%; reading the long prompt −12% to −25% (median −19%), first token after it −11% to −25%; tool task +17%, +41%, −42%, −34%; skill task −31% to +2%; long tool task 0% and +8% (Gemmas only); GPU memory −8% to −29%. Scored right: Vireo 15 of 20, llama.cpp 18. Both Qwens' long tool task ends on Vireo in "A tool call ran too long" (a `calculate` call counting the log's errors as `sum([1, 2, 3, …`; llama.cpp's Qwen 3.5 4B the same, cut off as an HTTP 500); Gemma 4 E2B on Vireo also missed the tool sum and never wrote the skill's file. `evals/speed/windows-rtx5060-2026-11-15-tasks.html` and `.pdf`, raw in `bench-2026-11-15-windows-tasks/` |
| 2026-11-15 | Windows against llama.cpp with the Mac's own harness (`evals/speed/bench-2026-11-13/bench.py`, merged in PR #167), on the merged build (7e00a19): Gemma 4 E2B and E4B and Qwen 3.5 4B and 9B at Q4, both servers, one run each, the Mac's prompts whole (a ~32K repository dump with two hidden facts, an eight-turn chat, the harbor-bikes tool and skill tasks, a ~56K-token log then tool calls), scored by its `aggregate.py`. The harness runs on Windows with changes behind `os.name == "nt"`: the Q4 files and llama.cpp b12262, no large or xlarge rows, the whole GPU's memory from `nvidia-smi` between workloads, the folder's path written with forward slashes (a backslashed path in a tool call lost llama.cpp's skill task five reads), contexts measured to fit: 65,536 for three models, 32,768 for Qwen 3.5 9B (at 59,392 its caches spilled into the PC's memory and it read at 145 tokens/s) with its long log cut to the first half. `bench_tasks.py` run again on the merged build for a before and after. `bench_mac_harness_report.py` builds the page | RTX 5060 Laptop at full speed (Turbo plan), llama.cpp b12262. Vireo's difference: tool task +5% to +125% (faster on all four), skill task −25% to +71% (three ahead; Qwen 3.5 9B made 17 calls to 8), long tool task −26% to −2%; reading ~32K −15% to −21% and the first token after it the same; writing after it +12%, +4%, −13%, −5%; short reply −10%, +27%, −8%, −10% (E2B's one reply; re-checked 190-198 tokens/s, ahead); GPU memory −15% to −42% (Vireo sets aside the whole context's caches: 7.5 GB on E4B at 65,536). The Mac's same four models are ahead on all of these but memory (reading ~32K +35% to +60%). Scored right: Vireo 16 of 20, llama.cpp 15; Vireo's misses are the Mac's exactly. Vireo's Gemmas and Qwen 3.5 9B answered the long tool task without a tool call, as on the Mac. The merge, on `bench_tasks.py`'s identical workloads: both Qwens' long tool task now finishes, right (26.9 s and 32.2 s; before, "A tool call ran too long"); Gemma tool task 25% and 11% faster, skill task 40% and 43%; Gemma 4 E2B's skill task now right; Vireo 15 → 18 of 20; llama.cpp unchanged (18, times within 3%). `evals/speed/windows-rtx5060-2026-11-15-mac-harness.html` and `.pdf`, raw in `bench-2026-11-15-windows-mac-harness/` and `bench-2026-11-15-windows-tasks-after-merge/` |
| 2026-11-15 | Windows, three changes after the Mac-harness run (uncommitted on 7e00a19): (1) Qwen keeps a second saved state of its DeltaNet layers a few tokens before the end of the system turn it reads (`Kept::turn` in vireo-server), so the next chat with the same instructions and tools reads none of them (the one saved state before was where the first two chats parted); (2) on CUDA a Gemma sliding-window layer keeps only its window and a batch's new positions, position p at row p mod ring (`ring_row` in vireo.cu, `attention_ring`), as the Mac did; (3) the prompt attention kernel turns a batch's queries to f16 where they lie (`q_rows_to_half`) instead of copying them into each block's shared memory, and uses larger blocks for batches over 64 tokens (a `_short` set for the rest). Measured with the Mac's harness exactly as that morning, the eval on both builds, and the Gemmas at 131,072 tokens | RTX 5060 Laptop at full speed (Turbo), one run each. GPU memory, highest: Gemma 4 E2B 3.5 → 2.7 GB, E4B 7.5 → 5.0 GB at 65,536 tokens (loaded 3.3 → 2.5, 7.2 → 4.7); Qwens unchanged. Reading ~32K: +10% (E2B, 9,447 → 10,431 tokens/s), +6% (E4B), +3% (Qwen 3.5 4B), +2% (9B); first token after it 2-6% sooner; long tool task 1-6% quicker. llama.cpp, unchanged, moved 0.8% (median) and 3.4% at most on its speeds. Against llama.cpp Vireo still reads ~32K 9-19% slower and uses 14-22% more GPU memory (was 15-42%); tool task faster on three. Cost of the rings, as on the Mac: after a prompt longer than a ring the next chat reads its ~1,300 tokens of instructions and tools again. Eval: 149 → 151 of 172 (E2B 30 → 32, E4B 39 → 38, Qwen 3.5 4B 39 → 40, 9B 41 → 41); four cases tipped on close choices, none a wrong answer (E4B's calc-tip gave the right sum without calling calculate); 110 of 172 answers identical. Qwen read 55% and 61% fewer tokens over the eval and took 20% and 18% less time (128 → 103 s, 196 → 161 s); the Gemmas read 79-90% more and E4B took 51 → 54 s. At 131,072 tokens with the ~95K prompt both Gemmas fit on both engines without spilling: Vireo read 5,371 (E2B) and 2,992 (E4B) tokens/s, llama.cpp 6,662 and 3,429. `evals/speed/windows-rtx5060-2026-11-15-after-changes.html` and `.pdf` |
| 2026-11-15 | Windows, evening: two more small gains and where the rest of the gap is (Nsight Systems, a kernel's time over a ~3K-token read). **The feed-forward's activation and its rounding in one pass** (`swiglu_q8`, `geglu_q8`, `Gpu::gated_quantize`, `GpuMatrix::mul_gated`; CUDA's 8-bit mode): the middle of a feed-forward is only ever the next product's input, so it is rounded to 8 bits as it is worked out and never written as f32. **Guesses from the chat need a run of 3 on CUDA** (`draft::LEAST_RUN`; a run of 2's one guess was seldom taken). `tools/reply_speed.py` loads one server at a time (two copies of Qwen 3.5 4B on 8 GB spilled, and a list read 51 tokens/s for 97). | `vireo-cli` on `prompts-mac/medium.txt`, before → after: Gemma 4 E4B 6,238 → 6,475 tokens/s, Qwen 3.5 4B 3,438 → 3,540, Gemma 4 E2B 17,450 → 17,830, Qwen 3.5 9B level. Qwen 3.5 4B's 400-token replies through the server, runs of 2 → 3 (no guesses): a list 96.9 → 98.4 (89.6), a code request 90.4 → 92.9 (90.2), a story 86.9 → 89.6 (89.8), an explanation 90.9 → 93.0 (89.7). Where a ~3K read's time goes: Qwen 3.5 9B's matrix products 79% (Q4_K 52%, Q5_K 14%, Q6_K 13%), DeltaNet 8%; Qwen 3.5 4B's 69% and 16%; Gemma 4 E4B's Q4_0 products 67%, at about twice the K-quants' rate a weight. The K-quants' minimum term (a fused multiply-add for every result and group of 32) is 8% of the 4B's read and 12% of the 9B's (timed without it: 652 → 598 ms, 1.19 → 1.05 s): **Tried and dropped:** that term on the tensor cores in f16 (the 6-bit minimums as whole numbers, a token's sums as shares of its largest, one f16 product 8 deep a tile and step, then one multiply-add a result): slower, 652 → 723 ms and 1.19 → 1.28 s, and off the dp4a kernels by 1.4e-4 of the largest value. The tensor cores are what the kernel is short of: 8 more products a step on top of its 32 cost more than the 128 multiply-adds they replaced. Once every 4 steps would be about level, not worth the rounding. A token's logits are 1.7 ms of Gemma 4 E4B's 11 (the Q6_K output matrix, 550 MB, read at ~320 GB/s). Engine 77, GPU 15, server 98, eval 12 tests pass. No full eval or benchmark run on these two changes yet |
| 2026-11-15 | Windows, night (branch `windows/speed-hunt`, uncommitted): the prompt attention kernel under Nsight Compute, and two changes to it. What `attention_mma_512` waits on at ~33K tokens (one block an SM, 8 warps, 218 registers a thread, 83 of the 100 KB of shared memory a block may have): the tensor cores 47% busy, an instruction issued every 6th cycle; stalls 30% the tensor pipe, 24% reads from global memory, 20% fixed waits, 12% the shared-memory queue. Timed with a phase taken out (`tests/prompt_attention_speed_cuda.rs`, a 1,024-token batch at 30,000 positions, 28.0 ms 512 wide): no key and value loads 21.2 ms, neither product nor the exponentials 19.3, none of those 10.7, the queries as constants 22.0. **The queries were the find**: read from global memory since that afternoon, a block's 64 KB of them don't fit what is left of the L1 cache (32% of reads found there), so each chunk read them from L2 again and every warp waited for each read. **Kept: (1) the queries read two steps ahead** (`attention_mma`: `q_rows_to_half` lays each 32 numbers out as a thread takes them, one 16-byte read for two steps of S = Q Kᵀ, asked for two steps before their products and carried over a chunk's end; the keys' fragments by `ldmatrix`), the same numbers to the last bit. **(2) A batch of 9 to 64 tokens with no window has its context in parts** (`attention_mma_*_part`, `PART` in the template): a block takes sets of 16 tokens of the query heads that share a key/value head (not 16 × rows tokens of one head, most past the batch's end), the context split into as many parts as give every SM a block, merged by `attention_combine` as one token's are. A draft's 2 to 8 tokens keep the one-token kernel (its numbers are one token's exactly). | RTX 5060 Laptop at full speed (Turbo). The kernel, 30,000 positions: 512 wide 28.0 → 24.6 ms, 256 wide 21.1 → 18.9 ms, the sliding layers' 0.315 → 0.305 ms (one step ahead about half of that; three or four no better). `vireo-cli` on `prompts-mac/long.txt`, before → after: Gemma 4 E2B 10,559 → 11,277 tokens/s (+6.8%), E4B 4,812 → 4,952 (+2.9%), Qwen 3.5 4B 2,761 → 2,795 (+1.2%), 9B 1,740 → 1,754 (+0.8%); `medium.txt` and writing level (E2B 17,900, E4B 6,498 and 88 written, Qwen 3.5 4B 3,521 and 88, 9B 1,950), the text written the same, GPU memory the same. A batch 27,000 tokens deep (`check_speed`), ms, before → after: Qwen 3.5 4B 12 tokens 40.8 → 23.7, 16: 40.7 → 24.0, 32: 44.0 → 30.7; Gemma 4 E2B 18.9 → 10.5, 18.8 → 10.6, 21.8 → 13.4; E4B 29.9 → 18.5, 29.7 → 18.7, 34.6 → 23.1; 64 tokens and over, and 2,000 tokens deep, level (4-6% quicker at 12-32 tokens). **Tried and dropped:** fewer and wider loads alone (the 16-byte query read and `ldmatrix` for the keys without reading ahead: 28.2 → 28.1 ms; loads cost latency, not instructions); two sets of sums in S = Q Kᵀ so a product doesn't wait for the one before (level: the tensor pipe is the limit there, not the chain); other block shapes with the queries in global memory (512 wide 4×2×16 31.4 ms, 6×2×16 53, 8×2×16 86, 4×4×16 43; 256 wide 8×1×48 21.5, 8×1×32 23.6, 12×1×32 32.9, 6×1×64 30.8: more warps spill registers, smaller chunks cost more per position); the softmax without its masks where a whole chunk is seen, and a ring's row worked out once a chunk (together 5% slower, the second alone 1.5%: this kernel's speed hangs on how its registers fall). **Not worth building:** the queries back in shared memory (64 KB beside 66 KB of keys and values and 16 KB of sums, 100 KB allowed); the K-quant products in f16 on the tensor cores (the 8-bit kernel uses 27% of the integer tensor pipe and does twice the multiply-adds an instruction; in f16 the pipe would be the limit). `matmul8_q4_k` under Nsight Compute: 2 instructions a cycle an SM, stalls 31-36% the shared-memory queue, 26% fixed waits, 9-14% global reads, 6% the tensor pipe. Where a ~34K read goes now: Gemma 4 E2B `attention_mma_512` 51%, `matmul8_q4_0` 32%; E4B 30% and 55%; Qwen 3.5 4B the K-quant products 55%, `attention_mma_256` 22%, DeltaNet 12%. Engine 77, GPU 15 (attention's test has six short-batch cases more), server 98, eval 12 tests pass. No full eval or benchmark run |
| 2026-11-15 | Windows, later that night (same branch, uncommitted): the 8-bit matrix product with a phase taken out, and the Gemmas' instructions after a long prompt. Qwen 3.5 4B's ~2.3K read, 653 ms, of which the products are ~450: without the tensor-core products 633, without the scaling ~600 (the minimum's term alone 598), without the reads from global memory 574, without those and the unpacking 499, without all four 436. So the products themselves are 5% of the kernel, the scaling 12%, reading and unpacking 17% each, and half is what is left: reading the fragments and scales back out of shared memory (Nsight Compute's 31-36% of stalls on the shared-memory queue). **Kept: (1) one side's fragments by `ldmatrix`** in `matmul8` (four reads in one: the weights' for Q4_0 and Q8_0, x's for Q4_K and Q5_K; the same numbers). **(2) Gemma's state saved at the system turn's end** (vireo-server, `Kept::head`, shared code): where the sliding-window layers keep rings, a prompt longer than a ring wrote over the window after the instructions and the next chat read all ~1,300 tokens again (`truncate` gave 0). The session's state there is now exported once when a system turn is read (`Gemma4CudaSession::has_rings`, `export`) and imported when the rings can't go back. | `vireo-cli`, start of the night → now, tokens/s: `medium.txt` Gemma 4 E2B 17,950 → 18,800 (+4.7%), E4B 6,498 → 6,848 (+5.4%), Qwen 3.5 4B 3,537 → 3,605 (+1.9%), 9B 1,986 → 2,028 (+2.1%); `long.txt` E2B 10,559 → 11,322 (+7.2%), E4B 4,812 → 5,148 (+7.0%), Qwen 3.5 4B 2,761 → 2,843 (+3.0%), 9B 1,740 → 1,788 (+2.8%); writing level (171, 87, 88, 55). Through vireo-server at 16,384 tokens, a one-line question in a new chat right after a ~2,400-token one: Gemma 4 E4B 0 → 1,267 of 1,307 tokens kept, first token 225 → 59 ms; E2B the same tokens, ~110 → 23-43 ms; the same answers; a system turn's first read 20-30 ms longer (the copy); Qwen 3.5 4B as before. **Tried and dropped:** both sides' fragments by `ldmatrix` (Gemma 4 E4B 366 ms for 357 with one side, Qwen 3.5 4B 675 for 638: slower than neither on the K-quants); the tokens' scales and sums by `ldmatrix` (633 for 638, E2B worse: noise); `ldmatrix` in Q6_K's products, either side or both (1-2% slower). Engine 77, GPU 15, server 98, eval 12 tests pass. Not compiled on a Mac. No full eval or benchmark run |
| 2026-11-15 | Windows against llama.cpp after PR 169 (`main` at 7b8bd4c), through both servers, one run each, reading, writing and first token only, at four prompt sizes: small (~40 tokens), medium (~3K), large (~32K; these three by `bench-2026-11-13/bench.py` with `WORK=short,medium,long` at the afternoon's contexts, 65,536 and 32,768 for Qwen 3.5 9B) and xlarge, the largest each model holds on 8 GB on both engines without spilling (`bench-2026-11-15-windows-after-pr169/xlarge.py`, the Mac's xlarge.txt cut to fit): Gemmas 131,072 with 125,561 tokens, Qwen 3.5 4B 114,688 with 113,190 (spills at 131,072), Qwen 3.5 9B 49,152 with 47,450 (fits 40,960 and 49,152, spills at 57,344; it was 32,768 before). Vireo against llama.cpp, Gemma 4 E2B / E4B / Qwen 3.5 4B / 9B. **Reading**: medium −18% (a one-off, re-check 18,024 and 19,150 tokens/s, +16% to +23%) / +1% / −11% / −20%; large 11,020 vs 11,986 (−8%) / 5,080 vs 5,181 (−2%) / 2,778 vs 3,169 (−12%) / 1,760 vs 2,052 (−14%), was −19% to −9% that afternoon, Vireo +5% to +8% since, llama.cpp ±1%; xlarge 4,908 vs 5,540 (−11%) / 2,856 vs 2,955 (−3%) / 1,744 vs 1,745 (0%) / 1,640 vs 1,878 (−13%). **Writing** (as shipped, the Gemmas with draft models): small −1% / +23% / −1% / −3%; medium 213 vs 153 (+40%) / 135 vs 88 (+54%) / +1% / −4%; large +11% / +5% / −11% / −6%; xlarge 64.6 vs 84.4 (−24%) / 40.6 vs 47.4 (−14%) / −9% / −5%, and with drafting off the Gemmas write 79.5 and 44.1 there, so the draft models cost speed after ~125K tokens. **First token**: small Vireo quicker on all four (+9% to +148%, its instructions kept); medium 0% to −17%; large −1% to −14% (3.0 / 6.4 / 10.7 / 16.9 s); xlarge 25.4 vs 22.6 s / 43.6 vs 42.2 / 64.1 vs 64.1 / 28.1 vs 24.5. Since PR 169 the Gemmas keep their 1,287-token instructions after a long prompt (0 that afternoon). | `evals/speed/windows-rtx5060-2026-11-15-after-pr169.html` / `.pdf` (`bench_after_pr169_report.py`), raw in `evals/speed/bench-2026-11-15-windows-after-pr169/` (results.jsonl 48 rows, xlarge.jsonl 12, xlarge-probes.jsonl, recheck-e2b-medium.jsonl); no errors; full speed before (6,907 / 6,902 read, 77.9 / 86.9 write) and after (6,921 / 6,893, 91.4 / 88.2) |
| 2026-11-15 | Windows, after that run (branch `windows/after-pr169`): why the Gemmas wrote slower than llama.cpp after ~125K tokens, and whether coarser activation scales would speed the Qwens' reads. **Guesses deep in a chat** (a 200-token description after a repository dump cut to size, vireo-server at 131,072 tokens, tokens/s with no guesses / the draft model's only / both as shipped): Gemma 4 E2B at 5K tokens 164 / 204 / 202, 32K 131 / 152 / 137, 63K 109 / 103 / 105, 93K 94 / 91 / 86, 119K 85 / 75 / 72; E4B 5K 87 / 142 / 144, 32K 74 / 98 / 91, 63K 61 / 71 / 61, 119K 48 / 49 / 42. Checking a round costs each guess a pass over the whole context, the chat's guesses are taken 10-25% of the time on new text there, the draft model's 52-78%. **Kept (vireo-server, CUDA only): guesses have to earn their keep deep in** (`guesses_pay`): past 24,000 tokens the chat's guesses stop for a reply once 8 were made and under 60% taken; past 48,000 a draft model's stop once 16 were made and under a share that grows from 50% to 75% at 120,000 (`mtp_share`). Copied text, nearly all taken, keeps its speed. **Tried and dropped, the Qwens' 8-bit product:** the scale reads asked for before the products (642 -> 637 ms, the wait isn't theirs); and, as speed-only mock-ups before building it, activation scales coarser than 32 columns: one scale a token a 128-column step, Qwen 3.5 4B's ~2.3K read 637 -> 656 ms and 9B's 1.14 -> 1.13 s; with whole-number scaling a group and floats once a step as well, 647 ms and 1.12 s. Not worth changing 8-bit numbers for, so it was not built. With the scaling's arithmetic removed the 4B's read is 616 ms, with everything that feeds it too (the scales fetched, decoded, stored and read) 508, and reading the fragments costs nothing (641 either way): the scaling is 21% of the 4B's read and 29% of the 9B's only as a whole. | As shipped before -> after, tokens/s: Gemma 4 E2B 32K 137 -> 144, 63K 105 -> 104, 93K 86 -> 91, 119K 72 -> 79; E4B 32K 91 -> 91-94, 63K 61 -> 60, 93K 54-57, 119K 42 -> 46; 5K unchanged (202, 144). Short of the best at each depth where a reply's first guesses mislead (E4B at 63K: 7 of 16 taken, then off, where the draft model alone gives 71). Qwen 3.5 4B made no guesses in this task either way (85 / 74 / 62 / 52 at 5K / 29K / 59K / 91K). Engine 77, GPU 15, server 98, eval 12 tests pass. No eval or benchmark run |
| 2026-11-15 | Windows (branch `windows/qwen-options`): the two exact options left for the Qwens' reads. **Kept: the K-quants' products in tiles of 128 tokens** (`matmul8_t128_*`, `WIDE_TOKENS`; a warp takes 32 rows by 64 tokens, a block an SM): a block reads and unpacks its rows' weights once for twice the tokens. Q4_K, Q5_K and Q6_K in batches of 128 or more; what a batch leaves over goes to the 64-, 32- or 16-token tile when one holds it (`matmul_8bit` runs the two parts, the second from its first token on). The same numbers to the last bit (`short_batch_tiles_match_the_64_token_tile` now reads 128, 140, 192 and 200 tokens too). Q4_0 and Q8_0 read no faster so (Gemma 4 E4B 355 ms either way) and keep 64. **Not built: the K-quants' scales stored ready-made** (speed-only mock-ups in `store`): as whole bytes beside d and dmin, Qwen 3.5 4B's ~2.3K read 637 -> 628 ms, 9B's 1.13 -> 1.11 s; ready in f16, 623 ms and 1.07 s (2% and 5%), and less than that after the wider tile, which halves the unpacking a token. It would take 11% more memory for those weights and slow writing, which reads every byte a token (or 22% more as a table beside them, which Qwen 3.5 9B has no room for), and f16 scales change the numbers a little. | `vireo-cli`, before -> after, tokens/s: `medium.txt` Qwen 3.5 4B 3,605 -> 3,854 (+6.9%), 9B 2,028 -> 2,206 (+8.8%); `long.txt` 2,843 -> 3,030 (+6.6%), 1,788 -> 1,941 (+8.6%); writing level (88 and 54, 68 and 46 after the long prompt); the Gemmas as they were (18,400 / 11,410 and 6,860 / 5,150). Qwen 3.5 4B's batches (`check_speed`), ms: 128 tokens 45.2 -> 44.4 at 2,000 deep and 71.2 -> 70.3 at 27,000; 256 tokens 81.1 -> 75.0 and 133.4 -> 127.5; 64 and under unchanged. The first try, with no smaller tile for what is left over, gave 615 ms and 1.06 s on `medium.txt` (its last batch is 264 tokens: two tiles and 8 tokens). Engine 77, GPU 15, server 98, eval 12 tests pass. No eval or benchmark run |
| 2026-11-15 | Mac, after the 14 November speed report (branch `prototype/mac-gains-nov15`, uncommitted): where a prompt read's time is and what the neural accelerators have left. Qwen 3.5 9B's ~3K read is 80% matrix products, and `matmul_tc` already runs them at the rate of an f16 matrix that needs no decoding (57 TFLOPS on real weights, `tests/big_matmul.rs`); read straight from device memory with no threadgroup copy the same product gives 63, and **8-bit by 8-bit gives 118** (a standalone `matmul2d` benchmark; half by 8-bit 60). **Kept: (1) the weights' tile decoded into two buffers in turn** (`sab` in `matmul_tc`): a step's decoding no longer waits for the last step's product, one barrier a step, the same numbers. **(2) A model's buffers made resident at load** (`Gpu::settle`, called by vireo-server and vireo-cli; a no-op on CUDA): Metal otherwise did it inside the first prompt's first command buffer (135 ms in the driver for Gemma 4 26B-A4B; `VIREO_METAL_TIMING=each` prints each buffer's driver, wait and GPU time). (3) The f16 copy of a batch is made by the first product that reads it. **Built and left off: 8-bit products** (`matmul_i8`, `to_i8`, `VIREO_I8=on`; Q6_K and Q8_0, batches of 128 tokens or more): activations rounded to 8 bits a token's 256 columns, weights to 8 bits a row's 256, whole-number sums scaled into f32 after each step. **Tried and dropped:** the f16 activations in blocks of 256 columns (level), 256 or 512 columns a step (level, slower), the 26B's two input norms and its closing norms and add in one dispatch each (writing 100 tokens/s either way: the small kernels' launch time isn't serial). `vireo-cli perplexity` writes each token's log p and top token with `VIREO_PPL_DUMP`; `tools/ppl_compare.py` compares two such files | Products, Qwen 3.5 9B, 1,024 tokens: ffn_up 56.4 -> 57.6 TFLOPS, attn_q 55.1 -> 56.5 (two buffers); with 8-bit 75.0 and 74.0. First prompt after a load, `vireo-cli`, ~3K tokens, main -> branch: Gemma 4 E2B 22.7K -> 25.9K tokens/s, E4B 7.7K -> 8.6K, Qwen 3.5 4B 4.8K -> 5.06K, 9B 2.78K -> 2.93K, 26B-A4B 3.19K -> 3.72K; the same 64 tokens generated on all five. **8-bit against exact, 20 chunks of 512 (`fixtures/ppl.txt`): the likeliest token differs at 4.9-6.4% of positions (f16: 0.06-0.12%), rms change in log p 0.11-0.32 (f16 0.002-0.017)**, perplexity -0.02% to +0.5%; scales every 32 columns still 2.5-3.5%. Reads with it: Qwen 3.5 9B ~3K 3.06K -> 3.79K tokens/s, 32K +10-20%; 4B 32K +12%. Too large a change for the default. Engine 76, GPU 17 (also with `VIREO_I8=on`), server 111 tests pass. No eval or benchmark run |
| 2026-11-15 | Qwen 3.5 against Qwen 3.8: what their model cards and chat templates ask for, against what vireo-server does (branch `prototype/qwen-38-config`, uncommitted). **Sampling by mode** (`Family::sampling(thinking)`): both cards give thinking 1.0 / 0.95 / top-k 20 and thinking off 0.7 / 0.8 / top-k 20; the server used 0.6 / 0.95 / 20 for both, which is Qwen 3.5's set for precise coding with thinking on (3.8's card has no such set, and its GGUF says `general.sampling.temp 1`). Not applied: the cards' presence penalty (1.5 with thinking off on both; with thinking on 1.5 for 3.5, 0 for 3.8), which the sampler doesn't have. **Past turns as 3.8's template writes them** (`chat::note_qwen_template`, set where the template has `preserve_thinking`): each earlier reply's turn opens with the empty thought (`<think>\n\n</think>\n\n`), after a tool's result too, as the template does for a reply whose thinking wasn't sent back; 3.5's template leaves past thoughts out, as before. **Found, not changed:** (1) 3.8's template opens the system turn with a reasoning-effort line when thinking is on (xhigh by default, none for medium, another for low); the server writes none, which is its medium. (2) With tools offered a request ends with the date note, which the next turn's history doesn't have, so every turn reads the last reply again from the end of the last message (all families; 235 + 53 tokens on the 27B's second turn): keeping the note in the history would let a Qwen 3.8 chat with thinking off go on from the end of its cache. (3) 3.8's card suggests far longer replies (131,072 tokens) and keeps thinking across turns when it is sent back. (4) This Q6 file of 3.8 has 153 tensors in Q8_0 (22% of its weights). **Drafting:** Qwen has no draft model of its own; its multi-token-prediction layer is the drafter and vireo-server uses it by default on 3.8 (3 guesses a round, 41-82% taken in a four-turn chat). The Qwen 3.5 4B and 9B files here (Unsloth, Q6_K) have no such layer, so they don't draft; files with it exist (Unsloth's and bartowski's MTP GGUFs). DFlash 2, a third party's ~2B block-diffusion drafter for 3.8, is not built: other engines measure it level with the layer at 2-4 guesses, it reads no images, and its files were reported broken | Server 112 tests pass (one new). Nothing measured for accuracy: no eval or gauntlet run with the new sampling or past turns |
| 2026-11-15 | Qwen 3.8, the three left open by the row above (same branch, uncommitted), each only for a Qwen whose template has `preserve_thinking` (`chat::qwen_keeps_thought`). **A presence penalty in the sampler** (`Settings::presence_penalty`, the request's `presence_penalty`): taken from the logit of each token the reply already has (its last 1,024), not from a greedy pick; 1.5 for Qwen 3.8 with thinking off, as its card asks, 0 with it on. **The reasoning-effort line** (`QWEN_EFFORT`): with thinking on the system turn opens with the template's default, xhigh. **Past requests' date notes kept** (`keep_notes`): a request with tools ends with the date, read apart from the history, and the next turn's history hadn't it, so the prompt parted from the cache at the last message's end and the last reply was read again every turn. Where the prompt parts from the chat's cache at a turn's end and the cache goes on with a date note (checked by its text), the note's tokens are put into the prompt as the cache has them, while it still fits. With the past turns' empty thought (row above) a Qwen 3.8 chat with thinking off now goes on from the end of its cache. Thinking on still reads the last reply again (its thinking isn't in the history) | Qwen 3.8 27B, four-turn chat, replies of 61-235 tokens: tokens read on turns 2-4 288 / 117 / 86 -> 44 / 47 / 42, first token 0.49 / 0.23 / 0.20 s -> 0.17 / 0.20 / 0.21 s. The benchmark's eight-turn chat reads only each new message (506-530 tokens a turn; 526-553 before). Its tool task 33.5 s and skill task 32.1 s, both right (31.5 and 30.6 on 14 November). Reasoning with and without the effort line, one sampled run each of three questions: 2,001 / 134 / 57 tokens against 1,776 / 67 / 100: no clear difference. Qwen 3.5 9B unchanged (254 / 121 / 90 read). Server 113 tests pass. No eval or gauntlet run |
| 2026-11-15 | Windows against llama.cpp after PRs 171 and 172 (`main` at 99c47ed), the after-PR-169 run repeated: both servers, one run each, reading, writing and first token at small (~40 tokens), medium (~3K), large (~32K; 65,536 context, 32,768 for Qwen 3.5 9B) and xlarge (Gemmas 131,072 with 125,561 tokens, Qwen 3.5 4B 114,688 with 113,192, Qwen 3.5 9B 49,152 with 47,448; no row spilled). Vireo against llama.cpp, Gemma 4 E2B / E4B / Qwen 3.5 4B / 9B, then Vireo's change since after PR 169. **Reading**: medium −4% (the first server's one-off again, 15,315 against 18,012 drafting off; re-check 18,018 both settings) / +3% / −6% / −12%, Vireo +7% to +20% since; large 11,328 vs 12,198 (−7%) / 5,185 vs 5,279 (−2%) / 2,982 vs 3,197 (−7%) / 1,928 vs 2,102 (−8%), was −14% to −2%, Vireo +3% / +2% / +7% / +10%; xlarge 4,994 vs 5,621 (−11%) / 2,893 vs 3,000 (−4%) / 1,817 vs 1,753 (**+4%**) / 1,780 vs 1,898 (−6%), Vireo +1% to +9%. **Writing** (as shipped): small +10% / +30% / −1% / −3%; medium 216 vs 155 (+40%) / 137 vs 89 (+54%) / +1% / −2%; large +11% / +5% / −8% / −2%; xlarge 72.9 vs 82.6 (−12%) / 46.0 vs 47.2 (−3%) / 43.4 vs 45.8 (−5%) / 44.8 vs 45.9 (−2%), was −24% to −5%: the Gemmas +13% each since (PR 171's guesses stopping deep in a chat), drafting off 78.5 and 45.7. **First token**: small Vireo quicker on all four (+3% to +68%, 30-120 ms); medium −11% to +5%; large −8% to −1% (2.9 / 6.3 / 10.0 / 15.4 s); xlarge 25.0 vs 22.2 s / 43.0 vs 41.6 / **61.5 vs 63.9** / 25.9 vs 24.3. llama.cpp within 3.7% of the after-PR-169 run outside the small size (median 1.5%). | `evals/speed/windows-rtx5060-2026-11-15-after-pr172.html` / `.pdf` (`bench_after_pr172_report.py`), raw in `evals/speed/bench-2026-11-15-windows-after-pr172/` (results.jsonl 48 rows, xlarge.jsonl 12, recheck-e2b-medium.jsonl); no errors; full speed before (6,869 / 6,901 read, 86.6 / 86.7 write) and after (6,870 / 6,829, 87.3 / 87.4) |
| 2026-11-15 | Windows, late (branch `windows/llama-kernels`): llama.cpp's own kernels under Nsight Systems beside ours, the first time (`nsys profile --trace=cuda`, `llama-completion` and `vireo-cli` on `prompts-mac/medium.txt`, Qwen 3.5 9B; the GPU is 99-100% busy at ~95 W through a read on both, so the gap is in the kernels). **A kernel's time summed over the read, llama.cpp / Vireo:** Q4_K products 457 / 548 ms, Q5_K 124 / 137, Q6_K 96 / 134, DeltaNet ~62 / ~130, activation and rounding ~27 / ~56; all kernels 874 / 1,096. llama.cpp ends only 12% ahead because it loses the rest between its launches. A block of 128 rows x 128 tokens x 4,096 columns: llama.cpp 68-74 us on every shape, ours 75-78 us on 4,096-row matrices and 81-87 on the feed-forward's 12,288 rows. **What its `mmq.cuh` does that ours doesn't** (read from the source): the rounded activations are laid out a 128-column step and then token, 144 bytes each with their scale and sum, so a tile's is one flat copy (ours reads bytes, scales and sums at three strides a token); a warp loads its rows' fragments for a 256-column step once and keeps them across every token; stream-k where a launch's blocks would fill under 90% of its waves; gate and up in one launch with the activation. **Kept: (1) Q6_K's 128-token tile reads ahead** (it runs one block an SM; only the smaller tiles' two blocks would spill). **(2) A feed-forward's gate and up products and its activation in one kernel** (`matmul8`'s `GLU`, `Gpu::glu_8bit`, `GpuMatrix::mul_ffn`; batches over 32 tokens): a block takes 64 rows of each matrix, 16 and 16 a warp, and writes act(gate) x up already rounded for the down product, into a second 8-bit buffer (`q8_ffn`, batch x ffn_dim x 1.375 bytes: 17 MB for Qwen 3.5 9B); neither product is written as f32. The same bytes, scales and sums as the two products and `gated_quantize` (`the_gated_feed_forward_kernel_matches_two_products_and_the_activation`, 33 to 264 tokens, four models). **Measured, not built: blocks that don't fill the SMs' waves** (`tests/tile_waves.rs`: 26 SMs, one block each): Q4_K runs at 48 TOPS where its tiles divide by 26 and 41 on Qwen 3.5 4B's 2,560-row matrices (160 blocks, 7 waves for 6.15), which stream-k would recover. **Tried and dropped:** token tiles as the grid's fast index, as llama.cpp has them (9B +1.5-2%, 4B -1%, Gemmas +0.5%; Q5_K 5% slower alone). | `vireo-cli`, start of the session -> now, tokens/s: `medium.txt` Gemma 4 E2B 18,275 -> 19,076 (+4%), E4B 6,857 -> 7,093 (+3%), Qwen 3.5 4B 3,838 -> 3,905 (+2%), 9B 2,206 -> 2,241 (+2%, the Q6_K change; the one kernel level there); `long.txt` E2B 11,595 -> 11,842, E4B 5,185 -> 5,361, 4B 3,049 -> 3,112, 9B 1,941 -> 1,990 (+2% to +3%); writing level. Q6_K, Qwen 3.5 9B's ffn_down, 1,024 tokens: 3.50 -> 3.14 ms (llama.cpp's 2.62). The one kernel alone against the three passes, 1,024 tokens: 5.36 -> 5.31 ms (9B), 2.68 -> 2.61 (E4B). Engine 77, GPU 16, server 98, eval 12 tests pass. Not compiled on a Mac (`mul_ffn` returns false there). No eval or benchmark run |
| 2026-11-15 | Windows, after PR 177 (branch `windows/delta-mma`): the three things that run found, sized on a trace of each model first (a ~2.3K read's kernel time: Qwen 3.5 4B products 71%, DeltaNet 21%; 9B 81% and 12%; Gemma 4 E4B products 80%, attention 10%; E2B 64% and 20%; blocks that don't fill the last wave of SMs 3-5% on all four). **Kept: the DeltaNet chunks' scan on the tensor cores** (`delta_chunk_scan_mma`, compute capability 8.0 on, states of 128; `VIREO_DELTA=f32` keeps the old kernel): every number in two bf16 parts in one word (`split_bf16`: the nearest bf16 and the nearest to what that left), every product as hi x hi + hi x lo + lo x hi with f32 sums (`mma.m16n8k16`), as llama.cpp's `gated-delta-net-mma` does; the state stays f32, in registers. A block of 4 warps a value head and 32 columns; `delta_chunk_prep` writes W, Q-hat, P and K-hat (transposed) already split, in the same scratch. The chunks' test now runs on CUDA too (`delta_net_in_chunks_matches_token_by_token`): off token by token by 2e-5 on values of 2 (the f32 chunks 1e-6). **Tried and dropped: (1) the last wave's tiles split among the idle SMs** (a stream-k for the leftover tiles only: 160 tiles on 26 SMs as 156, then 4 in 6 parts each, added up by a third kernel): Qwen 3.5 4B's 2,560-row products 4-6% quicker alone (1.124 -> 1.080 ms), but reads no faster, three alternating runs each: 9B -0.7%, 4B +0.3%, E4B 0%, E2B -1%; and those tiles' sums change in the last bit. **(2) In the new scan:** five blocks an SM instead of three (96 registers for 159: 700 us for 634), and its operands read a step ahead (665 us). It is the tensor cores it waits on: three products a product is ~17 TFLOPS of bf16, about 70% of what `attention_mma` showed this GPU has. **Not built:** `delta_chunk_prep` on the tensor cores too (its Gram products and W and U as T x B: now the larger half of DeltaNet, 710 us a 1,024-token batch to the scan's 634; llama.cpp's whole kernel ~1.0 ms); the products' activations laid out a tile at a time. | `vireo-cli`, f32 scan -> tensor cores, tokens/s: `medium.txt` Qwen 3.5 4B 3,920 -> 4,043 (+3.1%), 9B 2,206 -> 2,259 (+2.4%); `long.txt` 3,126 -> 3,177 (+1.6%), 1,973 -> 1,996 (+1.2%); writing level; the Gemmas don't run it. The scan, 1,024 tokens: 1,056 -> 634 us. **Against exact (`--exact`), 20 chunks of 512 (`evals/bench/fixtures/ppl.txt`), f32 scan / tensor cores:** Qwen 3.5 4B rms change in log p 0.0457 / 0.0472, same top token 98.29% / 98.00%, perplexity 22.491 / 22.471 (exact 22.459); 9B 0.0474 / 0.0482, 97.80% / 98.02%, 19.765 / 19.755 (19.762). The two scans differ from each other by as much (rms 0.043 and 0.040, 98.1% the same): the 8-bit activations turn any small change into that, and a 64-token greedy continuation parts ways after ~45 tokens. Engine 77, GPU 17, server 100, eval 12 tests pass. Not compiled on a Mac (the test, now on both, is). No eval or benchmark run |
| 2026-11-15 | Windows, the same branch, after that: `delta_chunk_prep`, then what both DeltaNet kernels wait on. **Tried and dropped: the prep on the tensor cores** (`delta_chunk_prep_mma`: k.k and q.k as one product [K; Q] K^T, T = (I + A)^-1 by 64 threads a column each, W and U as T [diag(beta gamma) K | diag(beta) V], all in two bf16 parts; right, 2.6-3.9e-5 off token by token): 692 us a 1,024-token batch for the f32 kernel's 710, and reads level. With the writes of Q-hat and K-hat taken out of it, 526: **it is the scratch they wait on**, 147 KB a chunk and value head, 75 MB a batch, written by the one kernel and read back by the other (which is also why five blocks an SM and reading ahead did nothing for the scan). **Kept: Q-hat and K-hat are no longer written.** They are q and k scaled a row, so `delta_chunk_prep` with `split` leaves only the scales (gamma_i and gamma_DC / gamma_i, in Q-hat's place) and `delta_chunk_scan_mma` reads q and k where they are (`load_q`, `load_kt`: scaled and split as a product takes them), 83 KB a chunk and head for 147. The same numbers as the scan before (the perplexity dumps are identical byte for byte). **Not built:** the whole of it in one kernel a head with chunks of 16, as llama.cpp's, whose W, U and P then never leave shared memory (its ~1.0 ms a batch to our 1.25 now). | A 1,024-token batch, Qwen 3.5 4B, us: prep 710 -> 559, scan 634 -> 688, together 1,344 -> 1,247 (2,050 with both in f32 at the start of the day's work on it, with `delta_out` and `delta_l2`'s 250 on top throughout). `vireo-cli`, f32 scan -> now, tokens/s, two alternating runs each: `medium.txt` Qwen 3.5 4B 3,932 -> 4,094 (+4.1%), 9B 2,212 -> 2,293 (+3.7%; 2,274 and 2,312); `long.txt` 3,106 -> 3,199 (+3.0%), 1,980 -> 2,012 (+1.6%). Against exact as the row above (0.0472 and 0.0482, 98.00% and 98.02%). Engine 77, GPU 17, server 100, eval 12 tests pass. No eval or benchmark run |
| 2026-11-15 | Windows against llama.cpp after PRs 177 and 178 (`main` at b5ff3ae), the after-PR-172 run repeated by an Opus agent: both servers, one run each, reading, writing and first token at small (~40 tokens), medium (~3K), large (~32K; 65,536 context, 32,768 for Qwen 3.5 9B) and xlarge (Gemmas 131,072 with 125,561 tokens, Qwen 3.5 4B 114,688 with 113,190, Qwen 3.5 9B 49,152 with 47,450; no row spilled). Vireo against llama.cpp, Gemma 4 E2B / E4B / Qwen 3.5 4B / 9B, then Vireo's change since after PR 172. **Reading**: medium −9% (the first server's one-off again, 14,576; re-checked alone 19,131-19,150, +19% to +20%) / +5% / 0% / −8%, Vireo 0% / +5% / +5% since (E2B aside); large 11,529 vs 11,923 (−3%) / 5,338 vs 5,193 (**+3%**) / 3,133 vs 3,180 (−1%) / 1,986 vs 2,081 (−5%), was −7% / −2% / −7% / −8%, Vireo +2% / +3% / +5% / +3%; xlarge 4,992 vs 5,533 (−10%) / 2,940 vs 2,963 (−1%) / 1,889 vs 1,757 (**+7%**) / 1,839 vs 1,842 (0%), Vireo 0% / +2% / +4% / +3%. **Writing** (as shipped): small +17% / +24% / −1% / −7% (9B's a one-off, re-checked 56.4-56.7, −1%); medium 217 vs 151 (+44%) / 135 vs 88 (+52%) / +1% / −2%; large +12% / +7% / −7% / −1%; xlarge 72.9 vs 82.7 (−12%) / 44.7 vs 47.0 (−5%) / 41.7 vs 45.1 (−8%) / 43.1 vs 45.9 (−6%); level with the last run within its noise (the xlarge answers are 33-63 tokens); drafting off the Gemmas write 77.5 and 45.3 at xlarge. **First token**: small Vireo quicker on three (9B's the one-off; 93-103 ms re-checked, ahead); medium +10% / +8% / 0% / −7%; large −3% / +4% / −1% / −4% (2.85 / 6.11 / 9.50 / 14.95 s); xlarge 24.9 vs 22.6 s / 42.3 vs 42.1 / **59.2 vs 63.7** / 25.0 vs 25.0. llama.cpp within 5.1% of the after-PR-172 run outside the small size (median 1.0%). | `evals/speed/windows-rtx5060-2026-11-15-after-pr178.html` / `.pdf` (`bench_after_pr178_report.py`), raw in `evals/speed/bench-2026-11-15-windows-after-pr178/` (results.jsonl 48 rows, xlarge.jsonl 12, recheck-e2b-medium.jsonl, recheck-q9b-short.jsonl); no errors; full speed before (5,908 right after the build, then 7,170 / 7,156 / 7,229 read, 87.4 / 87.8 / 87.0 write) and after (7,145 / 7,161, 87.3 / 87.1) |
| 2026-11-15 | Windows, night (branch `windows/product-kernel`): how the tensor-core products read a batch's rounded activations, which llama.cpp lays out a step and token at a time. **First what it could be worth** (a speed-only mock, x read from one always-cached spot): a ~2.3K read +10% on Qwen 3.5 9B, +5% on 4B, +2% and +1% on the Gemmas. **Kept: a batch's scales and sums as heads** (`X8_HEAD`, `matmul8`'s `HEAD`, `Q8::with_heads`, `quantize_q8_tile` and the gated ones): the bytes stay where they were (a token's 128 for a step on one 128-byte line) and, in the sums' buffer, each token and 128 numbers has 8 floats, its four scales and its four 32s' sums already added, so a step reads them in one place and half the sums. The K-quants' kernels read nothing else; Q4_0 and Q8_0 have kernels for both (`_h`), and a buffer is laid out for the product that reads it (`Q8::lay_out_for`, `Input::round`: heads for a K-quant model's batches and for any K-quant matrix, rounded again if a batch was rounded the other way, as for Gemma's one Q6_K matrix). The same numbers to the last bit (the sums added without contraction, `__fadd_rn`), the 64 tokens generated identical on all four models. **Tried and dropped: (1) everything in one block of 160 bytes** a token and 128 numbers, head then bytes, as llama.cpp's 144: the Qwens +2% to +4% but Gemma 4 E4B −6% and E2B −4% (its Q4_0 products 9-10% slower: the bytes off their line); **(2) heads for Q4_0 models too**: Gemma 4 E4B −3% (both its product kernels 3% slower, however the read was written), so Q4_0 models keep the scales and sums apart; **(3)** the rounding kernel finding its block by division against a grid a token (no difference). | `vireo-cli`, `main` -> branch, three alternating runs each, tokens/s: `medium.txt` Qwen 3.5 9B 2,319 -> 2,409 (+3.9%), 4B 4,087 -> 4,142 (+1.3%), Gemma 4 E4B 7,122 -> 7,107 (0%), E2B 18,964 -> 19,285 (+2%, the base's third run low; level); `long.txt` 2,013 -> 2,085 (+3.5%), 3,210 -> 3,244 (+1.1%), 5,369 -> 5,370, 11,759 -> 11,893 (level). Start to finish on a one-word prompt 800 ms either way (40 more kernels to compile). Engine 77, GPU 17 (`short_batch_tiles_match_the_64_token_tile` reads Q4_0 and Q8_0 with heads too), server 100, eval 12 tests pass. Not compiled on a Mac (`lay_out` is a no-op there). No eval or benchmark run |
| 2026-11-15 | Windows, late night (the same branch): the Q6_K files the Mac's work is on, starting with Gemma 4 E2B's (`gemma-4-E2B-it-Q6_K.gguf`, 4.5 GB: 246 Q6_K matrices). **It didn't load**: `per_layer_model_proj` is BF16 there and CUDA has no kernel for it; it now goes up as f32 (`GpuMatrix::upload`, every BF16 number exactly). **Against llama.cpp on it** (each engine's command-line tool, two alternating runs, so indicative): reading ~32K 10,093 vs 10,608 tokens/s (-5%), writing after it 89 vs 99.5 (-11%); ~3K 14,600 vs 8,800-9,400, which llama.cpp's tool spends partly warming up. Nsight Systems on both, the ~32K read's kernels: attention 1,627 vs 1,141 ms (`attention_mma_512` 1,456 to `flash_attn_ext_f16`'s 1,008), the Q6_K products 1,266 vs 988, the rest 350 vs 371 (ours 156 in `matmul_f32`: the file keeps each layer's `inp_gate` and `proj` in f32); all 3,243 vs 2,500. **The Q6_K product kernel with a part taken out** (speed-only, the ~3K read, 14,650 tokens/s): the weights' fragments not read +1%, x's +1.7% (so `ldmatrix` has nothing to give it), the scaling's multiply and multiply-add +15%, the tensor-core products +15%. **Kept: a 32's two sums scaled together.** Q6_K has a whole-number scale every 16 numbers and x one every 32: the two 16s' sums are multiplied by their scales and added as whole numbers (under 2^24, exact in f32) and scaled into the result once with d times x's scale (`matmul8`'s Q6_K branch; `store` keeps a run's two scales as halves of a word beside d), a token tile at a time so few sums are in registers at once (4 at a time +1.2%, 2 +2.6%, 1 +3.8%: the kernel is at 250 registers). Not the same f32 as scaling each 16 (one rounding where there were two). **Tried and dropped:** a row's d applied once a step, the step's sums gathered scaled by x alone (-4% with 4 tiles at a time, -1.5% with 1: 64 more registers). | `vireo-cli`, before -> after, three alternating runs, tokens/s: Gemma 4 E2B Q6_K ~3K 14,673 -> 15,157 (+3.3%), ~32K 10,061 -> 10,259 (+2.0%); the Q4 files, whose ffn_down is half Q6_K: Qwen 3.5 9B 2,400 -> 2,428 (+1%) and 2,084 -> 2,104 (+1%), 4B and Gemma 4 E4B level. Against exact on the Q6_K file (20 chunks of 512), before / after: rms change in log p 0.132 / 0.122, same top token 97.63% / 97.59%. Engine 77, GPU 17, server 100, eval 12 tests pass. Not compiled on a Mac. No eval or benchmark run |
| 2026-11-16 | Windows against llama.cpp on the Q6_K files, a new series (the Mac's work is on Q6_K; the Q4 runs end at "after PR 178"): Gemma 4 E2B and E4B and Qwen 3.5 4B (`bench.py`'s `WIN_MODELS`; `QUANT=q4` for the old four). No Qwen 3.5 9B: at Q6_K it is too large for this GPU. Contexts capped at 65,536 (nobody on 8 GB runs longer): both servers, one run each, reading, writing and first token at small (~40 tokens), medium (~3K), large (~32K) and full (`xlarge.py` at 65,536, the repository dump cut to 190,000 characters: 57,429 tokens for the Gemmas, 53,874 for Qwen 3.5 4B; no row spilled, Vireo 2.9 / 5.8 / 6.3 GB, llama.cpp 2.6 / 5.2 / 5.6). On the branch `windows/product-kernel`. Vireo against llama.cpp, E2B / E4B / Qwen 3.5 4B. **Reading**: medium 15,310 vs 14,045 (+9%) / 5,107 vs 5,359 (-5%) / 3,315 vs 3,538 (-6%); large 10,387 vs 11,178 (-7%) / 4,147 vs 4,380 (-5%) / 2,742 vs 2,842 (-4%); full 7,928 vs 8,794 (-10%) / 3,612 vs 3,846 (-6%) / 2,358 vs 2,362 (0%). **Writing** (as shipped): small 120 vs 117 (+3%) / 77.8 vs 67.6 (+15%) / 70.4 vs 77.9 (-10%); medium 138 vs 115 (+21%) / 86.8 vs 65.6 (+32%) / 68.1 vs 75.7 (-10%); large 103 vs 93.5 (+11%) / 73.0 vs 53.9 (+36%) / 54.4 vs 63.0 (-14%); full 77.3 vs 90.2 (-14%) / 52.2 vs 50.9 (+3%) / 48.6 vs 55.0 (-12%); drafting off 122.5 / 65.1 / 70.6 small and 83.6 / 50.0 / 49.5 full, so E2B's guesses cost it at 57K. **First token**: medium 0.20 vs 0.23 s / 0.62 vs 0.60 / 0.94 vs 0.89; large 3.15 vs 2.98 / 7.88 vs 7.51 / 10.85 vs 10.53; full 7.13 vs 6.52 / 15.6 vs 14.7 / 22.25 vs 22.27. Qwen 3.5 4B's small first token is 0.38 s to 0.11 (re-checked, both runs): it opens with a tool call there, which isn't streamed until whole (the Q4 file does on some runs: 0.30 and 0.07), not a speed. **Past the cap, for the record** (`over-64k.jsonl`): both Gemmas hold 131,072 at Q6_K (Vireo 3.3 and 6.8 GB) and read ~125K at 4,702 vs 5,379 and 2,564 vs 2,695; Qwen 3.5 4B spills at 114,688 on Vireo (0.53 GB shared) and fits 90,112 (7.0 GB). | `evals/speed/windows-rtx5060-2026-11-16-q6-bars.html` and `-dark.png` (`bench_q6_bars.py`: the 12 November page's layout, a panel a workload), raw in `evals/speed/bench-2026-11-16-windows-q6/` (results.jsonl 36 rows, xlarge.jsonl 9, over-64k.jsonl 7, recheck-q4b-short.jsonl); no errors; full speed before (Gemma 4 E4B Q6_K 5,023 / 4,805 read, 65.1 / 65.4 write) and after (4,980 / 5,073, 64.9 / 65.6) |
| 2026-11-16 | Windows (branch `windows/product-kernel`, uncommitted): writing on the Q6_K files, starting from the Mac's Q6_K work. Of that, what carries over to CUDA is the smaller blocks (the Mac's 212 bytes for 224; its reading gains are on the neural accelerators). **Kept: (1) Q6_K rows without padding** (`Q6K_BYTES` 208, `q6_k_row_bytes`, `Gpu::upload_weights`): a row's blocks without their d (ql, qh and the 16 scales, 208 bytes, so every block still starts 16-byte aligned for the tensor-core products' 16-byte loads), then the row's d's, padded to 16: 210 bytes a block and a few a row, 6% less to read a token (the GPU's memory in use fell by only ~35 MB on Qwen 3.5 4B, 4,362 -> 4,326 MiB after a load, where the bytes say ~180: not looked into). **(2) Fewer loads in the one-token and few-token kernels** (`matvec8_q6_k`, `matvec8_few_q6_k`, `matvec_q6_k`): a word in one load (`u32_at4`; it was two halves, as for a 2-byte-aligned block), a lane's four scales from one 8-byte load of its half's 8 (`Q6Scales`), x's four scales in one (`f4`): 5 loads a lane, row and step where there were 11. The same sums in the same order. **Tried and left at 1: 8 or 16 bytes of each run a lane** (`Q6_W` 2 or 4: 4 or 8 blocks a step): level to 2% slower at full power, and the sums are added in another order (Gemma 4 E4B's second token changed). **The laptop's GPU was held to ~45 W through the first half of this** (after a sleep; `nvidia-smi` gave the reason as a software power cap, Gemma 4 E4B read at 1,450 tokens/s for 5,050; a restart cleared it). Held so, writing waits on arithmetic, not memory, and the steps measured differently: the layout alone -3% (+4-6% at full power on Qwen 3.5 4B and Gemma 4 E2B), the loads +19-21% (+3-8%), `Q6_W` 2 another +6-9% (0 to -2%): a change timed under a cap says little about full power. | `vireo-cli generate`, 200 tokens after `prompts-mac/medium.txt`, before -> after, three alternating runs at full power, tokens/s: Qwen 3.5 4B 70.8 -> 76.1 (+7.5%), Gemma 4 E4B 66.8 -> 74.9 (+12%), E2B 125.1 -> 134.0 (+7%). By step (medians; 224 bytes with whole words / the new rows with whole words / all): Qwen 70.5 / 73.4 / 76.5, E4B 68.7 / 68.8 / 73.6, E2B 125.3 / 132.3 / 132.9. Reading level: medium 3,478 -> 3,489, 5,095 -> 5,101, 15,444 -> 15,410; `large.txt` Qwen 3.5 4B 1,990 -> 1,986 (88K tokens), E2B 5,502 -> 5,502 (103K). The same 128 tokens generated on all three, and the same 64 with `--exact` on Qwen 3.5 4B and Gemma 4 E4B; perplexity the same to every digit (20 chunks of 512, all three; `--exact` on E2B and Qwen 3.5 4B with the new rows). Engine 77, GPU 17, server 100, eval 12 tests pass. Not compiled on a Mac (CUDA only). No eval or benchmark run, and not yet against llama.cpp |
| 2026-11-16 | Windows against llama.cpp on the Q6_K files again, after the writing work in the row above (uncommitted on `windows/product-kernel`), the baseline for what comes next: Gemma 4 E2B and E4B and Qwen 3.5 4B, both servers, one run each, reading, writing and first token at small (~40 tokens), medium (~3K), large (~32K) and full (`xlarge.py` at 65,536: 57,428 tokens for the Gemmas, 53,875 for Qwen 3.5 4B; no row spilled, Vireo 2.9 / 5.8 / 6.2 GB, llama.cpp 2.6 / 5.2 / 5.6). Vireo against llama.cpp, E2B / E4B / Qwen 3.5 4B. **Writing** (as shipped): small 122.6 vs 124.1 (-1%) / 74.9 vs 68.8 (+9%) / 78.2 vs 79.0 (-1%); medium 156.0 vs 120.9 (+29%) / 85.0 vs 66.6 (+28%; 90.8 and 94.5 on two re-runs) / 77.1 vs 76.3 (+1%); large 92.9 vs 94.0 (-1%) / 74.2 vs 52.9 (+40%) / 57.1 vs 63.5 (-10%); full 85.0 vs 94.4 (-10%) / 57.1 vs 52.3 (+9%) / 52.9 vs 56.4 (-6%). **Drafting off, the first Q6_K run -> now**: small 122.5 -> 121.7 / 65.1 -> 74.3 (+14%) / 70.6 -> 76.3 (+8%); medium 117.5 -> 130.6 (+11%) / 64.8 -> 74.2 (+15%) / 67.2 -> 76.9 (+14%); large 94.2 -> 103.4 (+10%) / 52.9 -> 56.1 (+6%) / 53.4 -> 58.1 (+9%); full 83.6 -> 89.5 (+7%) / 50.0 -> 51.9 (+4%) / 49.5 -> 52.9 (+7%). So Qwen 3.5 4B, 10% behind at small and medium, is level there and still 6-10% behind after long prompts (each token reads a longer cache), and Gemma 4 E2B's guesses still cost it at 32K and 57K (92.9 and 85.0 as shipped against 103.4 and 89.5 without). **Reading** (not touched): medium 14,581 vs 12,206 (+19%) / 4,196 vs 5,409 in the run, 5,107 and 5,192 on two re-runs (-5%) / 3,389 vs 3,436 (-1%); large 10,288 vs 11,264 (-9%) / 4,173 vs 4,448 (-6%) / 2,770 vs 2,899 (-4%); full 7,984 vs 8,895 (-10%) / 3,596 vs 3,855 (-7%) / 2,386 vs 2,403 (-1%). **First token**: medium 0.23 vs 0.27 s / 0.76 (0.60 re-run) vs 0.59 / 0.91 vs 0.91; large 3.19 vs 2.95 / 7.83 vs 7.37 / 10.75 vs 10.31; full 7.09 vs 6.40 / 15.65 vs 14.63 / 22.01 vs 21.87; Qwen 3.5 4B's small 0.36 s to 0.10 is its opening tool call again, not a speed. | `evals/speed/windows-rtx5060-2026-11-16-q6-writing-bars.html` and `-dark.png` (`bench_q6_writing_bars.py`, a mark where Vireo was in the first Q6_K run), raw in `evals/speed/bench-2026-11-16-windows-q6-writing/` (results.jsonl 36 rows, xlarge.jsonl 9, recheck-e4b-medium.jsonl); no errors, both facts recalled in every full row; full speed before (Gemma 4 E4B Q6_K 5,087 / 5,080 read, 73.5 / 72.2 write) and after (5,085 / 5,074, 72.5 / 73.5), the laptop restarted that morning after its GPU came out of sleep held to 45 W |
| 2026-11-16 | Windows, after that benchmark (the same branch, uncommitted): where a token's time goes on the Q6_K files, kernel by kernel (`nsys profile --trace=cuda --cuda-graph-trace=node`, which shows the launches inside a token's graph), then the small things it found. **Every large Q6_K product already runs at the memory's speed** (310-350 GB/s; the output matrix alone is 1.0-1.5 ms a token), and so does one token's attention deep in a context (Qwen 3.5 4B at 30K: 8 layers of 371 us, 123 MB each); what was slow was small. **Kept: (1) Q8_0 rows as Q4_0's** (`q8_0_row_bytes`, `Gpu::upload_weights`): a row's numbers, 32 bytes a block, then its f16 scales, so the tensor-core tiles read a run of 16 in one load (it was eight halves, `unpack16<K_Q8_0>`, gone) and the one-token kernels whole words. **(2) The one-token and few-token Q8_0 kernels take 16 bytes a lane, 16 blocks a step** (`matvec8_q8_0`, `matvec8_few_q8_0`, `dot16`; 4 bytes and 4 blocks before): a step was 128 bytes of a row, so the reads' latency set the pace. Qwen 3.5 4B's `ssm_out` (11 MB) 49 -> 34 us and its 32-row `ssm_alpha` and `ssm_beta` 15.4 -> 3.9 us, 72 launches a token: 1.9 -> 1.0 ms of a 13 ms token. A block's 32 are summed as whole numbers before scaling, so not the same f32 as four at a time. **(3) Gemma 4's per-layer gate in Q8_0 and projection in f16 for 8-bit mode** (`Layer::per_layer_q8`, `GpuMatrix::upload_f16`): the Q6_K files keep both in f32 (2.6 MB each on E4B), which a token read at the memory's speed (0.70 of E4B's 13.4 ms, 0.40 of E2B's 7.5) and a prompt took the f32 tile kernel for. Against exact (20 chunks of 512), before / after: E2B rms change in log p 0.122 / 0.137, same top token 97.59% / 97.76%; E4B 0.100 / 0.109, 98.10% / 98.10%. **Tried and dropped: the projection in Q8_0 too** (reads 2.5% faster again; E4B 0.100 -> 0.138 and 98.1% -> 97.4%, E2B's perplexity 199.6 -> 202.3: its input, the gated per-layer vector, doesn't round well to 8 bits); **a token's first 8 launches as a graph of their own**, so the GPU starts while the driver queues the other ~800 (level on all three: the 0.45-0.53 ms between a token's last kernel and the next token's first is the logits coming down, 80 us for 1 MB, and the CPU's pick and uploads, not the queuing). **Looked at, not changed:** `delta_net` (19 us a layer, 4 MB of state read and written: 13 us at the memory's speed, ~1% of a token to gain); prompt attention, still the gap on long reads (`attention_mma_512` 24.7 ms for 1,024 tokens at 30,000 positions, 20 TFLOPS; llama.cpp's ~17 by its share of a 32K read). llama.cpp's `fattn-mma-f16.cuh` (read from its source) runs the same block as ours for 512-wide heads, 8 warps, 64 query rows, 32 positions a step, but the rows are 8 tokens of all 8 heads of a key/value head, its queries are in shared memory, and keys and values come half a head's numbers at a time; which of those is worth its 30% is the next thing to measure. | `vireo-cli`, the benchmark's build -> now, three alternating runs, tokens/s. Writing 250 tokens from a short prompt: Qwen 3.5 4B 76.1 -> 84.5 (+11%), Gemma 4 E4B 76.5 -> 78.0 (+2%), E2B 136.7 -> 140.4 (+3%); after ~30K tokens: Qwen 63.9 -> 66.5 (+4%), E2B (35K) 105.3 -> 107.0 (+2%). Reading `prompts-mac/medium.txt`: 3,472 -> 3,584 (+3%), 5,091 -> 5,231 (+3%), 15,344 -> 15,991 (+4%); ~30K: Qwen 2,824 -> 2,868 (+2%), E2B 10,294 -> 10,523 (+2%). Qwen 3.5 4B: the same 128 tokens generated and perplexity the same to every digit; the Gemmas' default-mode tokens differ (the rounded gate), `--exact` the same 64 on all three. Engine 77, GPU 17, server 100, eval 12 tests pass. Not compiled on a Mac (`per_layer_q8` is off there). No eval or benchmark run |
| 2026-11-16 | Windows (branch `windows/attention`, uncommitted): prompt attention, the gap on long reads, after reading llama.cpp's kernel (`fattn-mma-f16.cuh`, its main loop). Its block for 512-wide heads is ours (8 warps, 64 query rows, 32 positions a step); what differs is how two warps share 16 rows. Ours split the head's numbers, each warp working out half of S = Q Kᵀ, the halves added through shared memory every step, and each keeping half the output in f32. Its warps split a step's positions, each working out S whole for its own, with its own largest score and sum, and keep the whole head's output in f16, two numbers a register, merged once at the end. Our 256-wide kernel, with no split, ran 30% more work a second than our 512-wide one, the same as llama.cpp's lead. **Kept: (1) `attention_mma`'s `POS`** for 512-wide heads: the warps of a set of rows take half a chunk's positions each and the whole head, the output's sums in f16 (`mma_f16h`), the weights scaled down by 8 so the sums stay inside f16 however many positions a row sees, and the warps' largest scores, sums and outputs put together at the end as parts are. **(2) The 256-wide kernel's output in f16 too**, half the registers (Qwen's heads, and the Gemmas' sliding layers). The short-batch and part kernels are as they were. **Tried:** 2, 3, 5, 6 and 8 sets of rows a block 512 wide (all slower than 4), 12 and 16 sets 256 wide (slower than 8). | The kernel, 1,024 tokens at 30,000 positions (`prompt_attention_speed_cuda`): 512 wide 24.7 -> 20.0 ms (20.0 -> 24.8 TFLOPS), 256 wide 19.0 -> 15.3 (26.1 -> 32.3), the sliding layers 0.305 -> 0.252; its output's sum the same to four digits. `vireo-cli`, `main` -> branch, two alternating runs, tokens/s, a ~30K-35K read: Gemma 4 E2B 10,479 -> 11,573 (+10%), E4B 4,207 -> 4,438 (+5.5%), Qwen 3.5 4B 2,860 -> 2,954 (+3%); `medium.txt` and writing level. Against exact on a 16,384-token context (`long.txt`, 8,191 tokens scored), before / after: E2B rms change in log p 0.177 / 0.187, same top token 98.77% / 98.84%; E4B 0.124 / 0.109, 99.24% / 99.30%; Qwen 3.5 4B 0.115 / 0.123, 99.11% / 99.01%. On 20 chunks of 512: E2B 0.137 / 0.129, 97.76% / 97.53%; E4B 0.109 / 0.105, 98.10% / 98.12%. Engine 77, GPU 17, server 108, eval 12 tests pass. Not compiled on a Mac (CUDA only). No eval or benchmark run |
| 2026-11-16 | Windows against llama.cpp on the Q6_K files after PRs 182 and 183 (`main` at 02a72c0; branch `windows/bench-after-pr183`, uncommitted): Gemma 4 E2B and E4B and Qwen 3.5 4B, both servers, one run each, reading, writing and first token at small (~40 tokens), medium (~3K), large (~32K) and full (`xlarge.py` at 65,536: 57,428 tokens for the Gemmas, 53,875 for Qwen 3.5 4B; no row spilled, Vireo 2.9 / 5.9 / 6.2 GB, llama.cpp 2.6 / 5.2 / 5.6). Vireo against llama.cpp, E2B / E4B / Qwen 3.5 4B. **Reading**: medium 16,116 vs 14,668 (+10%) / 5,471 vs 5,408 (+1%) / 3,507 vs 3,556 (-1%); large 11,653 vs 11,328 (+3%) / 4,466 vs 4,439 (+1%) / 2,929 vs 2,907 (+1%); full 9,156 vs 8,907 (+3%) / 3,936 vs 3,847 (+2%) / 2,577 vs 2,415 (+7%). The morning's run had large and full 4-10% behind (Qwen's full level). **Writing** (as shipped): small 149.0 vs 124.0 (+20%) / 99.4 vs 68.7 (+45%) / 83.8 vs 77.5 (+8%); medium 186.5 vs 121.9 (+53%) / 115.3 vs 67.4 (+71%) / 81.5 vs 75.2 (+8%); large 108.3 vs 99.1 (+9%) / 86.8 vs 53.7 (+62%) / 60.6 vs 63.7 (-5%, a 43-token reply); full 87.9 vs 93.5 (-6%) / 54.5 vs 51.8 (+5%) / 56.1 vs 56.3 (0%). **Drafting off**: small 135.1 / 74.1 / 83.8; medium 135.9 / 75.3 / 81.5; large 101.6 / 55.4 / 60.0; full 87.9 / 51.9 / 52.2. The Gemmas' guesses gave more than in the morning's run (E4B small 74.9 -> 99.4 and medium 85-94 -> 115.3 as shipped, with drafting off level): not looked into; their draft models' matrices are Q8_0, whose one-token kernels are the faster ones now. **First token**: medium 0.20 vs 0.24 s / 0.58 vs 0.60 / 0.88 vs 0.86; large 2.85 vs 2.93 / 7.32 vs 7.40 / 10.16 vs 10.29; full 6.17 vs 6.37 / 14.33 vs 14.67 / 20.37 vs 21.78; Qwen 3.5 4B's small 0.35 s to 0.11 is its opening tool call, not a speed. | `evals/speed/windows-rtx5060-2026-11-16-q6-after-pr183-bars.html` and `-dark.png` (`bench_q6_after_pr183_bars.py`: the two engines only, no mark for an earlier run), raw in `evals/speed/bench-2026-11-16-windows-q6-after-pr183/` (results.jsonl 36 rows, xlarge.jsonl 9); no errors, both facts recalled in every full row; full speed before (Gemma 4 E4B Q6_K 5,174 / 5,211 read, 68.9 / 70.6 write, a third run 5,224 and 71.1) and after (5,273 / 5,288, 74.6 / 72.3) |
| 2026-11-17 | Windows (branch `windows/held-logits`): where a written token's time goes now, and the time around a round of guesses. **Writing without guesses is at the memory's speed**: `matvec8_q6_k` is 83-89% of a token on all three Q6_K files (`nsys`, 200 tokens), so two ways of reading fewer bytes were measured as throwaway kernels and dropped. **The output matrix from its high bits** (Q6_K keeps a weight's top 2 bits apart from its low 4: 39% of the bytes, then the low bits for the likely tokens only; the matrix is 12-15% of a token): the short pass is off by ~1 nat (0.7-1.6 by model, and 1.1-1.8 times what uniform low bits would give, since they aren't independent of x), so being sure of every token within 8 nats of the top (what a top-20 sampler needs) leaves, at the median, 50-1,300 tokens to finish on E4B but 10,000-135,000 of ~250,000 on E2B and Qwen 3.5 4B (the top token alone: 6-80 on E4B, 70-6,000 on the others). Not built. **Skipping the values of positions attention ignores** (one token deep in a ~30K context spends 19% of its time reading keys and values, 30% at 57K): 93-100% of positions have some query head within e^-16 of its largest weight (Qwen 3.5 4B 92.6%, E4B 99.3%, E2B 100%; e^-12: 50%, 78%, 96%), so nothing can be left out safely. Not built. **Where the time was: the server around a round of guesses.** The timeline of a drafted reply (`nsys` on vireo-server, kernels and copies in order, `timeline.py` and `rounds.py`) had the GPU idle 13% of the writing time on Gemma 4 E4B. After the main model's pass over a round, the engine brought every row of logits down (4 MB for 4 tokens, 0.33 ms), the server copied them into a new buffer and each guess's row into another (1.1-1.8 ms: fresh memory each round), and only the rows of guesses taken were ever looked at. **Kept: (1) a round's logits stay on the GPU** (`forward_batch_held`, `held_logits` on the three models, `LogitsBuffer::read_row`; the server's `read_held` and `held_row`, in the reply's loop and in a tool call's): a row comes down when its guess is taken, straight into the reply's logits (73 us a megabyte). **(2) f16 rows times a few tokens in one pass** (`matvec_few_f16`, `Gpu::matmul` for 2-8 tokens): Gemma 4's per-layer projection, f16 since 16 November, went through the tile kernel for a round of guesses, 18 us a layer where one token's pass is 4.6, on 35 or 42 layers; and its sums were the tile's, not the one-token kernel's, so a guess taken could change a reply (drafted and undrafted replies differed on `main`; the same again now, all six checked, and a new GPU test holds a batch's rows to one token at a time bit for bit on the Q6_K file). **Tried and dropped:** AVX2 for the draft model's centroid head and the sampler's pass over the logits (level: the head's ~115 us a guess reads 4 MB of centroids and rows); the head on the GPU, not built (a guess is ~135 us of ~55 small launches on the GPU and ~115 us on the CPU, and without the CPU's early stop a round would run more of them: ~1%). **Looked at, not changed:** a session's first rounds record a graph per batch length (4-6 ms each, four or five a first reply); a token never used since the PC started can wait 1.5 ms for its embedding rows to come off the disk (the tables are mapped, not read at load); a Qwen round of 7 costs 1.54 passes (`matvec8_few7_q6_k` 82 us against 53: past 4 tokens the few-token kernel waits on arithmetic) and each guessed token 0.46 ms of DeltaNet. Qwen 3.5 4B's files here still have no multi-token-prediction layer, so on new text it has no guesses to gain from. | vireo-server, `main`'s build -> now, three alternating rounds, greedy, 400 tokens (`tools/reply_speed.py`'s prompts through `ab.py`), tokens/s as shipped. Gemma 4 E2B: story 161.7 -> 180.5 (+12%), explain 208.7 -> 225.1 (+8%), list 237.1 -> 276.0 (+16%), code 421.1 -> 506.3 (+20%). E4B: 93.8 -> 97.7 (+4%), 128.3 -> 136.7 (+7%), 143.9 -> 154.6 (+8%), 269.7 -> 300.0 (+11%). Qwen 3.5 4B: 85.2 -> 85.7, 85.0 -> 85.0, 92.1 -> 93.9 (+2%), 159.6 -> 161.8 (+1%). At the models' own sampling, two rounds (replies differ run to run): E2B +11% to +18%; E4B story and explain level, list +10%, code 270 -> 294. With guesses off (`--mtp off --lookup 0`) both builds write the same (E2B 137.5, E4B 75.5) and the same replies. Eval, 43 cases, before / after (`evals/windows-held-logits-before.json` and `windows-held-logits.json`): E2B 35 / 36 passed, wrote at 192.7 / 211.4 tokens/s, a tool call written in 147 / 140 ms; E4B 35 / 36, 120.4 / 130.4, 232 / 225 ms; Qwen 3.5 4B 38 / 38 (the same cases), 93.3 / 94.9, 333 / 330 ms; reading level (12,662, 4,590, 2,980). Engine 77, GPU 18 (one new), grammar 2, real models 6, server 108, eval 12 tests pass. Raw rows and the scripts in `evals/speed/held-logits-2026-11-17/`. Full speed before (Gemma 4 E4B Q6_K 5,311 read, 74.2 write) and after (5,060, 67.4, the GPU at 57 C after the runs). Not compiled on a Mac: the models' new methods use only what both backends have (`read_f32_at`), the kernel is CUDA's. No llama.cpp benchmark run |
| 2026-11-17 | Windows against llama.cpp on the Q6_K files after PR 185 (`main` at 1bdde0a; branch `windows/bench-after-pr185`): Gemma 4 E2B and E4B and Qwen 3.5 4B, both servers, one run each, reading, writing and first token at small (~40 tokens), medium (~3K), large (~32K) and full (`xlarge.py` at 65,536: 57,428 tokens for the Gemmas, 53,876 for Qwen 3.5 4B; no row spilled, Vireo 2.9 / 5.9 / 6.2 GB, llama.cpp 2.6 / 5.2 / 5.6). Vireo against llama.cpp, E2B / E4B / Qwen 3.5 4B. **Writing** (as shipped; the last run's in brackets): small 178.0 vs 121.5 (+46%; 149.0) / 101.8 vs 67.8 (+50%; 99.4) / 84.1 vs 78.6 (+7%; 83.8); medium 210.4 vs 118.8 (+77%; 186.5) / 128.0 vs 65.5 (+95%; 115.3) / 82.1 vs 76.6 (+7%; 81.5); large 127.6 vs 99.4 (+28%; 108.3) / 93.9 vs 53.7 (+75%; 86.8) / 58.2 vs 64.4 (-10%, a 39-token reply; 60.6); full 92.7 vs 92.7 (0%; 87.9) / 55.0 vs 51.4 (+7%; 54.5) / 55.7 vs 55.1 (+1%; 56.1). So the Gemmas' checked guesses give 13-19% more on E2B and 2-11% on E4B up to ~32K, E2B's full row is level where it was 6% behind, and Qwen 3.5 4B, with no draft model, is where it was (still behind after ~32K). **Drafting off**: small 134.5 / 75.3 / 84.0; medium 134.7 / 74.6 / 80.3; large 103.2 / 56.8 / 60.2; full 91.1 / 52.8 / 52.8. **Reading** (not touched): medium 13,918 vs 14,972 (-7%) / 5,192 vs 5,426 (-4%) / 3,548 vs 3,574 (-1%) in the run, and on two re-runs E2B 17,006 and 17,011 vs 14,799 and 14,773 (+15%), E4B 5,283 and 5,283 vs 5,426 and 5,107; large 11,570 vs 11,442 (+1%) / 4,490 vs 4,474 (0%) / 2,949 vs 2,909 (+1%); full 9,201 vs 8,946 (+3%) / 3,925 vs 3,873 (+1%) / 2,590 vs 2,430 (+7%). **First token**: medium 0.25 vs 0.21 s / 0.60 vs 0.57 / 0.87 vs 0.88; large 2.84 vs 2.90 / 7.28 vs 7.34 / 10.09 vs 10.28; full 6.14 vs 6.36 / 14.33 vs 14.57 / 20.26 vs 21.65; Qwen 3.5 4B's small 0.34 s to 0.12 is its opening tool call, not a speed. | `evals/speed/windows-rtx5060-2026-11-17-q6-after-pr185-bars.html` and `-dark.png` (`bench_q6_after_pr185_bars.py`: the two engines only, no mark for an earlier run), raw in `evals/speed/bench-2026-11-17-windows-q6-after-pr185/` (results.jsonl 36 rows, xlarge.jsonl 9, recheck-medium.jsonl); no errors, recall in the full rows as in the last run on both engines (one of the two facts); full speed before (Gemma 4 E4B Q6_K 5,296 / 5,300 read, 73.8 / 72.7 write) and after (5,274 / 5,300, 74.3 / 74.2) |
| 2026-11-17 | Windows, agent tasks against llama.cpp on the Q6_K files (`main` at 01e0d96; branch `windows/tasks-after-pr186`): the Mac harness's four task workloads (`bench-2026-11-13/bench.py`, `WORK=multiturn,tools,skill,longtools`), Gemma 4 E2B and E4B and Qwen 3.5 4B, both servers and Vireo with drafting off, one run each at the models' own sampling. An eight-turn chat (96 tokens a turn), four quarters' sales read and totalled with `calculate`, the weekly-report skill (four notes read, hours summed, a file written), and a ~56K-token incident log with two files to check. Whole-task seconds, Vireo against llama.cpp, E2B / E4B / Qwen 3.5 4B. **Tools**: 4.75 vs 7.96 / 6.92 vs 14.00 / 11.78 vs 16.49, all right on both. **Skill**: 2.96 (wrong) vs 8.47 / 7.61 vs 15.39 (llama.cpp wrong) / 14.35 vs 12.59. **Long log with tools**: 6.36 vs 7.67 / 17.59 vs 17.81 / 31.30 vs 29.62, all right (Qwen 3.5 4B wrote 502 tokens to llama.cpp's 296). **Chat**: 2.40 vs 1.84 / 2.07 vs 3.58 / 4.78 vs 3.90, every turn right on both; Vireo's replies are longer on E2B (623 tokens to 134 over the eight turns) and Qwen 3.5 4B (261 to 152), so the times are not like for like. **Speeds in the tasks** (tokens/s, Vireo / drafting off / llama.cpp): writing in the tool task 223.5 / 124.8 / 120.1, 138.3 / 69.4 / 67.1, 81.1 / 80.8 / 75.3; in the skill task 215.5 / 133.0 / 118.9, 152.6 / 70.0 / 66.6, 79.4 / 79.7 / 75.8; reading the log 9,119 vs 8,779, 3,940 vs 3,748, 2,548 vs 2,302; a tool call written in ~190 ms on E2B (a `calculate` of nine products ~390). Right answers: Vireo 11 of 12, drafting off 10, llama.cpp 11. **Found: Gemma 4 E2B and the skill.** Three more runs of its skill and tool tasks on both servers (`recheck-e2b.jsonl`): on Vireo the skill task was wrong all four times, and each time the reply said the report was saved though `write_file` was never called (in the run's log it sent `calculate` the same malformed `sum([..], [..])` three times, then answered); on llama.cpp, where the skill's text is put in the prompt, it wrote the file all four times and had the hours right once. The tool task: Vireo right 3 of 4, llama.cpp 2 of 4. E4B and Qwen 3.5 4B wrote the file on Vireo. Not looked into further: whether the claim check should catch a save that wasn't made, and why E2B stops short of `write_file` after the skill's instructions arrive by `use_skill`. **Found: Qwen 3.5 4B makes no guesses at a 65,536 context.** The Gemmas' rows above had their draft models (79-91% of guesses taken in the tasks), but Qwen 3.5 4B's log has no guesses in any reply, the chat's included: its replies as shipped and with drafting off are the same speed. At a 32,768 context (`qwen-32k.jsonl`, the chat, tool and skill tasks; the log doesn't fit) it guesses from the chat, 51-54% taken in the tool and skill tasks: writing 116.1 and 110.9 tokens/s against 81.0 and 78.7 with drafting off and 76.5 and 76.7 on llama.cpp, the tool task in 8.94 s against 11.98 and 16.14, the skill task in 9.70 s against 13.39 and 12.64 (both Vireo rows wrong there, llama.cpp right; one run), the chat in 4.03 s against 4.48 and 3.85. The cause was not traced: the guesses need the DeltaNet steps kept (`steps_bytes`, 0.3 GB) with `STEPS_HEADROOM` (384 MB) free besides, and at 65,536 the session's cache is 2.1 GB of an 8 GB GPU. | Raw in `evals/speed/bench-2026-11-17-windows-q6-tasks/` (results.jsonl 45 rows, recheck-e2b.jsonl, qwen-32k.jsonl, the servers' logs beside them, not in git); no errors; full speed before (Gemma 4 E4B Q6_K 5,310 / 5,284 read, 74.1 / 74.3 write) and after (5,335 / 5,312, 73.0 / 74.2). One run each: a task's answer changes from run to run on these models, as the re-runs show. No page built |
| 2026-11-17 | Mac (branch `prototype/mac-gains-nov17`, uncommitted): a few tokens checked at once, and what is guessed. **A regression found and fixed: the 8- and 16-token tiles since PR #174.** Its two tiles of decoded weights (`sab`) are, at 256 columns a step, the whole 32 KB a threadgroup may have, and those tiles ran slower than the 32-token one; `matmul_tc` takes `BUFS` now, one tile for the 256-column tiles and two for the rest. **The 32- and 64-token tiles take 256 columns and one tile too.** **Qwen 3.5 4B and 9B guess from the chat in a reply's text on the Mac** (they did only inside tool calls: a check of a few tokens cost two tokens' time when that was measured). **A running copy is guessed further** (`draft::Lookup::guess_on`: when a round of 7 was all taken, 15, then 31; Gemma 4 and Llama, which go back any number of tokens; Qwen keeps 8 steps). **A model with a draft model or layer takes the chat's runs too on the Mac**, where 8 of the chat's last tokens appeared before (`CHAT_RUN_OVER_LAYER`; runs of 4 cost a document rewritten as a list 10%). `tests/check_speed.rs` takes `VIREO_CHECK_COUNTS`; `tools/long_run.py` takes `VIREO_SERVER_BIN`; `tests/gpu.rs`'s one-token-against-batch bit check is CUDA's (it failed on the Mac on `main`: the tiles' sums are near the one-token kernel's, not its bits). **Tried and dropped:** 128 columns and one tile for the 32- and 64-token tiles (6% slower than two), the chat's runs of 3 guesses over the draft (above) | M5 Max. A batch of 4 / 8 / 16 tokens 4,000 tokens into a chat, `main` → branch: Gemma 4 31B 102.8 / 107.1 / 113.5 → 67.5 / 70.1 / 75.0 ms (one token 51), Qwen 3.5 9B 24.9 / 27.2 / 29.4 → 18.6 / 20.4 / 22.2 (13.7), Gemma 4 E4B 15.0 / 16.0 / 17.4 → 12.5 / 13.5 / 14.5 (9.8); 32 / 64 tokens 31B 102 / 141 → 98 / 137, Qwen 3.5 9B 26.8 / 35.0 → 25.6 / 33.1. 400-token replies through the server, greedy, tokens/s, a document as a list / an edited code file returned / a new story / an explanation (`tools/reply_speed.py`, one run each; `evals/speed/mac-gains-2026-11-17/`): Gemma 4 31B with Google's draft model 30.1 / 35.5 / 20.8 / 28.5 → 40.9 / 165 / 28.5 / 39.1; Qwen 3.8 27B 39.3 / 44.7 / 24.8 / 33.2 → 50.2 / 88.5 / 30.7 / 40.9; Gemma 4 26B-A4B 91.1 / 207 / 88.7 / 87.7 → 92.4 / 282 / 91.5 / 88.1; Qwen 3.5 9B 73.8 / 73.7 / 73.3 / 71.9 → 77.9 / 256 / 69.1 / 68.5 (story and explanation level over two more alternating runs: 68-73 both); Qwen 3.5 4B 123 / 127 / 122 / 122 → 130 / 287 / 114 / 108 (again: 107-120 against 123); Gemma 4 E4B 109 / 282 / 100 / 88.7 → 116 / 690 / 99.8 / 88.7; E2B 226 / 513 / 175 / 170 → 246 / 1,053 / 170 / 159 (again: 177 / 174 both). The small tiles against the 128-token one on 31B's and Qwen 3.5 9B's weights: within 1e-5 of the largest sum. Engine 76, GPU 18, server 124 tests pass. No eval, gauntlet or Vireo Bench run |
| 2026-11-17 | Mac and shared (branch `prototype/prompt-cache-new-chats`, uncommitted): a new chat's system turn from the cache, and a test for short batches. **Saved starts** (`Kept::heads`, `keep_head`; up to 3, 4 GB, the least recently used dropped): the state at a system turn's end kept in system memory for the last few system turns, not one. Gemma with rings can only go back a batch's length from the furthest it has read, and kept one start, the last turn read: after a long chat, a new chat whose turn differed (the tools that change files declared or not) read its system turn again from token 0, then replaced the saved one, so chats that alternate never met theirs. A start is also saved where a turn parts from the one in the session, well inside it. Qwen kept checkpoints only for the turn in the session and went back to where two turns part (token 614 of 1,870: the tool that changes files is declared mid-list); it now saves the turn's end as well (`export` of its checkpoint) and loads it. **`vireo-cli batches`** times batches of 1 to 64 tokens 4,000 tokens into a chat, and Vireo Bench's S1 runs it (`b1` to `b64`): nothing timed a short batch, which is how PR #174's slower 8- and 16-token tiles went unseen. `evals/speed/prompt-cache-2026-11-17/probe.py`: 15 new chats in a row (plain, long, asking for a file change, a memory saved). **Not fixed:** the first chat after a memory is saved reads Gemma's whole system turn again (its memories come before the tools: after them would keep the tools, a change to the prompt, not made); the first chat with the other tool list reads from 0 on Qwen (no shared checkpoint yet). Seen on `main` too: Qwen 3.8 27B sometimes writes "[Tool call: write_file]" as text and calls nothing | M5 Max, the probe, `main` → branch, a new chat's prompt tokens kept / read / seconds to read. After a long chat or one with the other tool list, a plain chat: Gemma 4 31B 0 / 1,645 / 2.6 s → 1,598 / 47 / 0.27 s; one asking for a change 1,414 / 393 / 0.84 → 1,706 / 99 / 0.25; Qwen 3.8 27B 614 / 1,253 / 1.9-2.0 → 1,820 / 47 / 0.21-0.26 and 614 / 1,418 / 2.1-2.2 → 1,937 / 95 / 0.28; Qwen 3.5 9B 614 / 1,253 / 0.45 → 1,820 / 47 / 0.06; Gemma 4 E4B 0 / 1,641 / 0.21 → 1,598 / 43 / 0.04. Of all 15 chats' prompt tokens from the cache (the two long chats' pasted text is 12,000 of them): Gemma 39-42% → 52-54%, Qwen 38% → 60%. Replies the same in all but one row a model (the 27B's text-for-a-call above; E4B one word's quoting). A saved start: 31B 1.0 GB (three: 2.7), Qwen 27B 0.28, E4B 0.05. `vireo-cli batches`, E4B: 1 / 4 / 8 / 16 / 32 / 64 tokens 9.9 / 12.4 / 13.4 / 14.5 / 15.7 / 18.3 ms. Server 124 tests pass. No eval, gauntlet or Vireo Bench baseline run |
| 2026-11-17 | Mac and shared (branch `prototype/qwen-call-text-and-memory-order`, uncommitted): a call Qwen 3.8 27B wrote as text, Gemma's memories after its tools, and one more draft guess a round. **Qwen's tool-call line names thinking only when the reply thinks** (`tools::family_notes(family, thinking)`): with thinking off, "one <tool_call> block per call, after your thinking ends" named something that wasn't there, and Qwen 3.8 27B, asked to create a small file, began its reply with "[" (0.50-0.81) more often than with <tool_call> (0.15-0.44), wrote "[Tool call: write_file]" or "I'll create that file for you." and stopped. `VIREO_DEBUG_CALLS` prints a reply's three likeliest first tokens. Found by taking lines out one at a time: without the approval paragraph 1 of 12 still failed; without the tool-call line, or with "Keep the final reply short" as "Be concise.", none; without "Do not put tool calls ... in your answer's text" 6 (it began with "I"). **Gemma's memories go after its tools in the system turn** (`system_turn`): its template puts the tools after the instructions, so a memory saved changed the turn before the ~1,100 tokens of tools and the next new chat read them all again (the one case PR #189 left). **Gemma 4's draft model guesses 4 a round on the Mac** (`GEMMA_MTP_GUESSES`; Qwen's layer keeps 3). `evals/speed/qwen-call-text-2026-11-17/calls.py`: twelve requests to create a file. **Tried and dropped:** 5 and 7 guesses a round (a new story 6% and 13% slower on the 31B), two more guesses after a round all taken (less than 4 every round on the 31B, a story 3-6% slower on both) | M5 Max. Twelve new chats asking for a small file, without a write_file call, before → after: Qwen 3.8 27B thinking off 8 → 0 (<tool_call> at 0.81-0.86), thinking on 0; Qwen 3.5 4B and 9B, Gemma 4 E2B, E4B, 26B-A4B and 31B, Muse: 0 of 12 each. The probe's new chat after a memory is saved, tokens kept / read / seconds: Gemma 4 31B 0 / 1,686 / 2.9 → 1,604 / 82 / 0.16; E4B 0 / 1,679 / 0.22 → 1,604 / 75 / 0.02; both facts recalled in a later chat by E2B, E4B, 26B-A4B, 31B and Qwen 3.8 27B; from the cache over the probe 54% → 63%. Gemma 4 31B, 400-token replies, 3 → 4 guesses a round in the same sitting: a document as a list 41.8 → 48.8 tokens/s, a code file returned 170 → 176, an explanation 40.7 → 42.1, a new story 29.4 → 29.3. Engine 76, GPU 18, server 124 tests pass. vireo-eval, all 43 cases, thinking off, `main` at 6bc5d1d → branch (`evals/prompt-fixes-2026-11-17/`): Gemma 4 E2B 35 → 35, E4B 36 → 36, 26B-A4B 40 → 40, 31B 39 → 39, Qwen 3.5 4B 38 → 38, 9B 40 → 41, Qwen 3.8 27B 41 → 42, Muse 43 → 43; no case from pass to fail, `office-policy` from fail to pass on the 9B and the 27B. The eval's memory stays empty, so it doesn't read memories after the tools. No gauntlet run |
| 2026-11-17 | Mac (branch `prototype/draft-guesses-at-depth`, uncommitted): the fourth draft guess at depth. PR #190 gave Gemma 4's draft model 4 guesses a round on the Mac, measured on short prompts only. Deep in a chat it loses: a guess not taken costs its share of a check, a check costs more the more there is to attend to, and fewer of four are taken than of three (40-46% against 52-62%). **4 a round only while the chat is under 48,000 tokens** (`GEMMA_MTP_GUESSES_UNTIL`), 3 from there. `VIREO_MTP_GUESSES` sets one count at every depth, to measure | M5 Max, Gemma 4 31B with Google's draft model writing ~500 tokens after a prompt that fills the context (`long_bench.py`, two runs each, alternating, 45 s between), 3 / 4 guesses a round: 32K 27.8 and 28.6 / 30.4 and 28.2 tokens/s, 64K 29.2 and 27.9 / 28.9 and 26.6, 128K 20.8 and 20.4 / 19.6 and 19.6. As built: 64K 27.6 (582 guesses, 3 a round), 32K 30.6 (724 guesses, 4 a round). `evals/speed/draft-guesses-at-depth-2026-11-17/`. Server 124 tests pass. One kind of reply (a summary with two planted facts); 256K not run |

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

**Fourteenth round: the Mac's passes on CUDA (2026-11-12, afternoon):**

The Mac's fifth to twelfth passes were merged with "not done: the CUDA side of any of
it". Measured first on this PC (`tests/check_speed.rs`, now for both backends; the
Performance power scheme throughout), a batch of n tokens after a chat of 2,000, ms:

| `main` | 1 written | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| Gemma 4 E2B | 5.9 | 12.2 | 12.2 | 12.4 | 12.6 | 13.6 |
| Gemma 4 E4B | 10.7 | 22.7 | 21.9 | 22.3 | 22.5 | 24.5 |
| Qwen 3.5 4B | 10.8 | 25.6 | 26.1 | 26.4 | 26.8 | 33.1 |
| Qwen 3.5 9B | 17.8 | 44.5 | 47.0 | 47.5 | 46.6 | 54.9 |

Two tokens cost what 64 did, two to two and a half times one: the tensor-core tile is
unpacked through shared memory whatever it holds, and a batch's ~700 kernels were
launched one by one (2-5 ms of the CPU's time that the GPU waits out), where one token
replays a graph. So the chat's guesses (`--lookup`, on by default since the Mac's sixth
pass) cost more here than they saved: nine eval cases on Gemma 4 E4B, 22.3 s of model
time with them and 18.0 without, checking guesses 45% of it (28.6 ms a check against
13.1 for a token). And the three picker sessions of the fifth pass, each with a prompt's
buffers, left Qwen 3.5 9B's 8 GB with none free. Built:

- **`matvec8_few<T>_*`** (T = 2 to 8; Q4_0, Q8_0, Q4_K, Q5_K, Q6_K): the dp4a
  one-token kernels with T rows of x, each weight word multiplied by every token's while
  it's in registers. The same sums in the same order: the same bits as one token at a
  time (a test holds them to it). With several tokens' numbers in registers an SM holds
  fewer warps, so the K-quants' kernels read the next step's weights before this step's
  arithmetic (Q4_K, 9,216 × 2,560, 3 / 4 tokens: 65 → 47, 76 → 60 µs; two steps ahead no
  better). Per product, µs at 1 / 2 / 4 / 8 tokens against the tile's 9: Gemma's Q4_0
  gate 47 / 43 / 51 / 74 / 120, Qwen's Q4_K gate 40 / 42 / 60 / 110 / 100, Q6_K down
  78 / 77 / 77 / 135 / 220. Q4_0 stays near the memory's speed to 8 tokens; the K-quants'
  arithmetic (a scale and a minimum for each 32) costs ~10 µs a token past 4.
- **One token's attention for up to 8 tokens** (`attention_part_kv`, f16 cache): a grid
  layer a token, each at its own length, split into the parts one token is, so each gets
  exactly the numbers it would alone. The convolution and the DeltaNet step through
  their one-token kernels for up to 8 tokens (one launch each, the state in registers
  throughout), not the chunked prompt kernels.
- **A graph for every batch of 1 to 64 tokens**, by its length and the logits wanted,
  recorded the first time (4-6 ms, about what a replay saves; copies inside the GPU
  record too). Gemma 4 E4B, one token read as a batch: 16.0 → 10.8 ms.
- **Tiles of 16 and 32 tokens** for the tensor-core products (`matmul8_t16_*`,
  `_t32_*`: 8 warps of 16 rows, multiplying only the tokens there are): the 64-token
  tile's numbers to the bit, 25-38% less a product at 9-16 tokens (Q4_0 gate 120 → 74 µs,
  Q4_K 106 → 72). Two blocks an SM, which their fewer registers allow, was slower for
  Q4_0 (74 → 86) and level for the rest.
- **Qwen goes back on CUDA**: `delta_net` writes the state before a batch and after each
  token but the last (`steps`), as the Mac's `delta_net_8` does, so `keep_steps` and
  `rewind` work for states up to 128 wide (every Qwen 3.5): 0.3 GB for the 4B, 0.4 GB for
  the 9B. The Mac's rewind test runs here now (to the bit, exact and 8-bit).

The same batches now (ms; and 27,000 tokens deep, `main` → now):

| now | 1 written | 2 | 4 | 8 | 16 | 64 | at 27K: 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | 5.5 | 5.7 | 6.5 | 7.1 | 8.2 | 10.0 | 22.2 → 7.6 | 22.3 → 9.2 | 21.2 → 11.9 |
| Gemma 4 E4B | 10.8 | 10.9 | 13.1 | 14.4 | 15.5 | 21.4 | 35.6 → 15.1 | 34.8 → 18.4 | 33.5 → 25.6 |
| Qwen 3.5 4B | 10.7 | 11.7 | 14.7 | 25.4 | 22.4 | 29.6 | 43.2 → 17.3 | 43.3 → 25.5 | 43.4 → 44.1 |
| Qwen 3.5 9B | 17.8 | 20.1 | 24.1 | 41.7 | 36.7 | 54.6 | 62.4 → 25.5 | 62.9 → 35.1 | 63.1 → 58.9 |

In 8-bit mode a batch of up to 8 tokens now gives the logits of the same tokens read one
at a time to the bit (`reading_in_batches_matches_one_token_at_a_time`: worst difference
0.0 on Llama, Qwen 3.5 and Gemma 4, where the tile's rounding gave 0.4 to 6.7), so a
round of guesses checked in one pass picks what one token at a time would: nine eval
cases with the guesses on and off, every reply the same on Gemma 4 E4B and Qwen 3.5 4B
(7 of 9 on `main`). Longer batches still round as the tiles do.

In the server:

- **7 guesses a round from the chat on CUDA too**, after a match of 8 tokens (3 after 4
  while a check cost two tokens' time).
- **Qwen guesses from the chat on CUDA on every size**, where the GPU has its steps'
  memory free and 384 MB more when a reply starts (the Mac keeps it to large models: its
  4B's check costs 1.5 times a token, here a check of 4 costs 1.4). Qwen 3.5 9B on this
  PC doesn't: its 8 GB are full.
- **Picker sessions read in batches of 256 on CUDA** (1,024 on the Mac): a session's
  buffers grow with its batch, 0.4 GB each for Qwen 3.5 9B at 1,024. With three of them
  and the chat's the GPU reported no memory free, Windows kept part of it in the PC's
  memory, and a token took 27.3 ms over nine eval cases, not 19.5.

Against `main`, back to back, each at its own defaults. Three 400-token replies through
the server (tokens/s; guesses taken):

| | A document rewritten as a list | A code file returned with an edit | A new story |
|---|---|---|---|
| Gemma 4 E2B | 128 → 288 (77%) | 154 → 479 (96%) | 130 → 158 (10%) |
| Gemma 4 E4B | 74 → 156 (72%) | 96 → 305 (96%) | 72 → 86 (9%) |
| Qwen 3.5 4B | 86 → 199 (85%) | (it answers in 73 tokens) | 87 → 84 (13%) |
| Qwen 3.5 9B | 51 → 51 | 52 → 51 | 52 → 52 (no guesses: no memory for the steps) |

(With its steps forced on, the 9B wrote the list at 91 tokens/s and the code file at
157.) Gemma's story on `main` was slower than with no guesses at all (E4B 85): each of
its failed checks cost two tokens. Qwen's story is 1-3% slower with guesses than
without: a match of 2 tokens is right 13% of the time.

Nine eval cases (datetime, read, list, no-tool-writing, find-then-read, git,
three-steps, skill-summarize, skill-review), model time, the same passes: Gemma 4 E2B
9.2 → 6.2 s, E4B 22.0 → 15.8, Qwen 3.5 4B 19.9 → 17.6, 9B 43.9 → 32.3 (its tokens 27.3 →
19.5 ms: the pickers' memory). The full eval, 43 cases: 33, 39, 43 and 39 on both, no
case changed either way, 70 → 44, 98 → 73, 107 → 101 and 179 → 172 s for all 43
(`evals/windows-short-batches.json`; `main`'s run beside it for the times). Where an
eval case's time goes now (E4B): writing 52%, checking guesses 25%, reads of 65-256
tokens 10%, of 9-64 8%.

- **Tried and dropped:** the K-quants' weights read two steps ahead (no better than
  one); the small tiles at two blocks an SM; a session's buffers carved from one
  allocation (with a model loaded, 50 allocations of 2 MB took 431 MB of the GPU's
  memory and one of 400 MB took 400, but a session cost the same either way: the driver
  takes memory ~320 MB at a time, and gives it back).
- **Seen, not fixed:** a check of 7-8 tokens costs Qwen more than a read of 12 (4B: 25.4
  against 22.6 ms), since the tile's products are cheaper than the few-token kernels' at
  that many, but its numbers aren't one token's. Deep in a chat every token of a check
  walks the context itself (Qwen 3.5 4B at 27K: 8 tokens 44 ms, one 16): one pass for
  them all, as the Mac's `attention_tcg` does, is next. And the automatic context doesn't
  count the sessions' own buffers: Qwen 3.5 9B is given 21,504 tokens (0.7 GB) of 1.9 GB
  free, then takes ~1.3 GB more for the chat's and the pickers' buffers and its
  checkpoints, so the GPU still runs full (at full speed since the pickers shrank).
  Counting them would leave it ~7K tokens, or ~9K and the chat's guesses with fewer
  pickers: a choice, not made here.

**Fifteenth round: Google's draft models for the small Gemmas (2026-11-15):**

The Mac's draft model for Gemma 4 31B (13 November) was "not yet" on CUDA, and not for
E2B and E4B, whose draft models (`gemma-4-E2B-it-assistant`, `-E4B-`; 158 MB each, four
layers 256 wide) have a centroid head. Both run now, on this PC (RTX 5060 Laptop, mains,
Performance scheme, but the GPU held to ~33 W all day: one token of E2B took 10.6 ms
against 5.5 on 12 November, so compare within a row, not with earlier rounds).

- **The centroid head** (`Gemma4AssistantMaskedEmbedder`): 2,048 clusters of 128 tokens;
  the draft's final state scores the centroids and only the 32 best clusters' tokens
  get logits, 4,096 rows of 256 instead of 262,144. On the CPU, from the state read back
  (1 KB, not 1 MB of logits), the tied embedding kept in cluster order in the PC's
  memory (134 MB, bf16): 0.11 ms a guess.
- **Its heads are its own**: 4 query heads over the model's 1 or 2 key-value heads (the
  loader asked for the model's 8). And after a check (every token's logits) every row of
  a shared-layer model's states is final, so the next round drafts from the last one
  kept; the Mac's code gave such models none.
- **A guess's layers replay a graph** (CUDA): queuing its ~40 kernels took 0.65 of a
  guess's 0.92 ms; now 0.6 ms a guess, the GPU's work 0.1-0.3.
- **Guesses only where the draft is sure.** A guess checked costs these models a third
  of a token, not the tenth `check_speed` shows: a check wants every token's logits, so
  the 20 layers that share keys and values and the 262,144-row output run for each
  guess (E2B: one token 10.6 ms, 3.4 more a guess; E4B 19.8 and 5.3; for the last
  token's logits only, 0.8 and 2.4). The draft knows when it's guessing: of first
  guesses it gave 0.9 or more, 90% were the model's greedy pick (87% of the model's own
  probability, what sampling at temperature 1 takes); under 0.7, about one in five. So a
  round's guesses stop before the first under 0.7 (`mtp_draft_sure`, `MTP_FLOOR`; CUDA
  only, the Mac's 31B drafts as before). Writing greedy in the engine, 3 guesses a
  round, any / 0.5 / 0.7 / 0.9: E2B 1.01 / 1.14 / 1.25 / 1.21 times one token at a
  time, E4B 1.28 / 1.42 / 1.41 / 1.36, the same text every time; 5 or 7 guesses a round
  were no better.
- **The chat's guesses too**: with a draft model the server still takes the chat's own
  where it offers 3 or more (a run of 4 matched), and leaves shorter ones to the draft.

Through the server, 400-token replies, the chat's guesses alone (`--mtp off`, as
before) → with the draft model, tokens/s (`tools/reply_speed.py`, two rounds, the ways
taking turns; `evals/speed/windows-2026-11-15/`):

| | A document as a list | A code file with an edit | A new story | An explanation |
|---|---|---|---|---|
| Gemma 4 E2B, greedy | 85 → 105 | 170 → 179 | 79 → 86 | 78 → 97 |
| Gemma 4 E2B, its own sampling | 88 → 109 | 168 → 178 | 76 → 87 | 77 → 97 |
| Gemma 4 E4B, greedy | 46 → 66 | 104 → 109 | 42 → 53 | 46 → 62 |
| Gemma 4 E4B, its own sampling | 43 → 64 | 96 → 106 | 43 → 51 | 45 → 61 |

The full eval (43 cases, greedy), the draft model off → on, back to back
(`evals/windows-small-gemma-draft-off.json`, `-draft.json`): Gemma 4 E2B 34 and 34, E4B
39 and 39, every answer the same to the letter on both (43 of 43); writing 76.5 → 85.0
and 46.1 → 53.3 tokens/s (98 → 90 and 162 → 144 s for all 43), guesses taken 1,760 of
3,760 → 2,784 of 4,813 and 1,656 of 3,284 → 2,718 of 4,294. No case changed against 13
November's run on the Gemmas or Qwen 3.5 9B (41); Qwen 3.5 4B has 38 for 39
(`office-invoices`, in a full run only), and has it on `main` without this round too,
so since the Mac's merge of 13 November or the date, not from here. The same run's
speeds, as the eval now saves them:

| | Read (tokens, /s; kept) | Wrote (tokens, /s) | A tool call written / run |
|---|---|---|---|
| Gemma 4 E2B | 13,998 at 3,544 (87,765 kept) | 6,749 at 85.0 | 322 ms / 0.9 ms |
| Gemma 4 E4B | 16,397 at 1,310 (87,765) | 6,233 at 53.3 | 584 ms / 0.9 ms |
| Qwen 3.5 4B | 49,700 at 782 (72,034) | 8,212 at 36.2 | 824 ms / 1.0 ms |
| Qwen 3.5 9B | 49,630 at 451 (72,034) | 7,916 at 22.4 (no guesses: no memory for its steps) | 1,452 ms / 1.1 ms |

- **The eval saves its speeds** (both machines): each case's prompt tokens, those kept
  and read, the reading and writing speeds, each tool call's time to write and to run
  (the server's `Tool:` line now has both), and guesses made and taken, with a total for
  each model (`speeds`), read from the server's `Reply:` and `Tool:` lines.
- **Next:** a check that runs the shared layers and the logits only for guesses still
  in the running (two passes; a wrong first guess now pays for the ones after it); the
  draft's embedding and head on the GPU, so a round's guesses are one launch; the draft
  models downloaded by the app (they're found beside the model: `models/` here). Qwen
  3.5 has no draft model of its own; its gains here are the Mac's shared ones (the
  system turn's state kept for a second chat).

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
  state: 12.8 of 26 ms, at ~270 GB/s), then deep in a chat its attention. Past one token
  at a time it's guessing (fourteenth round): a check of 4 tokens costs 1.2-1.4 times
  one in a short chat. Next there: one pass over the cache for a check's tokens deep in
  a chat, and the K-quants' few-token arithmetic for 5-8 tokens.
- **Reading a prompt**, at full clocks, is at 73–87% of llama.cpp at ~2K tokens and
  78–87% at ~31K (the 9B furthest behind). For long ones
  prompt attention is 30–35% of the time there (`attention_mma_256` and `_512`, ~3–4
  TFLOPS), one block of 4 warps an SM (~70 KB of shared memory each), with "wait" the
  top stall. More warps an SM needs a different tiling (the queries out of shared
  memory, or smaller key/value chunks with more rows); the cheap changes measured flat.
- **The chat cache, for later passes** (alongside this tuning):
  - Done 2026-11-12 (see Progress): the change tools kept declared in a chat, long
    chats cut in steps, remembered facts told without changing a chat's start.
  - Tool calls and results from earlier turns aren't in the next turn's prompt (it's
    built from the messages' text), so a follow-up calls the tool again and the cache
    after the last request goes unused.
  - The 4 GB budget against the large models' chats (Gemma 4 31B at 32K is 4.4 GB on
    the GPU): measure a saved file's size and its write and load times first.
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
- **Short reads** (9-64 tokens: tool results, new turns, the server's own questions)
  cost 1.5-2 times one token (fourteenth round; 2-2.5 before): the tile's unpacking
  through shared memory is what's left, at one block an SM.
- **Memory on 8 GB**: the automatic context should count the sessions' own buffers (see
  the fourteenth round's last note).
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
