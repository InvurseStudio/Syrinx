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

**What limits speed now, and what's next:**

- **Reading a prompt** is at ~60–65% of llama.cpp on all three. What's left is the
  tensor-core kernels' instruction count (the per-32 scaling is half of it and is the
  same in llama.cpp, so it's the rest: unpacking, addresses, zeroing; and their
  occupancy), then cp.async double buffering (measured +6% on Q4_0). llama.cpp also runs
  the last layer's feed-forward only for output tokens (a few % on Llama and Qwen).
  For Qwen, the DeltaNet scan would need the chunked (WY) form.
- **Generating** is 19–25% ahead of llama.cpp on all three.
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
