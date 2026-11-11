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
Generating deep in a long chat falls behind, though: short chats are 19-25% ahead of
llama.cpp, a 25K-token one 27-48% behind. That's one token's attention
(`attention_part_*`), which reads the whole cache per token and needs tuning for long
contexts before the automatic limit goes past 32K.

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
- **Generating** is 19–25% ahead of llama.cpp on all three in short chats, but 27–48%
  behind 25K tokens deep (see "Long chats"): one token's attention (`attention_part_*`)
  reads the whole cache per token and needs tuning for long contexts, before the
  automatic context goes past 32K.
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
   Done 2026-11-11, read-only tools (see Progress); the tools that change things, with
   approvals, follow.
3. ~~The eval, and a baseline.~~ Done 2026-11-11 (`vireo-eval`, see Progress). Gemma 4 E2B
   passing all twelve means the cases are too easy for it: harder ones from the test
   gauntlet (several steps, long results, near-miss tool names) come with the next
   features, so there's room to measure them on capable models too.
4. Copying constraints, scored choices, compact tool text. First round done 2026-11-11
   (see Progress): scored names and fewer tools are switches, measured; what's left for
   the tiniest models is knowing what each tool is for, which neither fixes. Next to
   try, each against the eval: worked examples of calls in the system prompt (cached, so
   free after the first turn), the tools' text and names as the model sees them (the
   server can show its own names and map them back to the app's), and more test models
   between 1B and 4B (Llama 3.2 3B, Qwen 3.5 2B/4B, Gemma 4 E4B) to see where each
   feature stops mattering.
5. Compiled skills (above), then the per-model guidance settings from the eval.

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
