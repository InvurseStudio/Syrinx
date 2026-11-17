# Vireo Bench v1: a standing benchmark for local engines

A fixed set of tests we run the same way every time, so a result from today can sit next to
one from next month. It compares three engines on the same model weights:

- **Vireo**: `vireo-server` (and `vireo-cli` for the raw tests), the prototype's own engine
- **llama.cpp**: a pinned build (Mac `reference/llama-b12228`, Windows `reference/llama.cpp-b12262`, CUDA)
- **mlx-vlm**: its own server, pinned at 0.7.12 in `~/.venvs/mlxbench` (Mac only)

Before this, each benchmark was written for one question (bench-2026-11-13, the
long-context run, long_run.py, the Windows task bench). They overlapped, used different
prompts, and their numbers couldn't be put side by side. This suite folds them into one
and adds what none of them measured.

Public benchmarks for local engines mostly stop at `llama-bench`'s two numbers (pp512 and
tg128), or single scores built on them. Those say how fast an engine reads and writes a
bare prompt. They say nothing about what a person or an agent waits for: the first word, a
long chat, tool calls, memory, staying right at 128K, or the fiftieth reply. Most of this
suite measures those.

## Running it

Everything is in `evals/bench/`, standard-library Python, no installs. On the Mac:

```bash
cd VireoPrototype && cargo build --release -p vireo-server -p vireo-cli
cd evals/bench
python3 -I run.py --tier quick                 # after a tuning change, ~15 min
python3 -I run.py --tier standard              # before an engine PR merges, ~4-5 h
python3 -I run.py --tier full                  # overnight, before a release
sudo -v && python3 -I run.py --tier energy     # S10 on its own
```

On the Windows laptop (Git Bash or PowerShell, from `VireoPrototype\evals\bench`, after
`cargo build --release -p vireo-server -p vireo-cli`):

```bash
python -X utf8 -I run.py --tier quick
python -X utf8 -I run.py --tier standard
```

The platform is picked from the OS. `suite.json`'s `windows` block has the laptop's Q4
files, each model's largest context on the 8 GB GPU, and llama.cpp's CUDA build. mlx-vlm
is left out there. First check the laptop is at full speed: Turbo power plan, on mains,
and Gemma 4 E4B reading `evals/speed/prompts-mac/medium.txt` at ~6,000 tokens/s with
`vireo-cli generate --gpu`. On 15 November it ran at a third to half of that until it was
restarted.

Useful options:

| Option | What |
|---|---|
| `--models e4b,g31` | some of the roster (keys in `suite.json`) |
| `--engines vireo,llama` | some of the engines (`vireo`, `vireo-plain`, `llama`, `mlx`) |
| `--suites S1,S3` | some of the tier's suites |
| `--rungs 4k,16k` | some of S3's rungs |
| `--reps 1` | override the tier's reps (a smoke test) |
| `--set S8.soak_minutes=5` | override a suite option (value is JSON) |
| `--note "after the Q6_K kernel change"` | a line saved with the run, shown on its page |
| `--dry-run` | print what it would run and which files are missing, then stop |

Each run writes `results/<date>-<platform>-<tier>-<commit>.jsonl` (one row per test, model,
engine and rep) and the page beside it (`.html`). The page puts the previous run of the
same tier and platform beside every Vireo number and lists regressions at the top: a speed
more than 5% worse, or any score lower. Server logs go to the system's temporary folder
(`vireo-bench-logs`).

After a standard run you trust, save it as the baseline that quick runs compare with:

```bash
python3 -I report.py results/<that run>.jsonl --save-baseline
```

Quick runs only re-run Vireo. llama.cpp and mlx-vlm don't change between our builds, so
their baseline numbers stand in, marked "(base)", until a pinned version moves.

Sudo on the Mac: `sudo -v` before a run lets it drop the disk cache for S2 `load-cold`
(`purge`) and read power for S10 (`powermetrics`). Without it, those two are skipped and
the page says so.

---

## The ten suites

Every result row carries its suite code, the suite version (`v1`), the platform, tier,
commit, engine, model and rep.

### S1: Raw speed (comparable to public numbers)

The `llama-bench` shapes, so our numbers can be checked against anyone's posted results.

| Test | What |
|---|---|
| `pp512`, `pp4096` | reading speed, a 512- and 4,096-token prompt |
| `tg128` | writing speed, 128 tokens after a short prompt |
| `tg128@d8K`, `tg128@d32K` | writing speed with 8K / 32K already in the context |
| `b1`, `b4`, `b8`, `b16`, `b32`, `b64` | Vireo only: a batch of that many tokens read at once, 4,000 tokens into a chat (`vireo-cli batches`; tokens/s of the batch, `ms` beside it) |

The `b` tests are there because nothing else here runs a short batch: a draft's guesses
are checked 2 to 32 at a time and a short tool result is a batch of up to 64, so drafted
writing rests on them. A kernel change doubled the 4- to 16-token ones on the Mac for two
days in November 2026 and no test showed it.

Vireo uses `vireo-cli generate --gpu` and llama.cpp uses `llama-bench -fa 1 -ngl 99`, both
one token at a time with no drafting. mlx-vlm has no bench tool, so it runs through its
server: one user message (its chat template adds ~20 tokens), speeds from the response's
timings. Prompts are the frozen corpus cut to the exact token count on each model's
tokenizer. Depths beyond a model's context on that machine are skipped.

### S2: Feel (what a user waits for)

| Test | What |
|---|---|
| `load-cold` | start to first answer after the disk cache is dropped (Mac, needs sudo) |
| `load-warm` | the same, straight after |
| `ttft-short` | time to the first token, a 40-token question (3 times) |
| `ttft-3k` | the same with a ~2.5K document pasted in (2 times) |
| `reply-400` | four whole replies, 400 tokens at most: a list rewrite, a code edit, a story, an explanation |

Timed by the client, streaming, through each engine's server.

### S3: Context ladder

**4K, 16K, 32K, 64K, 128K**, and **256K** in the full tier for models that support it.
The server is restarted with each rung's context, so its memory is what that context
costs. The prompt is the frozen corpus filling ~92% of the context, leaving room for
Vireo's ~1,300-token system turn and the answer.

Each rung records reading speed, time to the first token, writing speed (a ~200-word
answer after the read), and **recall**: five facts planted at 5%, 25%, 50%, 75% and 95%
depth, all asked for (0-5). Five facts instead of two so that facts lost from the middle
of the context show up.

Vireo runs twice: `vireo` (drafting on, as users get it) and `vireo-plain` (drafting off),
so the ladder shows both.

### S4: Conversation

| Test | What |
|---|---|
| `chat-16` | 16 turns, each pasting a ~600-token shift report and asking about it: per-turn time to first token, tokens read vs kept, right answers |
| `regenerate` | the 16th turn sent again: should read almost nothing |
| `edit-turn` | turns 1-12 with turn 6 changed: how much is read again |
| `chat-64` | 64 turns in a 32K context, so it overflows; every 8th turn asks for an earlier report's till total (a recent report, then the second one): total time, tokens read, probes right |

Vireo trims the chat itself. llama.cpp and mlx-vlm get a plain client that drops the
oldest turn pair until the chat fits, as `tools/long_run.py` does.

### S5: Agent tasks

12 tasks on the `harbor-bikes` sandbox, each with a checker, so no one has to grade by
hand. Every engine gets the same tool catalog (Vireo's names and descriptions), run by
the harness with an 8,000-character cap on results.

| Task | Right when |
|---|---|
| `t-single` | it lists the sales folder and names the files |
| `t-parallel` | it reads all three files asked for |
| `t-multistep` | list, read, calculate: the year's total, 26,742.25 |
| `t-skill` | the weekly-report skill: report written with 78 hours, total in the reply |
| `t-longlog` | a ~55K-token log pasted in (half of it for Qwen 3.5 9B on Windows), then tools: names inventory-sync |
| `t-notfound` | asked for a file that isn't there: says so |
| `t-notool` | a plain question with tools offered: answers with no tool call |
| `t-json` | the answer as JSON (Q4, 26,742.25), worked out with tools |
| `t-approval` | a write the user refuses: nothing written, says it was refused |
| `t-recover` | the first read fails: tries again, 14 days |
| `t-calc` | sums through `calculate`: 9,252.24 |
| `t-deep-tool` | ~70% of a large context (128K on the Mac, the model's largest on Windows) filled, then a tool call: lists the folder, says 4 |

Recorded: right or wrong (and what was expected), wall time, tools called, tokens
written. The full tier runs them all again with thinking on.

### S6: Correctness of the engine itself

These catch kernel and quantisation bugs, not model mistakes.

| Test | What |
|---|---|
| `ppl` | perplexity on `fixtures/ppl.txt` (the project's own prose), 40 chunks of 512 tokens, Vireo and llama.cpp scored the same way. Should match within ~0.5% |
| `greedy-match` | 20 raw prompts continued greedily for 256 tokens by Vireo and llama.cpp: words that agree before the first difference (median), and how many are identical |
| `qa-50` | 50 short exact-answer questions |
| `grammar` | 20 requests for strict JSON: right, parses, parses as sent |

mlx-vlm has no perplexity tool or raw continuation here, so it only takes part in `qa-50`
and `grammar`. A gap here outranks any speed win.

### S7: Memory

Recorded alongside S3 and the chat server:

| Test | What |
|---|---|
| `idle` | after load, before any request (chat context) |
| per rung: `peak_gb` | the most used while the rung's request runs |
| per rung: `settle_gb` | 5 s after it: memory that doesn't come back is a leak or an oversized cache |

On the Mac: phys_footprint plus the resident part of mapped files (RSS misses MLX's KV
cache; the footprint misses an mmapped GGUF). On Windows: the whole GPU's memory in use,
read between steps.

### S8: Endurance (full tier)

| Test | What |
|---|---|
| `soak-30` | 30 minutes of back-to-back mixed requests: writing speed in the first vs last 5 minutes, memory at start and end, errors |
| `heat-10` | 10 minutes of continuous writing on the platform's heat model (Gemma 4 31B on the Mac, Qwen 3.5 9B on Windows): speed per minute, last as % of first |

### S9: Several at once (full tier)

2 and 4 requests sent together: combined writing speed, slowest and fastest first token,
errors. llama-server gets that many slots. Vireo serves one reply at a time today, so this
is a baseline to improve on.

### S10: Energy (its own tier)

Joules per 1K tokens and average watts while reading (the 4K prompt, 5 times) and writing
(512 tokens, 3 times), through each server. On the Mac this is `powermetrics` (CPU + GPU +
ANE), which needs `sudo -v`. On Windows it's the GPU's own energy counter, read before and
after, so it covers the GPU only. It never runs alongside other suites, because reading
power moves the clocks.

---

## Models

All 6-bit on the Mac (Q6_K GGUF / MLX 6-bit), the same files as the Nov 13-14 comparison;
Q4 on the Windows laptop (the 6-bit files don't fit 8 GB).

| Group | Mac | Windows |
|---|---|---|
| Small | Gemma 4 E2B, Gemma 4 E4B, Qwen 3.5 4B, Qwen 3.5 9B | the same four, Q4 |
| MoE | Gemma 4 26B-A4B | n/a |
| Large | Gemma 4 31B, Qwen 3.8 27B, Muse Glimmer 30B | n/a |

A model joins or leaves the roster only with a version bump (v1 → v2).

## Tiers

| Tier | When | What runs | Time (estimated) |
|---|---|---|---|
| **quick** | after a tuning change | Vireo only: S1, S3 to 32K, S5 `t-multistep` + `t-skill`, S6 `greedy-match`, S7. Mac: E4B and 31B; Windows: E4B and Qwen 4B | ~15 min |
| **standard** | before merging an engine PR, weekly | S1-S7, every model and engine, S3 to 128K (Windows: to each model's largest), 3 reps of speed tests | Mac ~4-5 h, Windows ~1.5 h |
| **full** | overnight, before a release | everything in standard plus 256K, S5 with thinking on, S8, S9 | Mac ~10-12 h |
| **energy** | now and then | S10 alone | ~1 h |

## Rules (so runs stay comparable)

1. **Pinned engines.** llama.cpp and mlx-vlm versions are in `suite.json`. Moving one is a
   deliberate step: rerun standard for that engine and say so with `--note`.
2. **One engine and model loaded at a time.** (Three large models loaded together halved
   llama.cpp's reading on Nov 12.)
3. **Order alternates** between reps (Vireo first, then last).
4. **Reps:** 3 for S1 and S3 (median), 1 for the rest (they're long, and their checkers
   make them stable). Run again if a result moves more than 10%.
5. **Every prompt starts with a unique line**, so no engine reuses a cache it shouldn't
   (mlx's prompt cache survives a reset).
6. **Temperature 0**, thinking off unless the test says on, the same system prompt and the
   same max tokens for every engine.
7. **Machine state recorded** with every run: power source, power mode, OS, free memory,
   uptime, commit and whether it had uncommitted changes. Run on mains, in high-power
   mode (Turbo on Windows).
8. **Nothing polls the GPU or power during a measured test** (except S10).
9. **Same prompt length**: Vireo adds ~1,300 tokens of system instructions, so long
   prompts are sized to ~92% of the context minus room for them, and every engine fits.

## Files

```
evals/bench/
  BENCH-SUITE.md      this file
  suite.json          version, pinned engines, model roster per platform, tiers
  run.py              runs a tier, writes results/<run>.jsonl, then the page
  report.py           a results file -> page; --save-baseline
  common.py           paths, config, result rows, memory, HTTP, machine state
  engines.py          the three servers and the command-line tools
  desk.py             the tools and the sandbox they run on
  prompts.py          prompts built from the fixtures; scoring helpers
  suites/s1_raw.py ... s10_energy.py   (S7 is recorded by s3_ladder.py and run.py)
  make_fixtures.py    rebuilds the generated fixtures (only on a version bump)
  fixtures/           corpus.txt (1.4 MB of the repo's source, frozen), cuts.json (token cuts
                      per model), needles, chat notes, qa50, json tasks, greedy prompts,
                      reply prompts, ppl text, the sandbox, the skill, the incident log
  results/            run files and pages
  baselines.json      the last saved standard run per platform
```

Absorbed from earlier harnesses (their result files stay where they are as history):
`evals/speed/bench-2026-11-13/bench.py` (engine drivers, tools, sandbox, workloads),
`evals/speed/long-context-2026-11-15/` (the ladder's upper rungs), `tools/long_run.py`
(S4 `chat-64`), `tools/reply_speed.py` (S2's reply prompts), `tools/concurrent_run.py`
(S9).

## Not in this suite

- **The Aurelia gauntlet** (the communications-coordinator run) stays separate. It tests
  the whole app and its prompts on a real workflow, is graded partly by hand, and runs
  through Vireo-Proto rather than a bare server. It answers "does the product work"; this
  suite answers "is the engine fast and right".
- **vireo-eval** (the 43 cases) stays the check for prompt and rule changes.
