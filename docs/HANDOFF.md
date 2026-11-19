# Where things stand

## Focus now (2026-11-19)

In order:

1. **The long-context slowdown.** The 19 November rerun found the 32x128 prompt attention
   tile (`attention_tcg`) 5-25% slower at reading from 64K to 256K and 8-16% slower at
   writing after 125K, which the README's numbers show. The suspect is the
   runtime-bounded fallback loop in its inner K-tile loop.
2. **An OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`, keys), so other
   tools can use Syrinx.
3. **A measurement pass:** the eval on the large models (it hasn't run on the 8-bit KV
   cache or the 0.9 tools), and an Aurelia gauntlet run in Vireo-Proto.
4. **CUDA catching up:** the CUDA side of the Mac's last pass (`LogitsBuffer::likeliest`,
   Llama's cache sizing) has never been compiled, and the mixture-of-experts model is
   refused on CUDA.
5. **Vireo 1.0 on Syrinx:** undo, approval settings and skill import, then the release
   that Vireo 0.9 updates into.

The notes below are the record by date, Windows and Mac, from 2026-11-11.

---

Written 2026-11-11 on the Windows machine (RTX 5060 Laptop), for whoever picks the
prototype up next on either machine, the Mac especially. [PLAN.md](PLAN.md) has the full
record; this is the short version and the to-do list.

## One workspace for both platforms

Until today the Mac had its own copy of the engine in `Mac/Engine/`. It's been folded back
in: there is one Cargo workspace, here in `Syrinx/`, and both machines build it.

| Part | Shared or per platform |
|---|---|
| `crates/syrinx-engine` model code, GGUF, tokenizers, grammar, CPU path | Shared |
| GPU backend | Per platform: CUDA on Windows (`src/cuda`, `kernels/syrinx.cu`), Metal on the Mac (`src/metal`, `kernels/*.metal`). On macOS `src/metal` stands in for the `cuda` module, with the same API |
| Backend tuning | Per platform, behind `cfg(target_os = "macos")`: the Mac's fused gate-up-activation for one token (`mul_glu`). The prompt batch (`DEFAULT_BATCH`) is 1,024 on both |
| `crates/syrinx-server` (tools, approvals, skills, attachments, chat cache) | Shared |
| `crates/syrinx-eval` (35 cases) | Shared |
| Folders | Shared code (`store::data_dir`): `%LOCALAPPDATA%\VireoPrototype` on Windows, `~/Library/Application Support/VireoPrototype` on the Mac |

The rule from here: only the Metal backend and its tuning are the Mac's own. Anything else
written on one machine is written for both, and measured with `syrinx-eval` on both.

## After pulling on the Mac

Done on 2026-11-11 for the one-workspace merge (PR #129): every test passes on the Mac
(engine 69, GPU 12 on the M5 Max, server 58, eval 9; swift test 132), the app builds
`syrinx-server` from the workspace, and the 35 cases on Metal (`evals/eval-runs/mac.json`) match
Windows but for three that tip between the two GPUs. The steps, for the next time:

1. Move the local, untracked folders that were in `Mac/Engine/` up to the workspace:
   `Mac/Engine/models/` to `Syrinx/models/` and `Mac/Engine/reference/` to
   `Syrinx/reference/` (git doesn't track them, so they stay behind in the old
   folder). `Mac/Engine/target/` can be deleted.
2. From `Syrinx/`: `cargo test --release`. Not run on a Mac since the merge.
   Three of `files.rs`'s tests are macOS-only now (`/tmp` and `/etc`, Git where the Mac
   keeps it) and run only there.
3. From the Vireo repository's `Mac/`: `swift test`, then `./build-test-app.sh`. The build script and
   `LocalEngine.swift` now build and find `syrinx-server` in `Syrinx/target/release`;
   `LocalEngineTests` was updated for that layout. Not run (no Swift here).
4. Run the eval on Metal and save it beside the others:
   `cargo run --release -p syrinx-eval -- --full --out ../evals/eval-runs/mac.json` (a plain run is
   the short one, two models). The Mac's last run,
   `evals/eval-runs/mac-baseline.json`, has only the first 24 cases.

## What the server does now (both platforms)

- **Tools that read:** files and Git. Run by the app when it says it runs them
  (`clientTools`, as the Windows app does) or by the server itself (`files.rs`, as for the
  Mac app, when the chat gives access).
- **Tools that change files:** `write_file`, `rename_file`, `move_file`, `delete_file`.
  Each asks the user first (`confirm_required`, answered at `/api/agent/confirm`), after
  the app checks the change can be made (`check_change`). Offered only when one scored
  question says the request asks for a change; once a chat's cache declares them they
  stay declared in that chat (callable only when asked for), so its cache is read again
  once, not at every such turn (2026-11-12). **App-run only for now:** `files.rs` doesn't
  make changes, so the Mac app doesn't get them yet (to-do 1).
- **Skills:** SKILL.md files, the Mac's ten built-in ones plus the user's. A skill is
  offered when this machine has the first tool it names (`summarize-document`,
  `translate-document`, `code-review` today), picked by its slash command or by scoring,
  and written into the reply as a `use_skill` call and its result.
- **A chat's cache kept (2026-11-12, PLAN.md Progress):** besides the change tools
  above, a chat too long for the context starts at a fixed cut (`chat::cut_points`) and
  reads everything again once in about a third of the context, not every turn; and a
  fact remembered mid-chat is told at the end of the request (`memory::since`) while the
  chat's instructions keep the facts they began with. Checked with scripted chats and
  14 eval cases on Gemma 4 E2B and Qwen 3.5 4B on the Mac. **On Windows:** run the eval
  (shared code). **Not measured:** the large models.
- **Attachments:** `POST /api/upload` reads text, Markdown, PDF and Word in Rust (`attach/`).

## Eval on Windows after the merge

All 35 cases on the four eval models, after the merge, against the last run before it
(`evals/eval-runs/skills.json`): every case passes or fails exactly as it did, on every model.

| Model | Older 30 | Change cases | Skill cases | All 35 | Time per case |
|---|---|---|---|---|---|
| Gemma 4 E2B | 26 | 5/6 | 3/5 | 29 | 1.65 s |
| Gemma 4 E4B | 27 | 5/6 | 5/5 | 32 | 3.18 s |
| Qwen 3.5 4B | 29 | 6/6 | 5/5 | 34 | 6.43 s |
| Qwen 3.5 9B | 29 | 5/6 | 4/5 | 33 | 10.64 s |

Qwen 3.5 0.8B is no longer one of the eval's default models (2026-11-11): too small to use
the tools well (it said yes to every skill, saved a poem to a file). It ran in this check
too (22/35, as before) and still runs on request (`--models Qwen3.5-0.8B-Q4_K_M.gguf`); it
stays one of the engine's own test models. Results: `evals/eval-runs/one-engine.json`.

## Larger models (Mac, 2026-11-11)

Gemma 4 31B runs: per-layer key-value heads, no per-layer inputs and keys used as values
are handled in the shared model code (`gemma4.rs`, `gemma4_cuda.rs`). It passes all 35
cases on the Mac, about 10 s a case. Tuned for it, on Metal only: a sliding-window
layer's cache kept as a ring (32K chats 29.5 → 4.4 GB), a 256-token prompt tile for whole
1,024-token batches, and a 2–8-token Q6_K kernel. In the server, for both: `--draft`,
Gemma 4 E2B guessing tokens for 31B to check (15–18% faster writing on longer answers).
**Not run on CUDA yet:** the 31B itself and `--draft`. CUDA keeps full caches
(`attention_ring` and `qkv_prep` assert a ring of 0 there; and Muse Glimmer's `sigmoid_mul`
has no CUDA kernel). Details:
`evals/eval-runs/gemma-4-31B.md`.

## After pulling on the Mac (the Windows pass of 2026-11-12)

Shared code changed behind `cfg(target_os = "macos")` switches and **not compiled on a
Mac**: `gemma4_cuda.rs` and `qwen35_cuda.rs` (`forward_batch` queues a batch through
`queue_batch`, replayed from a graph on CUDA only), `syrinx-server` (`PICKER_BATCH`, 1,024
on the Mac as before; Qwen's steps kept only if `keep_steps` succeeds, not an error
otherwise; `DEFAULT_LOOKUP` and `LONGEST_RUN` now one value for both, the Mac's),
`tests/check_speed.rs` (runs on both; Qwen goes back through a checkpoint). Run
`cargo test --release` and a few eval cases there; nothing should measure differently.

## After the gauntlet's first run (the Mac, 2026-11-13)

Built in answer to `evals/gauntlet/runs/Aurelia-Proto-Run-2026-11-12-Gemma4-31B-notes.md`
(PLAN.md's Progress has the detail). **Nothing here has been run on a model**: unit tests
only (server 103, Mac app 144). Before it's merged: the eval on the small and the large
models, thinking off and on, and the gauntlet (`evals/tools/aurelia_run.py`, which now plays all
32 prompts; `--thinking off` for a like-for-like with the 12 November run).

- **Thinking:** a request's `"thinking": true` (else `--thinking`, off by default, so the
  eval and the Windows app are as they were). The Mac app sends it on unless Settings,
  Chats says otherwise. Thoughts go to the app as `{"type":"thinking"}` events
  (`start`, `content`, `done`), never in the reply's text.
- **A reply's tool results, kept:** the reply ends with a `turn_record` event (a string:
  JSON with each call as written and its result as read). An app keeps it with the reply
  and sends it back as that message's `record`; without it a chat is as before. The
  Windows app doesn't yet.
- **What every prompt now says differently:** the preamble's lines on the user's rules;
  `(Today is ...)` at the end of a request with tools; `read_file`'s and `calculate`'s
  descriptions. The eval's cases will read these.
- **`clientTools` from the Mac app:** `run_tasks` (it runs due tasks, so `schedule_task`
  is offered), `chats_here` (its chats are on this machine, so the server searches them),
  and the calendar tools it may use. An app that lists none gets none of the new tools.
- **Tasks** are the server's (`tasks.json`; `VIREO_TASKS_FILE` for a test) and run only
  while an app asks `GET /api/tasks/due` (the Mac app, every 10 s). A link task's link is
  `POST /api/tasks/link/<token>`, which needs no pairing: the token is the key, and the
  server listens on this machine only.
- **Calendar and reminders** need the usage texts `build-test-app.sh` now writes into the
  app's Info.plist; a build without them (`swift run`) offers no calendar tools. The first
  real use asks for the Mac's permission. A test script's runs are dry.

## Working rules and Gemma's several calls (the Mac, 2026-11-13, after the second run)

In answer to `evals/gauntlet/runs/Aurelia-Proto-Run-2026-11-13-Gemma4-31B-notes.md`
(17 pass / 12 partial / 3 fail by hand, thinking on): the format and sampling already
matched Google's template; the gap was the instructions. **Unit tests only (server 105);
not run on a model.** To check: a few eval cases on Gemma 31B, thinking off and on (e.g.
`--cases two-files,three-steps,read-and-add,office-invoices,office-policy,calc-tip,injection,no-tool`),
the small models too, then the gauntlet against the 13 November run.

- **The preamble** (`tools::PREAMBLE`) is 0.9's "How to work" (`server/api_server.py`'s
  `SYSTEM_PROMPT`): files first, every file and row, read the policy, finish its steps and
  give totals, `calculate` for every sum, result first then what the user must know, pending
  not approved, no card numbers, approvals. `tools::family_notes`: Gemma gets 0.9's
  several-calls rule and tool-call note; Qwen its tool-call note and the short-reply line
  (no several-calls rule: the server runs its calls one at a time). Every eval prompt reads it.
- **Gemma's several calls in one step:** after each call the server reads `<tool_call|>` and
  looks at the model's next token: another `<|tool_call>` is written as the next call, up
  to 8 together (`MAX_TOGETHER`); otherwise the results follow, one `<|tool_response>`
  each, then the thought. A record's step has `"then": true` when the next call followed
  it at once, so the history writes them back the same way (`chat::with_calls`).
- **Thinking off, Gemma's turn opens with the empty thought** (`<|channel>thought\n<channel|>`),
  as its GGUF template and Google's July 2026 one do.
- **Thinking limits:** a thought is 2,000 tokens, or 400 after a step that only made
  changes (0.9's routine step) or once a reply has thought 8,000; before, past 8,000 the
  model stopped thinking between steps altogether.

## Google's draft model for Gemma 4 31B (the Mac, 2026-11-13)

`gemma-4-31B-it-assistant` (Google's, safetensors bf16, 926 MB; the one 0.9 drafts with)
runs on the engine: `syrinx-engine/src/models/gemma4_assistant.rs`, a child of
`gemma4_cuda.rs`. Four small layers with no keys or values of their own: they read the
main model's (its last sliding and last global layer's), from its final state for the last
position read and the token just picked. Weights rounded to Q8_0 on load (about 1 s).

- **Measured on the M5 Max** (`tests/gemma_assistant.rs`, both ignored, need
  `VIREO_GEMMA_MODEL`): first guess right 74%, 1.72 of 3 taken a round, 4.3 ms to guess 3;
  writing greedy the same text as one token at a time, 19-20 → 31-32 tokens/s (1.6-1.7×;
  2, 3 or 4 guesses a round about the same, 3 kept). The E2B draft model managed 23.
- **The server** loads it with the model when it finds one (`assistant::find`: beside the
  GGUF, or `~/Library/Application Support/InvurseChat/DraftModels/<name>-assistant-bf16`)
  and `--mtp on` (the default), or from `--draft <folder or .safetensors>`. It drafts as
  Qwen's layer does (`mtp_draft`, `MTP_GUESSES`). The prototype app doesn't download it
  yet: on a Mac without 0.9's draft models it runs without.
- **Not yet:** a run through the app and the gauntlet with it; CUDA (the code is the
  shared GPU path, never run there); the 26B-A4B's and E2B/E4B's draft models (the E2B/E4B
  ones have a centroid head, `use_ordered_embeddings`, not built).

## Draft models for Gemma 4 E2B and E4B (Windows, 2026-11-15)

Google's draft models for the small Gemmas run now, and the 31B's path runs on CUDA
(PLAN.md, "Fifteenth round"). Shared code, **not compiled or run on a Mac**:

- **Engine** (`gemma4_assistant.rs`): the centroid head (`use_ordered_embeddings`), a
  draft's own count of query heads, `mtp_draft_sure` (each guess with the draft's
  probability, stopping under a floor), and on CUDA a recorded graph for a guess's
  layers. `gemma4_cuda.rs`: a shared-layer model's states count as final after a
  one-token read or a check of every token (they were never, so E2B and E4B could not
  draft). `tests/gemma_assistant.rs` runs on both machines now (still ignored; it wants
  `VIREO_GEMMA_MODEL`), with a third test for a round's costs.
- **Server:** behind `cfg!(target_os = "macos")`, the Mac drafts as before (any guess,
  the draft model or the chat, not both). On CUDA: `MTP_FLOOR` 0.7, and the chat's
  guesses of 3 or more beside the draft's. To try on the Mac: the 31B with a floor
  (its first guess was right 74% without one), and E2B/E4B with their draft models
  (`mlx-community/gemma-4-E2B-it-assistant-bf16`, found beside the GGUF or in
  DraftModels as the 31B's is; `find` also reads Google's `gemma-4-E2B_q4_0-it` names).
- **The eval's results have speeds now** (`speeds` for a model, and each case's
  `readTokensPerSecond`, `writtenTokensPerSecond`, `toolTimes` and the rest), and the
  server's `Tool:` line says how long the tool ran. Run the Mac's evals with `--out` as
  before; older result files don't have them.
- **Measured on Windows:** E2B writes 6-26% faster and E4B 5-50% (copied code least,
  lists and explanations most), every eval case scoring as it did (`evals/eval-runs/windows-small-gemma-draft.json` against `-draft-off.json`: 34 and 39 of 43 either way, all 43 answers identical, writing 11% and 16% faster over the eval).

## The Mac's speed harness on Windows (2026-11-15)

`evals/speed/bench-2026-11-13/bench.py` now runs on both machines. On the Mac nothing
changed; on Windows (`os.name == "nt"`) it takes the four small models' Q4 GGUFs from
`models/`, `reference/llama.cpp-b12262/llama-server.exe`, no `large` or `xlarge` rows and
no mlx-vlm, and reads the whole GPU's memory with `nvidia-smi` after each workload (there
is no process footprint to sample). From `evals/speed/bench-2026-11-13`, one model at a
time with nothing else on the GPU, in Python's UTF-8 mode (Windows' default code page
would garble the prompts):

    ENGINES=vireo,llama MODELS=e4b REPS=1 OUT=<folder>/results.jsonl LOGS=<folder> python -X utf8 -I bench.py

(`MODELS` takes `e2b,e4b,q4b,q9b`; `python -X utf8 report.py <results.jsonl>` prints the
Mac's summary of it.) Things that matter:

- **Contexts are forced** (`WIN_MODELS`; `CONTEXT=` overrides): syrinx-server's own choice
  stops at 32,768, and the Mac's long prompt is 33,797 tokens on Vireo for the Gemmas and
  its long tool prompt ~57K. 65,536 fits Gemma 4 E2B, E4B and Qwen 3.5 4B with nothing in
  Windows' shared GPU memory; Qwen 3.5 9B spills past 32,768 on Vireo, so it runs at
  32,768 with its long tool log cut to the first half (`WIN_LOG_SHARE`), for both engines.
  To check a context, read `\GPU Process Memory(pid_<server>*)\Shared Usage` while the
  server holds its memory: if it grows, that context spills and reading slows ~10×.
- **Paths in the questions use forward slashes** (`desk.root.as_posix()`): a model copying
  `C:\Users\me\…` into a tool call can turn `\t` and `\n` into a tab and a new line.
- `vireo-plain` (no guesses) adds little here: on Windows it only turns off the Gemmas'
  draft models and the chat's guesses, measured on their own already.
- The page is `evals/speed/bench_mac_harness_report.py` (its text in the results folder's
  `report.json`); 15 November's run is `windows-rtx5060-2026-11-15-mac-harness.html` and
  `.pdf`, raw in `evals/speed/bench-2026-11-15-windows-mac-harness/`.

## Three engine and server changes on Windows (2026-11-15, afternoon)

Uncommitted on `windows/cuda-reading-memory`, on top of 7e00a19. Measured on Windows only
(`evals/speed/windows-rtx5060-2026-11-15-after-changes.html`, PLAN.md's Progress). For the Mac:

- **Shared code, both machines: Qwen's saved state at the system turn's end**
  (`crates/syrinx-server/src/main.rs`, `Kept::turn`). When Qwen reads a system turn, its
  DeltaNet layers are now saved a few tokens before that turn's end, beside the saved state
  where the first two chats parted (`Kept::base`), so a new chat with the same instructions
  and tools reads none of them. On Windows the eval's Qwens read 55-61% fewer tokens and ran
  18-20% quicker, the same cases passing. **It has not been compiled or run on a Mac**: run
  the workspace tests and the eval there before relying on it.
- **CUDA only: sliding-window rings** (`attention_ring` and `ring_row` in `syrinx.cu`,
  `src/cuda/mod.rs`; `gemma4_cuda.rs` drops its `cfg!(target_os = "macos")`). The Gemmas'
  sliding-window layers keep only their window on CUDA, as Metal already did. Nothing
  changes on the Mac.
- **CUDA only: the prompt attention kernel.** `Gpu::attention` on CUDA now leaves a batch's
  queries in f16 where they lie (`q_rows_to_half`, each row's halves at the start of its
  floats): a caller must not read `q` as f32 after it. Larger blocks for batches over 64
  tokens, a `_short` set for 64 or fewer. Metal's kernels are unchanged.

## Two more Windows changes (2026-11-15, evening)

After the three above, both passing every test but **not yet run through the eval or the
benchmark** (PLAN.md's row of that evening has the measurements and what was tried and
dropped). Shared code, not compiled on a Mac:

- **`GpuMatrix::mul_gated`** (`models/gpu.rs`) is now the feed-forward's last product in
  the Gemma, Qwen and Llama GPU code. On the Mac it does what was there before (the
  activation, then the product). On CUDA in 8-bit mode the activation is rounded to 8
  bits as it is worked out (`swiglu_q8`, `geglu_q8`) and `gate` is left unchanged.
- **`draft::LEAST_RUN`**: guesses from the chat need a run of 3 on CUDA; 2 on the Mac, as
  before.
- `evals/tools/reply_speed.py` loads one server at a time now.

## The prompt attention kernel on Windows (2026-11-15, night)

On the branch `windows/speed-hunt`, uncommitted; all tests pass, no eval or benchmark run
(PLAN.md's row of that night has the profile, the numbers and what was dropped). CUDA
only: nothing here changes on the Mac.

- **`q_rows_to_half` lays the queries out differently**: each 32 numbers of a row in the
  order `attention_mma`'s threads take them. Only `attention_mma` reads them (as before,
  a caller must not read `q` again after `Gpu::attention` on an f16 cache).
- **The queries are read two steps ahead** in `attention_mma`. They sit in global memory
  and mostly miss the L1 cache: reading ~34K tokens is 6.8% faster on Gemma 4 E2B, 2.9% on
  E4B, about 1% on the Qwens, with the same numbers.
- **A batch of 9 to 64 tokens with no sliding window** goes to `attention_mma_*_part`:
  heads that share keys and values fill a block, the context is split into parts and
  `attention_combine` merges them. 27,000 tokens deep a 16-token batch takes 40% less time
  (Qwen 3.5 4B 40.7 → 24.0 ms). The Mac has done the same in `attention_tcg` for a while.
- `tests/prompt_attention_speed_cuda.rs` times a batch's attention for the small models'
  shapes (the Mac's `prompt_attention_speed.rs` is for the large ones and Metal only).
- **Shared code, both machines, not compiled on a Mac: Gemma's state at the system
  turn's end** (`crates/syrinx-server/src/main.rs`, `Kept::head`;
  `Gemma4CudaSession::has_rings`). Where the sliding-window layers keep rings, the
  server exports the session's state a few tokens before the system turn's end when it
  reads one, and imports it when `truncate` can't go back (a prompt longer than a ring
  wrote over the window). A new chat after a long prompt keeps 1,267 of 1,307 tokens
  where it kept none (Gemma 4 E4B: first token 225 → 59 ms). The Mac's rings had the
  same cost, so it should gain the same; run the server's tests and the eval there.
- **CUDA only: `matmul8` reads one side's fragments with `ldmatrix`** (the weights' for
  Q4_0 and Q8_0, x's for Q4_K and Q5_K): ~3K reads 5% faster on the Gemmas, 2% on the
  Qwens, the same numbers.
- **CUDA only (server): guesses deep in a chat** (`guesses_pay`, `mtp_share`,
  `CHAT_GUESSES_FROM`, `MTP_GUESSES_FROM` in `crates/syrinx-server/src/main.rs`). Past
  24,000 tokens the chat's guesses, and past 48,000 a Gemma draft model's, stop for a
  reply once too few of them were taken: a round's check costs each guess a pass over
  the context there. `guesses_pay` is always true on the Mac; worth measuring there with
  the same script idea (a 200-token reply after a long prompt, `--mtp off`, `--lookup 0`).
- **CUDA only: the K-quants' 8-bit products take tiles of 128 tokens** in batches of
  128 or more (`matmul8_t128_*`, `WIDE_TOKENS` in `src/cuda/mod.rs`), the rest of a
  batch going to the smaller tiles. The Qwens read 7-9% faster at ~2.3K and ~30K tokens,
  with the same numbers.
- **Nsight Compute works here** (`ncu --kernel-name <kernel> --launch-skip <n>
  --launch-count 1 --set full`; kernels inside a recorded graph don't show, a prompt's
  batches do). It says what a kernel waits on, which Nsight Systems doesn't.

## To do, roughly in order

1. **Changes on the Mac:** done 2026-11-12 (write, rename, move, delete and `write_spreadsheet` in `files.rs`, to the Trash; no undo journal yet, and Word and PowerPoint are still the Windows app's only). The original note: `files.rs` makes the four changes (to the Trash, with undo as
   the Windows app's journal does) and checks them before asking, as `check_change` does
   in the app. Then `tools::offered` can let the server run them, and the six change cases
   apply on the Mac.
2. **Before the server listens on the network** (another device chatting with a Mac or PC
   host): tie each waiting approval and tool request to the device that asked. Today any
   paired device can answer any of them; the Python server lets another device answer only
   its own.
3. **Tool wording by who runs the tool:** the tools say "on this PC" and "Recycle Bin"
   everywhere. It should name the machine that runs each tool, and be measured on both,
   since it changes what every reply reads.
4. **Skills, faster:** done on 2026-11-12 (the Mac): the scoring questions now go on from
   the start they share with the last one, in up to 3 picker sessions, and Qwen keeps a
   checkpoint at the end of a system prompt new chats share (PLAN.md, "Fifth pass"). Six
   eval cases ran 20-49% faster on the Mac with the same scores and results. **On
   Windows:** done on 2026-11-12 (PLAN.md, "Fourteenth round"): the Mac's passes cost
   time on CUDA as merged (a check of the chat's guesses took two tokens' time), so their
   CUDA side was built: few-token kernels, graphs for short batches, 16- and 32-token
   tiles, Qwen going back. The full eval scores the same as before on all four models, in
   24-37% less time on the Gemmas, 4-6% on the Qwens. Next: a skill one step at a time,
   its answer's headings forced.
5. **Skills in the packaged Mac app:** done 2026-11-19: the built-in skills are in this
   repository (`builtin-skills/`), found above `target/release` or beside an installed
   program, where the apps' build scripts copy them. The slash-command list should also follow the asking device's
   tools.
6. **Small things:** three clippy warnings in `attach/` (from the Mac's code), one in
   `syrinx-cli`; prompt attention on CUDA for long prompts (78–87% of
   llama.cpp at ~31K tokens at full clocks; see PLAN.md, "What limits speed now"); the GPU performance counters on the Windows machine are open to all users
   for Nsight and should go back to administrators only when the tuning is done.
7. **From the Mac's all-Q6 comparison** (2026-11-13, eight models against llama.cpp and
   mlx-vlm on the build with PR #165; the rows are in
   `evals/speed/bench-2026-11-13/results-q6-merged-build.jsonl`, the numbers in PLAN.md's
   progress rows of that day and the next). Worked through on 2026-11-14 (PLAN.md's row):
   1. **E2B and E4B answered with their reasoning:** done. With thinking off, Gemma's
      system turn now says so (`tools::GEMMA_NO_THINKING`): E4B 1 → 6 of the 6 scored
      tasks, E2B 0 → 4 (its tool and skill tasks still fail; llama.cpp and mlx-vlm get 5),
      the 26B and 31B as before.
   2. **Gemma 4 31B's skill task:** done. The file-changing tools were kept from it (the
      request's own words didn't ask for a change); a skill that names one now makes them
      callable. 113 s and wrong → 27 s and right. **No drafting inside tool calls:** done
      (`write_call`: the draft layer's or the chat's guesses, each checked under the
      grammar): the 31B's tool task 56 → 28 s, Qwen 3.8 27B's 56 → 30, Muse's 66 → 50,
      Qwen 3.5 4B's 9.5 → 6.5 and 9B's 16 → 11 (the small Qwens guess from the chat in
      calls only).
   3. **Qwen 3.8 27B's long tool task:** done. A call past its limit (4,096 tokens; 768
      for `calculate`) is taken back and the model answers without more calls: 365 s and
      an error → 209 s and right (measured at a limit of 1,536).
   4. **Gemma 4 26B-A4B's long tool task:** right again in three later runs; the wrong
      answer was one run's count of a 57K-token log, not the kernels.
   5. **Muse's first token:** it is Muse's reasoning before the answer (83-391 tokens a
      reply in the log), as designed, and why only this engine answers Muse right. Left.
   6. **Loading with a draft model:** not the draft model (1.1-1.6 s). The engine ran
      first on each model, with the file not yet in the disk cache: the 31B loads in
      4.2-4.5 s with its draft model once it is.
   7. **Qwen reads prompts 8-20% slower than mlx-vlm** at every length (4B, 9B and 27B);
      the Gemmas and Muse are level or ahead. Open: at 3K tokens 69% of the GPU's time is
      the weights' products and 10% DeltaNet; at 96K half is attention.
   8. **Short replies on the smaller models** wrote 8-16% slower than mlx-vlm. Q6_K's
      blocks are now padded to 212 bytes on the GPU, not 224 (one token reads every weight
      once): Qwen 3.5 9B 63 → 66 tokens/s. The rest is open.
   9. **The 26B's default mode on a repeating text:** open (the next token's probability
      0.71 against exact mode's 0.95 with the f16 attention cache alone; perplexity on
      ordinary text within 2%).
   10. **E2B's tool and skill tasks** still fail with thinking off (the skill task ends in
      1 s with nothing done). Open.
