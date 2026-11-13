# Where things stand

Written 2026-11-11 on the Windows machine (RTX 5060 Laptop), for whoever picks the
prototype up next on either machine, the Mac especially. [PLAN.md](PLAN.md) has the full
record; this is the short version and the to-do list.

## One workspace for both platforms

Until today the Mac had its own copy of the engine in `Mac/Engine/`. It's been folded back
in: there is one Cargo workspace, here in `VireoPrototype/`, and both machines build it.

| Part | Shared or per platform |
|---|---|
| `crates/vireo-engine` model code, GGUF, tokenizers, grammar, CPU path | Shared |
| GPU backend | Per platform: CUDA on Windows (`src/cuda`, `kernels/vireo.cu`), Metal on the Mac (`src/metal`, `kernels/*.metal`). On macOS `src/metal` stands in for the `cuda` module, with the same API |
| Backend tuning | Per platform, behind `cfg(target_os = "macos")`: the Mac's fused gate-up-activation for one token (`mul_glu`). The prompt batch (`DEFAULT_BATCH`) is 1,024 on both |
| `crates/vireo-server` (tools, approvals, skills, attachments, chat cache) | Shared |
| `crates/vireo-eval` (35 cases) | Shared |
| Folders | Shared code (`store::data_dir`): `%LOCALAPPDATA%\VireoPrototype` on Windows, `~/Library/Application Support/VireoPrototype` on the Mac |

The rule from here: only the Metal backend and its tuning are the Mac's own. Anything else
written on one machine is written for both, and measured with `vireo-eval` on both.

## After pulling on the Mac

Done on 2026-11-11 for the one-workspace merge (PR #129): every test passes on the Mac
(engine 69, GPU 12 on the M5 Max, server 58, eval 9; swift test 132), the app builds
`vireo-server` from the workspace, and the 35 cases on Metal (`evals/mac.json`) match
Windows but for three that tip between the two GPUs. The steps, for the next time:

1. Move the local, untracked folders that were in `Mac/Engine/` up to the workspace:
   `Mac/Engine/models/` to `VireoPrototype/models/` and `Mac/Engine/reference/` to
   `VireoPrototype/reference/` (git doesn't track them, so they stay behind in the old
   folder). `Mac/Engine/target/` can be deleted.
2. From `VireoPrototype/`: `cargo test --release`. Not run on a Mac since the merge.
   Three of `files.rs`'s tests are macOS-only now (`/tmp` and `/etc`, Git where the Mac
   keeps it) and run only there.
3. From `Mac/App/`: `swift test`, then `./build-test-app.sh`. The build script and
   `LocalEngine.swift` now build and find `vireo-server` in `VireoPrototype/target/release`;
   `LocalEngineTests` was updated for that layout. Not run (no Swift here).
4. Run the eval on Metal and save it beside the others:
   `cargo run --release -p vireo-eval -- --full --out evals/mac.json` (a plain run is
   the short one, two models). The Mac's last run,
   `evals/mac-baseline.json`, has only the first 24 cases.

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
(`evals/skills.json`): every case passes or fails exactly as it did, on every model.

| Model | Older 30 | Change cases | Skill cases | All 35 | Time per case |
|---|---|---|---|---|---|
| Gemma 4 E2B | 26 | 5/6 | 3/5 | 29 | 1.65 s |
| Gemma 4 E4B | 27 | 5/6 | 5/5 | 32 | 3.18 s |
| Qwen 3.5 4B | 29 | 6/6 | 5/5 | 34 | 6.43 s |
| Qwen 3.5 9B | 29 | 5/6 | 4/5 | 33 | 10.64 s |

Qwen 3.5 0.8B is no longer one of the eval's default models (2026-11-11): too small to use
the tools well (it said yes to every skill, saved a poem to a file). It ran in this check
too (22/35, as before) and still runs on request (`--models Qwen3.5-0.8B-Q4_K_M.gguf`); it
stays one of the engine's own test models. Results: `evals/one-engine.json`.

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
`evals/gemma-4-31B.md`.

## After pulling on the Mac (the Windows pass of 2026-11-12)

Shared code changed behind `cfg(target_os = "macos")` switches and **not compiled on a
Mac**: `gemma4_cuda.rs` and `qwen35_cuda.rs` (`forward_batch` queues a batch through
`queue_batch`, replayed from a graph on CUDA only), `vireo-server` (`PICKER_BATCH`, 1,024
on the Mac as before; Qwen's steps kept only if `keep_steps` succeeds, not an error
otherwise; `DEFAULT_LOOKUP` and `LONGEST_RUN` now one value for both, the Mac's),
`tests/check_speed.rs` (runs on both; Qwen goes back through a checkpoint). Run
`cargo test --release` and a few eval cases there; nothing should measure differently.

## After the gauntlet's first run (the Mac, 2026-11-13)

Built in answer to `docs/dev/test-gauntlet/runs/Aurelia-Proto-Run-2026-11-12-Gemma4-31B-notes.md`
(PLAN.md's Progress has the detail). **Nothing here has been run on a model**: unit tests
only (server 103, Mac app 144). Before it's merged: the eval on the small and the large
models, thinking off and on, and the gauntlet (`tools/aurelia_run.py`, which now plays all
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

In answer to `docs/dev/test-gauntlet/runs/Aurelia-Proto-Run-2026-11-13-Gemma4-31B-notes.md`
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
5. **Skills in the packaged Mac app:** the server finds the built-in skills by walking up
   to the repository's `server/builtin-skills`. A packaged app has to copy them in and set
   `VIREO_BUILTIN_SKILLS`. The slash-command list should also follow the asking device's
   tools.
6. **Small things:** three clippy warnings in `attach/` (from the Mac's code), one in
   `vireo-cli`; prompt attention on CUDA for long prompts (78–87% of
   llama.cpp at ~31K tokens at full clocks; see PLAN.md, "What limits speed now"); the GPU performance counters on the Windows machine are open to all users
   for Nsight and should go back to administrators only when the tuning is done.
