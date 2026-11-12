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
| Backend tuning | Per platform, behind `cfg(target_os = "macos")`: the prompt batch (`DEFAULT_BATCH`, 1,024 on the Mac, 512 on CUDA) and the Mac's fused gate-up-activation for one token (`mul_glu`) |
| `crates/vireo-server` (tools, approvals, skills, attachments, chat cache) | Shared |
| `crates/vireo-eval` (35 cases) | Shared |
| Folders | Shared code (`store::data_dir`): `%LOCALAPPDATA%\VireoPrototype` on Windows, `~/Library/Application Support/VireoPrototype` on the Mac |

The rule from here: only the Metal backend and its tuning are the Mac's own. Anything else
written on one machine is written for both, and measured with `vireo-eval` on both.

## After pulling on the Mac

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
  question says the request asks for a change. **App-run only for now:** `files.rs` doesn't
  make changes, so the Mac app doesn't get them yet (to-do 1).
- **Skills:** SKILL.md files, the Mac's ten built-in ones plus the user's. A skill is
  offered when this machine has the first tool it names (`summarize-document`,
  `translate-document`, `code-review` today), picked by its slash command or by scoring,
  and written into the reply as a `use_skill` call and its result.
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

## To do, roughly in order

1. **Changes on the Mac:** `files.rs` makes the four changes (to the Trash, with undo as
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
4. **Skills, faster:** the scoring questions are read from scratch each time (0.2 s a case
   on Gemma 4 E2B, up to 2.4 s on Qwen 3.5 9B). Keep their shared start in the picker's
   session. Then: a skill one step at a time, its answer's headings forced.
5. **Skills in the packaged Mac app:** the server finds the built-in skills by walking up
   to the repository's `server/builtin-skills`. A packaged app has to copy them in and set
   `VIREO_BUILTIN_SKILLS`. The slash-command list should also follow the asking device's
   tools.
6. **Small things:** three clippy warnings in `attach/` (from the Mac's code), one in
   `vireo-cli`; the long-chat attention kernel on CUDA (27–48% behind llama.cpp at 25K
   tokens); the GPU performance counters on the Windows machine are open to all users
   for Nsight and should go back to administrators only when the tuning is done.
