# Developing Syrinx

Syrinx is Vireo's own engine. Since 2026-11-19 it has its own repository
(InvurseStudio/Syrinx), mounted in the Vireo repository as a git submodule at `Syrinx/`;
before that it was `Syrinx/` there (and `VireoPrototype/` until 2026-11-17), and its
history came along. The chat apps that run on it, Vireo for Mac (from 0.9.1) and Windows, keep the
Vireo name and live in the Vireo repository (`Mac/`, `Windows/`), as do the evals,
benchmarks and the Aurelia gauntlet (`evals/`, `../evals/` from here in a Vireo checkout).

Work on Vireo's own stack: a Rust server and inference engine that would replace the
Python server in `server/` and the inference servers it starts. GGUF models, read
natively. One Cargo workspace for both platforms: on Windows the GPU backend is CUDA
(NVIDIA), on the Mac it's Metal (`crates/syrinx-engine/src/metal`), and that backend and
its tuning are the only parts that differ. Other GPUs and NPUs come later.

- [docs/HANDOFF.md](docs/HANDOFF.md): where things stand, for whoever picks it up on either machine

Nothing here is used by the shipping apps.

- [docs/PLAN.md](docs/PLAN.md): the plan, the decisions so far, phases and open questions

## What's here

| Folder | What it is |
|---|---|
| `crates/syrinx-engine/` | The inference engine (a library): the GGUF reader, the tokenizers (Llama 3, Qwen 3.5, Gemma 4), the tool-call grammar, and Llama, Qwen 3.5 and Gemma 4 on the CPU and on the GPU (`--gpu`): CUDA on Windows (`src/cuda`, `kernels/syrinx.cu`), Metal on the Mac (`src/metal`, `kernels/*.metal`), the same model code on both. |
| `builtin-skills/` | The skills Syrinx comes with, in the open SKILL.md format (13: summarizing, translating, code review, invoices, reports and more). The server finds them above `target/release`, or beside an installed program |
| `docs/` | The plan and its full record (`PLAN.md`) and where things stand (`HANDOFF.md`) |
| `tools/` | Scripts that generate engine source, such as the Unicode tables (the eval and benchmark runners are in `../evals/tools/`) |
| `models/` | Test models (not in git; see `crates/syrinx-engine/tests/real_models.rs` for which) |
| `crates/syrinx-cli/` | Runs the engine from the command line, for testing and benchmarks |
| `crates/syrinx-server/` | The server: one model over the part of the app's API that chats use. Tools: files and Git (run by the app that asks for them, as the Windows app does, or by the server itself, `files.rs`, as for the Mac app), the tools that change files with the user's approval, the date; skills (SKILL.md, picked by scoring); attachments (`attach/`). Each prototype app starts it itself; `start-server.cmd` runs it by hand on Windows. It keeps each chat's cache in the prototype's folder (`%LOCALAPPDATA%\VireoPrototype`, `~/Library/Application Support/VireoPrototype` on the Mac), so going back to a chat reads only what's new. With `--standalone` (`serve.sh`, `serve.cmd`) it serves the model on its own instead (`standalone.rs`): the OpenAI API, a chat page (`standalone.html`) and the statistics (`stats.html`), no pairing or tools, caches in `%LOCALAPPDATA%\Syrinx` or `~/Library/Application Support/Syrinx` |
| `windows/SyrinxManager/` | The Syrinx app for Windows (WinUI 3, as Vireo's Windows app): runs `syrinx-server --standalone`, downloads models (its `Models/` folder came from Vireo's `Vireo.Core/Engine`, with `EngineSupport.cs` made Syrinx's own: keep the two in step), says how other apps connect, and shows the statistics. `dotnet run -c Release --project windows/SyrinxManager` |
| `macos/SyrinxManager/` | The Syrinx app for the Mac (SwiftUI, in Vireo's look): the Windows app's four pages, its `Models/` code ported to Swift (`EngineSupport.swift` checks against Metal's kernels: keep it in step with the Windows copy and the engine). `build-app.sh` makes `build/Syrinx.app` with `syrinx-server` inside; `swift test` runs its unit tests, including reading every GGUF in `models/` |
| `crates/syrinx-eval/` | How well each model uses the tools and skills: `cargo run --release -p syrinx-eval -- --out ../evals/eval-runs/latest.json` runs 39 cases (twelve easy, twelve harder, six that change files, five skills, four for the later tools) through syrinx-server against a stand-in PC and scores the tools, arguments and answers: by default on Gemma 4 E2B and Qwen 3.5 4B (about 6 minutes), with `--full` on Gemma 4 E2B and E4B and Qwen 3.5 4B and 9B (about 18), for in-depth testing and before a pull request. The latest full run is `../evals/eval-runs/windows-short-batches.json` (Windows); the Mac's is `../evals/eval-runs/mac.json` (35 cases). Larger models run on request: Gemma 4 31B passes all 35 on the Mac (`../evals/eval-runs/gemma-4-31B.md`) |

## Chatting with the engine in the prototype app

1. Install the app with `Windows\install-test-build.cmd` in the Vireo repository. It builds `syrinx-server` too
   and installs it beside the app.
2. Open **Vireo Prototype**, then Settings › Models: download one of the suggested models
   or add a folder of `.gguf` files (Llama, Qwen 3.5 or Gemma 4), make it the active model,
   and **Start**.

The app starts `syrinx-server` with that model (any free port on 127.0.0.1, a token it
accepts without pairing) and sends chats to it while it runs, keeping any Mac's pairing.
With *Start the model when Vireo opens* it starts with the app; it ends with the app. The
choice is kept in `%LOCALAPPDATA%\VireoPrototype\engine.json`.

`start-server.cmd` runs the server by hand instead (port 3000, paired with the 6-digit
code it prints). The full server (tools, files, skills, the scheduler) is the rest of
phase 3.

## Setting up

Install Rust with [rustup](https://rustup.rs) (the MSVC toolchain), then from this
folder:

```bash
cargo run -p syrinx-cli
```

The CUDA work in phase 1 also needs Visual Studio's **Desktop development with C++**
workload and the CUDA Toolkit (13.5 on the development PC). See "Development machine"
in the plan.
