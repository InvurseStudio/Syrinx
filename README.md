# Vireo Prototype

Work on Vireo's own stack: a Rust server and inference engine that would replace the
Python server in `server/` and the inference servers it starts. GGUF models, read
natively. One Cargo workspace for both platforms: on Windows the GPU backend is CUDA
(NVIDIA), on the Mac it's Metal (`crates/vireo-engine/src/metal`), and that backend and
its tuning are the only parts that differ. Other GPUs and NPUs come later.

- [HANDOFF.md](HANDOFF.md): where things stand, for whoever picks it up on either machine

Nothing here is used by the shipping apps.

- [PLAN.md](PLAN.md): the plan, the decisions so far, phases and open questions

## What's here

| Folder | What it is |
|---|---|
| `crates/vireo-engine/` | The inference engine (a library): the GGUF reader, the tokenizers (Llama 3, Qwen 3.5, Gemma 4), the tool-call grammar, and Llama, Qwen 3.5 and Gemma 4 on the CPU and on the GPU (`--gpu`): CUDA on Windows (`src/cuda`, `kernels/vireo.cu`), Metal on the Mac (`src/metal`, `kernels/*.metal`), the same model code on both. |
| `tools/` | Scripts that generate engine source, such as the Unicode tables |
| `models/` | Test models (not in git; see `crates/vireo-engine/tests/real_models.rs` for which) |
| `crates/vireo-cli/` | Runs the engine from the command line, for testing and benchmarks |
| `crates/vireo-server/` | The server: one model over the part of the app's API that chats use. Tools: files and Git (run by the app that asks for them, as the Windows app does, or by the server itself, `files.rs`, as for the Mac app), the tools that change files with the user's approval, the date; skills (SKILL.md, picked by scoring); attachments (`attach/`). Each prototype app starts it itself; `start-server.cmd` runs it by hand on Windows. It keeps each chat's cache in the prototype's folder (`%LOCALAPPDATA%\VireoPrototype`, `~/Library/Application Support/VireoPrototype` on the Mac), so going back to a chat reads only what's new |
| `crates/vireo-eval/` | How well each model uses the tools and skills: `cargo run --release -p vireo-eval -- --out evals/latest.json` runs 35 cases (twelve easy, twelve harder, six that change files, five skills) through vireo-server against a stand-in PC, on Gemma 4 E2B and E4B and Qwen 3.5 4B and 9B, and scores the tools, arguments and answers. The latest run is `evals/one-engine.json` (Windows); the Mac's last is `evals/mac-baseline.json` |
| `Windows/` | A copy of the native Windows app, the prototype's front end. It runs side by side with the Vireo app and keeps its own chats. See [Windows/README.md](Windows/README.md). |
| `Mac/` | The Mac app, the prototype's front end on the Mac (`Mac/App/`). It builds `vireo-server` from this workspace. See [Mac/README.md](Mac/README.md). |

## Chatting with the engine in the prototype app

1. Install the app with `Windows\install-test-build.cmd`. It builds `vireo-server` too
   and installs it beside the app.
2. Open **Vireo Prototype**, then Settings › Your Host › Run a model on this PC: choose a
   Llama, Qwen 3.5 or Gemma 4 `.gguf` file and **Load Model**.

The app starts `vireo-server` with that model (any free port on 127.0.0.1, a token it
accepts without pairing), connects to it, and starts it again whenever the app opens,
until **Stop**. The server ends with the app. The choice is kept in
`%LOCALAPPDATA%\VireoPrototype\local-engine.json`.

`start-server.cmd` runs the server by hand instead (port 3000, paired with the 6-digit
code it prints). The full server (tools, files, skills, the scheduler) is the rest of
phase 3.

## Setting up

Install Rust with [rustup](https://rustup.rs) (the MSVC toolchain), then from this
folder:

```bash
cargo run -p vireo-cli
```

The CUDA work in phase 1 also needs Visual Studio's **Desktop development with C++**
workload and the CUDA Toolkit (13.5 on the development PC). See "Development machine"
in the plan.
