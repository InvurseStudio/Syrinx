# Vireo Prototype

Work on Vireo's own stack: a Rust server and inference engine that would replace the
Python server in `server/` and the inference servers it starts. GGUF models, read
natively. The first version targets Windows with NVIDIA GPUs (CUDA), tested with small
models; the Mac, other GPUs and NPUs come later.

Nothing here is used by the shipping apps.

- [PLAN.md](PLAN.md): the plan, the decisions so far, phases and open questions

## What's here

| Folder | What it is |
|---|---|
| `crates/vireo-engine/` | The inference engine (a library). So far: the GGUF reader, the tokenizers (Llama 3, Qwen 3.5, Gemma 4), and Llama, Qwen 3.5 and Gemma 4 on the CPU and on NVIDIA GPUs (`--gpu`). |
| `tools/` | Scripts that generate engine source, such as the Unicode tables |
| `models/` | Test models (not in git; see `crates/vireo-engine/tests/real_models.rs` for which) |
| `crates/vireo-cli/` | Runs the engine from the command line, for testing and benchmarks |
| `crates/vireo-server/` | A first server: one model on this PC over the part of the app's API that plain chats use (no tools, files or skills yet). The prototype app starts it itself; `start-server.cmd` runs it by hand. It keeps each chat's cache (in `%LOCALAPPDATA%\VireoPrototype\chat-cache`), so going back to a chat reads only what's new |
| `Windows/` | A copy of the native Windows app, the prototype's front end. It runs side by side with the Vireo app and keeps its own chats. See [Windows/README.md](Windows/README.md). |

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
