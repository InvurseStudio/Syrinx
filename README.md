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
| `Windows/` | A copy of the native Windows app, the prototype's front end. It runs side by side with the Vireo app and keeps its own chats. See [Windows/README.md](Windows/README.md). |

The server (`crates/vireo-server`) joins the workspace in phase 3.

## Setting up

Install Rust with [rustup](https://rustup.rs) (the MSVC toolchain), then from this
folder:

```bash
cargo run -p vireo-cli
```

The CUDA work in phase 1 also needs Visual Studio's **Desktop development with C++**
workload and the CUDA Toolkit (13.5 on the development PC). See "Development machine"
in the plan.
