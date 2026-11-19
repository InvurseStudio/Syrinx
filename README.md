# Syrinx

**A local inference engine and agent server, written from scratch in Rust.**

Follow along on X: [@InvurseNorth](https://x.com/InvurseNorth)

Syrinx runs open-weight language models on your own machine (Metal on Apple silicon,
CUDA on NVIDIA) and serves them to [Vireo](https://invursestudio.com), a private AI
assistant that works with your files, calendar and apps without sending anything to the
cloud. It reads GGUF models directly, with its own GPU kernels, tokenizers, sampler and
tool-calling grammar. It doesn't wrap llama.cpp or MLX.

> In a syrinx, a songbird's voice box, two sides sing at once. A vireo is a songbird.

---

## Highlights

- **Fast long context.** Gemma 4 31B reads a 256K-token prompt and still remembers the
  facts planted in it, in 57 GB of memory.
- **Built for agents.** Tool calls are written under a grammar in each model's own format,
  so they always parse. Tools that change something wait for the user's approval, and the
  server checks the reply for changes it claims but never made.
- **Several people at once.** One Mac serves 64 chats together, about 250 tokens a second
  in total, with each reply starting within about 6 seconds.
- **Speculative decoding where it pays.** Draft models (Gemma 4 E2B for 31B, or Google's
  draft models), Qwen's own multi-token-prediction layer, and guesses taken from the chat
  itself. Every token is still the model's own pick.
- **Chats pick up where they left off.** Each chat's cache is saved, so going back to a
  conversation reads only what's new.
- **One codebase, two GPUs.** The same model code runs on Metal and CUDA. Only the
  backend and its tuning differ.

## Performance

Measured on 19 November 2026 on a MacBook Pro with M5 Max and 128 GB of memory, with
Q6_K models, on `main`.

### Long context: Gemma 4 31B

| Context | Reading | Time to first token | Writing | Peak memory |
|---|---:|---:|---:|---:|
| 32K | 629 tok/s | 47 s | 28.8 tok/s | 33.3 GB |
| 64K | 460 tok/s | 2.2 min | 23.3 tok/s | 36.6 GB |
| 128K | 300 tok/s | 6.9 min | 18.6 tok/s | 43.1 GB |
| 256K | 172 tok/s | 24.6 min | 15.0 tok/s | 57.1 GB |

At every size the model recalled both facts planted deep in the prompt.

### Writing after 125K tokens of context

| Model | Writing speed |
|---|---:|
| Gemma 4 E2B | 116.1 tok/s |
| Gemma 4 E4B | 69.6 tok/s |
| Qwen 3.5 9B | 56.2 tok/s |
| Qwen 3.5 4B | 55.8 tok/s |
| Gemma 4 26B-A4B (MoE) | 48.1 tok/s |
| Qwen 3.8 27B | 27.8 tok/s |
| Muse Glimmer 30B | 19.7 tok/s |
| Gemma 4 31B | 16.9 tok/s |

All eight recalled both planted facts.

### A multi-step agent task

The whole task, start to finish: finding a sales file, reading it, and working out each
quarter's revenue and the year's total with the calculator tool.

| Model | Total time |
|---|---:|
| Gemma 4 E2B | 4.3 s |
| Gemma 4 E4B | 6.6 s |
| Qwen 3.5 4B | 6.6 s |
| Gemma 4 26B-A4B | 9.4 s |
| Qwen 3.5 9B | 15.2 s |
| Gemma 4 31B | 27.8 s |
| Qwen 3.8 27B | 32.5 s |
| Muse Glimmer 30B | 49.7 s |

### Short replies, large models

| Model | Writing speed |
|---|---:|
| Qwen 3.8 27B | 35.6 tok/s |
| Gemma 4 31B | 35.4 tok/s |
| Muse Glimmer 30B | 24.7 tok/s |

### A long conversation

A 64-turn chat with Gemma 4 31B in a 32K context, which fills and overflows the context
as it goes: **242 seconds** in all, with 60 of 63 answers right.

### Many chats at once

Gemma 4 26B-A4B, 200-token replies to 3,000-token chats:

| People chatting | Total writing | Each person | Longest wait for a first token |
|---|---:|---:|---:|
| 1 at a time | 88.9 tok/s | 91–94 tok/s | |
| 8 | 206.8 tok/s | 26–29 tok/s | 0.7 s |
| 16 | 224.6 tok/s | 14–16 tok/s | 1.4 s |
| 32 | 240.9 tok/s | 7.6–8.5 tok/s | 3.1 s |
| 48 | 250.4 tok/s | 5.2–6.0 tok/s | 4.6 s |
| 64 (16K context each) | 253.7 tok/s | 4.0–4.5 tok/s | 6.2 s |

Gemma 4 31B: 4 people together write 42.9 tok/s in total (first token within 0.9 s), and
8 people 70.1 tok/s (within 1.9 s).

<sub>The long conversation and many-chats runs use the opt-in 8-bit KV cache
(`VIREO_KV_Q8=1`). All runs use 8-bit middle-layer feed-forward weights (`VIREO_I8=ffn`).
"1 at a time" is eight people taking turns in one slot.</sub>

## Models

| Family | Tested sizes |
|---|---|
| Gemma 4 | E2B, E4B, 26B-A4B (mixture of experts), 31B |
| Qwen 3.5 / 3.8 | 0.8B, 4B, 9B, 27B (with multi-token prediction) |
| Llama 3 | 3.2 1B |
| Muse Glimmer | 30B |

Models are GGUF files: tested with Q4_0, Q4_K_M and Q6_K. Thinking is supported for
Gemma 4, Qwen and Muse, with the thoughts streamed apart from the answer.

## What's inside

| Crate | What it is |
|---|---|
| [`syrinx-engine`](crates/syrinx-engine) | The engine: GGUF reader, tokenizers, grammar-constrained sampling, and the models on the CPU, Metal and CUDA |
| [`syrinx-server`](crates/syrinx-server) | The agent server: chats over HTTP with streamed replies, tools (files, Git, Office documents, PDFs, calendar, the web, shell), approvals, skills, memory, scheduled tasks, attachments, and several chats at once |
| [`syrinx-cli`](crates/syrinx-cli) | Runs a model from the command line, for testing and benchmarks |
| [`syrinx-eval`](crates/syrinx-eval) | Scores how well each model uses the tools and skills, case by case |

## Building

You need Rust ([rustup](https://rustup.rs)). On Windows, CUDA also needs the CUDA
Toolkit and Visual Studio's C++ workload.

```bash
cargo build --release
./target/release/syrinx-server path/to/model.gguf --port 3000
```

`syrinx-server --help` lists the options: context size, draft models, chat slots,
thinking and more. [DEVELOPMENT.md](DEVELOPMENT.md) covers the layout, the tests and the
eval.

## Use it from other apps

`syrinx-server` speaks the OpenAI API, so tools built for OpenAI, LM Studio or
llama.cpp's server can use it. Start it with the API on:

```bash
./target/release/syrinx-server path/to/model.gguf --port 8080 --openai on
```

Then point any OpenAI client at `http://127.0.0.1:8080/v1`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "local",
  "messages": [{"role": "user", "content": "Write a haiku about the north."}],
  "stream": true
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="local", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

- **Endpoints:** `GET /v1/models` and `POST /v1/chat/completions`, whole or streamed.
- **What carries across:** sampling settings (`temperature`, `top_p`, `top_k`, `min_p`,
  `presence_penalty`), `max_tokens`, up to four `stop` strings, and usage with cached
  prompt tokens.
- **Thinking:** with `"reasoning_effort": "low"` or higher, a model that thinks sends its
  thoughts as `reasoning_content`, apart from the answer.
- **Not yet:** tool calls, images and JSON mode.
- **Access:** the server listens only on this computer, and needs no API key unless
  Vireo's settings ask for one.

## Run it on its own

`--standalone` runs Syrinx without Vireo: the OpenAI API (always on), a chat page for your
browser, and a statistics dashboard, on port 8088. No pairing and no tools; its chats'
caches are kept in Syrinx's own folder (`%LOCALAPPDATA%\Syrinx`, or
`~/Library/Application Support/Syrinx` on the Mac), apart from Vireo's.

```bash
./target/release/syrinx-server --standalone path/to/model.gguf
```

On Windows, `serve.cmd path\to\model.gguf` builds it and starts it. Then:

| Address | What it is |
|---|---|
| <http://127.0.0.1:8088/> | Chat in your browser, in Vireo's look |
| `http://127.0.0.1:8088/v1` | The OpenAI API, as above |
| <http://127.0.0.1:8088/stats> | Statistics: requests, speed, the prompt cache, and which apps are connected |
| `http://127.0.0.1:8088/health` | Whether it's up, and which model on which GPU |

It listens only on this computer unless you give `--host` (for example `0.0.0.0` for your
network), and that needs an API key: `--api-key <key>` or `SYRINX_API_KEY`, which callers
send as a bearer token.

### The Windows app

`windows/SyrinxManager` is a Windows app that runs the server for you, in Vireo's look:

- **Server:** start and stop it, with the port, network access, API key, context length,
  thinking and chat slots, and the server's output.
- **Models:** Vireo's model manager. Suggested small models and a Hugging Face search that
  shows only repositories whose architecture Syrinx runs (as Hugging Face read it from
  their GGUF files), with resumable downloads checked against Hugging Face's SHA-256, and
  each file on this PC checked against what the engine reads.
- **Connect:** the base URL, key and model name other chat apps need, with examples.
- **Statistics:** the dashboard above.

Build the server first, then run the app:

```bash
cargo build --release -p syrinx-server
dotnet run -c Release --project windows/SyrinxManager
```

It finds `syrinx-server.exe` beside itself or in this checkout's `target/release`, and
keeps models in `%LOCALAPPDATA%\Syrinx\Models` and any folders you add.

## Status

Syrinx is young and moving fast. It runs every chat in the Vireo-Proto apps for Mac and
Windows, and will replace Vireo's current Python server and the inference runtimes it
starts. Other GPUs and NPUs come later.

Progress, benchmarks and what's next are posted on X: [@InvurseNorth](https://x.com/InvurseNorth).

## License

Copyright © 2026 Invurse Studio. All rights reserved.
