# Models the tools eval runs against

The models aren't in the repo (`VireoPrototype/.gitignore` ignores `/models/`; the
largest is 5.7 GB). Download these into `VireoPrototype/models/` under exactly these
names, and `vireo-eval` finds them by name. Check each with `shasum -a 256` (macOS) or
`sha256sum`: a different file under the same name gives different scores.

The eval's four models (`FULL_MODELS` in `crates/vireo-eval/src/main.rs`). A plain run
uses two of them, Gemma 4 E2B and Qwen 3.5 4B (`QUICK_MODELS`, about 6 minutes for every
case on the RTX 5060), as the check while developing; `--full` runs all four (about 18
minutes), for in-depth testing and the results kept here before a pull request:

| File | Model | Published by | Size (bytes) | SHA-256 |
|---|---|---|---|---|
| `gemma-4-E2B_q4_0-it.gguf` | Gemma 4 E2B instruct, QAT Q4_0 | Google | 3,349,516,256 | `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634` |
| `gemma-4-E4B_q4_0-it.gguf` | Gemma 4 E4B instruct, QAT Q4_0 | Google | 5,154,941,280 | `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee` |
| `Qwen3.5-4B-Q4_K_M.gguf` | Qwen 3.5 4B, Q4_K_M | unsloth | 2,740,937,888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| `Qwen3.5-9B-Q4_K_M.gguf` | Qwen 3.5 9B, Q4_K_M | unsloth | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |

On request only (`--models Llama-3.2-1B-Instruct-Q4_K_M.gguf`, and so on). Both are still
the engine's own test models; Qwen 3.5 0.8B left the defaults on 2026-11-11, too small to
use the tools well (earlier results with it stay in the older `evals/*.json`):

| File | Model | Published by | Size (bytes) | SHA-256 |
|---|---|---|---|---|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | Llama 3.2 1B Instruct, Q4_K_M | bartowski | 807,694,464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | Qwen 3.5 0.8B, Q4_K_M | unsloth | 532,517,120 | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |

All four together are about 17 GB; the short run's two, about 6 GB.

## Running it

From `VireoPrototype/`:

```bash
cargo run --release -p vireo-eval -- --out evals/latest.json
```

`--models a.gguf,b.gguf` runs only some; `--cases read,git` only some cases, and `--cases more` (or easy, hard, changes, skills) one category: while adding a tool or skill, its own cases are the check, about a minute. Compare
against `evals/more-tools.json` (and `run-more-tools.txt`, the same run as printed), the
latest full run, all 39 cases on the four models; `evals/baseline.json` is the first,
which the earlier switches were measured against. Replies
are greedy, so the same model on the same build should give the same scores; on the Mac
the GPU backend differs, so small differences in wording are possible.
