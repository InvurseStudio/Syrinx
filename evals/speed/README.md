# Speed reports

One page per machine comparing Vireo's own engine with llama.cpp on the same GGUF files:
four workloads (writing a reply, writing deep in a long chat, reading a short prompt,
reading a long one; the Windows page also has three whole replies through the servers),
each a bar chart per model, with the machine's specs, a table of
every number and how they were measured. Open a report in any browser; it needs no
server (fonts come from Google Fonts and fall back to system fonts offline).

| Report | Machine | Against |
|---|---|---|
| `windows-rtx5060-2026-11-12.html` (and a dark image of it without the table, `windows-rtx5060-2026-11-12-dark.png`) | ASUS TUF A14, RTX 5060 Laptop (8 GB), Windows 11 | llama.cpp b12262, CUDA build |
| `mac-m5max-2026-11-12.html` | MacBook Pro, M5 Max (40-core GPU, 128 GB), macOS 27.2 | llama.cpp b12228 (Metal) and the shipped Vireo 0.9.0 (MLX) |
| `mac-m5max-2026-11-12-pm.html` | The same Mac, after the afternoon's tuning (PLAN.md, "Twelfth pass") | The same; the Vireo engine and llama.cpp measured again (`prompts-mac/results-2026-11-12-pm.jsonl`), Vireo 0.9 from the morning |
| `windows-rtx5060-2026-11-15.html` (and `windows-rtx5060-2026-11-15.pdf`, the same page in dark, printed) | The same laptop, the GPU held to ~33 W that day | Gemma 4 E2B and E4B with Google's draft models against the chat's guesses alone (PLAN.md, "Fifteenth round"); also the eval's read, write and tool times as tables (`REPORT.tables`) |
| `windows-rtx5060-2026-11-15-llama.html` (and `.pdf`) | The same laptop at full speed (Turbo power plan, restarted) | llama.cpp b12262, the Mac benchmark's way: its three prompt files, read and write speeds at ~34, ~2,300 and 29-34K tokens for the five Gemma and Qwen models, whole replies through both servers for four, GPU memory. From `bench_windows.py` (raw results in `bench-2026-11-15-windows/`), the page built by `bench_report.py` |
| `windows-rtx5060-2026-11-15-tasks.html` (and `.pdf`) | The same laptop at full speed (Turbo power plan) | llama.cpp b12262, the Mac's 14 November benchmark rebuilt (`bench_tasks.py`): Gemma 4 E2B and E4B, Qwen 3.5 4B and 9B through both servers, one run each; a short question, ~3K and a long prompt (~15-26K, two facts planted), a three-turn chat, tool, skill and long-log tool tasks, GPU memory, answers scored. Raw rows, both servers' logs and two superseded runs in `bench-2026-11-15-windows-tasks/`, the page built by `bench_tasks_report.py` (its text in that folder's `report.json`) |
| `windows-rtx5060-2026-11-15-mac-harness.html` (and `.pdf`) | The same laptop at full speed (Turbo power plan), the merged build (7e00a19) | llama.cpp b12262, with the Mac's own harness (`bench-2026-11-13/bench.py`, its Windows changes behind `os.name == "nt"`): Gemma 4 E2B and E4B, Qwen 3.5 4B and 9B, one run each, the Mac's prompts whole in a 65,536-token context (32,768 for Qwen 3.5 9B, its long tool log cut to half), its workloads but the 96K and 128K ones, scored by its `aggregate.py`, the Mac's own percentages for the same models beside them; and Vireo before and after the merge from `bench_tasks.py` run twice. Raw rows, server logs, the context probe and re-checks in `bench-2026-11-15-windows-mac-harness/` (and `bench-2026-11-15-windows-tasks-after-merge/`), the page built by `bench_mac_harness_report.py` (its text in that folder's `report.json`) |
| `windows-rtx5060-2026-11-15-after-changes.html` (and `.pdf`) | The same laptop at full speed (Turbo power plan), 7e00a19 with three uncommitted changes (Qwen's saved state at the system turn's end, sliding-window rings on CUDA, the prompt attention kernel's f16 queries) | llama.cpp b12262, the Mac's own harness run again exactly as that morning (no Mac columns): Vireo against llama.cpp now; Vireo that morning against now, llama.cpp's own drift as the check; the project's eval (`vireo-eval`, 43 cases, four models) on both builds; and the Gemmas at a 131,072-token context with the ~95K `large` prompt. Raw rows, server logs, re-checks and the 131K probe (`probe_131k.py`) in `bench-2026-11-15-windows-after-changes/`, the eval in `evals/windows-cuda-changes.json` and `-before.json`, the page built by `bench_after_changes_report.py` (its text in that folder's `report.json`) |

To print a report to PDF: `msedge --headless=new --no-pdf-header-footer --print-to-pdf=<out.pdf> "file:///<report>.html?theme=dark"` (the 15 November page has the print styles).

## Making one for another machine (the Mac, say)

1. Copy the latest report to `<platform>-<gpu>-<date>.html`, e.g. `mac-m5max-2026-11-13.html`.
2. Edit only the `REPORT` block at the top of its `<script>`; nothing below it needs
   changing:
   - `eyebrow`, `headline`, `lede`: the page's text. `rig`: the spec strip, one entry per
     cell (`label`, `value`, optional `detail`).
   - `labels.other`: what Vireo is compared with (for example "llama.cpp (Metal)" or
     "mlx_vlm"). Every "vs" label, verdict and table header follows it.
   - `models`: the models in the order every number list uses.
   - `panels`: one per workload, with `vireo` and `other` speeds in tokens/s, one per
     model. `before` (Vireo before a tuning round) is optional: leave it out of every
     panel and the markers, legend entry and table columns disappear. Keep or rewrite
     each panel's `note` for that platform.
   - Both reports have a light/dark switch (they follow the system until one is chosen;
     `?theme=dark` or `?theme=light` in the address sets it).
   - `labels.beforeShort` (optional) names the `before` series in tooltips and the table
     (default "Vireo before"); `labels.beforeVerdict: true` gives each panel a second
     verdict against it. The Mac report uses both, with `before` holding the shipped
     Vireo 0.9, and draws `before` as a third (grey) bar instead of a marker, with a
     light/dark switch; copy the Mac report to get these.
   - `method`: how it was measured (may use `<code>`).
3. Add a row to the table above.

The verdict tag on each panel is worked out from the numbers: "Vireo N% faster" when
Vireo is at least 5% ahead on every model, "Level" from 94% up, otherwise "N% of …".

## Measuring

The Windows numbers came from:

- Vireo: `vireo-cli generate <model> <prompt> --file --gpu -n <tokens>` on a 27-token
  story opening (`-n 128`), ~2,100 tokens of `PLAN.md` (`-n 64`), and ~31,000 tokens of
  the repository's docs with the story opening appended (`-n 64`; the opening keeps
  Qwen 3.5 9B writing instead of ending its turn). The prompt and generation speeds are
  the CLI's last line.
- llama.cpp: `llama-bench -ngl 99 -fa 1 -p 2048,31000 -n 128` and `-p 0 -n 64 -d 30000`.
- The three reply panels (added after the afternoon's pass, PLAN.md "Fourteenth round"):
  whole replies through each engine's server at temperature 0, 400 tokens at most, the
  same user message to each (a README excerpt to rewrite as a list, 40 lines of Rust to
  return with a variable renamed, a story to write): `vireo-server` at its defaults
  (`/api/chat/completions`, the speed from its `Reply:` log line) against
  `llama-server -ngl 99 -fa on -c 8192` (`/v1/chat/completions`, `timings`), the second
  of two rounds.

Whole replies one way of running the server against another (a draft model on and off,
say), on any machine: `tools/reply_speed.py --model <gguf> --way "name=server args"` twice
or more, `--out` a JSON file. Four prompts (a document as a list, a code file with an
edit, a story, an explanation), greedy and at the model's own sampling, the ways taking
turns; each row has the writing speed, the guesses taken by source and the prompt's
reading speed. `windows-2026-11-15/` has Gemma 4 E2B and E4B with and without their
draft models (PLAN.md, "Fifteenth round"; the GPU was held to ~33 W that day).

The Windows benchmark against llama.cpp, the Mac's way (`bench_windows.py`, after
`prompts-mac/bench_mac.py`, with its prompt files): `python evals/speed/bench_windows.py
--out evals/speed/bench-<date>-windows`, then `bench_report.py <that folder> <page.html>`
(the page's text is the folder's `report.json`). The Gemma and Qwen models, one round
each, by default (`--reps 2` reverses the engines' order for a second; `--with-llama`
adds Llama 3.2 1B). First check the laptop is at full speed: Gemma 4 E4B reads
`prompts-mac/medium.txt` at ~6,000 tokens/s and writes at ~88 with `vireo-cli generate
--gpu`. On 15 November it ran at a third to a half of that until it was restarted.

Measure on mains power in the fastest power mode, and say which: on the Windows
laptop a slower power state made both engines ~3× slower and shifted the comparison
(PLAN.md, "Thirteenth round"). Don't poll `nvidia-smi` (or `powermetrics`) during a
measured run; it moves the clocks. Run old and new builds alternately, and compare
within a row.

The Mac numbers (`mac-m5max-2026-11-12.html`, and a dark image of it without the table,
`mac-m5max-2026-11-12-dark.png`) came from `prompts-mac/bench_mac.py`, whose raw output is
`prompts-mac/results-2026-11-12.jsonl`. They used fresh prompts, kept in `prompts-mac/`: `short.txt` (34 tokens), `medium.txt` (Vireo 0.9's release notes, ~2,300)
and `long.txt` (~126 KB of the Mac app's Swift source, 27-31K tokens), the long ones
ending with the short one. Every engine ran them with no drafting, one engine and
model loaded at a time (three ~25 GB models loaded together slowed llama.cpp's reading
by half), two rounds with the order reversed:

- Vireo: `vireo-cli generate <model> <prompt> --file --gpu -n <128 short, 64 otherwise>`.
- llama.cpp: `llama-server -ngl 99 -fa on -c 34000`, the prompt to `/completion` with
  `cache_prompt: false`; speeds from the response's `timings`.
- Vireo 0.9: the 0.9.0 DMG's own Python running its `server/mlx_server.py`, with the
  app's environment (`APC_ENABLED=1`, snapshots every 256 tokens, 8 kept), the prompt
  as a user message to `/v1/chat/completions` behind a unique first line (its prompt
  cache survives `/v1/cache/reset`); speeds from the response's `timings`.
