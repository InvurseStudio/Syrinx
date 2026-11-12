# Speed reports

One page per machine comparing Vireo's own engine with llama.cpp on the same GGUF files:
four workloads (writing a reply, writing deep in a long chat, reading a short prompt,
reading a long one), each a bar chart per model, with the machine's specs, a table of
every number and how they were measured. Open a report in any browser; it needs no
server (fonts come from Google Fonts and fall back to system fonts offline).

| Report | Machine | Against |
|---|---|---|
| `windows-rtx5060-2026-11-12.html` | ASUS TUF A14, RTX 5060 Laptop (8 GB), Windows 11 | llama.cpp b12262, CUDA build |

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

Measure on mains power in the fastest power mode, and say which: on the Windows
laptop a slower power state made both engines ~3× slower and shifted the comparison
(PLAN.md, "Thirteenth round"). Don't poll `nvidia-smi` (or `powermetrics`) during a
measured run; it moves the clocks. Run old and new builds alternately, and compare
within a row.
