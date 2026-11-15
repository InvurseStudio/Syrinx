# Q6 benchmark rerun with 128K context, 2026-11-14

Build: branch `prototype/q6-followups` (uncommitted overnight changes), release `vireo-server` built 2026-11-14 01:12. One run per model and engine (REPS=1), all models at Q6 (GGUF Q6_K for Vireo and llama.cpp, 6-bit MLX for mlx-vlm), 131,072-token context, MacBook Pro M5 Max. Rows: `results-q6-128k.jsonl`; numbers read with `report.py`'s definitions.

## Summary

- **Ahead of both:** writing after the 128K prompt on all 8 models (+5% to +104% vs mlx-vlm, +7% to +108% vs llama.cpp); tool task time on all 8 (+48% to +105% vs mlx-vlm, +43% to +92% vs llama.cpp); skill task on every model where the other engine finished (+31% to +244%).
- **Ahead of llama.cpp, behind mlx-vlm at reading:** Vireo reads +23% to +78% faster than llama.cpp at 32K/96K/128K on all 8, but slower than mlx-vlm at 128K on 6 of 8 (−1% to −17%; ahead only on Gemma 4 31B +14% and Muse +9%) and at 3K on 6 of 8 (down to −28%, Gemma 4 26B-A4B).
- **Short-reply writing:** ahead of llama.cpp on 7 of 8; vs mlx-vlm ahead on 4 (31B +72%, Qwen 27B +36%, Muse +20%, Qwen 4B +9%), behind on 4 (26B-A4B −18%, E4B −7%, Qwen 9B −5%, E2B −1%).
- **Memory at 128K:** more than llama.cpp on 7 of 8, level on 26B-A4B (E2B: 9.2 GB vs 6.2); less than mlx-vlm on 6 of 8, more on Qwen 27B (45.1 vs 39.0 GB), level on Qwen 9B.
- **Long tool task:** faster on 6 of 7 vs mlx-vlm and 7 of 8 vs llama.cpp; slower only on Muse, where the others stopped at the token cap with empty answers.
- **Before/after (Vireo):** tool task faster on 7 of 8 (Muse slower, 66.1 → 83.3 s), skill task faster on all 8 (31B 113.2 → 28.2 s), long tool task faster on the 7 with a clean before row (Qwen 27B errored on Nov 13 after 365.4 s; now 115.5 s and right). Scored tasks right went up on E2B (0 → 4), E4B (1 → 5), 26B-A4B, 31B and Qwen 27B (5 → 6), down on Muse (6 → 5).
- **Failed:** Qwen 3.8 27B on mlx-vlm, long tool task (incomplete tool call), and Muse on llama.cpp, skill task (HTTP 500, bad tool-call JSON); each rerun once, same error both times. Everything else ran: 240 rows, 8 models × 3 engines × (load + 9 workloads).

## Per model

Speeds in tokens/s, times in seconds, memory in GB (peak of the server process). `Right`: recall prompts out of 2, multi-turn the fraction of turns right, others 0 or 1.

### Gemma 4 E2B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 176.2 | 153.0 | 177.4 |
| Reading, medium ~3K (tok/s) | 16121 | 16707 | 19907 |
| Reading, long ~32K (tok/s) | 18469 | 11559 | 19298 |
| Reading, large ~96K (tok/s) | 10572 | 5925 | 10568 |
| Reading, xlarge ~125K (tok/s) | 7897 | 4783 | 8775 |
| Writing after xlarge (tok/s) | 119.6 | 67.2 | 99.8 |
| Time to first token, xlarge (s) | 15.95 | 26.1 | 14.68 |
| Tool task total (s) | 4.0 | 6.9 | 8.1 |
| Skill task total (s) | 2.5 | 8.0 | 8.6 |
| Long tool task total (s) | 4.2 | 8.7 | 5.4 |
| Multi-turn total (s) | 2.2 | 1.3 | 2.5 |
| Peak memory, xlarge (GB) | 9.2 | 6.2 | 15.0 |
| Load (s) | 1.52 | 2.03 | 4.08 |
| Right: long (of 2) | 2 | 2 | 2 |
| Right: large (of 2) | 2 | 2 | 2 |
| Right: xlarge (of 2) | 2 | 2 | 2 |
| Right: multiturn (fraction) | 1.0 | 1.0 | 1.0 |
| Right: tools (0/1) | 0 | 1 | 1 |
| Right: skill (0/1) | 0 | 0 | 0 |
| Right: longtools (0/1) | 1 | 1 | 1 |

### Gemma 4 E4B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 98.3 | 95.8 | 105.8 |
| Reading, medium ~3K (tok/s) | 7856 | 6355 | 7913 |
| Reading, long ~32K (tok/s) | 7477 | 5052 | 7264 |
| Reading, large ~96K (tok/s) | 5007 | 3111 | 4499 |
| Reading, xlarge ~125K (tok/s) | 3865 | 2613 | 3901 |
| Writing after xlarge (tok/s) | 77.9 | 50.4 | 39.3 |
| Time to first token, xlarge (s) | 32.54 | 47.73 | 32.43 |
| Tool task total (s) | 7.4 | 10.7 | 15.2 |
| Skill task total (s) | 7.5 | 12.8 | 12.2 |
| Long tool task total (s) | 11.1 | 17.4 | 13.6 |
| Multi-turn total (s) | 1.9 | 2.6 | 3.6 |
| Peak memory, xlarge (GB) | 15.2 | 11.3 | 19.0 |
| Load (s) | 0.71 | 2.02 | 3.07 |
| Right: long (of 2) | 2 | 2 | 2 |
| Right: large (of 2) | 2 | 1 | 1 |
| Right: xlarge (of 2) | 2 | 1 | 1 |
| Right: multiturn (fraction) | 1.0 | 1.0 | 1.0 |
| Right: tools (0/1) | 1 | 1 | 1 |
| Right: skill (0/1) | 0 | 1 | 1 |
| Right: longtools (0/1) | 1 | 1 | 1 |

### Qwen 3.5 4B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 119.4 | 110.9 | 109.3 |
| Reading, medium ~3K (tok/s) | 4640 | 4116 | 5472 |
| Reading, long ~32K (tok/s) | 3966 | 2914 | 4313 |
| Reading, large ~96K (tok/s) | 2440 | 1713 | 2844 |
| Reading, xlarge ~125K (tok/s) | 2035 | 1413 | 2426 |
| Writing after xlarge (tok/s) | 63.4 | 58.3 | 58.5 |
| Time to first token, xlarge (s) | 57.53 | 83.06 | 48.7 |
| Tool task total (s) | 7.1 | 10.3 | 11.9 |
| Skill task total (s) | 7.1 | 11.0 | 11.3 |
| Long tool task total (s) | 22.5 | 31.3 | 23.5 |
| Multi-turn total (s) | 3.5 | 2.9 | 4.5 |
| Peak memory, xlarge (GB) | 14.1 | 12.6 | 25.0 |
| Load (s) | 0.56 | 1.01 | 2.05 |
| Right: long (of 2) | 2 | 2 | 2 |
| Right: large (of 2) | 2 | 2 | 2 |
| Right: xlarge (of 2) | 2 | 2 | 2 |
| Right: multiturn (fraction) | 1.0 | 1.0 | 1.0 |
| Right: tools (0/1) | 1 | 1 | 1 |
| Right: skill (0/1) | 0 | 0 | 0 |
| Right: longtools (0/1) | 1 | 1 | 1 |

### Qwen 3.5 9B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 67.4 | 65.0 | 71.3 |
| Reading, medium ~3K (tok/s) | 2793 | 2462 | 3050 |
| Reading, long ~32K (tok/s) | 2458 | 1818 | 2546 |
| Reading, large ~96K (tok/s) | 1800 | 1266 | 1942 |
| Reading, xlarge ~125K (tok/s) | 1528 | 1085 | 1720 |
| Writing after xlarge (tok/s) | 46.0 | 43.0 | 43.7 |
| Time to first token, xlarge (s) | 76.64 | 108.11 | 68.54 |
| Tool task total (s) | 11.1 | 18.3 | 18.6 |
| Skill task total (s) | 12.5 | 24.3 | 25.2 |
| Long tool task total (s) | 30.6 | 52.8 | 33.1 |
| Multi-turn total (s) | 4.0 | 4.6 | 6.4 |
| Peak memory, xlarge (GB) | 17.1 | 16.1 | 17.0 |
| Load (s) | 2.9 | 2.02 | 3.05 |
| Right: long (of 2) | 2 | 2 | 2 |
| Right: large (of 2) | 2 | 2 | 2 |
| Right: xlarge (of 2) | 2 | 2 | 2 |
| Right: multiturn (fraction) | 1.0 | 1.0 | 1.0 |
| Right: tools (0/1) | 1 | 1 | 1 |
| Right: skill (0/1) | 1 | 1 | 1 |
| Right: longtools (0/1) | 1 | 1 | 1 |

### Gemma 4 26B-A4B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 92.1 | 94.8 | 112.7 |
| Reading, medium ~3K (tok/s) | 3128 | 2970 | 4347 |
| Reading, long ~32K (tok/s) | 2879 | 2123 | 3158 |
| Reading, large ~96K (tok/s) | 1906 | 1331 | 2026 |
| Reading, xlarge ~125K (tok/s) | 1576 | 1105 | 1709 |
| Writing after xlarge (tok/s) | 56.7 | 42.5 | 43.6 |
| Time to first token, xlarge (s) | 79.73 | 112.61 | 73.32 |
| Tool task total (s) | 8.9 | 12.7 | 13.2 |
| Skill task total (s) | 10.2 | 14.0 | 13.4 |
| Long tool task total (s) | 27.4 | 38.2 | 48.7 |
| Multi-turn total (s) | 3.3 | 3.7 | 5.4 |
| Peak memory, xlarge (GB) | 30.1 | 30.0 | 37.0 |
| Load (s) | 12.32 | 2.02 | 4.07 |
| Right: long (of 2) | 2 | 2 | 2 |
| Right: large (of 2) | 2 | 2 | 2 |
| Right: xlarge (of 2) | 2 | 2 | 2 |
| Right: multiturn (fraction) | 1.0 | 1.0 | 1.0 |
| Right: tools (0/1) | 1 | 1 | 1 |
| Right: skill (0/1) | 1 | 0 | 0 |
| Right: longtools (0/1) | 1 | 1 | 1 |

### Gemma 4 31B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 31.6 | 17.9 | 18.4 |
| Reading, medium ~3K (tok/s) | 725 | 524 | 590 |
| Reading, long ~32K (tok/s) | 579 | 407 | 472 |
| Reading, large ~96K (tok/s) | 402 | 270 | 339 |
| Reading, xlarge ~125K (tok/s) | 342 | 231 | 301 |
| Writing after xlarge (tok/s) | 18.8 | 10.1 | 9.2 |
| Time to first token, xlarge (s) | 367.28 | 537.39 | 414.24 |
| Tool task total (s) | 30.1 | 57.9 | 57.6 |
| Skill task total (s) | 28.2 | 59.5 | 60.3 |
| Long tool task total (s) | 130.7 | 197.0 | 295.6 |
| Multi-turn total (s) | 14.4 | 18.8 | 25.1 |
| Peak memory, xlarge (GB) | 52.1 | 44.0 | 76.0 |
| Load (s) | 20.89 | 2.02 | 5.05 |
| Right: long (of 2) | 2 | 2 | 2 |
| Right: large (of 2) | 2 | 2 | 2 |
| Right: xlarge (of 2) | 2 | 2 | 2 |
| Right: multiturn (fraction) | 1.0 | 1.0 | 1.0 |
| Right: tools (0/1) | 1 | 1 | 1 |
| Right: skill (0/1) | 1 | 1 | 1 |
| Right: longtools (0/1) | 1 | 1 | 1 |

### Qwen 3.8 27B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 30.8 | 20.1 | 22.6 |
| Reading, medium ~3K (tok/s) | 809 | 694 | 851 |
| Reading, long ~32K (tok/s) | 705 | 545 | 699 |
| Reading, large ~96K (tok/s) | 490 | 387 | 543 |
| Reading, xlarge ~125K (tok/s) | 422 | 344 | 509 |
| Writing after xlarge (tok/s) | 30.4 | 14.6 | 16.4 |
| Time to first token, xlarge (s) | 277.16 | 340.5 | 230.73 |
| Tool task total (s) | 31.5 | 53.8 | 56.8 |
| Skill task total (s) | 30.6 | 55.1 | 55.2 |
| Long tool task total (s) | 115.5 | 151.5 | 117.6 (task errored) |
| Multi-turn total (s) | 9.8 | 14.1 | 14.3 |
| Peak memory, xlarge (GB) | 45.1 | 40.0 | 39.0 |
| Load (s) | 13.45 | 3.03 | 4.08 |
| Right: long (of 2) | 2 | 2 | 2 |
| Right: large (of 2) | 2 | 2 | 2 |
| Right: xlarge (of 2) | 2 | 2 | 2 |
| Right: multiturn (fraction) | 1.0 | 1.0 | 1.0 |
| Right: tools (0/1) | 1 | 1 | 1 |
| Right: skill (0/1) | 1 | 1 | 1 |
| Right: longtools (0/1) | 1 | 0 | 1 (task errored) |

### Muse Glimmer 30B

| | Vireo | llama.cpp | mlx-vlm |
|---|---|---|---|
| Writing, short reply (tok/s) | 25.3 | 21.2 | 21.0 |
| Reading, medium ~3K (tok/s) | 881 | 678 | 815 |
| Reading, long ~32K (tok/s) | 786 | 575 | 697 |
| Reading, large ~96K (tok/s) | 673 | 477 | 607 |
| Reading, xlarge ~125K (tok/s) | 620 | 448 | 569 |
| Writing after xlarge (tok/s) | 23.0 | 17.7 | 17.8 |
| Time to first token, xlarge (s) | 184.53 | 238.59 | 188.17 |
| Tool task total (s) | 83.3 | 121.8 | 129.3 |
| Skill task total (s) | 79.3 | 176.0 (task errored) | 143.8 |
| Long tool task total (s) | 107.7 | 95.5 | 82.7 |
| Multi-turn total (s) | 41.0 | 17.4 | 28.0 |
| Peak memory, xlarge (GB) | 28.1 | 26.5 | 33.0 |
| Load (s) | 13.09 | 2.03 | 4.06 |
| Right: long (of 2) | 2 | 0 | 0 |
| Right: large (of 2) | 2 | 0 | 0 |
| Right: xlarge (of 2) | 2 | 0 | 0 |
| Right: multiturn (fraction) | 1.0 | 0.0 | 0.25 |
| Right: tools (0/1) | 0 | 0 | 0 |
| Right: skill (0/1) | 1 | 0 (task errored) | 0 |
| Right: longtools (0/1) | 1 | 0 | 0 |

## Vireo before/after (Nov 13 merged build → this build)

Before: `results-q6-merged-build.jsonl`. Scored tasks right = fully right of the six (32K and 96K recall 2/2, multi-turn all turns, tool, skill, long tool).

| Model | Short writing | Medium read | Long read | Large read | Tool task (s) | Skill task (s) | Long tool task (s) | Right of 6 |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | 177 → 176 | 18041 → 16121 | 18348 → 18469 | 10143 → 10572 | 6.9 → 4.0 | 7.2 → 2.5 | 6.9 → 4.2 | 0 → 4 |
| Gemma 4 E4B | 87.1 → 98.3 | 7302 → 7856 | 5892 → 7477 | 3946 → 5007 | 14.1 → 7.4 | 15.8 → 7.5 | 20.5 → 11.1 | 1 → 5 |
| Qwen 3.5 4B | 114 → 119 | 4568 → 4640 | 3337 → 3966 | 2113 → 2440 | 9.5 → 7.1 | 11.1 → 7.1 | 29.4 → 22.5 | 5 → 5 |
| Qwen 3.5 9B | 62.8 → 67.4 | 2791 → 2793 | 2242 → 2458 | 1666 → 1800 | 16.0 → 11.1 | 20.1 → 12.5 | 31.3 → 30.6 | 6 → 6 |
| Gemma 4 26B-A4B | 94.8 → 92.1 | 3161 → 3128 | 2938 → 2879 | 1829 → 1906 | 11.6 → 8.9 | 13.8 → 10.2 | 30.1 → 27.4 | 5 → 6 |
| Gemma 4 31B | 28.8 → 31.6 | 688 → 725 | 533 → 579 | 340 → 402 | 56.2 → 30.1 | 113 → 28.2 | 144 → 131 | 5 → 6 |
| Qwen 3.8 27B | 28.9 → 30.8 | 750 → 809 | 630 → 705 | 446 → 490 | 56.0 → 31.5 | 59.5 → 30.6 | error → 116 | 5 → 6 |
| Muse Glimmer 30B | 22.2 → 25.3 | 837 → 881 | 716 → 786 | 589 → 673 | 66.1 → 83.3 | 92.1 → 79.3 | 120 → 108 | 6 → 5 |

## Vireo's difference from mlx-vlm and llama.cpp

Positive = Vireo better. Speeds: Vireo ÷ other − 1. Times and memory: other ÷ Vireo − 1.

**vs mlx-vlm**

| Model | Short reply writing | Reading ~3K | Reading ~32K | Reading ~96K | Reading ~128K | Writing after 128K | Tool task time | Skill task time | Long tool task time | Peak memory 128K |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | -1% | -19% | -4% | 0% | -10% | +20% | +102% | +244% | +29% | +63% |
| Gemma 4 E4B | -7% | -1% | +3% | +11% | -1% | +98% | +105% | +63% | +23% | +25% |
| Qwen 3.5 4B | +9% | -15% | -8% | -14% | -16% | +8% | +68% | +59% | +4% | +77% |
| Qwen 3.5 9B | -5% | -8% | -3% | -7% | -11% | +5% | +68% | +102% | +8% | -1% |
| Gemma 4 26B-A4B | -18% | -28% | -9% | -6% | -8% | +30% | +48% | +31% | +78% | +23% |
| Gemma 4 31B | +72% | +23% | +23% | +19% | +14% | +104% | +91% | +114% | +126% | +46% |
| Qwen 3.8 27B | +36% | -5% | +1% | -10% | -17% | +85% | +80% | +80% | – | -14% |
| Muse Glimmer 30B | +20% | +8% | +13% | +11% | +9% | +29% | +55% | +81% | -23% | +17% |

**vs llama.cpp**

| Model | Short reply writing | Reading ~3K | Reading ~32K | Reading ~96K | Reading ~128K | Writing after 128K | Tool task time | Skill task time | Long tool task time | Peak memory 128K |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | +15% | -4% | +60% | +78% | +65% | +78% | +73% | +220% | +107% | -33% |
| Gemma 4 E4B | +3% | +24% | +48% | +61% | +48% | +55% | +45% | +71% | +57% | -26% |
| Qwen 3.5 4B | +8% | +13% | +36% | +42% | +44% | +9% | +45% | +55% | +39% | -11% |
| Qwen 3.5 9B | +4% | +13% | +35% | +42% | +41% | +7% | +65% | +94% | +73% | -6% |
| Gemma 4 26B-A4B | -3% | +5% | +36% | +43% | +43% | +33% | +43% | +37% | +39% | 0% |
| Gemma 4 31B | +77% | +38% | +42% | +49% | +48% | +86% | +92% | +111% | +51% | -16% |
| Qwen 3.8 27B | +53% | +17% | +29% | +27% | +23% | +108% | +71% | +80% | +31% | -11% |
| Muse Glimmer 30B | +19% | +30% | +37% | +41% | +38% | +30% | +46% | – | -11% | -6% |

## Errors and oddities

1. **Qwen 3.8 27B · mlx-vlm · long tool task**: "Generation ended with an incomplete tool call." Rerun once (mlx-vlm only): same error. Same as on Nov 13. The scorer marks it right because the broken tool-call text names inventory-sync; it is not counted as right here, and its time is shown as missing.
2. **Muse Glimmer 30B · llama.cpp · skill task**: HTTP 500 "Failed to parse tool call arguments as JSON … missing closing quote" on the model's write_file call. Rerun once (llama.cpp only): same error. Shown as missing.
3. **Muse Glimmer 30B · llama.cpp and mlx-vlm · every workload** but llama.cpp's tool task: empty answers. Each reply used its whole token cap (256, 128, 96 or 512) on reasoning, so their 0 scores and their shorter long-tool times are not like-for-like. Vireo's Muse also reasons despite thinking off (140–230 reasoning tokens a reply in its log), which likely explains its short-reply first token of 4.9 s against 0.5 s.
4. **Gemma 4 E2B · Vireo · skill task**: said the report was saved but never called write_file (nothing written) and gave 48 hours (right: 78). The tool task gave the quarters but not the year's total. Both right on llama.cpp and mlx-vlm for the tool task.
5. **Gemma 4 E4B · Vireo · skill task**: wrote the file but gave 82 hours. **Qwen 3.5 4B · all three engines · skill task**: wrong totals (67, 72, 72). **Muse · all three · tool task**: scored wrong (the year's total, $26,742.25, not in the answer).
6. **Vireo's tool-task rows** carry one server stats line for the whole task (llama.cpp and mlx-vlm one per round trip), so their read and write speeds are not comparable; the whole-task times are.
7. **Vireo load times** with a draft model: 26B-A4B 12.3 s (3.1 s on Nov 13), 31B 20.9 s, Qwen 27B 13.5 s, Muse 13.1 s, against 2–5 s on the other engines.
8. **Gemma 4 31B · Vireo**: writing drops from 30.7 tok/s after the 96K prompt to 18.8 after the 128K one (still ahead of 10.1 and 9.2). Vireo's peak memory then climbs to 61.1 GB in later tasks.
9. **Gemma 4 E2B · Vireo**: reading the ~3K prompt 16,121 tok/s, down from 18,041 on Nov 13 and below mlx-vlm's 19,907.
