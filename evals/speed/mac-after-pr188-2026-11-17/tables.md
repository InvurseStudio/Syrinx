## Cards 2, 3, 8: Gemma 4 31B at long context

### Card 2: reading speed (tok/s)

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 573 | 595 |  | +3.7% | 444 | 526 | 2/2, 2/2 |
| 64K | 460 | 478 |  | +3.9% | 333 | 429 | 2/2, 2/2 |
| 128K | 346 | 340 | 345 | -0.2% | 232 | 324 | 2/2, 2/2 |

### Card 3: writing speed after the prompt (tok/s)

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 29.7 | 29.9 |  | +0.4% | 16.0 | 12.7 | 2/2, 2/2 |
| 64K | 24.1 | 26.6 |  | +10.4% | 12.5 | 9.1 | 2/2, 2/2 |
| 128K | 20.3 | 20.0 | 19.8 | -2.7% | 9.7 | 6.0 | 2/2, 2/2 |

### Card 8: peak memory (GB, lower is better)

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 32.3 | 33.3 |  | +3.3% | 30.8 | 37.6 | 2/2, 2/2 |
| 64K | 35.5 | 36.6 |  | +3.0% | 33.6 | 41.9 | 2/2, 2/2 |
| 128K | 42.0 | 43.1 | 43.1 | +2.6% | 39.7 | 54.8 | 2/2, 2/2 |

### Time to first token (s), not on a card

| Context | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | Facts now |
|---|---|---|---|---|---|---|---|
| 32K | 52 | 50 |  | -3.6% | 67 | 57 | 2/2, 2/2 |
| 64K | 133 | 128 |  | -3.8% | 184 | 143 | 2/2, 2/2 |
| 128K | 360 | 367 | 361 | +0.2% | 537 | 385 | 2/2, 2/2 |

## Cards 4, 5, 9: eight models, 131,072-token context

### Card 4: writing after the ~125K-token prompt (tok/s)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Facts (of 2) deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 18.8 | 20.1 |  | +7.1% | 10.1 | 9.2 | 1.98× | 2 → 2 (llama 2, mlx 2) |
| Qwen 3.8 27B | 30.4 | 30.2 |  | -0.8% | 14.6 | 16.4 | 1.84× | 2 → 2 (llama 2, mlx 2) |
| Muse Glimmer 30B | 23.0 | 21.7 | 22.3 | -3.2% | 17.7 | 17.8 | 1.25× | 2 → 2 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 56.7 | 53.1 | 56.7 | +0.0% | 42.5 | 43.6 | 1.30× | 2 → 2 (llama 2, mlx 2) |
| Gemma 4 E4B | 77.9 | 77.9 |  | +0.0% | 50.4 | 39.3 | 1.55× | 2 → 2 (llama 1, mlx 1) |
| Gemma 4 E2B | 119.6 | 109.8 |  | -8.2% | 67.2 | 99.8 | 1.10× | 2 → 2 (llama 2, mlx 2) |
| Qwen 3.5 4B | 63.4 | 59.7 |  | -5.8% | 58.3 | 58.5 | 1.02× | 2 → 2 (llama 2, mlx 2) |
| Qwen 3.5 9B | 46.0 | 61.3 |  | +33.1% | 43.0 | 43.7 | 1.40× | 2 → 2 (llama 2, mlx 2) |

### Card 5: tool task, total time (s, lower is better)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Right deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 30.1 | 30.1 |  | -0.2% | 57.9 | 57.6 | 1.92× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.8 27B | 31.5 | 31.0 |  | -1.6% | 53.8 | 56.8 | 1.74× | 1 → 1 (llama 1, mlx 1) |
| Muse Glimmer 30B | 83.3 | 47.3 |  | -43.2% | 121.8 | 129.3 | 2.57× | 0 → 1 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 8.9 | 8.7 |  | -2.8% | 12.7 | 13.2 | 1.47× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E4B | 7.4 | 7.3 |  | -1.7% | 10.7 | 15.2 | 1.47× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E2B | 4.0 | 4.3 |  | +8.1% | 6.9 | 8.1 | 1.59× | 0 → 1 (llama 1, mlx 1) |
| Qwen 3.5 4B | 7.1 | 7.3 |  | +3.0% | 10.3 | 11.9 | 1.41× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.5 9B | 11.1 | 10.0 |  | -9.8% | 18.3 | 18.6 | 1.84× | 1 → 1 (llama 1, mlx 1) |

### Skill task, total time (s), the other agent task

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Right deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 28.2 | 27.7 |  | -1.7% | 59.5 | 60.3 | 2.15× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.8 27B | 30.6 | 30.2 |  | -1.4% | 55.1 | 55.2 | 1.82× | 1 → 1 (llama 1, mlx 1) |
| Muse Glimmer 30B | 79.3 | 68.5 |  | -13.5% | 176.0 | 143.8 | 2.10× | 1 → 1 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 10.2 | 10.2 |  | -0.0% | 14.0 | 13.4 | 1.31× | 1 → 1 (llama 0, mlx 0) |
| Gemma 4 E4B | 7.5 | 7.3 |  | -1.8% | 12.8 | 12.2 | 1.66× | 0 → 1 (llama 1, mlx 1) |
| Gemma 4 E2B | 2.5 | 0.9 |  | -63.6% | 8.0 | 8.6 | 8.99× | 0 → 0 (llama 0, mlx 0) |
| Qwen 3.5 4B | 7.1 | 7.0 |  | -2.0% | 11.0 | 11.3 | 1.57× | 0 → 0 (llama 0, mlx 0) |
| Qwen 3.5 9B | 12.5 | 12.2 |  | -2.1% | 24.3 | 25.2 | 1.99× | 1 → 1 (llama 1, mlx 1) |

### Long tool task, total time (s)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other | Right deck → now |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 130.7 | 134.6 |  | +2.9% | 197.0 | 295.6 | 1.46× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.8 27B | 115.5 | 131.4 |  | +13.7% | 151.5 | 117.6 | 0.90× | 1 → 1 (llama 0, mlx 1) |
| Muse Glimmer 30B | 107.7 | 132.7 |  | +23.2% | 95.5 | 82.7 | 0.62× | 1 → 1 (llama 0, mlx 0) |
| Gemma 4 26B-A4B | 27.4 | 28.6 |  | +4.3% | 38.2 | 48.7 | 1.34× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E4B | 11.1 | 11.3 |  | +1.0% | 17.4 | 13.6 | 1.21× | 1 → 1 (llama 1, mlx 1) |
| Gemma 4 E2B | 4.2 | 4.7 |  | +11.8% | 8.7 | 5.4 | 1.14× | 1 → 0 (llama 1, mlx 1) |
| Qwen 3.5 4B | 22.5 | 23.3 |  | +3.5% | 31.3 | 23.5 | 1.00× | 1 → 1 (llama 1, mlx 1) |
| Qwen 3.5 9B | 30.6 | 30.0 |  | -2.2% | 52.8 | 33.1 | 1.10× | 1 → 1 (llama 1, mlx 1) |

### Card 9: writing a short reply (tok/s)

| Model | Deck (new engine) | Main run | Rerun after cool-down | Change vs deck (rerun where run) | llama.cpp (stored) | mlx-vlm (stored) | New vs better other |
|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 31.6 | 27.9 | 33.7 | +6.6% | 17.9 | 18.4 | 1.84× |
| Qwen 3.8 27B | 30.8 | 27.2 | 34.5 | +12.0% | 20.1 | 22.6 | 1.53× |
| Muse Glimmer 30B | 25.3 | 24.3 | 25.7 | +1.8% | 21.2 | 21.0 | 1.21× |
| Gemma 4 26B-A4B | 92.1 | 87.6 |  | -4.9% | 94.8 | 112.7 | 0.78× |
| Gemma 4 E4B | 98.3 | 98.3 |  | +0.0% | 95.8 | 105.8 | 0.93× |
| Gemma 4 E2B | 176.2 | 176.2 |  | +0.0% | 153.0 | 177.4 | 0.99× |
| Qwen 3.5 4B | 119.4 | 116.7 |  | -2.3% | 110.9 | 109.3 | 1.05× |
| Qwen 3.5 9B | 67.4 | 65.8 |  | -2.3% | 65.0 | 71.3 | 0.92× |

### Card 6: 64-turn chat, Gemma 4 31B, 32K context

| | Deck (new engine) | Now | Change | llama.cpp (stored) |
|---|---|---|---|---|
| Whole chat (s) | 319.9 | 259.6 | -18.8% | 1212.0 |
| Full re-reads | 2 | 1 | | 12 (its client drops old turns; per the deck) |
| Answers right | 60/63 | 60/63 | | 62/63 |
| Turns / errors | 64 / 0 | 64 / 0 | | 63 / 0 |
| vs llama.cpp | 3.79× | 4.67× | | |

### Card 7: same answers (Gemma 4 E2B)

| Check | Deck | Now | Notes |
|---|---|---|---|
| Perplexity, Vireo vs llama.cpp | 120.78 vs 120.76 (0.01%) | 120.78 vs 120.76 (0.01%) | llama-perplexity rerun (b12228) |
| Greedy 256-token continuations identical | 19/20 | 19/20 | median words agreeing 109.5, min 7 |
| Planted facts, 32K context | 5/5 | 5/5 | read 19044 tok/s, write 172.9 tok/s |
| Planted facts, 128K context | 5/5 | 5/5 | read 7535 tok/s, write 118.0 tok/s |
| Gemma 4 31B facts at every size | 2/2 to 256K | 32K 2/2, 2/2, 64K 2/2, 2/2, 128K 2/2 (256K not run) | card 2's run |
