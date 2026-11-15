# Nov 13 benchmark summary

## Gemma 4 E2B

| Workload | Metric | Vireo | Vireo (no drafting) | llama.cpp | mlx-vlm |
|---|---|---|---|---|---|
| short | ttft | 0.013 | 0.017 | 0.03 | 0.188 |
| short | gen_tps | 213 | 212 | 183 | 168 |
| medium | ttft | 0.139 | 0.139 | 0.177 | 0.317 |
| medium | read_tps | 2.19e+04 | 2.19e+04 | 1.78e+04 | 2.01e+04 |
| medium | gen_tps | 216 | 206 | 176 | 156 |
| long | ttft | 1.94 | 1.93 | 2.78 | 1.86 |
| long | read_tps | 1.85e+04 | 1.86e+04 | 1.18e+04 | 1.97e+04 |
| long | gen_tps | 183 | 168 | 132 | 137 |
| long | right | 0 | 0 | 1 | 1 |
| large | prompt_tokens | 95954 | 9.6e+04 | 94706 | 94706 |
| large | ttft | 10.3 | 10.8 | 15.9 | 8.38 |
| large | read_tps | 9.32e+03 | 8.9e+03 | 5.99e+03 | 1.18e+04 |
| large | gen_tps | 140 | 125 | 84.9 | 108 |
| large | right | 0 | 0 | 1 | 1 |
| xlarge | prompt_tokens |  |  |  |  |
| xlarge | ttft |  |  |  |  |
| xlarge | read_tps |  |  |  |  |
| xlarge | gen_tps |  |  |  |  |
| xlarge | right |  |  |  |  |
| multiturn | ttft | 0.112 | 0.122 | 0.046 | 0.232 |
| multiturn | read_per_turn | 578 | 575 | 489 | 505 |
| multiturn | right | 0.812 | 0.75 | 0.875 | 0.875 |
| tools | total | 5.46 | 6.09 | 5.32 | 7.59 |
| tools | requests | 1 | 1 | 7 | 8 |
| tools | calls | 6 | 6 | 6 | 7 |
| tools | right | 0.5 | 0.5 | 1 | 0.5 |
| skill | total | 5.97 | 6.29 | 5.44 | 9.11 |
| skill | calls | 9 | 9 | 7 | 8 |
| skill | right | 0 | 0 | 0 | 0 |
| longtools | ttft | 4.43 | 4.43 | 6.62 | 3.85 |
| longtools | total | 6.51 | 7.86 | 8.37 | 5.81 |
| longtools | right | 0 | 0 | 0.5 | 0 |
| large | peak_gb | 8.07 | 8.04 | 4.45 | 12 |
| xlarge | peak_gb |  |  |  |  |
| all | load_s | 0.395 | 0.405 | 1.01 | 3.055 |
| all | peak_gb | 8.1 | 8.0 | 5.4 | 12.0 |

## Gemma 4 E4B

| Workload | Metric | Vireo | Vireo (no drafting) | llama.cpp | mlx-vlm |
|---|---|---|---|---|---|
| short | ttft | 0.029 | 0.028 | 0.054 | 0.204 |
| short | gen_tps | 125 | 125 | 117 | 94.4 |
| medium | ttft | 0.384 | 0.372 | 0.529 | 0.546 |
| medium | read_tps | 7.98e+03 | 8.29e+03 | 5.97e+03 | 7.3e+03 |
| medium | gen_tps | 124 | 113 | 114 | 90.8 |
| long | ttft | 4.52 | 5.05 | 6.26 | 5.47 |
| long | read_tps | 7.49e+03 | 6.76e+03 | 5.22e+03 | 6.22e+03 |
| long | gen_tps | 120 | 92.4 | 91.3 | 66.6 |
| long | right | 0 | 0 | 1 | 1 |
| large | prompt_tokens | 95955 | 95955 | 94706 | 9.47e+04 |
| large | ttft | 22.2 | 23.3 | 30.1 | 20.7 |
| large | read_tps | 4.34e+03 | 4.14e+03 | 3.16e+03 | 4.67e+03 |
| large | gen_tps | 89.1 | 72 | 62.9 | 43.4 |
| large | right | 0.25 | 0.5 | 1 | 1 |
| xlarge | prompt_tokens |  |  |  |  |
| xlarge | ttft |  |  |  |  |
| xlarge | read_tps |  |  |  |  |
| xlarge | gen_tps |  |  |  |  |
| xlarge | right |  |  |  |  |
| multiturn | ttft | 0.099 | 0.096 | 0.179 | 0.241 |
| multiturn | read_per_turn | 611 | 611 | 489 | 506 |
| multiturn | right | 1 | 1 | 1 | 1 |
| tools | total | 9.48 | 12.3 | 9.21 | 12.9 |
| tools | requests | 1 | 1 | 10 | 11 |
| tools | calls | 2.5 | 3.5 | 9 | 10 |
| tools | right | 0 | 0 | 1 | 1 |
| skill | total | 10.1 | 12.7 | 10 | 13.4 |
| skill | calls | 7 | 10.5 | 8 | 8.5 |
| skill | right | 0 | 0 | 1 | 0 |
| longtools | ttft | 10.4 | 10.7 | 13.8 | 10.4 |
| longtools | total | 14.4 | 16.8 | 16.9 | 14.4 |
| longtools | right | 0 | 0 | 0.5 | 0 |
| large | peak_gb | 12.8 | 12.8 | 7.59 | 16.5 |
| xlarge | peak_gb |  |  |  |  |
| all | load_s | 1.425 | 0.605 | 1.515 | 3.06 |
| all | peak_gb | 13.8 | 13.8 | 10.4 | 16.5 |

## Qwen 3.5 4B

| Workload | Metric | Vireo | Vireo (no drafting) | llama.cpp | mlx-vlm |
|---|---|---|---|---|---|
| short | ttft | 0.333 | 0.333 | 0.074 | 0.194 |
| short | gen_tps | 127 | 126 | 132 | 150 |
| medium | ttft | 0.647 | 0.648 | 0.744 | 0.702 |
| medium | read_tps | 4.67e+03 | 4.67e+03 | 4.06e+03 | 5.23e+03 |
| medium | gen_tps | 138 | 138 | 131 | 134 |
| long | ttft | 7.47 | 7.61 | 10.2 | 7.16 |
| long | read_tps | 3.97e+03 | 3.9e+03 | 2.92e+03 | 4.27e+03 |
| long | gen_tps | 104 | 102 | 102 | 107 |
| long | right | 1 | 1 | 1 | 1 |
| large | prompt_tokens | 89707 | 8.97e+04 | 8.82e+04 | 8.82e+04 |
| large | ttft | 36.6 | 37.2 | 51.9 | 29.2 |
| large | read_tps | 2.41e+03 | 2.37e+03 | 1.7e+03 | 3.06e+03 |
| large | gen_tps | 74.4 | 74.4 | 72.8 | 80.6 |
| large | right | 1 | 1 | 1 | 1 |
| xlarge | prompt_tokens |  |  |  |  |
| xlarge | ttft |  |  |  |  |
| xlarge | read_tps |  |  |  |  |
| xlarge | gen_tps |  |  |  |  |
| xlarge | right |  |  |  |  |
| multiturn | ttft | 0.134 | 0.135 | 0.211 | 0.37 |
| multiturn | read_per_turn | 533 | 533 | 514 | 1.02e+03 |
| multiturn | right | 1 | 1 | 1 | 1 |
| tools | total | 10.2 | 10.8 | 12.7 | 11.2 |
| tools | requests | 1 | 1 | 5 | 3.5 |
| tools | calls | 12 | 12 | 14 | 8.5 |
| tools | right | 1 | 1 | 1 | 0 |
| skill | total | 9.28 | 10.1 | 8.61 | 9.19 |
| skill | calls | 9.5 | 10 | 8 | 8 |
| skill | right | 0.5 | 1 | 0 | 1 |
| longtools | ttft | 19.7 | 19.8 | 25.9 | 15.9 |
| longtools | total | 48.7 | 49.2 | 30.9 | 19 |
| longtools | right | 0.5 | 0.5 | 1 | 1 |
| large | peak_gb | 11.3 | 11.3 | 7.95 | 10.5 |
| xlarge | peak_gb |  |  |  |  |
| all | load_s | 1.345 | 0.46 | 1.01 | 2.035 |
| all | peak_gb | 14.6 | 14.6 | 13.6 | 12.8 |

## Qwen 3.5 9B

| Workload | Metric | Vireo | Vireo (no drafting) | llama.cpp | mlx-vlm |
|---|---|---|---|---|---|
| short | ttft | 0.568 | 0.567 | 0.092 | 0.219 |
| short | gen_tps | 76.4 | 77.1 | 80.6 | 101 |
| medium | ttft | 1.05 | 1.06 | 1.26 | 1.02 |
| medium | read_tps | 2.87e+03 | 2.85e+03 | 2.38e+03 | 3.43e+03 |
| medium | gen_tps | 84.8 | 83.6 | 81.7 | 88.6 |
| long | ttft | 12.6 | 12.8 | 16.5 | 11 |
| long | read_tps | 2.36e+03 | 2.32e+03 | 1.8e+03 | 2.74e+03 |
| long | gen_tps | 68.8 | 69.3 | 69.7 | 75.3 |
| long | right | 1 | 1 | 1 | 1 |
| large | prompt_tokens | 8.97e+04 | 8.97e+04 | 8.82e+04 | 88230 |
| large | ttft | 51.2 | 51.7 | 70.5 | 44.5 |
| large | read_tps | 1.72e+03 | 1.71e+03 | 1.25e+03 | 2e+03 |
| large | gen_tps | 56 | 55.8 | 54.3 | 58.9 |
| large | right | 1 | 1 | 1 | 1 |
| xlarge | prompt_tokens |  |  |  |  |
| xlarge | ttft |  |  |  |  |
| xlarge | read_tps |  |  |  |  |
| xlarge | gen_tps |  |  |  |  |
| xlarge | right |  |  |  |  |
| multiturn | ttft | 0.215 | 0.214 | 0.307 | 0.503 |
| multiturn | read_per_turn | 535 | 535 | 513 | 1.02e+03 |
| multiturn | right | 1 | 1 | 1 | 1 |
| tools | total | 13.3 | 13.2 | 15 | 13.1 |
| tools | requests | 1 | 1 | 6.5 | 5 |
| tools | calls | 8.5 | 7 | 10.5 | 10 |
| tools | right | 1 | 1 | 1 | 1 |
| skill | total | 19.6 | 20.5 | 16.6 | 15.5 |
| skill | calls | 17 | 17.5 | 8 | 8 |
| skill | right | 0 | 1 | 1 | 0 |
| longtools | ttft | 28.5 | 28.7 | 37.8 | 24.8 |
| longtools | total | 30.5 | 29.4 | 43.3 | 38.1 |
| longtools | right | 1 | 1 | 1 | 0 |
| large | peak_gb | 16.8 | 16.8 | 10.6 | 14.5 |
| xlarge | peak_gb |  |  |  |  |
| all | load_s | 2.075 | 0.745 | 1.01 | 2.035 |
| all | peak_gb | 19.8 | 19.8 | 16.3 | 17.5 |

## Gemma 4 26B-A4B

| Workload | Metric | Vireo | Vireo (no drafting) | llama.cpp | mlx-vlm |
|---|---|---|---|---|---|
| short | ttft | 0.14 | 0.141 | 0.094 | 0.262 |
| short | gen_tps | 61.1 | 61.4 | 93.9 | 111 |
| medium | ttft | 1.78 | 1.73 | 1.07 | 1.02 |
| medium | read_tps | 1.73e+03 | 1.77e+03 | 2.88e+03 | 4e+03 |
| medium | gen_tps | 63.2 | 60.2 | 93.2 | 102 |
| long | ttft | 20.2 | 20.6 | 15.6 | 10.4 |
| long | read_tps | 1.67e+03 | 1.64e+03 | 2.09e+03 | 3.24e+03 |
| long | gen_tps | 68 | 53 | 70.1 | 73 |
| long | right | 1 | 1 | 1 | 1 |
| large | prompt_tokens | 9.6e+04 | 95955 | 9.47e+04 | 94711 |
| large | ttft | 75.7 | 76.4 | 72.2 | 48.4 |
| large | read_tps | 1.27e+03 | 1.26e+03 | 1.31e+03 | 1.98e+03 |
| large | gen_tps | 50.4 | 43.2 | 48.9 | 49.2 |
| large | right | 1 | 1 | 1 | 1 |
| xlarge | prompt_tokens |  |  |  |  |
| xlarge | ttft |  |  |  |  |
| xlarge | read_tps |  |  |  |  |
| xlarge | gen_tps |  |  |  |  |
| xlarge | right |  |  |  |  |
| multiturn | ttft | 0.326 | 0.326 | 0.282 | 0.497 |
| multiturn | read_per_turn | 531 | 531 | 509 | 1.01e+03 |
| multiturn | right | 1 | 1 | 1 | 1 |
| tools | total | 17.3 | 18.4 | 12.6 | 13.4 |
| tools | requests | 1 | 1 | 11 | 11.5 |
| tools | calls | 5.5 | 7 | 10 | 10.5 |
| tools | right | 1 | 1 | 1 | 0.5 |
| skill | total | 25 | 39.3 | 14.4 | 13 |
| skill | calls | 10 | 8 | 5 | 5 |
| skill | right | 0 | 0 | 0.5 | 0 |
| longtools | ttft | 42.1 | 41.4 | 33.8 | 21.8 |
| longtools | total | 45 | 45 | 38.6 | 47.7 |
| longtools | right | 1 | 1 | 1 | 1 |
| large | peak_gb | 50 | 50 | 27 | 33 |
| xlarge | peak_gb |  |  |  |  |
| all | load_s | 8.95 | 2.925 | 2.015 | 3.555 |
| all | peak_gb | 53.0 | 53.0 | 38.0 | 33.0 |

## Gemma 4 31B

| Workload | Metric | Vireo | Vireo (no drafting) | llama.cpp |
|---|---|---|---|---|
| short | ttft | 0.177 | 0.173 | 0.357 |
| short | gen_tps | 27.5 | 17.7 | 17.2 |
| medium | ttft | 4.6 | 4.63 | 6.06 |
| medium | read_tps | 667 | 664 | 505 |
| medium | gen_tps | 30.9 | 17.8 | 17.3 |
| long | ttft | 64 | 65.4 | 81.5 |
| long | read_tps | 528 | 517 | 399 |
| long | gen_tps | 36.9 | 16.1 | 16 |
| long | right | 1 | 1 | 1 |
| large | prompt_tokens | 95956 | 95954 |  |
| large | ttft | 277 | 287 |  |
| large | read_tps | 346 | 335 |  |
| large | gen_tps | 28.5 | 12.4 |  |
| large | right | 1 | 1 |  |
| xlarge | prompt_tokens |  |  |  |
| xlarge | ttft |  |  |  |
| xlarge | read_tps |  |  |  |
| xlarge | gen_tps |  |  |  |
| xlarge | right |  |  |  |
| multiturn | ttft | 1.1 | 1.23 |  |
| multiturn | read_per_turn | 530 | 531 |  |
| multiturn | right | 1 | 1 |  |
| tools | total | 55.5 | 59.9 |  |
| tools | requests | 1 | 1 |  |
| tools | calls | 7 | 7 |  |
| tools | right | 1 | 1 |  |
| skill | total | 97.9 | 173 |  |
| skill | calls | 31 | 31 |  |
| skill | right | 0 | 0 |  |
| longtools | ttft | 143 | 144 |  |
| longtools | total | 145 | 148 |  |
| longtools | right | 1 | 1 |  |
| large | peak_gb | 65 | 65 |  |
| xlarge | peak_gb |  |  |  |
| all | load_s | 16.97 | 3.5 | 2.02 |
| all | peak_gb | 76.0 | 76.0 | 37.0 |

## Errors

- Qwen 3.5 4B / vireo / longtools: A tool call ran too long.
- Qwen 3.5 4B / vireo-plain / longtools: A tool call ran too long.
- Qwen 3.5 4B / mlx / tools: "Generation ended with an incomplete tool call."