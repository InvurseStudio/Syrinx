# Several people taking turns on one server, Gemma 4 31B, 2026-11-17

`alternate.py`: each person has a chat of their own that starts with a different stretch of
text, and they take turns. MacBook Pro M5 Max, 128 GB, Q6_K, main at 97c4964, then the same
with the store holding saved states in memory.

## What going from one person's chat to another's costs (main)

| People | Chat length | First token, turns after the first | Of which loading the chat |
|---|---|---|---|
| 1 | 9.4K tokens | 0.18 s | none |
| 2 | 9.4-10.4K | 0.31 s (worst 0.39) | 80-194 ms |
| 6 | 9.4-10.8K | 0.34 s (worst 0.44) | 146-243 ms |
| 3 | 31-33K | 0.57 s (worst 0.66) | 268-406 ms |

No chat was read again: every later turn read its 54-92 new tokens. A first turn reads the
whole prompt (9-10K tokens in 13-16 s, 31K in 55-59 s), and nobody else is answered meanwhile.

## Holding saved states in memory (six people, 9.4-10.8K-token chats, 3 turns each)

| Build | First token, turns after the first | Chat files on disk at the end |
|---|---|---|
| main | 0.34 s (median 0.35, worst 0.45) | 6 files, 10.0 GB, each rewritten at every turn of its chat |
| states in memory | 0.27 s (median 0.28, worst 0.34) | none |

Main wrote a chat's state (about 1.7 GB here) each time its turn ended and another person's
began: twelve times in this run after the first round, about 20 GB. The new build writes a
state when it's pushed out of memory (an eighth of the machine's memory: 16 GB here), when
the server has had nothing to do for 30 s, and when it stops.

## Batches of tokens (what batched writing across chats could carry, upper bound)

`vireo-cli batches`, ms a step: at 4,000 tokens 1: 49.3, 2: 52.1, 4: 66.7, 8: 69.0, 16: 74.0,
32: 88.2, 64: 120.3; at 32,000 tokens 1: 56.2, 2: 58.2, 4: 75.4, 8: 80.2, 16: 90.4, 32: 116.7,
64: 194.9. These are one chat's tokens over one cache; separate chats each read their own.

    M=~/Library/Application\ Support/InvurseChat/models/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q6_K.gguf
    python3 -I evals/speed/multi-chat-2026-11-17/alternate.py "$M" --chats 6 --turns 3 [--server <main's build>]
    python3 -I evals/speed/multi-chat-2026-11-17/alternate.py "$M" --chats 3 --turns 3 --chars 110000 --context 131072
