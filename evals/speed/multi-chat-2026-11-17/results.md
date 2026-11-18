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

## Chats written together (`vireo-cli together`, branch `prototype/write-for-several-chats`)

One token for each of several chats in one pass (`Gemma4Cuda::forward_each`), every chat's
logits read back. Gemma 4 31B Q6_K, each chat a different stretch of the corpus. A step, and
tokens a second for each chat / for all of them:

| Chats | 4,000 tokens each | 16,000 tokens each |
|---|---|---|
| 1 | 50.3 ms (19.9 / 20) | 54.4 ms (18.4 / 18) |
| 2 | 56.8 ms (17.6 / 35) | 78.0 ms (12.8 / 26) |
| 4 | 80.8 ms (12.4 / 49) | 102.7 ms (9.7 / 39) |
| 8 | 99.1 ms (10.1 / 81) | 128.0 ms (7.8 / 63) |
| 1, the usual way | 51.3 ms (19.5) | 55.1 ms (18.1) |

The same likeliest token as each chat written alone for 8 of 8 chats at both depths; logits
differ by 0.036 / 0.050 at most. Against one chat's batch of 8 tokens (69 ms at 4,000) the 30 ms
more is each chat's own attention and key/value step, a dispatch a chat a layer.

    ./target/release/vireo-cli together "$M" evals/bench/fixtures/corpus.txt [--depth 16000] [--chats 8]

## People asking at the same moment (`together.py`, `vireo-server --slots <n>`)

Gemma 4 31B Q6_K, 32K context, each person's chat starting with 12,000 characters (about
3,000 tokens) of a different text. Round 2: everyone asks for three paragraphs at once
(200-token replies).

| People | Slots | First tokens | Replies done | Each reply written at | In all |
|---|---|---|---|---|---|
| 4 | 1 (one at a time) | 0.3, 6.7, 12.6, 18.4 s | 6.4, 12.3, 18.2, 24.6 s | 32-36 tok/s (drafted) | 32.5 tok/s |
| 4 | 4 | 0.3, 0.7, 1.0, 1.3 s | 19.7-19.8 s | 10.4-10.9 tok/s | 40.3 tok/s |
| 8 | 8 | 0.4-2.7 s | 23.7-24.1 s | 8.5-9.5 tok/s | 66.5 tok/s |

Round 1, everyone's 3,000-token prompt at once (60-token replies): prompts are read one
after another, a batch at a time with the replies being written given the GPU between
batches. 4 people, 4 slots: first tokens at 6.5, 14.9, 24.1, 32.9 s, replies done at 15.0,
24.2, 33.0, 34.9 s (one slot: first tokens 5.9, 11.9, 17.5, 23.8 s, done 7.5, 13.1, 19.1,
25.0 s). 8 people, 8 slots: first tokens 8.2 to 71.5 s, done 27.3 to 73.4 s. A reply written
while other prompts are read gets about half the GPU, and the prompts take that much longer:
the share wants tuning.

vireo-eval, Gemma 4 31B, 43 cases (`eval/`): 39 passed with one slot and with `--slots 2`, the
same four failing as on main this afternoon (`two-files`, `recover`, `office-note`,
`office-policy`); the same 4,581 tokens written and 10,830 read both ways.

    python3 -I evals/speed/multi-chat-2026-11-17/together.py "$M" --people 8 --slots 8 --rounds 1
    ./target/release/vireo-eval --models "$M" --server-arg --slots --server-arg 2
