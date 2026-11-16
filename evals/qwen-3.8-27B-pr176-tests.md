# PR #176 tests: Qwen sampling by mode, Qwen 3.8 template behaviour

Tested 2026-11-15 on the MacBook Pro M5 Max. AFTER = main at c7eadca (PR #176 merged). BEFORE = b8b341c, built from a temporary git worktree (since removed). Model: Qwen 3.8 27B Q6_K, plus Qwen 3.5 9B Q6_K as a sanity check. Nothing else was using the GPU during the runs. No source code was changed.

## Verdict

Nothing looks broken or worse. On Qwen 3.8 27B the eval scores the same before and after (42/43). The sampled chats, tool tasks and thinking questions were all answered correctly on AFTER. The presence penalty caused no defects I could find. Tool-call arguments can't be affected by it: they are written greedily under the grammar in `write_call`, and the `Sampler` isn't used there. `keep_notes` works as intended. In thinking-off chats with tools, AFTER reads about 40-55 tokens a turn where BEFORE re-read about 130-230. Edits, retries, alternating chats and a user message ending in "(Today is Monday.)" all gave correct replies.

The one real defect is older than this PR: in thinking-off mode Qwen 3.8 sometimes writes `</think>` mid-reply and then repeats its answer. It happened at the same rate on BEFORE and AFTER (1 in 24 each). With the presence penalty turned off it happened more often (3 in 24), so the penalty isn't the cause.

How sure I am: high for the eval, tool-call and cache results, which are greedy and repeatable. For the quality of sampled replies, only medium, because each sampled chat ran once per build (see the notes on noise).

## Tests and results, before vs after

| Test | Runs | BEFORE (b8b341c) | AFTER (c7eadca) |
|---|---|---|---|
| 1. `cargo test --release -p vireo-server` | 1 | - | 113 passed, 0 failed, 4 ignored |
| 2. vireo-eval, Qwen 3.8 27B, all 43 cases (temperature 0, thinking off) | 1 each | 42/43, 6.54 s/case, 281.4 s | 42/43, 6.69 s/case, 287.7 s |
| 2. same: cases whose answer differs | | | 3 of 43, all still passing (below) |
| 3a. Thinking-off chats, no client tools (3 chats, 17 turns, sampled) | 1 each | all right; 4,427 tokens read; total TTFT 11.9 s | all right; 2,520 tokens read; total TTFT 7.8 s |
| 3a. Thinking-off chats with client tools (same 17 turns) | 1 each | 1 `</think>` leak (offsite t1); 8,500 read; TTFT 30.4 s | 1 `</think>` leak (offsite t1); 6,929 read; TTFT 28.6 s |
| 3b. Tool tasks, sampled, thinking off (8 tasks) | 1 each | 8/8 right; "write" task wandered off: 16 tool calls, 2 `</think>` leaks, 44 s | 8/8 right; "write" in 1 call, 6 s; total 115 s vs 153 s |
| 3c. Thinking on: 5 questions + a 3-turn chat (sampled) | 1 each | 8/8 right; 710 reasoning tokens; 42 s | 8/8 right; 1,760 reasoning tokens; 83 s |
| 3d. `</think>` leak repeat (3 first-turn prompts x 8, tools, thinking off) | 24 each | 1/24 leaked | 1/24 leaked; with `presence_penalty: 0` sent: 3/24 |
| 4. Cache edge cases (temperature 0) | 1 each | all right except 2 `</think>` leaks | all right except 1 `</think>` leak (same reply as BEFORE's) |
| 5. vireo-eval, Qwen 3.5 9B, all 43 cases | 2 BEFORE, 1 AFTER | 41/43 (both runs), 2.77-2.8 s/case | 41/43, 2.89 s/case; every answer and call identical to BEFORE |
| 5. Qwen 3.5 9B, sampled 4-turn chat and 3 chats with tools | 1 each | all right apart from model errors seen on both builds | the same |

### 2. vireo-eval on Qwen 3.8 27B

Files: `evals/qwen-3.8-27B-before-pr176.json` and `evals/qwen-3.8-27B-after-pr176.json`. The eval runs every case at temperature 0 with thinking off. So it doesn't exercise the new sampling, the presence penalty (not applied at temperature 0), the effort line (thinking on only) or `keep_notes` (one request per case). What it does exercise is the empty thought block in past assistant turns. Both builds fail `office-note` the same way ("looked for files it didn't need"). 40 of 43 cases gave the same answer and the same tool calls, word for word. The three that differ all still pass:

| Case | BEFORE answer | AFTER answer |
|---|---|---|
| `datetime` | "Today is Sunday." | "Today is Sunday, November 15, 2026." |
| `follow-up-file` (has history) | "plan.md says: # Plan / Ship the new app in December." | "plan.md says: \"Ship the new app in December.\"" |
| `follow-up-no-tool` (has history) | "The code word \"PELICAN\" is for **Friday**. Since today is Sunday, November 15, 2026, that would be **Friday, November 20, 2026**." | "The code word \"PELICAN\" is for **Friday**." |

Inference: the two `follow-up-*` changes come from the empty thought now written into the past assistant turn. They are the only cases with an earlier assistant turn. `datetime` has no history. Its difference most likely comes from the same change applied to the turn the server rebuilds after a tool result, but I haven't confirmed that.

Time per case: AFTER was 0.1-0.5 s slower on most cases (6.69 vs 6.54 s on average). The tokens kept, read and written are almost identical (read 28,404 vs 28,396; written 6,833 vs 6,859). On the 9B, a second BEFORE run came out between the first BEFORE and the AFTER run (read 2,097/s vs 2,150 and 1,917). So I read the timing gap as run-to-run noise or heat (AFTER always ran second), not a slowdown. Each figure is from one run.

A first AFTER run is not counted. In it I pointed `VIREO_BUILTIN_SKILLS` at a wrong folder, so the skill cases failed (35/43). The first BEFORE run had the same problem, because the BEFORE binary lives outside the repo and can't find `server/builtin-skills` itself. Both runs were redone with the right folder.

### 3. Sampled behaviour (no temperature in the request, so the new defaults apply)

Script: `evals/qwen-3.8-27B-pr176-sampled/sampled.py`, built on bench.py's `Desk`, `post` and `sse`. Raw results are in the same folder. Every server started with a fresh chat cache, the bench's skills folder and the built-in skills. Thinking-off requests sent `"thinking": false` and only `max_tokens`. The script does not send a reply's `turn_record` back in the 3a/3b/3c chats; the app does. So in those chats, a turn after a tool call re-reads from the tool call onward on both builds. The edge cases in section 4 do send records back.

**3a. Thinking-off chats.** Three chats: a Merkle-tree explanation then list, rewrite, back-reference, Markdown table and summary (6 turns); `parse_duration` code then pytest tests, adding days, a number question and a JavaScript port (5 turns); a 14-person offsite with groups, a budget, a back-reference, an email, a German subject line and alphabetical names (6 turns). Each ran once with no client tools and once with the bench's client tools.

- Things a presence penalty of 1.5 could cause: none found on AFTER. No odd word swaps, no language switch (the German subject line was asked for), no broken Markdown tables or lists, no trailing off. Names and numbers stayed the same across turns: the groups held from turn 1 to turns 3 and 6, €2,121 was right and carried into the email, and turn 4's 90061 was right. The Python that the model wrote ran and gave correct results for every input (`1h30m`, `45s`, `2h5s`, `1d2h`, `1d1h1m1s`) on both builds.
- The model's own pytest tests had one wrong test on AFTER in each run: `parse_duration("S5M1H") == 3665` (its parser rejects a leading unit), and in the tools run an empty-string test that expected an error the function doesn't raise. BEFORE had one wrong test too (an invalid-input test). These are model mistakes on both builds, one run each.
- Pre-existing, on both builds and all four runs: the email says "**Thursday**, 12 March". 12 March 2027 is a Friday.
- Tokens read per turn after the first, no client tools: AFTER 41-73, BEFORE 52-481 (for example the Merkle chat, turns 2-6: AFTER 46/51/49/52/41, BEFORE 161/227/126/143/145). "Kept" on AFTER was the whole previous cache on every turn without a tool call (for example turn 2 kept 1,622 = 1,513 prompt + 109 written). TTFT on later turns: AFTER 0.18-0.23 s, BEFORE 0.25-0.52 s, except turns after a tool call.
- Pre-existing, both builds: in the code chat with client tools, turn 2 ("Add tests ... pytest") re-read the whole prompt (0 kept, about 2,360 read). The server's log says "Prompt as the chat's cache has it: the tools that change files declared": the file-change tools were added to the system prompt there.
- With client tools the TTFT on later turns is 0.6-1.2 s even when about 50 tokens are read, on both builds. Something else takes that time per request. Not related to this PR.

**3b. Tool tasks (thinking off, sampled, `VIREO_CLIENT_TOOLS` on the bench's harbor-bikes desk), 8 tasks.**

| Task | Expected | BEFORE | AFTER |
|---|---|---|---|
| revenue | Q4 best, $26,742.25 | right (also flagged future-dated Q4 rows) | right (same flag) |
| weekly-report skill | report with hours by person, 78 total | right (all hours right) | right (all six people's hours checked against the notes) |
| terms | Kestrel, 14 days | right | right |
| notes | 5 notes, 9-13 Nov | right | right |
| search | 10 Nov, card reader dropped twice | right | right |
| helmets | Q3 18 vs Q4 4 | right | right |
| copy | exact copy at backup/suppliers-copy.md | exact copy | exact copy |
| write | three given lines in orders/reorder-2026-11-16.txt | right file, but first 16 tool calls reading every file, `</think>` twice in the reply, 1,475 tokens | right, 1 call |

Tool-call text on AFTER: every path was right and complete, including the 5-path `read_file` in the skill task with the same long root repeated five times. Arguments repeated across calls were intact, and the copied file matched byte for byte. As expected, because tool calls are written greedily under the grammar (`write_call`, main.rs), so the presence penalty never touches them. BEFORE's "write" spiral is a single sample; it shows that BEFORE's sampling can wander too, not that AFTER is better.

**3c. Thinking on (no temperature: AFTER uses 1.0 / 0.95 / 20 plus the "Reasoning effort is set to xhigh" line; BEFORE 0.6 / 0.95 / 20, no line).**

| Question | Want | BEFORE: reasoning / total tokens, s | AFTER: reasoning / total tokens, s |
|---|---|---|---|
| train times | 20:10 | 81 / 214, 8.4 s, right | 153 / 284, 10.0 s, right |
| r's in "strawberry raspberry" | 6 | 64 / 124, 3.7 s, right | 319 / 400, 11.7 s, right |
| second_largest code | works | 30 / 70, 2.0 s, right | 145 / 184, 5.7 s, right |
| weekday of 1 Jan 2027 | Friday | 148 / 257, 7.4 s, right | 346 / 451, 13.8 s, right |
| sum of multiples of 3 or 5 to 200 | 9368 | 276 / 519, 14.4 s, right | 417 / 638, 17.5 s, right |
| chat t1: apples total | 38 | 14 / 54, 1.9 s | 34 / 74, 2.3 s |
| chat t2: give away a third of 19, rounded down | 32 | 37 / 62, 2.0 s | 191 / 368, 11.8 s (2 calculate calls) |
| chat t3: split among 4 | 8 each, 0 left | 60 / 74, 2.4 s | 155 / 344, 10.2 s (2 calculate calls) |
| Total | | 710 / 1,374, 42 s | 1,760 / 2,743, 83 s |

No reasoning leaked into any answer on either build. Both builds start thinking-on answers with 2-6 blank lines, in the replies after tool calls (pre-existing, cosmetic). AFTER reasons about 2.5x as long and takes about 2x the time, as expected from the xhigh effort line and the higher temperature. It also called `calculate` more (it tried `floor()`, which the calculator doesn't support, then used `//`). One run per question.

**3d. The `</think>` leak, repeated.** The three first-turn prompts from 3a, 8 times each, with client tools and thinking off, no temperature:

| Server | Leaks | Where |
|---|---|---|
| BEFORE | 1/24 | offsite prompt |
| AFTER | 1/24 | offsite prompt |
| AFTER, request sends `presence_penalty: 0` | 3/24 | offsite prompt |

Example (AFTER, 3a with tools, offsite turn 1), with the prompt "I'm planning a team offsite for 14 people: Ana, ... Noor. Split them into 3 groups as evenly as possible and list the groups.". The reply was the full answer, then a line `</think>`, then the same answer again:

```
14 people ÷ 3 groups = 4, 5, 5.
**Group A (5):** Ana, Bilal, Chen, Dara, Emeka
...
All 14 are covered, with two groups of 5 and one of 4 — the most even split possible.
</think>

14 people ÷ 3 groups = 4, 5, 5.
...
```

It happens at temperature 0 too: in 4(d) the chat-A first message "Chat A: I'm cooking a risotto with mushrooms. What wine goes with it? One sentence." gives "A crisp, unoaked white like a Pinot Grigio or a dry Vermentino will cut through the creamy mushroom risotto beautifully. `</think>` A crisp, unoaked white ..." on both builds. The server sends the `</think>` text through as answer text in thinking-off mode.

### 4. Cache reuse edge cases (AFTER, thinking off, temperature 0; BEFORE also run for comparison)

Run twice: a first pass (`edge-*.json`) and a cleaner second pass (`edge2-*.json`) that sends each reply's `turn_record` back as the app does and avoids memory-tool wording.

| Case | Reply right? (AFTER) | Kept / read (AFTER) | BEFORE |
|---|---|---|---|
| (a) 6 turns, no client tools | yes, all 6 (name, hobby recalled; 391) | turns 3-5 kept the whole previous cache (1,607 = 1,601 + 6, and so on), reading 39-45 | right; read 51-59 |
| (a) 6 turns, client tools, records sent | yes | turns 3, 4 and 6 kept all (after `calculate` in turn 5: kept 2,226 = all, read 43). Turn 2 kept only the 1,806-token system start (read 207): turn 1 called `remember`, which changes the system prompt | right; read 53/54/106 on those turns; turn 2 same as AFTER |
| (b) edit an earlier message (Lisbon to Porto, neutral "OK." reply kept, turn 2 resent) | yes: "Porto" | kept 1,806 (system start), read 124 | the right city, but each reply carried a `</think>` leak: "Lisbon. `</think>` Lisbon", "Porto. `</think>` Porto" |
| (b') edit the last message instead (city to month) | yes: "May" | kept 1,806, read 124 | right |
| (c) retry the same last message | yes, identical reply | kept 1,542 of 1,568, read 26 | same |
| (d) two chats alternated A1 B1 A2 B2 | each reply answered its own chat; A1 had the `</think>` leak | each switch kept only the 1,463-token system start; small chats are not swapped back in | same, including the A1 leak |
| (e) tool call in turn 1 (records sent), then 2 plain turns | yes (Alder & Finch; 45 days; Kestrel shortest) | turn 2 kept 2,145, read 40; turn 3 kept 2,198, read 41 | right; read 226 then 59 |
| (f) user text ending "(Today is Monday.)" (records sent) | yes: "Monday", then the leave week, then "OK", "OK" | after the "Thanks. Reply with just: OK. (Today is Monday.)" turn, the next kept 2,217 of 2,217 (all), read 38: `keep_notes` did not splice the user's own text by mistake | right; read 72 and 45 |

The first pass's edit test isn't counted. Its first message ("My favourite colour is green. Remember it.") made the model call `remember`, and the kept assistant reply said "green". The edited chat then answered "Green" on both builds, which is no proof of a cache problem.

### 5. Qwen 3.5 9B (sampling change only)

vireo-eval on all 43 cases: `evals/qwen-3.5-9B-before-pr176.json` and `evals/qwen-3.5-9B-after-pr176.json`. Both 41/43, failing `skill-none` and `office-note`. Every case's answer and tool calls are identical, as expected at temperature 0 with no change to 3.5's prompt (`qwen_keeps_thought()` is false for its template).

Sampled (one run each): the 4-turn offsite chat without client tools was right on both. Groups, €2,121 and Freya's group were consistent. Both called 12 March a "Sunday", which is wrong. With client tools:
- The 9B, on both builds, answered `parse_duration('1d1h1m1s')` wrongly (BEFORE 86401, AFTER 94261) even though its own function returns 90061. A model error on both.
- On AFTER, turn 2 ("Add tests ...") made 3 `write_file` calls to save test files nobody asked for. On BEFORE, turns 3 and 5 ran long (1,058 tokens, and the 1,200-token cap on the JavaScript port).
- No `</think>` leaks on either.
- One run each, so none of this separates the builds.

## Problems found

1. **`</think>` written mid-reply in thinking-off mode, with the answer repeated (Qwen 3.8 27B). Pre-existing, not from this PR.** Prompts and replies are quoted above. Measured: 1/24 on BEFORE and 1/24 on AFTER (sampled), 3/24 on AFTER with the presence penalty off. At temperature 0 it happens on both builds for the risotto prompt. Two more BEFORE-only cases at temperature 0 (edit test) and two in BEFORE's sampled "write" task. Inferred: the model sometimes treats its first answer as reasoning and "closes" a thought that the prompt's empty `<think>\n\n</think>\n\n` already closed. It isn't the presence penalty (fewer leaks with it on) and not the empty thought in past turns (it happens on first turns, and BEFORE leaked more in multi-turn cases). Possible fix, not made: in thinking-off mode, drop `</think>` and the text before it from the answer, or stop on it.
2. **Thinking on now takes about 2x as long on Qwen 3.8.** Measured: 1,760 vs 710 reasoning tokens and 83 vs 42 s over 8 requests, all answers right on both. This is by design (the xhigh line and temperature 1.0). Nothing broke, but it's the cost users will notice. One run each.
3. **Small things on both builds, not from this PR:** blank lines at the start of thinking-on answers after tool calls; a full re-read when a turn makes the server declare the file-change tools; the system prompt changing after a `remember` call (so the next turn re-reads from the system start); about 0.5-1 s of extra TTFT per request when client tools are offered, even with about 50 tokens read.

Nothing found that points at the presence penalty, the empty thought in past turns, the effort line (beyond longer thinking) or `keep_notes` as a cause of a wrong reply.

## Measured vs inferred, and noise

- Measured: eval scores, answers and tool calls (greedy, repeatable); kept/read/TTFT per turn from the server's `Reply:` lines; correctness of every sampled answer (checked by hand and by running the model's Python); leak counts.
- Inferred: which change explains each eval difference; the leak's cause; that the AFTER eval timing gap is heat or ordering.
- Noise: every sampled chat, tool task and thinking question ran once per build. Differences such as BEFORE's 16-call "write" task, the 9B's unasked-for `write_file` calls, reply lengths and per-task times are single samples and shouldn't be read as a difference between builds. The leak counts (24 per setting) are small samples too: 1 vs 1 vs 3 can't tell the three settings apart statistically. It only shows the leak exists without the penalty.

## Not run or cut short

- bench.py itself was not run. It uses temperature 0 and thinking off, which the eval and section 4 already cover. Its `Desk`, sandbox and skill were used in section 3.
- Qwen 3.5 9B was not tested with thinking on.
- The Aurelia gauntlet was not run (out of scope).
- The 3a/3b/3c chats do not send `turn_record`s back (sections 4 and 2 cover that), so their read counts after tool turns overstate what the app would read, on both builds alike.

## Files

- `evals/qwen-3.8-27B-before-pr176.json`, `evals/qwen-3.8-27B-after-pr176.json`: eval, 27B
- `evals/qwen-3.5-9B-before-pr176.json`, `evals/qwen-3.5-9B-after-pr176.json`: eval, 9B
- `evals/qwen-3.8-27B-pr176-sampled/`: `sampled.py` (the test script) and the raw results: `before-27b.json`/`after-27b.json` (3a-3c), `leak-*.json` (3d), `edge-*.json` and `edge2-*.json` (4), `9b-*.json` (5)
