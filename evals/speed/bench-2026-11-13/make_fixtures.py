"""Builds the Nov 13 benchmark's inputs, all fresh: prompts, the long and large context
documents (with facts hidden at known depths), a small business folder for the tool
tasks, and one skill. Deterministic (fixed seed); run again to rebuild.

    python3 -I make_fixtures.py
"""
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
rng = random.Random(1113)

PROMPTS = HERE / "prompts"
SANDBOX = HERE / "sandbox" / "harbor-bikes"
SKILLS = HERE / "skills"
for d in (PROMPTS, SANDBOX, SKILLS):
    d.mkdir(parents=True, exist_ok=True)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ---- Plain prompts -------------------------------------------------------------------

write(PROMPTS / "short.txt",
      "In about 150 words, explain to a new bike-shop employee why a chain wears out "
      "faster when it is left dry, and what they should tell customers about it.")

# Medium: ~2.5K tokens of the repo's own engine plan (real prose), then a task.
plan = (REPO / "PLAN.md").read_text()
medium_body = plan[:9500]
write(PROMPTS / "medium.txt",
      "Here is part of an engineering plan.\n\n<document>\n" + medium_body +
      "\n</document>\n\nSummarise the plan above as six short bullet points for a "
      "product manager, then name the single biggest risk it mentions.")

# Long and large: real source and docs from this repository, with five made-up facts
# placed at known depths; the question asks for two of them.
sources = []
for root in ["crates/vireo-server/src", "crates/vireo-engine/src", "crates/vireo-eval/src",
             "Mac", "tools"]:
    for p in sorted((REPO / root).rglob("*")):
        if p.suffix in (".rs", ".swift", ".py", ".md") and p.is_file() and p.stat().st_size < 400_000:
            sources.append(p)
rng.shuffle(sources)

NEEDLES = [
    ("The harbour depot's spare-parts code word is OSPREY-41.", "OSPREY-41"),
    ("Marisol Quint signed the cold-store lease for 14 months.", "14 months"),
    ("The test rig in bay 3 is calibrated to 2.375 newton-metres.", "2.375"),
    ("The fallback ferry to Skerra leaves at 06:50 on weekdays.", "06:50"),
    ("Invoice batch 7731 was approved by Dagny Vell.", "Dagny Vell"),
]


def corpus(chars, tag):
    out, total = [], 0
    for p in sources:
        text = p.read_text(errors="replace")
        piece = f"\n\n===== {p.relative_to(REPO)} =====\n{text}"
        out.append(piece)
        total += len(piece)
        if total >= chars:
            break
    body = "".join(out)[:chars]
    # Facts at 10%, 30%, 50%, 70%, 90% depth, each on its own line.
    for i, (fact, _) in enumerate(NEEDLES):
        at = body.find("\n", int(len(body) * (0.1 + 0.2 * i)))
        body = body[:at] + f"\n// NOTE ({tag}): {fact}\n" + body[at:]
    question = ("\n\nAbove is a dump of a software repository. Somewhere in it are notes "
                "marked NOTE. Answer in two lines: (1) the harbour depot's spare-parts code "
                "word, and (2) who approved invoice batch 7731. Then, in one sentence, say "
                "what the repository's server program does.")
    return "<repository>" + body + "\n</repository>" + question


# ~3.6 characters per token on this mix (checked against the servers' prompt counts).
write(PROMPTS / "long.txt", corpus(112_000, "long"))      # ~31K tokens
write(PROMPTS / "large.txt", corpus(325_000, "large"))    # ~90K tokens
write(PROMPTS / "needles.json", json.dumps({"expect": ["OSPREY-41", "Dagny Vell"]}))

# Multi-turn: eight turns, each pasting a ~600-token shift report and asking about it,
# the last two asking about earlier turns.
CREW = ["Ines", "Bram", "Yuki", "Tobias", "Amara", "Leif", "Noor", "Caspian"]
JOBS = ["wheel truing", "brake bleeds", "drivetrain swaps", "e-bike diagnostics",
        "tubeless conversions", "fork services", "fittings", "warranty claims"]
FILL = [
    "The front counter was busy from opening until noon, mostly with walk-in repairs.",
    "A delivery of tyres arrived short by two boxes and the supplier has been told.",
    "{p} reorganised the parts wall so the common sizes are at eye height.",
    "One customer asked for a same-day service and was booked in for the morning instead.",
    "The card reader dropped its connection twice and was restarted each time.",
    "{p} finished the online orders before lunch and packed them for the courier.",
    "The workshop stand in bay two needs a new clamp pad before the weekend.",
    "Rain kept the test-ride area quiet, so the team caught up on cleaning.",
    "{p} spoke to the landlord about the leaking gutter above the back door.",
    "Three rental bikes came back late and the late fees were waived as a courtesy.",
]
turns = []
facts = []
for i in range(8):
    p, job = rng.choice(CREW), rng.choice(JOBS)
    n = rng.randrange(6, 31)
    facts.append((p, job, n))
    lines = [f"Shift report {i + 1}: {p} led the shift and the team did {n} {job}."]
    lines += [rng.choice(FILL).format(p=rng.choice(CREW)) for _ in range(28)]
    if i < 6:
        q = f"From shift report {i + 1}: who led it, and how many {job} were done? One sentence."
    else:
        j = 1 if i == 6 else 3
        q = (f"Going back to shift report {j + 1}: who led it and how many {facts[j][1]} "
             f"were done? One sentence.")
        expect = str(facts[j][2])
    turns.append({"text": " ".join(lines) + "\n\n" + q,
                  "expect": str(n) if i < 6 else expect})
write(PROMPTS / "multiturn.json", json.dumps(turns, indent=1))

# ---- The tool tasks' folder ----------------------------------------------------------

write(SANDBOX / "README.md", """# Harbor Bikes back office

Folders:
- sales/   one CSV per quarter of 2026: date, item, units, unit_price (dollars)
- notes/   the workshop's daily notes for the week of 9 November 2026
- suppliers.md  who we buy from and their terms

Revenue for a quarter is the sum of units * unit_price over its rows.
""")
ITEMS = [("chain", 24.0), ("tube", 7.5), ("tyre", 39.0), ("brake pads", 18.25),
         ("helmet", 64.0), ("light set", 45.5), ("service", 85.0), ("saddle", 52.0)]
for q, month0 in zip(range(1, 5), (1, 4, 7, 10)):
    rows = ["date,item,units,unit_price"]
    for k in range(9):
        item, price = rng.choice(ITEMS)
        rows.append(f"2026-{month0 + k // 3:02d}-{rng.randrange(1, 28):02d},{item},"
                    f"{rng.randrange(2, 40)},{price:.2f}")
    write(SANDBOX / "sales" / f"2026-q{q}.csv", "\n".join(rows) + "\n")
for d, day in enumerate(["09", "10", "11", "12", "13"]):
    people = rng.sample(CREW, 3)
    body = [f"# Workshop notes, 2026-11-{day}", ""]
    for p in people:
        body.append(f"- {p}: {rng.randrange(3, 9)} hours, {rng.randrange(2, 9)} "
                    f"{rng.choice(JOBS)}")
    body += ["", rng.choice(FILL).format(p=rng.choice(CREW)),
             rng.choice(FILL).format(p=rng.choice(CREW))]
    if day == "11":
        body.append("Problem: the hydraulic bleed kit is leaking and needs replacing.")
    write(SANDBOX / "notes" / f"2026-11-{day}.md", "\n".join(body) + "\n")
write(SANDBOX / "suppliers.md", """# Suppliers

| Supplier | Supplies | Terms |
|---|---|---|
| Northwind Cycle Parts | chains, tubes, tyres | 30 days |
| Kestrel Components | brake pads, light sets | 14 days |
| Alder & Finch | helmets, saddles | 45 days |
""")

# A long incident log pasted into the chat for the long-context tool task (~20K tokens).
COMP = ["card-reader", "booking-api", "label-printer", "inventory-sync", "email-relay"]
log = []
t = 0
for i in range(1500):
    t += rng.randrange(5, 90)
    h, m, s = 8 + t // 3600, (t // 60) % 60, t % 60
    c = rng.choice(COMP)
    level = rng.choices(["INFO", "WARN", "ERROR"], [80, 14, 6])[0]
    if c == "inventory-sync" and level == "INFO" and rng.random() < 0.08:
        level = "ERROR"
    msg = {"INFO": "ok", "WARN": "slow response", "ERROR": "request failed"}[level]
    log.append(f"2026-11-12 {h:02d}:{m:02d}:{s:02d} {level:5} {c}: {msg} (req {rng.randrange(10**5, 10**6)})")
write(PROMPTS / "incident-log.txt", "\n".join(log))

# ---- The skill -------------------------------------------------------------------------

write(SKILLS / "weekly-report" / "SKILL.md", """---
name: weekly-report
description: Write the workshop's weekly status report from the daily notes, with hours totalled, and save it as a file.
---

# Weekly workshop report

Follow these steps in order.

1. Call `list_directory` on the notes folder to see which daily notes exist.
2. Call `read_file` once with every note's path, one path a line.
3. Call `calculate` once to total the hours worked per person and overall, one expression a line.
4. Call `write_file` to save the report as `weekly-report.md` in the folder above the notes folder, with exactly these sections:
   - `# Weekly report, week of <first date>`
   - `## Hours` - a table with one row per person and a total row.
   - `## Work done` - one bullet per day, a short sentence each.
   - `## Problems` - every problem the notes mention, or "None".
5. Reply with two sentences: where the report was saved and the total hours.

Do not make up hours or names; use only what the notes say.
""")
print("fixtures written to", HERE)
