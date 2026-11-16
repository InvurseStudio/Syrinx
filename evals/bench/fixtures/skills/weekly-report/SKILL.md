---
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
