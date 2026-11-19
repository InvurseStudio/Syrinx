---
name: weekly-review
description: Review the past week and plan the next from reminders and the calendar. Use when asked for a weekly review or to plan the week.
---

# Weekly review

1. Gather everything in one step, calling these together (today's date is in your instructions):
   - `list_reminders` with `include_completed` true, for what got done in the last seven days and what is still open or overdue;
   - `list_events` for the past seven days, to see what happened;
   - `list_events` for the next seven days;
   - `list_reminders` for what is due in the next seven days.
2. Write the review:
   - **Done this week:** the main completed items and meetings, grouped where it helps.
   - **Still open:** overdue and unfinished items, most urgent first.
   - **Next week:** the busiest days, important meetings and deadlines.
   - **Suggestions:** two or three concrete things, such as moving an overdue item, blocking time before a deadline, or dropping something that keeps slipping.
3. Offer to make this a weekly habit. If the user agrees, call `schedule_task` with a name like "Weekly review", the prompt "Do my weekly review", a time (for example "16:00") and the day they choose (for example ["fri"]). The app asks them to approve it.

Keep it to what fits on one screen. If reminders or calendar aren't available, say so and review from what the user tells you.
