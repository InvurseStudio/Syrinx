---
name: plan-my-day
description: Plan the user's day from their calendar and reminders and offer to block focus time. Use when asked to plan the day or fit something in today.
---

# Plan my day

1. In one step, call all three together: `list_events` for today, `list_reminders` with `due_before` set to the end of today, and `find_free_time` for the rest of today (default working hours unless the user says otherwise). Today's date is in your instructions; call `get_current_datetime` in that same step only if you need the exact time. Note anything overdue.
2. Write the plan:
   - **Fixed:** today's events in time order, with times.
   - **To do:** reminders due today and overdue ones, most urgent first.
   - **Suggested schedule:** put the to-dos into the free slots, hardest or most urgent work in the longest slot. Leave short gaps between back-to-back items.
   - One line on anything that looks tight or double-booked.
3. Offer to add the suggested blocks to the calendar. Only call `create_event` for the blocks the user agrees to (the app asks them to approve each one).

Keep it short: a plan they can glance at, not an essay. Past events from earlier today can be left out.

If calendar or reminder tools aren't available (for example on a paired device without calendar access), say so in one line and plan from whatever the user tells you instead.
