#!/usr/bin/env python3
"""Plays the Aurelia gauntlet (docs/dev/test-gauntlet/aurelia_full.json, 32 prompts) in the
prototype Mac app, on vireo-server, and grades each prompt with the suite's own checks.

The prompts are typed into the real app by its test-script runner (Test > Run Test Script,
here started with --run-script), so the run can be watched: the chats, tool steps and
approval cards are the app's own. After each prompt the app writes that step's record and
waits; this script grades it against the files as that prompt left them, then lets the
next one start.

    python3 tools/aurelia_run.py --model <file.gguf> [--label gemma4-31b] [--only 2.1,2.2]
    python3 tools/aurelia_run.py --compare <new run folder> <old run folder> [--grades gemma]

What a run touches:

- The app's chats and chat cache (~/Library/Application Support/VireoPrototype/Chats and
  chat-cache) are DELETED before the run, so it starts with none. --keep-chats leaves them.
- Everything else is the run's own, in a scratch folder: a fresh copy of the test files
  (made by make_aurelia.py), the memory file, the approvals file and an empty skills
  folder. The user's memory, skills, saved approvals and ~/Documents are not used. The
  prompts that name "~/Documents/Vireo App Test/Aurelia Minerals" are sent with the
  scratch copy's path in its place (each record keeps the original as `prompt_as_written`).
- The chats get "Folders I choose" access to the scratch copy only, so nothing else on the
  Mac (the answer key included) can be read or changed. A file the model deletes goes to
  the Mac's Trash.
- The app's model becomes --model, and stays so afterwards.

Grading is run_full.py's: the suite's checks are hints (text in the answer, tools called,
text in the files, facts in memory), ported as they are. The grade here is worked out from
the hints alone: pass when every hint that applies passes, partial when at least half do,
fail otherwise. The 0.9 runs were graded by a person reading the answers, so compare hints
with hints (--compare does) and read the answers before calling a prompt better or worse.

Prompts the prototype can't run fairly are marked "not runnable" with the reason, and
still sent so what the model did is on record (none are, since 2026-11-13).

The calendar prompts (4.4, 9.1) run dry: the app pretends to add the events and reminders
(nothing goes on the Mac's own calendar) and each record lists what would have been added.
The automation prompts (11.1 to 11.3) use the server's tasks, kept in the scratch folder:
after 11.1 the waiting files are moved into the watched folder, after 11.2 the link task's
link is called, and 11.3 is made as a task and started with Run now. The app runs each task
as a chat of its own and leaves its record in the gate folder (task-run-<n>.json).

--thinking on|off sets the model's thinking for the run (on by default, as the 0.9 runs
had it).

Standard library only; openpyxl and python-docx are used to read the files when they are
there (server/venv/bin/python has them), as run_full.py did.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent
REPO = PROTO.parent
GAUNTLET = REPO / 'docs/dev/test-gauntlet'
SUITE = GAUNTLET / 'aurelia_full.json'
APP = PROTO / 'Mac/App/release/Vireo-Proto.app'
BUNDLE_ID = 'com.invurseai.vireo.prototype'
APP_DATA = Path.home() / 'Library/Application Support/VireoPrototype'
WRITTEN_FOLDER = '~/Documents/Vireo App Test/Aurelia Minerals'
PROMPT_TIMEOUT = 30 * 60
START_TIMEOUT = 12 * 60

# What vireo-server has no way to do on the Mac yet, so the prompt can't be run fairly.
# (Calendar, reminders, tasks and chat search came on 2026-11-13: none left.)
NOT_RUNNABLE: dict = {}
# Tools a check names that the Mac prototype doesn't offer: a check that wants one of
# only these is left out of the count.
NOT_ON_THE_MAC = {'run_shell_command'}
# How long the app's task runner may take to pick up a task that's due (it asks the server
# every 10 s, and a new file is left 3 s to settle).
TASK_PICKUP = 40


def log(*parts) -> None:
    print(time.strftime('%H:%M:%S'), *parts, flush=True)


# ─── Reading the files (run_quick.py's file_text) ────────────────────────────
def _xml(archive: zipfile.ZipFile, name: str):
    return ElementTree.fromstring(archive.read(name))


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _xlsx_text(path: Path) -> str:
    """Every sheet's rows, cells joined with " | " (what openpyxl gives, near enough)."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shared = []
        if 'xl/sharedStrings.xml' in names:
            for item in _xml(archive, 'xl/sharedStrings.xml'):
                shared.append(''.join(t.text or '' for t in item.iter() if _local(t.tag) == 't'))
        lines = []
        for name in sorted(n for n in names if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n)):
            for row in (e for e in _xml(archive, name).iter() if _local(e.tag) == 'row'):
                cells = []
                for cell in (e for e in row if _local(e.tag) == 'c'):
                    kind = cell.get('t')
                    value = next((e.text or '' for e in cell if _local(e.tag) == 'v'), '')
                    if kind == 's' and value.isdigit():
                        value = shared[int(value)]
                    elif kind == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.iter() if _local(t.tag) == 't')
                    cells.append(value)
                lines.append(' | '.join(cells))
        return '\n'.join(lines)


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        body = _xml(archive, 'word/document.xml')
    return '\n'.join(''.join(t.text or '' for t in p.iter() if _local(t.tag) == 't')
                     for p in body.iter() if _local(p.tag) == 'p')


def file_text(path: Path) -> str:
    if path.suffix == '.xlsx':
        try:
            import openpyxl  # noqa: PLC0415
        except ImportError:
            return _xlsx_text(path)
        book = openpyxl.load_workbook(path, data_only=True)
        return '\n'.join(' | '.join('' if c is None else str(c) for c in row)
                         for sheet in book.worksheets for row in sheet.iter_rows(values_only=True))
    if path.suffix == '.docx':
        try:
            import docx  # noqa: PLC0415
        except ImportError:
            return _docx_text(path)
        doc = docx.Document(path)
        parts = [p.text for p in doc.paragraphs]
        parts += [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
        return '\n'.join(parts)
    return path.read_text(errors='replace')


def readers() -> str:
    found = []
    for name in ('openpyxl', 'docx'):
        try:
            __import__(name)
            found.append(name)
        except ImportError:
            pass
    return ', '.join(found) if len(found) == 2 else f'standard library (found: {", ".join(found) or "neither"})'


# ─── Checks (run_quick.py's check, with run_full.py's tools_any_*) ───────────
def plain(text: str) -> str:
    """Lower case, thousands separators and currency signs removed, so 38,981.55 matches 38981.55."""
    text = re.sub(r'(?<=\d),(?=\d{3})', '', text or '')
    return text.replace('$', '').lower()


def check(item: dict, record: dict, folder: Path, docx_before: set, memories: list) -> list[str]:
    """One line per hint: ✓ passed, ✗ failed, – left out (it wants a tool the Mac
    prototype doesn't have)."""
    results = []
    checks = item.get('checks', {})
    answer = plain(record['answer'])
    tools = [t['tool'] for t in record['tools']]

    def mark(ok: bool, text: str) -> None:
        results.append(f'{"✓" if ok else "✗"} {text}')

    for key, wanted in checks.items():
        if key.startswith('answer_all'):
            for w in wanted:
                mark(plain(w) in answer, f'answer has "{w}"')
        elif key.startswith('answer_any'):
            mark(any(plain(w) in answer for w in wanted), f'answer has one of {wanted}')
        elif key.startswith('answer_none'):
            for w in wanted:
                mark(plain(w) not in answer, f'answer avoids "{w}"')
        elif key.startswith('tools_any'):
            if all(w in NOT_ON_THE_MAC for w in wanted):
                results.append(f'– used one of {wanted} (not a tool of the Mac prototype)')
            else:
                mark(any(t in tools for t in wanted), f'used one of {wanted}')
        elif key == 'tools_none':
            mark(not any(t in tools for t in wanted), f'did not use {wanted}')
        elif key.startswith('memory_'):
            # The scratch folder's own name is not something the model chose to keep (a
            # temporary folder's name can hold "4000", which 12.2 looks for).
            texts = plain('\n'.join(memories).replace(str(folder), WRITTEN_FOLDER))
            if key == 'memory_all':
                for w in wanted:
                    mark(plain(w) in texts, f'a memory has "{w}"')
            elif key == 'memory_any':
                mark(any(plain(w) in texts for w in wanted), f'a memory has one of {wanted}')
            elif key == 'memory_none':
                for w in wanted:
                    mark(plain(w) not in texts, f'no memory has "{w}"')
        elif key == 'file':
            for spec in wanted:
                path = folder / spec['path']
                text = plain(file_text(path)) if path.exists() else ''
                for w in spec.get('all', []):
                    mark(plain(w) in text, f'{spec["path"]} has "{w}"')
                for w in spec.get('none', []):
                    mark(plain(w) not in text, f'{spec["path"]} avoids "{w}"')
                for w, n in spec.get('count', {}).items():
                    found = text.count(plain(w))
                    mark(found == n, f'{spec["path"]} has "{w}" {n}x (found {found})')
        elif key == 'new_docx':
            new = sorted(set(glob.glob(str(folder / '**/*.docx'), recursive=True)) - docx_before)
            record['new_files'] = [str(Path(p).relative_to(folder)) for p in new]
            mark(wanted['min'] <= len(new) <= wanted['max'],
                 f'{wanted["min"]}-{wanted["max"]} new .docx (found {len(new)}: {", ".join(record["new_files"])})')
            together = plain('\n'.join(file_text(Path(p)) for p in new))
            for w in wanted.get('all_together', []):
                mark(plain(w) in together, f'new .docx files have "{w}"')
    return results


def hint_counts(hints: list) -> tuple[int, int]:
    """Hints passed, and hints that count (those left out with "–" don't)."""
    return sum(h.startswith('✓') for h in hints), sum(h.startswith(('✓', '✗')) for h in hints)


def grade_from_hints(hints: list, error) -> str:
    """The grade the hints alone give. The 0.9 runs' grades were a person's, from the answers."""
    passed, counted = hint_counts(hints)
    if error:
        return 'fail'
    if counted == 0:
        return 'no hints'
    if passed == counted:
        return 'pass'
    return 'partial' if passed * 2 >= counted else 'fail'


# ─── The app ─────────────────────────────────────────────────────────────────
APP_PROCESS = 'Vireo-Proto.app/Contents/MacOS/VireoMac'


def app_running() -> bool:
    return subprocess.run(['pgrep', '-f', APP_PROCESS],
                          capture_output=True).returncode == 0


def quit_app() -> None:
    if not app_running():
        return
    log('Quitting Vireo-Proto')
    # Asked as a user would; then, for a copy started some other way (which doesn't hear
    # that), told to end. Its chats are saved as they go, and its server stops with it.
    for ask in (['osascript', '-e', f'tell application id "{BUNDLE_ID}" to quit'],
                ['pkill', '-TERM', '-f', APP_PROCESS], ['pkill', '-KILL', '-f', APP_PROCESS]):
        subprocess.run(ask, capture_output=True)
        for _ in range(20):
            if not app_running():
                time.sleep(1)
                return
            time.sleep(1)
    raise SystemExit('Vireo-Proto did not quit; quit it by hand and run again.')


def delete_chats() -> list[str]:
    """The app's chats and their caches, so a run starts with none (the app is quit)."""
    gone = []
    for name in ('Chats', 'chat-cache'):
        path = APP_DATA / name
        if path.exists():
            shutil.rmtree(path)
            gone.append(str(path))
    return gone


def make_files(folder: Path, python: str | None, copy_from: str | None) -> str:
    """A fresh copy of the test files: built by make_aurelia.py, or copied."""
    if copy_from:
        shutil.copytree(Path(copy_from).expanduser(), folder)
        return f'copied from {copy_from}'
    python = python or str(REPO / 'server/venv/bin/python')
    made = subprocess.run([python, '-I', str(GAUNTLET / 'make_aurelia.py'), str(folder)], capture_output=True, text=True)
    if made.returncode != 0:
        raise SystemExit(f'make_aurelia.py failed (it needs python-docx, openpyxl and reportlab; '
                         f'pass --python or --copy-from):\n{made.stderr[-800:]}')
    return f'built by make_aurelia.py with {python}'


def as_sent(text: str, folder: Path) -> str:
    return text.replace(WRITTEN_FOLDER, str(folder))


def server_api(method: str, path: str, body=None, timeout: float = 30):
    """A call to the app's vireo-server, with the app's own token (both from its settings)."""
    def setting(key: str) -> str:
        return subprocess.run(['defaults', 'read', BUNDLE_ID, key], capture_output=True, text=True).stdout.strip()
    base = (setting('invurse_server_url') or 'http://localhost:3100').rstrip('/')
    request = urllib.request.Request(
        base + path, method=method, data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization': f'Bearer {setting("vireo_engine_server_token")}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode() or 'null')


def task_runs(gate: Path) -> tuple[int, list]:
    """How many task runs the app has started, and the records of those that have ended."""
    started = len(list(gate.glob('task-run-*.started')))
    ended = sorted(gate.glob('task-run-*.json'), key=lambda p: int(re.findall(r'\d+', p.stem)[-1]))
    return started, ended


def wait_for_task_runs(gate: Path, seen: int, seconds: float) -> list:
    """The records of the task runs since `seen` of them, once the app has gone quiet: at
    least one has ended, none is under way, and no other starts for a while after."""
    began = time.time()
    deadline = began + seconds
    quiet_since = None
    while time.time() < deadline and app_running():
        started, ended = task_runs(gate)
        if started <= seen and time.time() - began > 2 * TASK_PICKUP:
            break  # nothing was set off: no task watches for it
        if len(ended) > seen and started == len(ended):
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= TASK_PICKUP:
                break
        else:
            quiet_since = None
        time.sleep(1)
    return [json.loads(p.read_text()) for p in task_runs(gate)[1][seen:]]


def after_steps(item: dict, folder: Path, gate: Path, record: dict) -> None:
    """What the suite does after a prompt's reply (run_full.py's after_steps), for the
    prompts that set up a task: something lands in the watched folder, or the link is
    called, and the task's run is waited for. Its tools and answer go in `after`."""
    after = item.get('after') or {}
    seen = len(task_runs(gate)[1])
    began = time.time()
    if 'move_folder' in after:
        src, dst = (folder / part for part in after['move_folder'])
        dst.mkdir(parents=True, exist_ok=True)
        moved = []
        for path in sorted(src.iterdir()):
            if path.is_file():
                shutil.move(str(path), dst / path.name)
                moved.append(path.name)
        log(f'  moved into {dst.name}: {", ".join(moved)}')
        record['after'] = {'moved': moved}
    elif 'hook' in after:
        try:
            links = [t for t in server_api('GET', '/api/tasks').get('tasks', []) if t.get('link')]
        except (OSError, ValueError) as error:
            record['after'] = {'error': f'The server\'s tasks could not be read: {error}'}
            return
        record['after'] = {'link_tasks': [{k: t.get(k) for k in ('name', 'prompt', 'schedule', 'link')} for t in links]}
        if not links:
            record['after']['error'] = 'No link task was created.'
            log('  no link task was created')
            return
        url = links[-1]['link']
        record['after']['link_in_reply'] = url in record['answer']
        try:
            request = urllib.request.Request(url, data=after['hook']['text'].encode(), method='POST',
                                             headers={'Content-Type': 'text/plain; charset=utf-8'})
            with urllib.request.urlopen(request, timeout=30) as response:
                record['after']['post'] = response.read().decode()[:300]
        except OSError as error:
            record['after']['error'] = f'The link could not be called: {error}'
            return
        log(f'  posted to the link: {record["after"]["post"]}')
    else:
        return
    runs = wait_for_task_runs(gate, seen, after.get('wait_task_seconds', 600))
    record['after']['task_runs'] = runs
    record['after']['seconds'] = round(time.time() - began, 1)
    if not runs:
        record['after']['error'] = 'No task ran.'
    log(f'  {len(runs)} task run(s): ' + '; '.join(f'{len(r.get("tools", []))} tools' for r in runs))


def run_task_item(item: dict, folder: Path, gate: Path) -> dict:
    """11.3: a task made as the app's own form would make it, then Run now. Its record is
    the run's: the answer and tools of the chat the app played it in."""
    record = {'id': item['id'], 'title': item['title'], 'chatId': None, 'prompt': as_sent(item['prompt'], folder),
              'tools': [], 'approvals': [], 'events': [], 'answer': '', 'error': None, 'stats': {}, 'seconds': 0}
    seen = len(task_runs(gate)[1])
    began = time.time()
    schedule = item['task'].get('schedule', {})
    try:
        task = server_api('POST', '/api/tasks', {
            'name': item['task']['name'], 'prompt': record['prompt'],
            'when': f'daily {schedule.get("time", "08:00")}', 'access': 'folders', 'folders': [str(folder)]})
        server_api('POST', f'/api/tasks/{task["id"]}/run', {})
    except (OSError, ValueError, KeyError) as error:
        record['error'] = f'the task could not be made or started: {error}'
        return record
    record['task'] = task
    log(f'  task "{task["name"]}" ({task.get("schedule")}) started with Run now')
    runs = wait_for_task_runs(gate, seen, (item.get('after') or {}).get('wait_task_seconds', 600))
    record['seconds'] = round(time.time() - began - (TASK_PICKUP if runs else 0), 1)
    if not runs:
        record['error'] = 'the task did not run'
        return record
    run = runs[-1]
    record['answer'] = run.get('reply', '')
    record['events'] = run.get('toolSteps') or []
    record['tools'] = [{'tool': s.get('tool'), 'detail': (s.get('detail') or '')[:200]}
                       for s in record['events'] if s.get('type') == 'tool_call']
    record['approvals'] = [{'tool': a.get('tool'), 'approved': a.get('approved'),
                            'preview': str(a.get('preview', ''))[:4000], 'args': a.get('args')}
                           for a in run.get('approvals') or []]
    if run.get('error'):
        record['error'] = 'the task\'s reply ended in an error'
    return record


def build_script(suite: dict, prompts: list, folder: Path, gate: Path, thinking: bool) -> dict:
    steps, chat = [], None
    for item in prompts:
        if item.get('task'):  # made as a task and started with Run now, not typed
            continue
        if not (item.get('chat') == 'same' and chat):
            chat = f'{item["id"]} {item["title"]}'
        step = {'prompt': as_sent(item['prompt'], folder), 'title': f'{item["id"]} {item["title"]}', 'chat': chat}
        policy = item.get('approvals', 'approve')
        if isinstance(policy, dict):
            step['approveIf'] = policy['approve_if']
        else:
            step['approve'] = policy != 'reject'
        steps.append(step)
    return {'name': 'Aurelia', 'access': 'folders', 'folders': [str(folder)], 'gate': str(gate),
            'thinking': thinking, 'steps': steps}


def record_from(step: dict, item: dict, folder: Path) -> dict:
    """The app's record of a step, in the shape of run_full.py's."""
    tool_steps = step.get('toolSteps') or []
    record = {
        'id': item['id'], 'title': item['title'], 'chatId': step.get('chat'),
        'prompt': step.get('prompt', ''), 'tools': [], 'approvals': [], 'events': tool_steps,
        'answer': step.get('reply', ''), 'error': 'the reply ended in an error' if step.get('error') else None,
        'stats': {'tokens': step.get('tokens'), 'tokensPerSecond': step.get('tokensPerSecond'),
                  'firstWordSeconds': step.get('firstWordSeconds')},
        'seconds': round(step.get('seconds', 0), 1),
    }
    if record['prompt'] != item['prompt']:
        record['prompt_as_written'] = item['prompt']
    calls = [s for s in tool_steps if s.get('type') == 'tool_call']
    if not calls:  # an app without the fuller record
        calls = [{'tool': t} for t in step.get('tools', [])]
    record['tools'] = [{'tool': c.get('tool'), 'detail': (c.get('detail') or '')[:200]} for c in calls]
    record['approvals'] = [{'tool': a.get('tool'), 'approved': a.get('approved'),
                            'preview': str(a.get('preview', ''))[:4000], 'args': a.get('args')}
                           for a in step.get('approvalsAsked') or []]
    if step.get('calendar'):  # what a dry run would have put on the calendar
        record['calendar'] = step['calendar']
    return record


def wait_for(path: Path, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.exists():
            time.sleep(0.2)  # written whole (atomically), but let the app settle
            return True
        if not app_running():
            return False
        time.sleep(0.5)
    return False


# ─── Report ──────────────────────────────────────────────────────────────────
def summary_rows(records: list) -> list[dict]:
    rows = []
    for r in records:
        passed, counted = hint_counts(r.get('checks', []))
        rows.append({'id': r['id'], 'title': r['title'], 'grade': r['grade'], 'hints_passed': passed,
                     'hints': counted, 'seconds': r['seconds'], 'tools': len(r['tools']),
                     'approvals': len(r['approvals']),
                     'approved': sum(1 for a in r['approvals'] if a['approved']),
                     **({'not_runnable': r['not_runnable']} if r.get('not_runnable') else {})})
    return rows


def table(rows: list) -> str:
    lines = ['| # | Title | Grade (hints) | Hints | Seconds | Tools | Approvals |', '|---|---|---|---|---|---|---|']
    for r in rows:
        asked = f'{r["approvals"]}' + (f' ({r["approvals"] - r["approved"]} declined)' if r['approvals'] != r['approved'] else '')
        lines.append(f'| {r["id"]} | {r["title"]} | {r["grade"]} | {r["hints_passed"]}/{r["hints"]} '
                     f'| {r["seconds"]:.0f} | {r["tools"]} | {asked} |')
    return '\n'.join(lines)


def totals(rows: list) -> dict:
    count = {g: sum(1 for r in rows if r['grade'] == g) for g in ('pass', 'partial', 'fail', 'not runnable', 'no hints')}
    ran = [r for r in rows if r['grade'] != 'not runnable']
    return {**count, 'prompts': len(rows),
            'hints_passed': sum(r['hints_passed'] for r in ran), 'hints': sum(r['hints'] for r in ran),
            'minutes': round(sum(r['seconds'] for r in rows) / 60, 1), 'tool_calls': sum(r['tools'] for r in rows)}


def report(out: Path, suite: dict, meta: dict, records: list) -> None:
    rows = summary_rows(records)
    total = totals(rows)
    (out / 'summary.json').write_text(json.dumps({'meta': meta, 'totals': total, 'prompts': rows}, indent=1, default=str))
    lines = [f'# {suite["name"]} on the prototype, {meta["started"]}, {meta["label"]}', '',
             'Played in the prototype Mac app (vireo-server) by `VireoPrototype/tools/aurelia_run.py`. '
             'The grade is worked out from the hints alone; see that script for what it does and does not show.', '',
             '## Setup', '', '```json', json.dumps(meta, indent=2, default=str), '```', '',
             '## Summary', '', table(rows), '',
             f'Pass {total["pass"]}, partial {total["partial"]}, fail {total["fail"]}, not runnable {total["not runnable"]} '
             f'of {total["prompts"]} · hints {total["hints_passed"]}/{total["hints"]} on the prompts that ran · '
             f'{total["minutes"]} min · {total["tool_calls"]} tool calls', '']
    for r in records:
        item = next(p for p in suite['prompts'] if p['id'] == r['id'])
        lines += [f'## {r["id"]} {item["title"]}' + (' (same chat)' if item.get('chat') == 'same' else ''), '',
                  f'**Prompt:** {r["prompt"]}', '', f'**Expected:** {item["expect"]}', '',
                  f'**Time:** {r["seconds"]} s · **Stats:** {r.get("stats")}' + (f' · **Error:** {r["error"]}' if r.get('error') else ''), '',
                  '**Tools:** ' + (', '.join(f'{t["tool"]}({t["detail"]})' if t['detail'] else t['tool'] for t in r['tools']) or 'none'), '']
        if r.get('not_runnable'):
            lines += [f'**Not runnable:** this prompt {r["not_runnable"]}.', '']
        for a in r['approvals']:
            lines += [f'- {"Approved" if a["approved"] else "Declined"} `{a["tool"]}`:', '', '```', a['preview'], '```', '']
        if r.get('calendar'):
            lines += ['**Calendar (pretended, not added to the Mac\'s own):**', ''] + [f'- {c}' for c in r['calendar']] + ['']
        for run in (r.get('after') or {}).get('task_runs', []):
            lines += [f'**Task run "{run.get("task")}"** ({run.get("seconds", 0):.0f} s, tools: {", ".join(run.get("tools", [])) or "none"}):',
                      '', (run.get('reply') or '(empty)').strip(), '']
        if (r.get('after') or {}).get('error'):
            lines += [f'**After the reply:** {r["after"]["error"]}', '']
        lines += ['**Hints:**', ''] + [f'- {h}' for h in r.get('checks', [])] + [
            '', f'**Grade from the hints:** {r["grade"]}', '', '**Answer:**', '', r['answer'].strip() or '(empty)', '']
    (out / 'report.md').write_text('\n'.join(lines))


# ─── A run ───────────────────────────────────────────────────────────────────
def run(args) -> None:
    model = Path(args.model).expanduser()
    if not model.is_file():
        raise SystemExit(f'No model at {model}')
    binary = Path(args.app) / 'Contents/MacOS/VireoMac'
    if not binary.is_file():
        raise SystemExit(f'No app at {args.app}; build it with Mac/App/build-test-app.sh')
    suite = json.loads(SUITE.read_text())
    only = set(args.only.split(',')) if args.only else None
    prompts = [p for p in suite['prompts'] if not only or p['id'] in only]
    if not prompts:
        raise SystemExit('No prompt has those ids.')

    stamp = dt.datetime.now().strftime('%Y-%m-%d-%H%M')
    label = args.label or re.sub(r'[^a-z0-9.]+', '-', model.stem.lower()).strip('-')
    out = Path(args.out).expanduser() if args.out else GAUNTLET / 'runs' / f'proto-{stamp}-{label}'
    work = Path(os.path.realpath(tempfile.mkdtemp(prefix='vireo-aurelia-', dir=args.work)))
    folder, gate = work / 'Aurelia Minerals', work / 'gate'
    memory_file, skills = work / 'memories.txt', work / 'skills'
    gate.mkdir()
    memory_file.write_text('')
    if args.skills:
        shutil.copytree(Path(args.skills).expanduser(), skills)
    else:
        skills.mkdir()
    built_in = REPO / 'server/builtin-skills'
    if args.builtin_skills == 'off':
        built_in = work / 'no-builtin-skills'
        built_in.mkdir()

    quit_app()
    deleted = [] if args.keep_chats else delete_chats()
    for path in deleted:
        log(f'Deleted {path}')
    files_from = make_files(folder, args.python, args.copy_from)
    log(f'Test files: {folder} ({files_from})')
    thinking = args.thinking == 'on'
    script = build_script(suite, prompts, folder, gate, thinking)
    if not script['steps']:
        raise SystemExit('A task prompt (11.3) needs a typed prompt in the same run to start the app\'s '
                         'script with, e.g. --only 8.2,11.3')
    (work / 'script.json').write_text(json.dumps(script, indent=1))

    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = {
        'label': label, 'started': stamp, 'model': model.name, 'app': str(args.app), 'commit': commit,
        'played_in': 'the prototype Mac app (its test-script runner), on vireo-server inside the app',
        'access': 'folders', 'folder': str(folder), 'files': files_from, 'thinking': args.thinking,
        'calendar': 'pretended by the app (a dry run): nothing is added to the Mac\'s calendar or reminders',
        'path_rewritten': f'"{WRITTEN_FOLDER}" in a prompt is sent as "{folder}"',
        'chat_settings': 'the app\'s defaults for a new chat (system prompt, sampling, reply length)',
        'chats_deleted_before_run': deleted, 'memory_file': str(memory_file),
        'user_skills': args.skills or 'none (an empty folder)',
        'builtin_skills': str(built_in) if args.builtin_skills == 'on' else 'none',
        'file_readers': readers(), 'prompts': [p['id'] for p in prompts],
        'grading': 'pass = every hint that applies passed; partial = at least half; fail = fewer, or the reply ended in an error',
        'seconds_note': 'as the app times a reply; each approval card is left up 1.5 s before it is answered',
    }
    # Opened as the Finder opens it (a new copy, in front), with the run's own memory,
    # approvals and skills; the server the app starts inherits them.
    env = {'VIREO_MEMORY_FILE': memory_file, 'VIREO_APPROVALS_FILE': work / 'approvals.txt',
           'VIREO_TASKS_FILE': work / 'tasks.json',
           'INVURSE_SKILLS_DIR': skills, 'VIREO_BUILTIN_SKILLS': built_in}
    opened = subprocess.run(
        ['open', '-n', '-a', str(args.app), '--stdout', str(work / 'app.log'), '--stderr', str(work / 'app.log')]
        + [part for key, value in env.items() for part in ('--env', f'{key}={value}')]
        + ['--args', '--run-script', str(work / 'script.json'), '--model', str(model)],
        capture_output=True, text=True)
    if opened.returncode != 0:
        raise SystemExit(f"The app didn't open: {opened.stderr.strip()}")
    time.sleep(3)
    log(f'Vireo-Proto started with {model.name}; {len(prompts)} prompts. Results: {out}')

    out.mkdir(parents=True, exist_ok=True)
    records: list = []

    def graded(item: dict, record: dict, docx_before: set) -> None:
        memories = [line for line in memory_file.read_text(errors='replace').splitlines() if line.strip()]
        record['memories'] = [m[:400] for m in memories]
        try:
            record['checks'] = check(item, record, folder, docx_before, memories)
        except Exception as error:  # noqa: BLE001
            record['checks'] = [f'✗ check error: {type(error).__name__}: {error}']
        record['grade_from_hints'] = grade_from_hints(record['checks'], record['error'])
        if item['id'] in NOT_RUNNABLE:
            record['not_runnable'] = NOT_RUNNABLE[item['id']]
        record['grade'] = 'not runnable' if record.get('not_runnable') else record['grade_from_hints']
        records.append(record)
        (out / f'{item["id"]}.json').write_text(json.dumps(record, indent=1, default=str))
        passed, counted = hint_counts(record['checks'])
        log(f'  {record["grade"]}: {record["seconds"]} s, {len(record["tools"])} tools, '
            f'{len(record["approvals"])} approvals, hints {passed}/{counted}'
            + (f', ERROR {record["error"]}' if record['error'] else ''))
        report(out, suite, meta, records)

    # The app waits at each typed prompt's gate until its `.done` is written. That is held
    # back until the next typed prompt is wanted, so a task prompt in between (11.3) runs
    # while the app is waiting, not alongside the next reply.
    held_gate = None
    step = 0
    try:
        for item in prompts:
            docx_before = set(glob.glob(str(folder / '**/*.docx'), recursive=True))
            log(f'{item["id"]} {item["title"]}')
            if item.get('task'):
                if step == 0:  # nothing typed yet: the app is still starting its model
                    deadline = time.time() + START_TIMEOUT
                    while time.time() < deadline and app_running():
                        try:
                            server_api('GET', '/api/tasks', timeout=5)
                            break
                        except (OSError, ValueError):
                            time.sleep(2)
                graded(item, run_task_item(item, folder, gate), docx_before)
                continue
            if held_gate:
                held_gate.write_text('')
            step += 1
            step_file = gate / f'step-{step}.json'
            if not wait_for(step_file, PROMPT_TIMEOUT + (START_TIMEOUT if step == 1 else 0)):
                reason = 'the app quit' if not app_running() else 'no reply in time'
                log(f'  stopped: {reason}')
                meta['stopped'] = f'at {item["id"]}: {reason}'
                held_gate = None
                break
            held_gate = gate / f'step-{step}.done'
            record = record_from(json.loads(step_file.read_text()), item, folder)
            after_steps(item, folder, gate, record)
            graded(item, record, docx_before)
        if held_gate:
            held_gate.write_text('')
    finally:
        if records:
            report(out, suite, meta, records)
            if folder.exists():
                shutil.copytree(folder, out / 'files', dirs_exist_ok=True)
        if args.quit_after:
            quit_app()
    if records:
        rows = summary_rows(records)
        print('\n' + table(rows) + '\n')
        print(json.dumps(totals(rows)))
    log(f'Report: {out / "report.md"}. The run\'s own files stay in {work}'
        + ('' if args.quit_after else '; the app is left open with the chats to read.'))


# ─── Comparing two runs ──────────────────────────────────────────────────────
def load_run(folder: Path) -> dict:
    records = {}
    for path in folder.glob('*.json'):
        if re.fullmatch(r'\d+\.\d+', path.stem):
            records[path.stem] = json.loads(path.read_text())
    return records


def compare(new: Path, old: Path, grades_key: str | None) -> None:
    """Prompt by prompt: the same hint rule applied to both runs' recorded hints, and the
    old run's grades as a person gave them (grades_after.py) when asked for."""
    new_run, old_run = load_run(new), load_run(old)
    by_hand = {}
    if grades_key:
        sys.path.insert(0, str(GAUNTLET))
        import grades_after  # noqa: PLC0415
        by_hand = {k: {'P': 'pass', 'H': 'partial', 'F': 'fail'}[v] for k, v in grades_after.AFTER[grades_key].items()}
    suite = json.loads(SUITE.read_text())
    order = {'fail': 0, 'partial': 1, 'pass': 2}
    print(f'| # | Title | Old: by hand | Old: hints | New: hints | Old s | New s | Old tools | New tools | By the hints |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    count = {'same': 0, 'better': 0, 'worse': 0, 'not runnable': 0}
    for item in suite['prompts']:
        key = item['id']
        a, b = old_run.get(key), new_run.get(key)
        if not b:
            continue
        old_hints = a.get('checks', []) if a else []
        old_grade = grade_from_hints(old_hints, a.get('error')) if a else '-'
        op, oc = hint_counts(old_hints)
        np_, nc = hint_counts(b.get('checks', []))
        new_grade = b.get('grade_from_hints') or grade_from_hints(b.get('checks', []), b.get('error'))
        if b.get('not_runnable'):
            verdict = 'not runnable'
        elif old_grade in order and new_grade in order:
            verdict = 'same' if order[old_grade] == order[new_grade] else 'better' if order[new_grade] > order[old_grade] else 'worse'
        else:
            verdict = '-'
        if verdict in count:
            count[verdict] += 1
        old_seconds = (a['seconds'] + (a.get('after') or {}).get('seconds', 0)) if a else 0
        print(f'| {key} | {item["title"]} | {by_hand.get(key, "-")} | {old_grade} {op}/{oc} | {new_grade} {np_}/{nc} '
              f'| {old_seconds:.0f} | {b["seconds"]:.0f} | {len(a["tools"]) if a else "-"} | {len(b["tools"])} | {verdict} |')
    print(f'\nBy the hints: {json.dumps(count)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', help='the GGUF file the app should run')
    parser.add_argument('--label', help='the model\'s name in the run folder (default: from the file name)')
    parser.add_argument('--only', help='comma-separated prompt ids, e.g. 2.1,2.2,7.1,12.3')
    parser.add_argument('--out', help='the run folder (default: docs/dev/test-gauntlet/runs/proto-<date>-<label>)')
    parser.add_argument('--app', default=str(APP), help='Vireo-Proto.app (default: Mac/App/release)')
    parser.add_argument('--work', help='where the run\'s scratch folder is made (default: the temporary folder)')
    parser.add_argument('--python', help='a Python with python-docx, openpyxl and reportlab, for make_aurelia.py '
                                         '(default: server/venv/bin/python)')
    parser.add_argument('--copy-from', help='copy the test files from this folder instead of building them')
    parser.add_argument('--skills', help='a folder of the user\'s skills to run with (default: none)')
    parser.add_argument('--builtin-skills', choices=['on', 'off'], default='on',
                        help='the repository\'s server/builtin-skills (default: on)')
    parser.add_argument('--keep-chats', action='store_true', help="don't delete the app's chats and chat cache first")
    parser.add_argument('--thinking', choices=['on', 'off'], default='on',
                        help='whether the model thinks before it answers (default: on)')
    parser.add_argument('--quit-after', action='store_true', help='quit the app when the run ends')
    parser.add_argument('--compare', nargs=2, metavar=('NEW', 'OLD'), help='compare two run folders and stop')
    parser.add_argument('--grades', help='with --compare: the old run\'s grades by hand, by model key in grades_after.py (gemma, muse, qwen)')
    args = parser.parse_args()
    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]), args.grades)
        return
    if not args.model:
        parser.error('--model is needed (or --compare)')
    run(args)


if __name__ == '__main__':
    main()
