"""The tools every engine is offered (Vireo's catalog names and descriptions), run here by
the harness on a fresh copy of the harbor-bikes sandbox, with Vireo's 8,000-character cap
on a result applied for all. From evals/speed/bench-2026-11-13/bench.py, plus switches
for the tasks that need a refused write or a failing first read."""
import ast
import operator
import os
import re
import shutil
import tempfile
from pathlib import Path

from common import FIX

TOOL_SPECS = [
    ("read_file", "Read a text file, PDF, or Word, Excel or PowerPoint document on this PC. To read several files, give all their paths in one call, one a line, rather than a call for each.",
     {"path": "The file's path: a full path, or one relative to the home folder (Documents/report.pdf). Several files: one path a line. Don't make one up."}),
    ("list_directory", "List the files and folders in a folder on this PC.",
     {"path": "The folder's path: a full path, or one relative to the home folder (Documents)."}),
    ("search_files", "Find files and folders by name, or files containing some text, in a folder on this PC and its subfolders.",
     {"pattern": "A name or part of one, or the text to look for.",
      "directory": "The folder to search: a full path, or one relative to the home folder."}),
    ("calculate", "Work out arithmetic exactly. Use it for every total, sum, difference, percentage or average instead of doing the maths yourself, e.g. \"412.50 + 129.30 + 400\" or \"sum([12.40, 8.99, 31.00]) * 1.13\". Dollar signs, thousands commas and % are fine. Work out everything you need in one call, one expression a line, rather than a call for each.",
     {"expression": "Numbers with + - * / % ** and brackets, or round, abs, min, max, sum. For several results, one expression a line."}),
    ("write_file", "Create a text file on this PC, or replace one's whole text. Asks the user for approval.",
     {"path": "The file's path: a full path, or one relative to the home folder.",
      "content": "The file's whole new text."}),
]
OPENAI_TOOLS = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in p.items()},
                   "required": list(p)}}} for n, d, p in TOOL_SPECS]
# vireo-server has calculate built in; the rest it asks the client (us) to run.
VIREO_CLIENT_TOOLS = ["read_file", "list_directory", "search_files", "write_file"]


class Desk:
    def __init__(self, deny_writes=False, fail_first_read=False):
        self.root = Path(tempfile.mkdtemp(prefix="vbench-desk-")) / "harbor-bikes"
        shutil.copytree(FIX / "sandbox" / "harbor-bikes", self.root)
        self.written, self.read, self.calls = {}, [], []
        self.deny_writes, self.fail_first_read = deny_writes, fail_first_read

    @property
    def posix(self):
        return self.root.as_posix()

    def path(self, p):
        p = (p or "").strip().strip("'\"")
        q = Path(p) if os.path.isabs(p) else self.root.parent / p
        q = q.resolve()
        if not os.path.normcase(str(q)).startswith(os.path.normcase(str(self.root.resolve().parent))):
            raise ValueError(f"{p}: no such file or folder")
        return q

    def run(self, tool, args):
        self.calls.append(tool)
        try:
            if tool == "read_file":
                if self.fail_first_read:
                    self.fail_first_read = False
                    return False, "The file is locked by another program right now. Try again."
                paths = [x for x in str(args.get("path", "")).splitlines() if x.strip()]
                for p in paths:
                    self.read.append(self.path(p).name)
                if len(paths) > 1:
                    share = 8000 * min(3, len(paths)) // len(paths) - 120
                    out = []
                    for p in paths[:10]:
                        try:
                            t = self.path(p).read_text()
                        except Exception as e:
                            t = f"Error: {e}"
                        out.append(f"=== {p} ===\n" + (t[:share] + f"\n[... {len(t) - share} more characters not shown]" if len(t) > share else t.rstrip()))
                    return True, "\n\n".join(out)
                return True, self.path(paths[0] if paths else "").read_text()[:8000]
            if tool == "list_directory":
                d = self.path(args.get("path", ""))
                return True, "\n".join(f"{x.name}/" if x.is_dir() else f"{x.name} ({x.stat().st_size} bytes)"
                                       for x in sorted(d.iterdir()))
            if tool == "search_files":
                d, pat = self.path(args.get("directory", "")), str(args.get("pattern", "")).lower()
                hits = []
                for f in sorted(d.rglob("*")):
                    if pat in f.name.lower():
                        hits.append(str(f))
                    elif f.is_file():
                        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                            if pat in line.lower():
                                hits.append(f"{f}:{i}: {line.strip()}")
                return True, "\n".join(hits[:60]) or "No matches."
            if tool == "calculate":
                return True, calc(str(args.get("expression", "")))
            if tool == "write_file":
                if self.deny_writes:
                    return False, "The user declined this change. Nothing was written."
                p = self.path(args.get("path", ""))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(args.get("content", "")))
                self.written[str(p)] = str(args.get("content", ""))
                return True, f"Wrote {p}"
            return False, f"Unknown tool {tool}"
        except Exception as e:
            return False, str(e)

    def close(self):
        shutil.rmtree(self.root.parent, ignore_errors=True)


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sum": sum}


def calc(expr):
    out = []
    for line in [x for x in expr.splitlines() if x.strip()]:
        e = line.split("=")[-1] if "=" in line and "==" not in line else line
        e = e.replace("$", "").replace("%", "/100").replace("×", "*")
        e = re.sub(r"(?<=\d),(?=\d{3})", "", e)

        def ev(n):
            if isinstance(n, ast.Expression):
                return ev(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp):
                return OPS[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp):
                return OPS[type(n.op)](ev(n.operand))
            if isinstance(n, (ast.List, ast.Tuple)):
                return [ev(x) for x in n.elts]
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS:
                return FUNCS[n.func.id](*[ev(a) for a in n.args])
            raise ValueError("unsupported")
        try:
            v = ev(ast.parse(e.strip(), mode="eval"))
            v = round(v, 6) if isinstance(v, float) else v
            out.append(f"{line.strip()} = {v}")
        except Exception:
            out.append(f"{line.strip()} = error: can't work that out")
    return "\n".join(out) or "error: empty expression"
