"""Builds the bench's prompts from the frozen fixtures, and the small scoring helpers."""
import json
import re

import common

_corpus = None
_cuts = None


def corpus():
    global _corpus
    if _corpus is None:
        _corpus = common.fixture_text("corpus.txt")
    return _corpus


def cuts(key):
    global _cuts
    if _cuts is None:
        _cuts = common.fixture_json("cuts.json")
    if key not in _cuts:
        raise RuntimeError(f"no token cuts for {key}: run make_fixtures.py --cuts-only")
    return _cuts[key]


def body(key, name):
    """corpus.txt cut to the size `name` (an S1 size, a ladder rung, or 'deep...') makes on
    model `key`'s tokenizer. Returns (text, tokens)."""
    end, n = cuts(key)[name]
    return corpus()[:end], n


def ladder(key, rung):
    """A ladder prompt: the corpus, five facts planted at their depths, the question."""
    text, n = body(key, rung)
    nd = common.fixture_json("needles.json")
    for i, f in enumerate(sorted(nd["facts"], key=lambda f: -f["at"])):
        at = text.find("\n", int(len(text) * f["at"]))
        at = len(text) if at < 0 else at
        text = text[:at] + f"\n// NOTE: {f['text']}\n" + text[at:]
    return "<repository>" + text + "\n</repository>\n\n" + nd["question"], n


def recall(text):
    nd = common.fixture_json("needles.json")
    t = norm(text)
    return sum(any(norm(e) in t for e in f["expect"]) for f in nd["facts"])


def norm(text):
    return re.sub(r"(?<=\d),(?=\d{3})", "", (text or "")).lower()


def has_any(text, options):
    t = norm(text)
    return any(norm(o) in t for o in options)


def parse_json(text):
    """(strict, value): strict when the reply alone parses; else a ```json fence taken off."""
    t = (text or "").strip()
    try:
        return True, json.loads(t)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    cand = m.group(1) if m else t[t.find("{"): t.rfind("}") + 1]
    try:
        return False, json.loads(cand)
    except Exception:
        return False, None


def same(a, b):
    if isinstance(b, bool) or b is None:
        return a == b
    if isinstance(b, (int, float)) and isinstance(a, (int, float)) and not isinstance(a, bool):
        return round(float(a), 2) == round(float(b), 2)
    if isinstance(b, str):
        return isinstance(a, str) and a.strip().lower() == b.strip().lower()
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(b, dict):
        return isinstance(a, dict) and all(k in a and same(a[k], v) for k, v in b.items())
    return a == b


def words_agree(a, b):
    """Words two continuations share before their first difference."""
    n = 0
    for x, y in zip((a or "").split(), (b or "").split()):
        if x != y:
            break
        n += 1
    return n
