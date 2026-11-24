"""gen2.py — fact / question generator for floppy-cognition (train2.py).

Loads DigestedUnit JSON from training/digested/ when present.
Falls back to built-in pool if digested/ is empty.
"""
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DIGESTED = _ROOT / "training" / "digested"

# ---------------------------------------------------------------------------
# Built-in fallback pool (canonical answer per fact)
# ---------------------------------------------------------------------------

_CAPITALS = [
    (["what is the capital of france", "paris is the capital of which country",
      "what city is the capital of france"], "Paris.", "geo"),
    (["what is the capital of japan", "what city is the capital of japan"],
     "Tokyo.", "geo"),
    (["what is the capital of peru"], "Lima.", "geo"),
    (["what is the capital of texas"], "Austin.", "geo"),
    (["what is the capital of canada"], "Ottawa.", "geo"),
    (["what is the capital of australia"], "Canberra.", "geo"),
    (["what is the capital of egypt"], "Cairo.", "geo"),
    (["what is the capital of germany"], "Berlin.", "geo"),
]

_MATH = [
    (["what is 2 plus 2", "what is 2 + 2"], "4.", "math"),
    (["what is 7 times 8", "what is 7 x 8"], "56.", "math"),
    (["what is 15 minus 6"], "9.", "math"),
    (["what is 12 divided by 3"], "4.", "math"),
    (["is 31 even or odd"], "Odd.", "math"),
    (["is 42 even or odd"], "Even.", "math"),
]

_ANIMALS = [
    (["how many legs does a spider have"], "Eight.", "animals"),
    (["is a whale a fish"], "No.", "animals"),
    (["what sound does a cow make"], "Moo.", "animals"),
    (["what is a baby kangaroo called"], "A joey.", "animals"),
]

_LANGUAGE = [
    (["what is the opposite of hot"], "Cold.", "language"),
    (["plural of mouse"], "Mice.", "language"),
    (["past tense of go"], "Went.", "language"),
]

_WORLD = [
    (["what is the biggest ocean"], "The Pacific Ocean.", "geo"),
    (["how many continents are there"], "Seven.", "geo"),
    (["what is the highest mountain in the world"], "Mount Everest.", "geo"),
]

_SCIENCE = [
    (["what is the symbol for gold"], "Au.", "science"),
    (["how many moons does mars have"], "Two.", "science"),
    (["what is the chemical symbol for oxygen"], "O.", "science"),
]

_MISC = [
    (["who invented the telephone"], "Alexander Graham Bell.", "facts"),
    (["what is dos"], "Disk Operating System.", "facts"),
]

_UNKNOWN = [
    (["what time is it"], "I do not know.", "unknown"),
    (["what is the capital of xyzland"], "I do not know.", "unknown"),
    (["what is the capital of zorp"], "I do not know.", "unknown"),
    (["who is the president of atlantis"], "I do not know.", "unknown"),
    (["what is the population of narnia"], "I do not know.", "unknown"),
    (["when did the battle of foo occur"], "I do not know.", "unknown"),
]

_BUILTIN = (_CAPITALS + _MATH + _ANIMALS + _LANGUAGE + _WORLD + _SCIENCE + _MISC + _UNKNOWN)


def load_digested(path=None):
    """Load DigestedUnit JSON files -> list of (variants, answer, kind)."""
    root = Path(path) if path else _DIGESTED
    if not root.is_dir():
        return []
    units = []
    for f in sorted(root.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        # support both wrapped {units:[...]} and bare list
        items = data.get("units", data if isinstance(data, list) else [data])
        for d in items:
            if not isinstance(d, dict):
                continue
            q = d.get("canonical_q")
            a = d.get("canonical_a")
            if not q or not a:
                continue
            variants = [q] + list(d.get("variants") or [])
            # de-dupe preserve order
            seen = set()
            uniq = []
            for v in variants:
                k = v.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(v.strip())
            kind = d.get("kind") or "facts"
            units.append((uniq, a if a.endswith(".") else a + ".", kind))
    return units


def _build_all():
    digested = load_digested()
    if digested:
        return digested
    return list(_BUILTIN)


_ALL = _build_all()

# TXT / GENS for train2.eval_set compatibility
TXT = []
for variants, answer, kind in _ALL:
    TXT.append({
        "test_q": [variants[0]],
        "answers": [answer],
        "kind": kind,
    })

GENS = [
    ("facts", None, None),
    ("math", None, None),
    ("geo", None, None),
    ("animals", None, None),
    ("language", None, None),
    ("science", None, None),
    ("unknown", None, None),
]


def reload():
    """Re-scan training/digested and rebuild pools (call after digest)."""
    global _ALL, TXT
    _ALL = _build_all()
    TXT = []
    for variants, answer, kind in _ALL:
        TXT.append({
            "test_q": [variants[0]],
            "answers": [answer],
            "kind": kind,
        })
    return len(_ALL)


def sample(rng, test=False, name=None):
    """Return (question, answer, kind)."""
    if name is not None:
        pool = [(v, a, k) for v, a, k in _ALL if k == name]
        if not pool:
            pool = _ALL
    else:
        pool = _ALL

    variants, answer, kind = rng.choice(pool)
    if test and len(variants) > 1:
        q = rng.choice(variants[1:])
    else:
        q = rng.choice(variants)
    return q, answer, kind


def sample_many(rng, n=10, name=None):
    return [sample(rng, name=name) for _ in range(n)]


def pool_stats():
    from collections import Counter
    c = Counter(k for _, _, k in _ALL)
    return {"total": len(_ALL), "by_kind": dict(c), "source": "digested" if load_digested() else "builtin"}


if __name__ == "__main__":
    n = reload()
    stats = pool_stats()
    print("pool:", stats)
    rng = random.Random(42)
    print("=== samples ===")
    for _ in range(6):
        q, a, k = sample(rng)
        print("[%s] %s -> %s" % (k, q, a))
    print("=== unknown ===")
    for _ in range(2):
        print(sample(rng, name="unknown"))
