"""Four-audit runner for floppy-cognition (Digestion Blueprint V5 §8).

  python3 -I audit.py MODEL.pt|MODEL.riskytensor
  python3 -I audit.py MODEL.pt --n 10

Audits:
  1. Recall      — exact canonical questions
  2. Transfer    — reworded variants (held-out style)
  3. Abstention  — unknown / zorp-style
  4. Calibration — placeholder ECE (greedy has no confidence yet)

Keeps generation short for pure-Python laser.
"""
import sys
import os
import time
import random
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen2
import train2

MAX_NEW = 16
CTX = 128


def _generate(model, question, max_new=MAX_NEW):
    """Greedy decode, truncated prompt, short max_new."""
    budget = CTX - max_new - 1
    q = question if len(question) <= budget else question[-budget:]
    model.eval()
    ids = train2.encode(q + "\n")
    import laser as torch
    x = torch.tensor([ids])
    out = []
    for _ in range(max_new):
        logits = model(x)
        nxt = int(logits[0, -1].argmax(-1).item())
        if nxt <= 1:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    model.train()
    return train2.decode(out)


def _match(gen, expected):
    """Loose match: key content words of expected appear in gen."""
    g = set(train2.words_of(gen))
    e = train2.words_of(expected)
    # strip trailing period word artifacts
    key = [w for w in e if w not in train2.STOP]
    if not key:
        key = e
    if not key:
        return False
    return all(w in g for w in key)


def _is_abstain(gen):
    gl = gen.lower().strip()
    return gl.startswith("i do not") or gl.startswith("i don't know") or "do not know" in gl


def build_audit_sets(n=10, seed=0):
    rng = random.Random(seed)
    gen2.reload()

    recall = []
    transfer = []
    abstention = []

    # Recall: first variant (canonical) of non-unknown facts
    facts = [(v, a, k) for v, a, k in gen2._ALL if k != "unknown"]
    rng.shuffle(facts)
    for variants, answer, kind in facts[:n]:
        recall.append((variants[0], answer, kind))

    # Transfer: later variants when available
    for variants, answer, kind in facts:
        if len(transfer) >= n:
            break
        if len(variants) > 1:
            transfer.append((variants[1], answer, kind))
    # pad transfer with reworded-style if needed
    while len(transfer) < min(n, len(facts)):
        variants, answer, kind = rng.choice(facts)
        q = variants[0]
        # cheap surface distortion for transfer probe
        q2 = q.replace("what is", "what's") if "what is" in q else q + "?"
        if q2 != q:
            transfer.append((q2, answer, kind))
        else:
            break

    # Abstention: unknown pool + zorp
    unknowns = [(v, a, k) for v, a, k in gen2._ALL if k == "unknown"]
    rng.shuffle(unknowns)
    for variants, answer, kind in unknowns[:n]:
        abstention.append((variants[0], answer, kind))
    if len(abstention) < n:
        abstention.append(("what is the capital of zorp", "I do not know.", "unknown"))

    return recall, transfer, abstention


def run_audit(model, n=10, seed=0):
    recall, transfer, abstention = build_audit_sets(n=n, seed=seed)
    results = {
        "recall": [],
        "transfer": [],
        "abstention": [],
        "meta": {
            "n": n,
            "pool": gen2.pool_stats(),
            "max_new": MAX_NEW,
        },
    }

    def run_set(name, items, checker):
        print("\n=== %s audit (%d) ===" % (name, len(items)), flush=True)
        ok = 0
        for i, (q, expected, kind) in enumerate(items):
            t0 = time.time()
            gen = _generate(model, q)
            dt = time.time() - t0
            hit = checker(gen, expected)
            ok += int(hit)
            mark = "OK" if hit else "x"
            print("  %s [%s] %s" % (mark, kind, q), flush=True)
            print("      -> %r  (want %r)  %.1fs" % (gen, expected, dt), flush=True)
            results[name].append({
                "q": q, "expected": expected, "got": gen,
                "ok": hit, "kind": kind, "sec": round(dt, 2),
            })
        acc = ok / max(1, len(items))
        print("  score: %d/%d  (%.0f%%)" % (ok, len(items), 100 * acc), flush=True)
        return acc

    r = run_set("recall", recall, _match)
    t = run_set("transfer", transfer, _match)

    def abs_check(gen, expected):
        return _is_abstain(gen)

    a = run_set("abstention", abstention, abs_check)

    # Calibration: greedy has no probabilities yet — report placeholder
    print("\n=== calibration audit ===", flush=True)
    print("  (greedy decode has no confidence scores; ECE not computed)", flush=True)
    print("  placeholder ECE: n/a  — add temperature/top-k or logit probe later", flush=True)

    print("\n=== summary ===", flush=True)
    print("  recall:     %.0f%%  (target 80%%)" % (100 * r), flush=True)
    print("  transfer:   %.0f%%  (target 50%%)" % (100 * t), flush=True)
    print("  abstention: %.0f%%  (target 80%%)" % (100 * a), flush=True)
    print("  calibration: n/a", flush=True)
    print("  pool: %s" % results["meta"]["pool"], flush=True)

    results["scores"] = {
        "recall": r,
        "transfer": t,
        "abstention": a,
        "calibration_ece": None,
    }
    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    n = 10
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    print("loading", path, flush=True)
    t0 = time.time()
    model = train2.load_model(path)
    print("loaded in %.2fs  %s" % (time.time() - t0, model.c), flush=True)

    results = run_audit(model, n=n)

    out = "audit_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
