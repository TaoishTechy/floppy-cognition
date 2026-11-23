"""Minimal gen2 stub for floppy-cognition / train2.py.

Provides just enough surface for train2.batch, eval_set and answer_many to run.
Replace with the full fact generator from the upstream Tiny training kit for real training.
"""
import random

# Minimal fact pool so sample() never fails
_FACTS = [
    ("what is the capital of france", "Paris."),
    ("what is 2 plus 2", "4."),
    ("what colour is the sky", "Blue."),
    ("who invented the telephone", "Alexander Graham Bell."),
    ("what is the largest ocean", "The Pacific Ocean."),
    ("how many legs does a spider have", "Eight."),
    ("what is the capital of japan", "Tokyo."),
    ("what is 7 times 8", "56."),
    ("is a whale a fish", "No."),
    ("what is the opposite of hot", "Cold."),
]

# Surfaces required by train2.py
TXT = [
    {"test_q": [q], "answers": [a]} for q, a in _FACTS
]

GENS = [
    ("facts", None, None),
    ("math", None, None),
    ("geo", None, None),
    ("unknown", None, None),
]

def sample(rng, test=False, name=None):
    """Return (question, answer, kind)."""
    if name == "unknown":
        return ("what is the capital of xyzland", "I do not know.", "unknown")
    q, a = rng.choice(_FACTS)
    if test:
        # slight wording variation for held-out style
        q = q.replace("what is", "tell me").replace("who", "which person")
    return q, a, name or "facts"

if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(5):
        print(sample(rng))
