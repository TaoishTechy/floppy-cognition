"""gen2.py — fact / question generator for floppy-cognition (train2.py).

Self-contained. Supplies sample(), TXT and GENS so train2 can train and evaluate
without the full upstream Tiny training kit. Expand the pools as needed.
"""
import random
import re

# ---------------------------------------------------------------------------
# Core fact pools (question templates + answers)
# Each entry: (list of question variants, list of acceptable answers, kind)
# ---------------------------------------------------------------------------

_CAPITALS = [
    (["what is the capital of france", "paris is the capital of which country",
      "what city is the capital of france"], ["Paris.", "It is Paris."], "geo"),
    (["what is the capital of japan", "what city is the capital of japan"],
     ["Tokyo.", "It is Tokyo."], "geo"),
    (["what is the capital of peru", "what city is the capital of peru"],
     ["Lima.", "It is Lima."], "geo"),
    (["what is the capital of texas", "what city is the capital of texas"],
     ["Austin.", "It is Austin."], "geo"),
    (["what is the capital of canada", "what city is the capital of canada"],
     ["Ottawa.", "It is Ottawa."], "geo"),
    (["what is the capital of australia", "what city is the capital of australia"],
     ["Canberra.", "It is Canberra."], "geo"),
    (["what is the capital of egypt", "what city is the capital of egypt"],
     ["Cairo.", "It is Cairo."], "geo"),
    (["what is the capital of germany", "what city is the capital of germany"],
     ["Berlin.", "It is Berlin."], "geo"),
]

_MATH = [
    (["what is 2 plus 2", "what is 2 + 2", "2 plus 2 equals"], ["4.", "Four."], "math"),
    (["what is 7 times 8", "what is 7 x 8", "7 times 8 equals"], ["56.", "Fifty-six."], "math"),
    (["what is 15 minus 6", "what is 15 - 6"], ["9.", "Nine."], "math"),
    (["what is 12 divided by 3", "what is 12 / 3"], ["4.", "Four."], "math"),
    (["is 31 even or odd", "is 31 odd or even"], ["Odd.", "It is odd."], "math"),
    (["is 42 even or odd"], ["Even.", "It is even."], "math"),
    (["which is bigger, 405 or 450", "which is larger 405 or 450"],
     ["450.", "450 is bigger."], "math"),
    (["what comes after 199", "what number comes after 199"], ["200.", "Two hundred."], "math"),
    (["write 47 in words", "how do you write 47 in words"], ["Forty-seven.", "Forty seven."], "math"),
    (["what is 14 in roman numerals", "write 14 in roman numerals"], ["XIV.", "It is XIV."], "math"),
]

_ANIMALS = [
    (["how many legs does a spider have", "how many legs has a spider"],
     ["Eight.", "A spider has eight legs."], "animals"),
    (["is a whale a fish", "are whales fish"], ["No.", "No, a whale is a mammal."], "animals"),
    (["what sound does a cow make", "what noise does a cow make"],
     ["Moo.", "A cow says moo."], "animals"),
    (["what is a baby kangaroo called", "what do you call a baby kangaroo"],
     ["A joey.", "Joey."], "animals"),
    (["what is a group of lions called", "what do you call a group of lions"],
     ["A pride.", "Pride."], "animals"),
    (["which animal says hoot", "what animal hoots"], ["An owl.", "Owl."], "animals"),
    (["what is the fastest land animal", "which is the fastest land animal"],
     ["The cheetah.", "Cheetah."], "animals"),
    (["how many legs does a spider have"], ["Eight."], "animals"),
]

_LANGUAGE = [
    (["what is the opposite of hot", "what is the opposite of the word hot"],
     ["Cold.", "The opposite is cold."], "language"),
    (["what is another word for big", "what is a synonym for big"],
     ["Large.", "Huge.", "Large or huge."], "language"),
    (["what is the superlative of good", "what is the superlative form of good"],
     ["Best.", "The superlative is best."], "language"),
    (["plural of mouse", "what is the plural of mouse"], ["Mice.", "The plural is mice."], "language"),
    (["past tense of go", "what is the past tense of go"], ["Went.", "The past tense is went."], "language"),
    (["spell elephant", "how do you spell elephant"], ["E-L-E-P-H-A-N-T.", "Elephant."], "language"),
    (["what is cat backwards", "spell cat backwards"], ["Tac.", "T-A-C."], "language"),
    (["what letter comes after k", "which letter comes after k"], ["L.", "The letter L."], "language"),
]

_WORLD = [
    (["what is the biggest ocean", "which ocean is the largest",
      "what is the largest ocean on earth"], ["The Pacific Ocean.", "The Pacific."], "geo"),
    (["what is the biggest country in the world", "which country is the largest by area"],
     ["Russia.", "Russia is the biggest country."], "geo"),
    (["what is the smallest country in the world", "which country is the smallest"],
     ["Vatican City.", "It is Vatican City."], "geo"),
    (["how many continents are there", "how many continents does the world have"],
     ["Seven.", "There are seven continents."], "geo"),
    (["what is the highest mountain in the world", "which mountain is the tallest"],
     ["Mount Everest.", "It is Mount Everest."], "geo"),
    (["where is the eiffel tower", "in which city is the eiffel tower"],
     ["In Paris.", "Paris, France."], "geo"),
    (["where are the pyramids of giza", "in which country are the great pyramids"],
     ["In Egypt.", "Egypt, near Cairo."], "geo"),
    (["what language do they speak in canada", "what languages are spoken in canada"],
     ["English and French.", "English and French."], "geo"),
]

_SCIENCE = [
    (["what is the symbol for gold", "what is the chemical symbol for gold"],
     ["Au.", "The symbol is Au."], "science"),
    (["how many moons does mars have", "how many moons has mars"],
     ["Two.", "Mars has two moons."], "science"),
    (["what is the chemical symbol for oxygen"], ["O.", "The symbol is O."], "science"),
    (["what is the chemical symbol for carbon"], ["C.", "The symbol is C."], "science"),
    (["how many strings does a guitar have"], ["Six.", "A guitar has six strings."], "science"),
]

_MISC = [
    (["who invented the telephone", "who invented the telephone"],
     ["Alexander Graham Bell.", "Bell."], "facts"),
    (["what is dos", "what does dos stand for"],
     ["Disk Operating System.", "It is an old operating system."], "facts"),
    (["what time is it"], ["I do not know the current time.", "I cannot tell the time."], "unknown"),
    (["tell me a joke"], ["I am still learning jokes.", "Why did the computer go to school? To improve its byte-size."], "facts"),
    (["how do you say thank you in spanish", "what is thank you in spanish"],
     ["Gracias.", "It is gracias."], "language"),
    (["how do you say thank you in german"], ["Danke.", "It is danke."], "language"),
    (["how do you say thank you in french"], ["Merci.", "It is merci."], "language"),
]

# Flatten into master list of (variants, answers, kind)
_ALL = (_CAPITALS + _MATH + _ANIMALS + _LANGUAGE + _WORLD + _SCIENCE + _MISC)

# Build TXT for eval_set (first question of each entry becomes a test_q)
TXT = []
for variants, answers, kind in _ALL:
    TXT.append({
        "test_q": [variants[0]],
        "answers": answers,
        "kind": kind,
    })

# GENS surface expected by train2.eval_set
GENS = [
    ("facts", None, None),
    ("math", None, None),
    ("geo", None, None),
    ("animals", None, None),
    ("language", None, None),
    ("science", None, None),
    ("unknown", None, None),
]

def _pick(rng, pool):
    return rng.choice(pool)

def sample(rng, test=False, name=None):
    """Return (question, answer, kind).

    test=True selects a slightly altered wording when possible.
    name restricts the pool to a given kind (or 'unknown').
    """
    if name == "unknown":
        return ("what is the capital of xyzland",
                "I do not know.",
                "unknown")

    if name is not None:
        pool = [(v, a, k) for v, a, k in _ALL if k == name]
        if not pool:
            pool = _ALL
    else:
        pool = _ALL

    variants, answers, kind = _pick(rng, pool)
    if test and len(variants) > 1:
        q = rng.choice(variants[1:])          # held-out style wording
    else:
        q = rng.choice(variants)
    a = rng.choice(answers)
    return q, a, kind

def sample_many(rng, n=10, name=None):
    """Convenience: list of (q, a, kind)."""
    return [sample(rng, name=name) for _ in range(n)]

if __name__ == "__main__":
    rng = random.Random(42)
    print("=== sample facts ===")
    for _ in range(8):
        q, a, k = sample(rng)
        print(f"[{k}] {q}  ->  {a}")
    print("\n=== test-style (held-out wording) ===")
    for _ in range(4):
        q, a, k = sample(rng, test=True)
        print(f"[{k}] {q}  ->  {a}")
    print("\n=== unknown ===")
    print(sample(rng, name="unknown"))
