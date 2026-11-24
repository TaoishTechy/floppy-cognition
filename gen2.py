"""gen2.py — fact / question generator for floppy-cognition (train2.py).

Canonical answer per fact (one target, not alternating variants).
Unknown examples appear in training, not only eval.
"""
import random

# ---------------------------------------------------------------------------
# Core fact pools: (question variants, CANONICAL answer, kind)
# One answer string only — greedy models cannot learn two targets.
# ---------------------------------------------------------------------------

_CAPITALS = [
    (["what is the capital of france", "paris is the capital of which country",
      "what city is the capital of france"], "Paris.", "geo"),
    (["what is the capital of japan", "what city is the capital of japan"],
     "Tokyo.", "geo"),
    (["what is the capital of peru", "what city is the capital of peru"],
     "Lima.", "geo"),
    (["what is the capital of texas", "what city is the capital of texas"],
     "Austin.", "geo"),
    (["what is the capital of canada", "what city is the capital of canada"],
     "Ottawa.", "geo"),
    (["what is the capital of australia", "what city is the capital of australia"],
     "Canberra.", "geo"),
    (["what is the capital of egypt", "what city is the capital of egypt"],
     "Cairo.", "geo"),
    (["what is the capital of germany", "what city is the capital of germany"],
     "Berlin.", "geo"),
]

_MATH = [
    (["what is 2 plus 2", "what is 2 + 2", "2 plus 2 equals"], "4.", "math"),
    (["what is 7 times 8", "what is 7 x 8", "7 times 8 equals"], "56.", "math"),
    (["what is 15 minus 6", "what is 15 - 6"], "9.", "math"),
    (["what is 12 divided by 3", "what is 12 / 3"], "4.", "math"),
    (["is 31 even or odd", "is 31 odd or even"], "Odd.", "math"),
    (["is 42 even or odd"], "Even.", "math"),
    (["which is bigger, 405 or 450", "which is larger 405 or 450"], "450.", "math"),
    (["what comes after 199", "what number comes after 199"], "200.", "math"),
    (["write 47 in words", "how do you write 47 in words"], "Forty-seven.", "math"),
    (["what is 14 in roman numerals", "write 14 in roman numerals"], "XIV.", "math"),
]

_ANIMALS = [
    (["how many legs does a spider have", "how many legs has a spider"],
     "Eight.", "animals"),
    (["is a whale a fish", "are whales fish"], "No.", "animals"),
    (["what sound does a cow make", "what noise does a cow make"], "Moo.", "animals"),
    (["what is a baby kangaroo called", "what do you call a baby kangaroo"],
     "A joey.", "animals"),
    (["what is a group of lions called", "what do you call a group of lions"],
     "A pride.", "animals"),
    (["which animal says hoot", "what animal hoots"], "An owl.", "animals"),
    (["what is the fastest land animal", "which is the fastest land animal"],
     "The cheetah.", "animals"),
]

_LANGUAGE = [
    (["what is the opposite of hot", "what is the opposite of the word hot"],
     "Cold.", "language"),
    (["what is another word for big", "what is a synonym for big"], "Large.", "language"),
    (["what is the superlative of good", "what is the superlative form of good"],
     "Best.", "language"),
    (["plural of mouse", "what is the plural of mouse"], "Mice.", "language"),
    (["past tense of go", "what is the past tense of go"], "Went.", "language"),
    (["spell elephant", "how do you spell elephant"], "Elephant.", "language"),
    (["what is cat backwards", "spell cat backwards"], "Tac.", "language"),
    (["what letter comes after k", "which letter comes after k"], "L.", "language"),
]

_WORLD = [
    (["what is the biggest ocean", "which ocean is the largest",
      "what is the largest ocean on earth"], "The Pacific Ocean.", "geo"),
    (["what is the biggest country in the world", "which country is the largest by area"],
     "Russia.", "geo"),
    (["what is the smallest country in the world", "which country is the smallest"],
     "Vatican City.", "geo"),
    (["how many continents are there", "how many continents does the world have"],
     "Seven.", "geo"),
    (["what is the highest mountain in the world", "which mountain is the tallest"],
     "Mount Everest.", "geo"),
    (["where is the eiffel tower", "in which city is the eiffel tower"],
     "In Paris.", "geo"),
    (["where are the pyramids of giza", "in which country are the great pyramids"],
     "In Egypt.", "geo"),
    (["what language do they speak in canada", "what languages are spoken in canada"],
     "English and French.", "geo"),
]

_SCIENCE = [
    (["what is the symbol for gold", "what is the chemical symbol for gold"],
     "Au.", "science"),
    (["how many moons does mars have", "how many moons has mars"],
     "Two.", "science"),
    (["what is the chemical symbol for oxygen"], "O.", "science"),
    (["what is the chemical symbol for carbon"], "C.", "science"),
    (["how many strings does a guitar have"], "Six.", "science"),
]

_MISC = [
    (["who invented the telephone"], "Alexander Graham Bell.", "facts"),
    (["what is dos", "what does dos stand for"], "Disk Operating System.", "facts"),
    (["tell me a joke"], "I am still learning jokes.", "facts"),
    (["how do you say thank you in spanish", "what is thank you in spanish"],
     "Gracias.", "language"),
    (["how do you say thank you in german"], "Danke.", "language"),
    (["how do you say thank you in french"], "Merci.", "language"),
]

# Unknown / abstention — appear in training so the model can learn to refuse
_UNKNOWN = [
    (["what time is it"], "I do not know.", "unknown"),
    (["what is the capital of xyzland"], "I do not know.", "unknown"),
    (["what is the capital of zorp"], "I do not know.", "unknown"),
    (["who is the president of atlantis"], "I do not know.", "unknown"),
    (["how many moons does pluto's twin have"], "I do not know.", "unknown"),
    (["what is the boiling point of unobtainium"], "I do not know.", "unknown"),
    (["who invented the perpetual motion machine"], "I do not know.", "unknown"),
    (["what is the population of narnia"], "I do not know.", "unknown"),
    (["when did the battle of foo occur"], "I do not know.", "unknown"),
    (["what language do they speak on mars"], "I do not know.", "unknown"),
]

_ALL = (_CAPITALS + _MATH + _ANIMALS + _LANGUAGE + _WORLD + _SCIENCE + _MISC + _UNKNOWN)

# TXT for eval: first variant held as test_q; answers is a one-element list
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


if __name__ == "__main__":
    rng = random.Random(42)
    print("=== sample facts ===")
    for _ in range(8):
        q, a, k = sample(rng)
        print("[%s] %s  ->  %s" % (k, q, a))
    print("\n=== unknown ===")
    for _ in range(3):
        print(sample(rng, name="unknown"))
