"""Phase 1 digest: RawUnit -> DigestedUnit using core approaches A1,B7,F36,G37,G38.

  python3 -I training/digest.py
  python3 -I training/digest.py --limit 20

Reads training/raw/*.json, writes training/digested/*.json
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "digested"

# Simple capital/schema patterns for F36
_CAPITAL_RE = re.compile(
    r"(?P<city>[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*)\s+is\s+the\s+capital\s+of\s+(?P<country>[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*)",
    re.I,
)
_IS_RE = re.compile(
    r"(?P<sub>[A-Z][\w\s\-]+?)\s+is\s+(?P<obj>.+?)\.\s*$",
    re.I,
)


def _fact_id(text: str) -> str:
    return "f_" + hashlib.sha256(text.encode()).hexdigest()[:10]


def approach_f36_schema(text: str):
    """Structure mapping: X is capital of Y, etc."""
    m = _CAPITAL_RE.search(text)
    if m:
        city, country = m.group("city"), m.group("country")
        return {
            "schema": "X_is_capital_of_Y",
            "X": city,
            "Y": country,
            "canonical_q": "what is the capital of %s" % country.lower(),
            "canonical_a": city + ".",
            "kind": "geo",
        }
    return None


def approach_g37_questions(canonical_q, canonical_a, schema_info):
    """Self-question generation: fact + variants."""
    variants = [canonical_q]
    if schema_info and schema_info.get("schema") == "X_is_capital_of_Y":
        y = schema_info["Y"]
        x = schema_info["X"]
        variants.extend([
            "%s is the capital of which country" % x.lower(),
            "what city is the capital of %s" % y.lower(),
        ])
    return variants


def approach_g38_distractors(schema_info):
    if schema_info and schema_info.get("schema") == "X_is_capital_of_Y":
        # generic geo distractors; real system would be entity-aware
        city = schema_info["X"]
        pool = ["London.", "Berlin.", "Madrid.", "Rome.", "Lyon.", "Marseille."]
        return [d for d in pool if not d.startswith(city)][:3]
    return []


def approach_g41_negative(schema_info):
    if schema_info and schema_info.get("schema") == "X_is_capital_of_Y":
        return [{
            "q": "what is the capital of zorp",
            "expected": "I do not know.",
        }]
    return []


def approach_b7_prediction_placeholder(text: str):
    """Offline digester cannot run the tiny model yet; store a stub."""
    return {
        "predicted_next": None,
        "note": "run with model later for real prediction error",
    }


def approach_a1_calibration_stub():
    return {"predicted_confidence": 0.5, "note": "set after audit"}


def digest_unit(raw: dict):
    text = (raw.get("text") or "").strip()
    if len(text) < 8:
        return []

    schema = approach_f36_schema(text)
    out = []

    if schema:
        variants = approach_g37_questions(schema["canonical_q"], schema["canonical_a"], schema)
        unit = {
            "digested_id": "d_" + raw["id"][:8],
            "fact_id": _fact_id(schema["canonical_q"]),
            "canonical_q": schema["canonical_q"],
            "canonical_a": schema["canonical_a"],
            "variants": variants[1:],
            "distractors": approach_g38_distractors(schema),
            "counterfactuals": [
                "If %s had no capital, what would you answer? I do not know."
                % schema["Y"]
            ],
            "causal_graph": {
                "cause": "",
                "mechanism": "capital is seat of government",
                "effect": schema["canonical_a"],
            },
            "uncertainty": "known",
            "calibration": 0.5,
            "kind": schema["kind"],
            "source": {
                "file": raw.get("source_file"),
                "page": raw.get("page"),
                "line": raw.get("line_start"),
                "format": raw.get("source_format"),
                "raw_id": raw.get("id"),
            },
            "approaches_used": ["A1", "B7", "F36", "G37", "G38", "G41"],
            "teaching_script": "%s is the capital of %s." % (schema["X"], schema["Y"]),
            "tests": [
                {"q": schema["canonical_q"], "expected": schema["canonical_a"]},
            ] + approach_g41_negative(schema),
            "prediction": approach_b7_prediction_placeholder(text),
            "calibration_meta": approach_a1_calibration_stub(),
        }
        out.append(unit)
        return out

    # Generic prose: one weak Q/A from first sentence
    sent = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    if len(sent) < 12:
        return []
    q = "what does the following state: %s" % sent[:80]
    unit = {
        "digested_id": "d_" + raw["id"][:8],
        "fact_id": _fact_id(sent),
        "canonical_q": q.lower(),
        "canonical_a": sent if sent.endswith(".") else sent + ".",
        "variants": [],
        "distractors": [],
        "counterfactuals": [],
        "causal_graph": {},
        "uncertainty": "inferred",
        "calibration": 0.3,
        "kind": "facts",
        "source": {
            "file": raw.get("source_file"),
            "page": raw.get("page"),
            "line": raw.get("line_start"),
            "format": raw.get("source_format"),
            "raw_id": raw.get("id"),
        },
        "approaches_used": ["A1", "B7", "G37"],
        "teaching_script": sent,
        "tests": [{"q": q.lower(), "expected": sent if sent.endswith(".") else sent + "."}],
        "prediction": approach_b7_prediction_placeholder(text),
        "calibration_meta": approach_a1_calibration_stub(),
    }
    out.append(unit)
    return out


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.is_dir():
        print("no raw units — run: python3 -I training/ingest.py parse-all")
        sys.exit(1)

    n_in = n_out = 0
    for raw_path in sorted(RAW_DIR.glob("*.json")):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        digested = []
        for unit in payload.get("units", []):
            if limit is not None and n_out >= limit:
                break
            n_in += 1
            for d in digest_unit(unit):
                digested.append(d)
                n_out += 1
                if limit is not None and n_out >= limit:
                    break
        if digested:
            out_path = OUT_DIR / (raw_path.stem + ".digested.json")
            out_path.write_text(
                json.dumps({"source_raw": str(raw_path.name), "units": digested}, indent=2),
                encoding="utf-8",
            )
            print("wrote %s (%d units)" % (out_path, len(digested)))

    print("raw units seen: %d  digested: %d" % (n_in, n_out))


if __name__ == "__main__":
    main()
