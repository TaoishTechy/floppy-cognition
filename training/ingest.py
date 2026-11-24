"""Phase 0 ingest: discover + parse training sources into RawUnit JSON.

  python3 -I training/ingest.py scan
  python3 -I training/ingest.py parse <file>
  python3 -I training/ingest.py parse-all

Supported: .md .json .xml .pdf (pdf needs pypdf or falls back to text extract)
Output: training/raw/<sha>.json
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
MANIFEST = ROOT / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"version": 1, "files": []}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def discover():
    found = []
    for sub in ("md", "json", "xml", "pdf"):
        d = ROOT / sub
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".md", ".json", ".xml", ".pdf"}:
                found.append(p)
    return found


def parse_md(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    units = []
    heading = []
    buf = []
    line_start = 1

    def flush(end_line):
        nonlocal buf, line_start
        body = "\n".join(buf).strip()
        if body:
            units.append({
                "kind": "prose",
                "text": body,
                "heading_path": list(heading),
                "line_start": line_start,
                "line_end": end_line,
                "structured": None,
                "trust": 1.0,
            })
        buf = []

    for i, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush(i - 1)
            level = len(m.group(1))
            title = m.group(2).strip()
            heading = heading[: level - 1] + [title]
            line_start = i + 1
            continue
        if line.strip().startswith("```"):
            flush(i - 1)
            # skip code block body lightly
            buf = []
            line_start = i
            continue
        buf.append(line)
    flush(len(text.splitlines()) or 1)
    return units


def parse_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    units = []

    def walk(obj, prefix):
        if isinstance(obj, dict):
            # fact-like arrays
            if "facts" in obj and isinstance(obj["facts"], list):
                for i, item in enumerate(obj["facts"]):
                    units.append({
                        "kind": "fact",
                        "text": json.dumps(item, ensure_ascii=False),
                        "heading_path": [prefix, "facts", str(i)],
                        "line_start": 0,
                        "line_end": 0,
                        "structured": item,
                        "trust": 1.0,
                    })
            for k, v in obj.items():
                walk(v, prefix + "." + k if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, "%s[%d]" % (prefix, i))
        else:
            units.append({
                "kind": "metadata" if "config" in prefix.lower() else "fact",
                "text": "%s = %s" % (prefix, obj),
                "heading_path": prefix.split("."),
                "line_start": 0,
                "line_end": 0,
                "structured": {"path": prefix, "value": obj},
                "trust": 1.0,
            })

    walk(data, "")
    if not units:
        units.append({
            "kind": "metadata",
            "text": path.name,
            "heading_path": [],
            "line_start": 0,
            "line_end": 0,
            "structured": data,
            "trust": 1.0,
        })
    return units


def parse_xml(path: Path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(path))
    root = tree.getroot()
    units = []

    def walk(el, path_parts):
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        parts = path_parts + [tag]
        text = (el.text or "").strip()
        if text:
            units.append({
                "kind": "definition" if tag.lower() in {"definition", "def", "term"} else "prose",
                "text": text,
                "heading_path": parts,
                "line_start": 0,
                "line_end": 0,
                "structured": {"tag": tag, "attrs": dict(el.attrib)},
                "trust": 1.0,
            })
        for child in el:
            walk(child, parts)

    walk(root, [])
    return units


def parse_pdf(path: Path):
    units = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                units.append({
                    "kind": "prose",
                    "text": "",
                    "heading_path": ["page %d" % i],
                    "line_start": 0,
                    "line_end": 0,
                    "structured": {"page": i},
                    "trust": 0.7,
                    "note": "empty or needs OCR",
                })
                continue
            units.append({
                "kind": "prose",
                "text": text,
                "heading_path": ["page %d" % i],
                "line_start": 0,
                "line_end": 0,
                "structured": {"page": i},
                "trust": 1.0,
            })
    except Exception as e:
        units.append({
            "kind": "prose",
            "text": "",
            "heading_path": [],
            "line_start": 0,
            "line_end": 0,
            "structured": {"error": str(e)},
            "trust": 0.5,
            "note": "pdf parse failed",
        })
    return units


def parse_file(path: Path):
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".md":
        chunks = parse_md(path)
    elif suf == ".json":
        chunks = parse_json(path)
    elif suf == ".xml":
        chunks = parse_xml(path)
    elif suf == ".pdf":
        chunks = parse_pdf(path)
    else:
        raise ValueError("unsupported format: %s" % suf)

    file_sha = sha256_file(path)
    raw_units = []
    for i, c in enumerate(chunks):
        uid = hashlib.sha256(
            (file_sha + "|" + str(i) + "|" + c.get("text", "")[:200]).encode()
        ).hexdigest()[:16]
        raw_units.append({
            "id": uid,
            "source_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "source_format": suf.lstrip("."),
            "page": (c.get("structured") or {}).get("page"),
            "line_start": c.get("line_start", 0),
            "line_end": c.get("line_end", 0),
            "heading_path": c.get("heading_path") or [],
            "kind": c.get("kind", "prose"),
            "text": c.get("text", ""),
            "structured": c.get("structured"),
            "trust": c.get("trust", 1.0),
            "language": "en",
        })
    return file_sha, raw_units


def write_raw(file_sha, units, source_path):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / (file_sha[:16] + ".json")
    payload = {
        "source": str(source_path),
        "file_sha256": file_sha,
        "units": units,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def cmd_scan():
    files = discover()
    m = load_manifest()
    by_path = {e["path"]: e for e in m.get("files", [])}
    for p in files:
        rel = str(p.relative_to(ROOT))
        h = sha256_file(p)
        if rel not in by_path:
            by_path[rel] = {
                "path": rel,
                "sha256": h,
                "status": "raw",
                "parser_version": "0.1",
            }
        else:
            if by_path[rel]["sha256"] != h:
                by_path[rel]["sha256"] = h
                by_path[rel]["status"] = "raw"
    m["files"] = sorted(by_path.values(), key=lambda x: x["path"])
    save_manifest(m)
    print("scanned %d files" % len(m["files"]))
    for e in m["files"]:
        print("  %s  %s  %s" % (e["status"], e["sha256"][:12], e["path"]))


def cmd_parse(path):
    path = Path(path)
    if not path.is_file():
        # try relative to training/
        alt = ROOT / path
        if alt.is_file():
            path = alt
        else:
            print("not found:", path)
            sys.exit(1)
    file_sha, units = parse_file(path)
    out = write_raw(file_sha, units, path)
    print("wrote %s  (%d units)" % (out, len(units)))
    # update manifest
    m = load_manifest()
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    found = False
    for e in m.get("files", []):
        if e["path"] == rel:
            e["status"] = "parsed"
            e["sha256"] = file_sha
            found = True
    if not found:
        m.setdefault("files", []).append({
            "path": rel, "sha256": file_sha, "status": "parsed", "parser_version": "0.1"
        })
    save_manifest(m)


def cmd_parse_all():
    cmd_scan()
    m = load_manifest()
    for e in m.get("files", []):
        if e["status"] in ("raw", "parsed", "failed"):
            try:
                cmd_parse(ROOT / e["path"])
            except Exception as ex:
                print("FAIL", e["path"], ex)
                e["status"] = "failed"
    save_manifest(m)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "scan":
        cmd_scan()
    elif cmd == "parse" and len(sys.argv) >= 3:
        cmd_parse(sys.argv[2])
    elif cmd == "parse-all":
        cmd_parse_all()
    else:
        print(__doc__)
        sys.exit(1)
