"""riskytensor Phase-0 skeleton — header + manifest + info().

Full Phase-1 float32 save/load and later phases are still to be implemented
per the Consolidated Blueprint v4.0 and Riskytensor-Blueprint-V4.0.

No pickle, no eval, no exec. Stdlib only for the parser.
"""
import struct
import json
import hashlib
import os
import sys

MAGIC = b"RISKY4\n"
VERSION = 4
HEADER_LEN = 64

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def info(path: str) -> dict:
    """Return the manifest only, without loading weights.
    Raises ValueError on bad magic, truncated file, or manifest SHA mismatch.
    """
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
    if len(header) < HEADER_LEN:
        raise ValueError("riskytensor: file too short for header")
    if header[:7] != MAGIC:
        raise ValueError("riskytensor: bad magic (expected RISKY4)")
    version = struct.unpack_from("<H", header, 7)[0]
    if version != VERSION:
        raise ValueError(f"riskytensor: unsupported version {version}")
    header_len = struct.unpack_from("<H", header, 9)[0]
    if header_len != HEADER_LEN:
        raise ValueError(f"riskytensor: unexpected header_len {header_len}")
    manifest_len = struct.unpack_from("<Q", header, 11)[0]
    manifest_sha = header[19:51]
    # payload_sha is header[51:83] but we do not need it for info()

    with open(path, "rb") as f:
        f.seek(HEADER_LEN)
        manifest_bytes = f.read(manifest_len)
    if len(manifest_bytes) != manifest_len:
        raise ValueError("riskytensor: truncated manifest")
    if _sha256(manifest_bytes) != manifest_sha:
        raise ValueError("riskytensor: manifest SHA-256 mismatch")

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"riskytensor: invalid manifest JSON: {e}") from e
    return manifest

def _write_header(manifest_bytes: bytes, payload_sha: bytes) -> bytes:
    """Build the 64-byte header."""
    manifest_sha = _sha256(manifest_bytes)
    # magic(7) + version(u16) + header_len(u16) + manifest_len(u64)
    # + manifest_sha(32) + payload_sha(32) + reserved(7) = 64
    header = bytearray(HEADER_LEN)
    header[0:7] = MAGIC
    struct.pack_into("<H", header, 7, VERSION)
    struct.pack_into("<H", header, 9, HEADER_LEN)
    struct.pack_into("<Q", header, 11, len(manifest_bytes))
    header[19:51] = manifest_sha
    header[51:83] = payload_sha
    # remaining bytes already zero
    return bytes(header)

def save(state_dict, path, recipe=None):
    """Phase-0 stub: only supports an empty state_dict for testing the header/manifest path.
    Full float32 / int8_per_row implementation is Phase 1+.
    """
    if state_dict:
        raise NotImplementedError(
            "riskytensor Phase-0 only supports empty state_dict; "
            "full save/load arrives in Phase 1"
        )
    manifest = {
        "format": "riskytensor",
        "version": VERSION,
        "created_by": "riskytensor Phase-0 skeleton",
        "tensors": {},
        "quantization": {"default": "float32", "exceptions": {}},
        "metadata": {},
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload = b""  # nothing yet
    payload_sha = _sha256(payload)
    header = _write_header(manifest_bytes, payload_sha)
    with open(path, "wb") as f:
        f.write(header)
        f.write(manifest_bytes)
        f.write(payload)
    return os.path.getsize(path)

def load(path):
    """Phase-0 stub: only returns empty dict after verifying header + manifest."""
    manifest = info(path)
    if manifest.get("tensors"):
        raise NotImplementedError("riskytensor Phase-0 cannot load tensors yet")
    return {}

def verify(path):
    """Re-hash and print OK / FAIL for the manifest."""
    try:
        info(path)
        print(f"OK  {path}")
        return True
    except ValueError as e:
        print(f"FAIL  {path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -I riskytensor.py info|verify|save-empty <file>")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "info" and len(sys.argv) == 3:
        print(json.dumps(info(sys.argv[2]), indent=2))
    elif cmd == "verify" and len(sys.argv) == 3:
        sys.exit(0 if verify(sys.argv[2]) else 1)
    elif cmd == "save-empty" and len(sys.argv) == 3:
        n = save({}, sys.argv[2])
        print(f"wrote {n} bytes")
    else:
        print("Usage: python3 -I riskytensor.py info|verify|save-empty <file>")
        sys.exit(1)
